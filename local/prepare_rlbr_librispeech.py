#!/usr/bin/env python3
"""Prepare contextual SFT/RLBR JSONL from LibriSpeech and Rare5k files.

The evaluation protocol follows the contextual-biasing convention used by the
RLBR paper: the 5,000 most frequent training words are common, all remaining
training words form the rare-word pool.  Official Rare5k TSVs are preferred for
fixed test lists.  Otherwise equivalent lists are generated deterministically.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.rlbr_utils import build_biasing_prompt, normalize_reward_text  # noqa: E402


DEFAULT_TRAIN_SUBSETS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEFAULT_DEV_SUBSETS = ["dev-clean", "dev-other"]
DEFAULT_TEST_SUBSETS = ["test-clean", "test-other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare LibriSpeech for the RLBR reproduction")
    parser.add_argument("--librispeech_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_subsets", nargs="+", default=DEFAULT_TRAIN_SUBSETS)
    parser.add_argument("--dev_subsets", nargs="+", default=DEFAULT_DEV_SUBSETS)
    parser.add_argument("--test_subsets", nargs="+", default=DEFAULT_TEST_SUBSETS)
    parser.add_argument("--common_word_count", type=int, default=5000)
    parser.add_argument(
        "--biasing_benchmark_root",
        type=Path,
        default=None,
        help=(
            "Path to facebookresearch/fbai-speech/is21_deep_bias. When given, "
            "its official Rare5k word lists and fixed test TSVs are used."
        ),
    )
    parser.add_argument(
        "--train_num_positive",
        type=int,
        default=3,
        help="Positive reference words sampled per training item. RLBR does not report this value.",
    )
    parser.add_argument(
        "--train_distractors",
        nargs="+",
        type=int,
        default=[100, 500, 1000],
        help=(
            "One listed size is sampled per training item. RLBR does not report "
            "this distribution."
        ),
    )
    parser.add_argument("--eval_bias_sizes", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument(
        "--eval_distractors",
        nargs="+",
        type=int,
        default=None,
        help="Deprecated alias for --eval_bias_sizes.",
    )
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--seed", type=int, default=66)
    return parser.parse_args()


def resolve_corpus_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    nested = path / "LibriSpeech"
    if nested.is_dir():
        return nested
    return path


def transcript_words(text: str) -> List[str]:
    return normalize_reward_text(text, keep_bias_markers=False).split()


def find_audio(transcript_path: Path, utterance_id: str) -> Path:
    for suffix in (".flac", ".wav"):
        candidate = transcript_path.parent / f"{utterance_id}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No .flac or .wav found for {utterance_id} next to {transcript_path}"
    )


def read_subset(corpus_root: Path, subset: str) -> List[Dict[str, str]]:
    subset_root = corpus_root / subset
    if not subset_root.is_dir():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_root}")

    rows: List[Dict[str, str]] = []
    for transcript_path in sorted(subset_root.rglob("*.trans.txt")):
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed transcript at {transcript_path}:{line_number}: {line!r}"
                    )
                utterance_id, reference = parts
                rows.append(
                    {
                        "text_id": utterance_id,
                        "audio": str(find_audio(transcript_path, utterance_id)),
                        "reference": reference.strip(),
                        "subset": subset,
                    }
                )
    if not rows:
        raise RuntimeError(f"No LibriSpeech transcripts found under {subset_root}")
    return rows


def resolve_biasing_benchmark_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    nested = path / "is21_deep_bias"
    if nested.is_dir():
        path = nested
    required = [
        path / "words" / "common_words_5k.txt",
        path / "words" / "all_rare_words.txt",
        path / "words" / "all_words.count.txt",
        path / "ref",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete is21_deep_bias benchmark directory; missing: "
            + ", ".join(missing)
        )
    return path


def read_word_list(path: Path) -> List[str]:
    words = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip().split("\t", maxsplit=1)[0]
            if word:
                words.append(word)
    if not words:
        raise RuntimeError(f"No words found in {path}")
    return words


def read_biasing_reference(path: Path) -> List[Dict[str, object]]:
    """Read the four-column TSV released with the Rare5k benchmark."""

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 4:
                raise ValueError(
                    f"Expected four TSV columns at {path}:{line_number}, got {len(columns)}"
                )
            text_id, reference, positive_json, bias_list_json = columns
            try:
                positives = json.loads(positive_json)
                bias_list = json.loads(bias_list_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid list JSON at {path}:{line_number}") from exc
            if not isinstance(positives, list) or not isinstance(bias_list, list):
                raise ValueError(f"Bias columns must be JSON lists at {path}:{line_number}")
            if not set(positives).issubset(set(bias_list)):
                raise ValueError(
                    f"Bias list does not contain every positive at {path}:{line_number}"
                )
            rows.append(
                {
                    "text_id": text_id,
                    "reference": reference,
                    "bias_words": positives,
                    "bias_list": bias_list,
                }
            )
    if not rows:
        raise RuntimeError(f"No benchmark rows found in {path}")
    return rows


def unique_in_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def sample_without_reference(
    pool: Sequence[str], reference_words: Sequence[str], count: int, rng: random.Random
) -> List[str]:
    excluded = set(reference_words)
    candidates = [word for word in pool if word not in excluded]
    if count > len(candidates):
        raise ValueError(
            f"Requested {count} distractors but only {len(candidates)} candidates are available"
        )
    return rng.sample(candidates, count)


def mark_reference(reference: str, bias_words: Sequence[str]) -> str:
    bias_set = set(bias_words)
    marked = []
    for token in reference.split():
        normalized = transcript_words(token)
        key = normalized[0] if len(normalized) == 1 else ""
        marked.append(f"*{token}*" if key in bias_set else token)
    return " ".join(marked)


def make_output_row_from_list(
    source: Dict[str, str],
    positives: Sequence[str],
    bias_list: Sequence[str],
    language: str,
    condition: str,
) -> Dict[str, object]:
    positives = unique_in_order(positives)
    bias_list = list(bias_list)
    marked_reference = mark_reference(source["reference"], positives)
    return {
        "text_id": source["text_id"],
        "audio": source["audio"],
        "prompt": build_biasing_prompt(bias_list),
        "text": f"language {language}<asr_text>{marked_reference}",
        "reference": source["reference"],
        "marked_reference": marked_reference,
        "bias_words": list(positives),
        "bias_list": bias_list,
        "bias_list_size": len(bias_list),
        "distractor_count": len([word for word in bias_list if word not in positives]),
        "condition": condition,
        "subset": source["subset"],
    }


def make_sampled_output_row(
    source: Dict[str, str],
    positives: Sequence[str],
    distractors: Sequence[str],
    language: str,
    rng: random.Random,
    condition: str,
) -> Dict[str, object]:
    bias_list = list(unique_in_order(positives)) + list(distractors)
    rng.shuffle(bias_list)
    return make_output_row_from_list(
        source=source,
        positives=positives,
        bias_list=bias_list,
        language=language,
        condition=condition,
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"[info] wrote {count} rows: {path}")
    return count


def stable_rng(seed: int, *parts: object) -> random.Random:
    return random.Random(":".join([str(seed), *(str(part) for part in parts)]))


def main() -> None:
    args = parse_args()
    if args.eval_distractors is not None:
        print("[warning] --eval_distractors is deprecated; use --eval_bias_sizes")
        args.eval_bias_sizes = [size for size in args.eval_distractors if size > 0]
    if args.common_word_count < 0:
        raise ValueError("--common_word_count must be non-negative")
    if args.train_num_positive <= 0:
        raise ValueError("--train_num_positive must be positive")
    if not args.train_distractors or min(args.train_distractors) < 0:
        raise ValueError("--train_distractors must contain non-negative values")
    if not args.eval_bias_sizes or min(args.eval_bias_sizes) <= 0:
        raise ValueError("--eval_bias_sizes must contain positive values")

    corpus_root = resolve_corpus_root(args.librispeech_root)
    train_by_subset = {name: read_subset(corpus_root, name) for name in args.train_subsets}
    train_rows = [row for name in args.train_subsets for row in train_by_subset[name]]

    frequencies = Counter(
        word for row in train_rows for word in transcript_words(row["reference"])
    )
    ranked_vocabulary = [
        word
        for word, _ in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    benchmark_root = None
    if args.biasing_benchmark_root is not None:
        if args.common_word_count != 5000:
            raise ValueError(
                "The official Rare5k files require --common_word_count 5000"
            )
        benchmark_root = resolve_biasing_benchmark_root(args.biasing_benchmark_root)
        common_words = set(read_word_list(benchmark_root / "words/common_words_5k.txt"))
        rare_pool = read_word_list(benchmark_root / "words/all_rare_words.txt")
        training_vocabulary = read_word_list(
            benchmark_root / "words/all_words.count.txt"
        )
    else:
        common_words = set(ranked_vocabulary[: args.common_word_count])
        rare_pool = [word for word in ranked_vocabulary if word not in common_words]
        training_vocabulary = ranked_vocabulary
    rare_words = set(rare_pool)
    if max([*args.train_distractors, *args.eval_bias_sizes]) > len(rare_pool):
        raise ValueError(
            "The rare-word pool is smaller than the requested distractor list. "
            "Reduce list sizes or --common_word_count."
        )

    def training_items() -> Iterable[Dict[str, object]]:
        for row in train_rows:
            words = unique_in_order(transcript_words(row["reference"]))
            rng = stable_rng(args.seed, "train", row["text_id"])
            positives = rng.sample(words, min(args.train_num_positive, len(words)))
            distractor_count = rng.choice(args.train_distractors)
            distractors = sample_without_reference(
                training_vocabulary, words, distractor_count, rng
            )
            yield make_sampled_output_row(
                row, positives, distractors, args.language, rng, "train"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: Dict[str, int] = {}
    split_counts["train"] = write_jsonl(args.output_dir / "train.jsonl", training_items())

    def write_generated_evaluation(subset: str) -> None:
        source_rows = read_subset(corpus_root, subset)
        short_name = subset.replace("-", "_")
        positives_by_id = {
            row["text_id"]: [
                word
                for word in unique_in_order(transcript_words(row["reference"]))
                if word in rare_words
            ]
            for row in source_rows
        }
        local_name = f"{short_name}_local"
        split_counts[local_name] = write_jsonl(
            args.output_dir / f"{local_name}.jsonl",
            (
                make_output_row_from_list(
                    row,
                    positives_by_id[row["text_id"]],
                    positives_by_id[row["text_id"]],
                    args.language,
                    "local",
                )
                for row in source_rows
            ),
        )

        for bias_list_size in args.eval_bias_sizes:
            def evaluation_items() -> Iterable[Dict[str, object]]:
                for row in source_rows:
                    words = unique_in_order(transcript_words(row["reference"]))
                    positives = positives_by_id[row["text_id"]]
                    distractor_count = bias_list_size - len(positives)
                    if distractor_count < 0:
                        raise ValueError(
                            f"{row['text_id']} has more positives than list size "
                            f"{bias_list_size}"
                        )
                    rng = stable_rng(args.seed, subset, bias_list_size, row["text_id"])
                    distractors = sample_without_reference(
                        rare_pool, words, distractor_count, rng
                    )
                    yield make_sampled_output_row(
                        row, positives, distractors, args.language, rng, "global"
                    )

            name = f"{short_name}_n{bias_list_size}"
            split_counts[name] = write_jsonl(
                args.output_dir / f"{name}.jsonl", evaluation_items()
            )

    def write_official_test_evaluation(subset: str) -> None:
        source_rows = read_subset(corpus_root, subset)
        source_by_id = {row["text_id"]: row for row in source_rows}
        official_by_size: Dict[int, List[Dict[str, object]]] = {}
        for bias_list_size in args.eval_bias_sizes:
            reference_path = (
                benchmark_root / "ref" / f"{subset}.biasing_{bias_list_size}.tsv"
            )
            official_rows = read_biasing_reference(reference_path)
            if len(official_rows) != len(source_rows):
                raise ValueError(
                    f"{reference_path} has {len(official_rows)} rows, but {subset} "
                    f"contains {len(source_rows)} utterances"
                )
            for benchmark_row in official_rows:
                text_id = str(benchmark_row["text_id"])
                if text_id not in source_by_id:
                    raise KeyError(f"Benchmark utterance {text_id} is absent from {subset}")
                if normalize_reward_text(str(benchmark_row["reference"]), False) != \
                        normalize_reward_text(source_by_id[text_id]["reference"], False):
                    raise ValueError(f"Reference mismatch for benchmark utterance {text_id}")
                if len(benchmark_row["bias_list"]) != bias_list_size:
                    raise ValueError(
                        f"Official list for {text_id} has {len(benchmark_row['bias_list'])} "
                        f"items, expected {bias_list_size}"
                    )
            official_by_size[bias_list_size] = official_rows

        local_rows = official_by_size[min(args.eval_bias_sizes)]
        local_positives = {
            str(row["text_id"]): row["bias_words"] for row in local_rows
        }
        for bias_list_size, official_rows in official_by_size.items():
            for row in official_rows:
                text_id = str(row["text_id"])
                if row["bias_words"] != local_positives[text_id]:
                    raise ValueError(
                        f"Rare reference words differ across official list sizes "
                        f"for {text_id} at N={bias_list_size}"
                    )
        short_name = subset.replace("-", "_")
        local_name = f"{short_name}_local"
        split_counts[local_name] = write_jsonl(
            args.output_dir / f"{local_name}.jsonl",
            (
                make_output_row_from_list(
                    source_by_id[str(row["text_id"])],
                    row["bias_words"],
                    row["bias_words"],
                    args.language,
                    "local",
                )
                for row in local_rows
            ),
        )

        for bias_list_size, official_rows in official_by_size.items():
            name = f"{short_name}_n{bias_list_size}"
            split_counts[name] = write_jsonl(
                args.output_dir / f"{name}.jsonl",
                (
                    make_output_row_from_list(
                        source_by_id[str(row["text_id"])],
                        row["bias_words"],
                        row["bias_list"],
                        args.language,
                        "global",
                    )
                    for row in official_rows
                ),
            )

    for subset in args.dev_subsets:
        write_generated_evaluation(subset)
    for subset in args.test_subsets:
        if benchmark_root is None:
            write_generated_evaluation(subset)
        else:
            write_official_test_evaluation(subset)

    manifest = {
        "paper": (
            "RLBR: Reinforcement Learning with Biasing Rewards for Contextual "
            "Speech Large Language Models"
        ),
        "corpus_root": str(corpus_root),
        "biasing_list_source": (
            "official_is21_deep_bias" if benchmark_root is not None else "generated"
        ),
        "biasing_benchmark_root": str(benchmark_root) if benchmark_root else None,
        "seed": args.seed,
        "common_word_count": args.common_word_count,
        "vocabulary_size": len(training_vocabulary),
        "rare_vocabulary_size": len(rare_pool),
        "train_num_positive": args.train_num_positive,
        "train_distractors": args.train_distractors,
        "eval_bias_sizes": args.eval_bias_sizes,
        "splits": split_counts,
        "paper_unreported_implementation_choices": [
            "train_num_positive",
            "training distractor-count distribution",
        ],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[info] wrote manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
