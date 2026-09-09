#!/usr/bin/env python3
"""Text normalization, RLBR rewards, and contextual-ASR error metrics.

The reward follows Eq. (4) of Ren et al. (ICASSP 2026):

    r = -(ED(reference, hypothesis) + lambda * ED_b(reference, hypothesis))

``ED_b`` aligns every marked bias term in the reference to its best matching
contiguous span in the hypothesis.  The special ``*`` markers are deliberately
kept in reward calculation because the paper treats a missing marker as a
formatting error (Fig. 1).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_QWEN_ASR_PREFIX_RE = re.compile(
    r"^\s*language\s+[^<\n]+<asr_text>", flags=re.IGNORECASE
)
_NON_REWARD_CHAR_RE = re.compile(r"[^\w'*\s]+", flags=re.UNICODE)
BIASING_PROMPT_TEMPLATE = (
    "Transcribe the audio clip into text with extra attention to the following words: "
    "{bias_list}"
)


def build_biasing_prompt(words: Sequence[str]) -> str:
    """Format a local contextual word list for Qwen3-ASR."""

    formatted = ", ".join(f"*{word}*" for word in words)
    return BIASING_PROMPT_TEMPLATE.format(bias_list=f"[{formatted}]")


def resolve_evaluation_prompt(row: Dict[str, Any], prompt_mode: str) -> str:
    """Select a leakage-safe direct-ASR or local-biasing evaluation prompt."""

    if prompt_mode == "none":
        return ""
    if prompt_mode != "biasing":
        raise ValueError(
            f"Unsupported prompt_mode={prompt_mode!r}; expected 'none' or 'biasing'"
        )

    prompt = str(row.get("prompt", "") or "").strip()
    if prompt:
        return prompt
    if "bias_list" not in row:
        raise KeyError(
            "Biasing evaluation requires either a non-empty `prompt` or `bias_list`"
        )

    bias_list = row["bias_list"]
    if isinstance(bias_list, str):
        bias_list = [bias_list]
    return build_biasing_prompt([str(word) for word in bias_list])


def strip_qwen_asr_prefix(text: str) -> str:
    """Remove Qwen3-ASR's generated language prefix, if present."""

    return _QWEN_ASR_PREFIX_RE.sub("", str(text or ""), count=1).strip()


def normalize_reward_text(text: str, keep_bias_markers: bool = True) -> str:
    """Normalize English ASR text while optionally retaining ``*`` markers."""

    text = unicodedata.normalize("NFKC", strip_qwen_asr_prefix(text)).casefold()
    if not keep_bias_markers:
        text = text.replace("*", "")
    text = _NON_REWARD_CHAR_RE.sub(" ", text)
    return " ".join(text.split())


def reward_units(text: str, edit_level: str) -> List[str]:
    normalized = normalize_reward_text(text, keep_bias_markers=True)
    if edit_level == "word":
        return normalized.split()
    if edit_level == "char":
        return list(normalized.replace(" ", ""))
    raise ValueError(f"Unsupported edit_level={edit_level!r}; expected 'word' or 'char'")


def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return unit-cost Levenshtein distance using O(len(hypothesis)) memory."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def best_matching_span_distance(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> int:
    """Distance from ``reference`` to its best contiguous span in ``hypothesis``.

    Initializing the first DP row to zero gives free deletion of a hypothesis
    prefix.  Taking the minimum over the final row gives free deletion of a
    suffix, which is the standard semi-global alignment used for best-span
    matching.
    """

    if not reference:
        return 0
    if not hypothesis:
        return len(reference)

    previous = [0] * (len(hypothesis) + 1)
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return min(previous)


def _normalized_bias_terms(reference: str, bias_words: Iterable[str]) -> List[str]:
    terms: List[str] = []
    seen = set()
    for word in bias_words or []:
        normalized = normalize_reward_text(str(word), keep_bias_markers=False)
        if normalized and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)

    if terms:
        return terms

    # A prepared RLBR reference is self-describing, which is useful for reward
    # verification and for externally produced JSONL without ``bias_words``.
    for marked in re.findall(r"\*([^*]+)\*", strip_qwen_asr_prefix(reference)):
        normalized = normalize_reward_text(marked, keep_bias_markers=False)
        if normalized and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


@dataclass(frozen=True)
class RLBRReward:
    reward: float
    edit_distance: int
    bias_edit_distance: int


def compute_rlbr_reward(
    reference: str,
    hypothesis: str,
    bias_words: Iterable[str],
    bias_weight: float = 5.0,
    edit_level: str = "char",
) -> RLBRReward:
    """Compute the unnormalized edit-distance reward reported in the paper."""

    reference_items = reward_units(reference, edit_level)
    hypothesis_items = reward_units(hypothesis, edit_level)
    overall_distance = levenshtein_distance(reference_items, hypothesis_items)

    bias_distance = 0
    for term in _normalized_bias_terms(reference, bias_words):
        marked_term = f"*{term}*"
        term_items = reward_units(marked_term, edit_level)
        bias_distance += best_matching_span_distance(term_items, hypothesis_items)

    reward = -(overall_distance + float(bias_weight) * bias_distance)
    return RLBRReward(float(reward), overall_distance, bias_distance)


def align_words(reference: Sequence[str], hypothesis: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Return the weighted alignment used by the public Rare5k scorer.

    Each item is ``(operation, reference_index, hypothesis_index)``.  An absent
    side is represented by ``-1``.  Following ``is21_deep_bias/score.py``,
    insertion/deletion cost 3, substitution costs 4, and ties retain the
    substitution path before insertion and deletion.
    """

    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    backtrace = [["match"] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = 3 * i
        backtrace[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = 3 * j
        backtrace[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            is_match = reference[i - 1] == hypothesis[j - 1]
            best = dp[i - 1][j - 1] + (0 if is_match else 4)
            operation = "match" if is_match else "substitute"
            insertion = dp[i][j - 1] + 3
            if insertion < best:
                best = insertion
                operation = "insert"
            deletion = dp[i - 1][j] + 3
            if deletion < best:
                best = deletion
                operation = "delete"
            dp[i][j] = best
            backtrace[i][j] = operation

    alignment: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        operation = backtrace[i][j]
        if operation in {"match", "substitute"}:
            alignment.append((operation, i - 1, j - 1))
            i -= 1
            j -= 1
        elif operation == "insert":
            alignment.append((operation, -1, j - 1))
            j -= 1
        else:
            alignment.append(("delete", i - 1, -1))
            i -= 1

    alignment.reverse()
    return alignment


def _plain_words(text: str) -> List[str]:
    return normalize_reward_text(text, keep_bias_markers=False).split()


def compute_corpus_error_rates(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute WER, biasing WER (BWER), and unbiased WER (UWER).

    Substitutions and deletions inherit the class of their reference word.
    An insertion is a biasing error when the inserted hypothesis word occurs in
    the utterance's bias list, matching the public Rare5k scoring script.  This
    makes total errors equal biasing plus unbiased errors.
    """

    counts = {
        "utterances": 0,
        "reference_words": 0,
        "bias_reference_words": 0,
        "unbiased_reference_words": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "bias_errors": 0,
        "unbiased_errors": 0,
    }

    for row in rows:
        reference = _plain_words(row.get("reference", row.get("text", "")))
        hypothesis = _plain_words(row.get("hypothesis", ""))
        bias_set = {
            token
            for word in row.get("bias_words", [])
            for token in _plain_words(str(word))
        }
        bias_mask = [word in bias_set for word in reference]

        counts["utterances"] += 1
        counts["reference_words"] += len(reference)
        counts["bias_reference_words"] += sum(bias_mask)
        counts["unbiased_reference_words"] += len(reference) - sum(bias_mask)

        for operation, ref_index, hyp_index in align_words(reference, hypothesis):
            if operation == "match":
                continue
            if operation == "insert":
                counts["insertions"] += 1
                if hypothesis[hyp_index] in bias_set:
                    counts["bias_errors"] += 1
                else:
                    counts["unbiased_errors"] += 1
                continue

            if operation == "substitute":
                counts["substitutions"] += 1
            else:
                counts["deletions"] += 1

            if bias_mask[ref_index]:
                counts["bias_errors"] += 1
            else:
                counts["unbiased_errors"] += 1

    def rate(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    total_errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {
        **counts,
        "wer": rate(total_errors, counts["reference_words"]),
        "bwer": rate(counts["bias_errors"], counts["bias_reference_words"]),
        "uwer": rate(counts["unbiased_errors"], counts["unbiased_reference_words"]),
    }


def _evaluation_bias_sizes(model_name: str, bias_sizes: Sequence[int]) -> List[Optional[int]]:
    if model_name in {"baseline", "local"}:
        return [None]
    if model_name in {"global", "base", "sft", "rlbr"}:
        return list(bias_sizes)
    raise ValueError(f"Unsupported report model: {model_name}")


def _relative_reduction(baseline: float, value: float) -> Optional[float]:
    if baseline == 0.0:
        return None
    return 100.0 * (baseline - value) / baseline


def collect_evaluation_report(
    eval_root: Path,
    model_names: Sequence[str],
    bias_sizes: Sequence[int],
    splits: Sequence[str] = ("test_clean", "test_other"),
) -> Dict[str, Any]:
    """Collect Stage 3 metrics and compare RLBR against global at each N."""

    eval_root = Path(eval_root)
    results: List[Dict[str, Any]] = []
    missing: List[str] = []
    for split in splits:
        for model_name in model_names:
            for bias_size in _evaluation_bias_sizes(model_name, bias_sizes):
                condition = model_name if bias_size is None else f"{model_name}-{bias_size}"
                result_dir = split if bias_size is None else f"{split}_n{bias_size}"
                metrics_path = eval_root / model_name / result_dir / "metrics.json"
                if not metrics_path.is_file():
                    missing.append(str(metrics_path))
                    continue
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                results.append(
                    {
                        "split": split,
                        "system": model_name,
                        "condition": condition,
                        "bias_size_n": bias_size,
                        "wer": float(metrics["wer"]),
                        "bwer": float(metrics["bwer"]),
                        "uwer": float(metrics["uwer"]),
                        "utterances": int(metrics["utterances"]),
                        "decode_failures": int(metrics.get("decode_failures", 0)),
                        "decode_failure_rate": float(
                            metrics.get("decode_failure_rate", 0.0)
                        ),
                        "metrics_path": str(metrics_path),
                    }
                )

    if missing:
        raise FileNotFoundError(
            "Cannot create a complete evaluation report; missing metrics:\n"
            + "\n".join(missing)
        )

    by_key = {
        (row["system"], row["split"], row["bias_size_n"]): row for row in results
    }
    comparisons: List[Dict[str, Any]] = []
    if "global" in model_names and "rlbr" in model_names:
        for split in splits:
            for bias_size in bias_sizes:
                global_row = by_key[("global", split, bias_size)]
                rlbr_row = by_key[("rlbr", split, bias_size)]
                comparison: Dict[str, Any] = {
                    "split": split,
                    "bias_size_n": bias_size,
                }
                for metric in ("wer", "bwer", "uwer"):
                    global_value = float(global_row[metric])
                    rlbr_value = float(rlbr_row[metric])
                    comparison[f"global_{metric}"] = global_value
                    comparison[f"rlbr_{metric}"] = rlbr_value
                    comparison[f"{metric}_absolute_change"] = (
                        rlbr_value - global_value
                    )
                    comparison[f"{metric}_relative_reduction"] = _relative_reduction(
                        global_value, rlbr_value
                    )
                comparisons.append(comparison)

    return {
        "eval_root": str(eval_root),
        "models": list(model_names),
        "bias_sizes": list(bias_sizes),
        "splits": list(splits),
        "results": results,
        "rlbr_vs_global": comparisons,
    }


def _format_report_number(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def write_evaluation_report(
    eval_root: Path,
    output_dir: Path,
    model_names: Sequence[str],
    bias_sizes: Sequence[int],
    splits: Sequence[str] = ("test_clean", "test_other"),
) -> Dict[str, str]:
    """Write machine-readable and Markdown summaries for Stage 3 evaluation."""

    report = collect_evaluation_report(eval_root, model_names, bias_sizes, splits)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    csv_path = output_dir / "summary.csv"
    result_fields = [
        "split",
        "system",
        "condition",
        "bias_size_n",
        "wer",
        "bwer",
        "uwer",
        "utterances",
        "decode_failures",
        "decode_failure_rate",
        "metrics_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(report["results"])

    markdown_path = output_dir / "summary.md"
    lines = [
        "# RLBR Evaluation Report",
        "",
        "## Results",
        "",
        "| Split | System | Bias condition | WER (%) | BWER (%) | UWER (%) | Decode failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        condition = "none" if row["system"] == "baseline" else (
            "local" if row["bias_size_n"] is None else f"N={row['bias_size_n']}"
        )
        lines.append(
            f"| {row['split']} | {row['system']} | {condition} | "
            f"{row['wer']:.3f} | {row['bwer']:.3f} | {row['uwer']:.3f} | "
            f"{row['decode_failures']} |"
        )

    lines.extend(
        [
            "",
            "## RLBR vs. Global",
            "",
            "Absolute change is RLBR minus Global, so a negative value is better. "
            "Relative reduction is positive when RLBR reduces the error rate.",
            "",
            "| Split | N | WER change | WER reduction (%) | BWER change | "
            "BWER reduction (%) | UWER change | UWER reduction (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if report["rlbr_vs_global"]:
        for row in report["rlbr_vs_global"]:
            lines.append(
                f"| {row['split']} | {row['bias_size_n']} | "
                f"{row['wer_absolute_change']:.3f} | "
                f"{_format_report_number(row['wer_relative_reduction'])} | "
                f"{row['bwer_absolute_change']:.3f} | "
                f"{_format_report_number(row['bwer_relative_reduction'])} | "
                f"{row['uwer_absolute_change']:.3f} | "
                f"{_format_report_number(row['uwer_relative_reduction'])} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "markdown": str(markdown_path),
        "csv": str(csv_path),
        "json": str(json_path),
    }
