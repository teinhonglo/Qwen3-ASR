#!/usr/bin/env python3
"""Run Qwen3-ASR with the global hotword list and evaluate the benchmark.

The benchmark keeps two different hotword files:

* ``all_hotwords.json`` is the global candidate list supplied to every audio.
* ``hotwords.json`` contains per-audio ground truth and is used only to obtain the
  expected audio IDs. The benchmark's ``evaluate.py`` reads it for scoring.

Using the global list avoids leaking each recording's exact target words during
inference. Candidate transcripts are written in the layout expected by the
benchmark: ``<output_dir>/candidate/<audio_id>/transcription.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-ASR with global hotwords, then evaluate the result."
    )
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        required=True,
        help=(
            "Extracted hotword_benchmark directory containing audio/, "
            "all_hotwords.json, hotwords.json, pseudo_transcripts.json, and evaluate.py"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Experiment directory. Defaults to "
            "exp/hotword_benchmark/<model_name>/global_hotwords"
        ),
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Hugging Face model ID or local checkpoint path",
    )
    parser.add_argument(
        "--hotwords_file",
        type=Path,
        default=None,
        help="Global JSON list of candidate hotwords (default: <benchmark_dir>/all_hotwords.json)",
    )
    parser.add_argument(
        "--evaluation_script",
        type=Path,
        default=None,
        help="Benchmark evaluator (default: <benchmark_dir>/evaluate.py)",
    )
    parser.add_argument(
        "--report_name",
        default="report.xlsx",
        help="Evaluation report filename inside output_dir (default: report.xlsx)",
    )
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Forced Qwen3-ASR language. Pass an empty string to enable detection.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype (default: bfloat16 on CUDA, float32 on CPU)",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="Transformers attention implementation (default: auto)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens per model call (default: 4096 for long recordings)",
    )
    parser.add_argument(
        "--max_inference_batch_size",
        type=int,
        default=1,
        help="Maximum number of Qwen audio chunks inferred together (default: 1)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="First stage to run: 1=inference, 2=evaluation (default: 1)",
    )
    parser.add_argument(
        "--stop_stage",
        type=int,
        default=2,
        help="Last stage to run: 1=inference, 2=evaluation (default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run audio IDs that already have transcription.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def audio_id_sort_key(audio_id: str) -> tuple[int, int | str]:
    return (0, int(audio_id)) if audio_id.isdigit() else (1, audio_id)


def model_slug(model_path: str) -> str:
    name = Path(model_path.rstrip("/")).name or "qwen3-asr"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).lower()


def validate_benchmark(
    benchmark_dir: Path,
    hotwords_file: Path,
    evaluation_script: Path,
) -> tuple[list[str], list[str]]:
    required_paths = [
        benchmark_dir / "audio",
        benchmark_dir / "hotwords.json",
        benchmark_dir / "pseudo_transcripts.json",
        hotwords_file,
        evaluation_script,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark paths:\n  " + "\n  ".join(missing))

    ground_truth = read_json(benchmark_dir / "hotwords.json")
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("hotwords.json must be a non-empty object keyed by audio ID")

    audio_ids = sorted((str(key) for key in ground_truth), key=audio_id_sort_key)
    missing_audio = [
        str(benchmark_dir / "audio" / f"{audio_id}.wav")
        for audio_id in audio_ids
        if not (benchmark_dir / "audio" / f"{audio_id}.wav").is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(
            "Missing benchmark audio files:\n  " + "\n  ".join(missing_audio)
        )

    hotwords = read_json(hotwords_file)
    if not isinstance(hotwords, list) or not hotwords:
        raise ValueError("The global hotwords file must contain a non-empty JSON list")
    if not all(isinstance(word, str) and word.strip() for word in hotwords):
        raise ValueError("Every global hotword must be a non-empty string")

    normalized = [word.strip() for word in hotwords]
    if len(set(normalized)) != len(normalized):
        raise ValueError("The global hotwords file contains duplicate entries")

    return audio_ids, normalized


def resolve_runtime(args: argparse.Namespace):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise RuntimeError(
            "Inference dependencies are unavailable. Install this repository first, "
            "for example with `pip install -e .`."
        ) from error

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device == "cuda" else "float32"
    dtype = getattr(torch, dtype_name)

    attention = args.attn_implementation
    if attention == "auto":
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attention = "flash_attention_2"
            except ImportError:
                attention = "sdpa"
        else:
            attention = "sdpa"

    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device,
        attn_implementation=attention,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    return model, device, dtype_name, attention


def run_inference(
    args: argparse.Namespace,
    benchmark_dir: Path,
    candidate_dir: Path,
    audio_ids: list[str],
    hotwords: list[str],
    hotwords_file: Path,
) -> None:
    pending_ids = []
    for audio_id in audio_ids:
        transcription_path = candidate_dir / audio_id / "transcription.json"
        if args.overwrite or not transcription_path.is_file():
            pending_ids.append(audio_id)

    if not pending_ids:
        print(f"[stage 1] All {len(audio_ids)} transcripts already exist; skipping inference.")
        return

    # Qwen3-ASR documents context as plain biasing text. Newlines keep the 139
    # entries separate without adding task instructions that could alter decoding.
    context = "\n".join(hotwords)
    model, device, dtype_name, attention = resolve_runtime(args)
    language = args.language.strip() or None

    print(f"[stage 1] Audio files : {len(audio_ids)} ({len(pending_ids)} pending)")
    print(f"[stage 1] Hotwords    : {len(hotwords)} from {hotwords_file}")
    print(f"[stage 1] Model       : {args.model_path}")
    print(f"[stage 1] Runtime     : device={device}, dtype={dtype_name}, attention={attention}")

    for index, audio_id in enumerate(pending_ids, start=1):
        audio_path = benchmark_dir / "audio" / f"{audio_id}.wav"
        transcription_path = candidate_dir / audio_id / "transcription.json"
        print(f"[{index}/{len(pending_ids)}] Transcribing {audio_id}: {audio_path}")

        results = model.transcribe(
            audio=str(audio_path),
            context=context,
            language=language,
            return_time_stamps=False,
        )
        if len(results) != 1:
            raise RuntimeError(
                f"Expected one transcription for audio {audio_id}, got {len(results)}"
            )

        write_json_atomic(transcription_path, {"text": results[0].text or ""})

    config = {
        "benchmark_dir": str(benchmark_dir),
        "candidate_dir": str(candidate_dir),
        "model_path": args.model_path,
        "language": language,
        "device": device,
        "dtype": dtype_name,
        "attn_implementation": attention,
        "max_new_tokens": args.max_new_tokens,
        "max_inference_batch_size": args.max_inference_batch_size,
        "hotwords_file": str(hotwords_file),
        "hotword_count": len(hotwords),
        "hotwords_sha256": hashlib.sha256(
            hotwords_file.read_bytes()
        ).hexdigest(),
        "context_format": "newline-separated global hotword list",
    }
    write_json_atomic(candidate_dir.parent / "run_config.json", config)


def validate_candidates(candidate_dir: Path, audio_ids: list[str]) -> None:
    missing = [
        audio_id
        for audio_id in audio_ids
        if not (candidate_dir / audio_id / "transcription.json").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Evaluation requires a complete candidate set. Missing transcription.json "
            f"for {len(missing)} audio IDs: {', '.join(missing)}"
        )


def run_evaluation(
    evaluation_script: Path,
    candidate_dir: Path,
    report_path: Path,
) -> None:
    command = [
        sys.executable,
        str(evaluation_script),
        "--candidate",
        str(candidate_dir),
        "--output",
        str(report_path),
    ]
    print("[stage 2] Running benchmark evaluator:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.stage not in {1, 2} or args.stop_stage not in {1, 2}:
        raise ValueError("--stage and --stop_stage must be 1 or 2")
    if args.stage > args.stop_stage:
        raise ValueError("--stage cannot be greater than --stop_stage")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.max_inference_batch_size == 0:
        raise ValueError("--max_inference_batch_size cannot be zero")

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    hotwords_file = (
        args.hotwords_file.expanduser().resolve()
        if args.hotwords_file
        else benchmark_dir / "all_hotwords.json"
    )
    evaluation_script = (
        args.evaluation_script.expanduser().resolve()
        if args.evaluation_script
        else benchmark_dir / "evaluate.py"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            Path("exp")
            / "hotword_benchmark"
            / model_slug(args.model_path)
            / "global_hotwords"
        ).resolve()
    )
    candidate_dir = output_dir / "candidate"
    report_path = output_dir / args.report_name

    audio_ids, hotwords = validate_benchmark(
        benchmark_dir=benchmark_dir,
        hotwords_file=hotwords_file,
        evaluation_script=evaluation_script,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage <= 1 <= args.stop_stage:
        run_inference(
            args=args,
            benchmark_dir=benchmark_dir,
            candidate_dir=candidate_dir,
            audio_ids=audio_ids,
            hotwords=hotwords,
            hotwords_file=hotwords_file,
        )

    if args.stage <= 2 <= args.stop_stage:
        validate_candidates(candidate_dir, audio_ids)
        run_evaluation(
            evaluation_script=evaluation_script,
            candidate_dir=candidate_dir,
            report_path=report_path,
        )


if __name__ == "__main__":
    main()
