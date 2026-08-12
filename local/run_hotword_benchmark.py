#!/usr/bin/env python3
"""Run Qwen3-ASR with or without global hotwords and evaluate the benchmark.

The benchmark keeps two different hotword files:

* ``all_hotwords.json`` is the global candidate list supplied verbatim to every
  audio in the hotword condition.
* ``hotwords.json`` contains per-audio ground truth and is used only to obtain the
  expected audio IDs. The benchmark's ``evaluate.py`` reads it for scoring.

In global-hotword mode, using the same list for every recording avoids leaking
each recording's exact target words during inference. Candidate transcripts are
written in the layout and segment schema expected by the benchmark:
``<output_dir>/candidate/<audio_id>/transcription.json`` contains
``[{"text": "..."}]``.

The runner also records real-time factor (RTF), peak GPU memory, and peak process
resident memory in ``inference_metrics.json``. An optional explicit hotword
prediction file can be forwarded to ``evaluate.py`` for Precision/Recall/F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any


EXPECTED_AUDIO_COUNT = 71
EXPECTED_HOTWORD_TYPE_COUNT = 139


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-ASR with or without global hotwords, then evaluate."
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
            "exp/hotword_benchmark/<model_name>/<context_mode>"
        ),
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Hugging Face model ID or local checkpoint path",
    )
    parser.add_argument(
        "--context_mode",
        choices=["none", "global"],
        default="global",
        help=(
            "none: no hotword context; global: supply all_hotwords.json to every "
            "recording (default: global)"
        ),
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
        "--predicted_keywords",
        type=Path,
        default=None,
        help=(
            "Optional per-audio detected-hotword JSON passed to evaluate.py for "
            "keyword Precision/Recall/F1"
        ),
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
    if len(audio_ids) != EXPECTED_AUDIO_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_AUDIO_COUNT} audio IDs in hotwords.json, "
            f"found {len(audio_ids)}"
        )

    expected_audio_ids = set(audio_ids)
    actual_audio_ids = {path.stem for path in (benchmark_dir / "audio").glob("*.wav")}
    missing_audio_ids = sorted(
        expected_audio_ids - actual_audio_ids, key=audio_id_sort_key
    )
    extra_audio_ids = sorted(
        actual_audio_ids - expected_audio_ids, key=audio_id_sort_key
    )
    if missing_audio_ids or extra_audio_ids:
        raise ValueError(
            "audio/ and hotwords.json contain different audio IDs. "
            f"Missing WAVs: {missing_audio_ids}; extra WAVs: {extra_audio_ids}"
        )

    ground_truth_union = set()
    per_audio_case_variants = []
    for audio_id in audio_ids:
        keywords = ground_truth[audio_id]
        if not isinstance(keywords, list):
            raise ValueError(f"hotwords.json[{audio_id!r}] must be a JSON list")
        if not all(isinstance(word, str) and word.strip() for word in keywords):
            raise ValueError(
                f"hotwords.json[{audio_id!r}] contains a non-string or empty hotword"
            )
        cleaned = [word.strip() for word in keywords]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(
                f"hotwords.json[{audio_id!r}] contains exact duplicate hotwords"
            )
        folded_groups = {}
        for word in cleaned:
            folded_groups.setdefault(word.casefold(), []).append(word)
        collisions = [
            group for group in folded_groups.values() if len(group) > 1
        ]
        if collisions:
            per_audio_case_variants.append((audio_id, collisions))
        ground_truth_union.update(cleaned)

    if len(ground_truth_union) != EXPECTED_HOTWORD_TYPE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_HOTWORD_TYPE_COUNT} unique hotwords in the "
            f"hotwords.json union, found {len(ground_truth_union)}"
        )

    pseudo_transcripts = read_json(benchmark_dir / "pseudo_transcripts.json")
    if not isinstance(pseudo_transcripts, dict):
        raise ValueError("pseudo_transcripts.json must be an object keyed by audio ID")
    pseudo_ids = {str(key) for key in pseudo_transcripts}
    if pseudo_ids != expected_audio_ids:
        raise ValueError(
            "pseudo_transcripts.json and hotwords.json contain different audio IDs. "
            f"Missing pseudo transcripts: {sorted(expected_audio_ids - pseudo_ids, key=audio_id_sort_key)}; "
            f"extra pseudo transcripts: {sorted(pseudo_ids - expected_audio_ids, key=audio_id_sort_key)}"
        )

    hotwords = read_json(hotwords_file)
    if not isinstance(hotwords, list) or not hotwords:
        raise ValueError("The global hotwords file must contain a non-empty JSON list")
    if not all(isinstance(word, str) and word.strip() for word in hotwords):
        raise ValueError("Every global hotword must be a non-empty string")

    normalized = [word.strip() for word in hotwords]
    if len(set(normalized)) != len(normalized):
        raise ValueError("The global hotwords file contains exact duplicate entries")
    if len(normalized) != EXPECTED_HOTWORD_TYPE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_HOTWORD_TYPE_COUNT} global hotwords, "
            f"found {len(normalized)}"
        )
    if set(normalized) != ground_truth_union:
        print(
            "[benchmark warning] all_hotwords.json is not the exact union of "
            "hotwords.json. "
            f"Missing globally: {sorted(ground_truth_union - set(normalized))}; "
            f"extra globally: {sorted(set(normalized) - ground_truth_union)}. "
            "Inference will still use all_hotwords.json verbatim; evaluate.py "
            "will score against hotwords.json as defined by the benchmark.",
            file=sys.stderr,
        )

    global_folded_groups = {}
    for word in normalized:
        global_folded_groups.setdefault(word.casefold(), []).append(word)
    global_case_variants = [
        group for group in global_folded_groups.values() if len(group) > 1
    ]
    if global_case_variants:
        print(
            "[benchmark warning] Case-only global hotword variants are counted as "
            f"separate entries in the 139-item file: {global_case_variants}. "
            "evaluate.py matches case-insensitively.",
            file=sys.stderr,
        )
    if per_audio_case_variants:
        print(
            "[benchmark warning] Some audio IDs contain case-only target variants: "
            f"{per_audio_case_variants}. Presence-based Recall may count one textual "
            "occurrence more than once.",
            file=sys.stderr,
        )

    return audio_ids, normalized


def validate_predicted_keywords(path: Path, audio_ids: list[str]) -> None:
    predicted = read_json(path)
    if not isinstance(predicted, dict):
        raise ValueError("--predicted_keywords must contain an object keyed by audio ID")

    expected_ids = set(audio_ids)
    predicted_ids = {str(key) for key in predicted}
    if predicted_ids != expected_ids:
        raise ValueError(
            "--predicted_keywords must explicitly contain every benchmark audio ID. "
            f"Missing: {sorted(expected_ids - predicted_ids, key=audio_id_sort_key)}; "
            f"extra: {sorted(predicted_ids - expected_ids, key=audio_id_sort_key)}"
        )

    for audio_id in audio_ids:
        keywords = predicted[audio_id]
        if not isinstance(keywords, list):
            raise ValueError(
                f"predicted_keywords[{audio_id!r}] must be a JSON list"
            )
        if not all(isinstance(word, str) and word.strip() for word in keywords):
            raise ValueError(
                f"predicted_keywords[{audio_id!r}] contains an invalid hotword"
            )
        cleaned = [word.strip() for word in keywords]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(
                f"predicted_keywords[{audio_id!r}] contains exact duplicate hotwords"
            )


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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid WAV frame rate in {path}: {frame_rate}")
        return wav_file.getnframes() / frame_rate


def peak_process_rss_mib() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak_rss / divisor


def prepare_gpu_measurement(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def finish_gpu_measurement(device: str) -> tuple[float | None, float | None]:
    if device != "cuda":
        return None, None
    import torch

    torch.cuda.synchronize()
    to_mib = 1024 * 1024
    return (
        torch.cuda.max_memory_allocated() / to_mib,
        torch.cuda.max_memory_reserved() / to_mib,
    )


def build_efficiency_metrics(
    per_audio: dict[str, dict[str, Any]],
    benchmark_audio_count: int,
    model_load_seconds: float,
) -> dict[str, Any]:
    records = list(per_audio.values())
    total_audio_seconds = sum(record["audio_seconds"] for record in records)
    total_inference_seconds = sum(record["inference_seconds"] for record in records)
    gpu_allocated = [
        record["peak_gpu_allocated_mib"]
        for record in records
        if record.get("peak_gpu_allocated_mib") is not None
    ]
    gpu_reserved = [
        record["peak_gpu_reserved_mib"]
        for record in records
        if record.get("peak_gpu_reserved_mib") is not None
    ]
    process_rss = [record["peak_process_rss_mib"] for record in records]

    return {
        "benchmark_audio_count": benchmark_audio_count,
        "measured_audio_count": len(records),
        "model_load_seconds_latest_run": round(model_load_seconds, 6),
        "total_audio_seconds": round(total_audio_seconds, 6),
        "total_inference_seconds": round(total_inference_seconds, 6),
        "overall_rtf": (
            round(total_inference_seconds / total_audio_seconds, 6)
            if total_audio_seconds
            else None
        ),
        "audio_seconds_per_inference_second": (
            round(total_audio_seconds / total_inference_seconds, 6)
            if total_inference_seconds
            else None
        ),
        "peak_gpu_allocated_mib": max(gpu_allocated) if gpu_allocated else None,
        "peak_gpu_reserved_mib": max(gpu_reserved) if gpu_reserved else None,
        "peak_process_rss_mib": max(process_rss) if process_rss else None,
        "per_audio": per_audio,
    }


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

    # Qwen3-ASR documents context as plain biasing text. In global mode, newlines
    # keep the entries separate without adding task instructions that could alter
    # decoding. In none mode, the system context is empty.
    if args.context_mode == "global":
        context = "\n".join(hotwords)
        context_format = "newline-separated global hotword list"
        context_hotword_count = len(hotwords)
    else:
        context = ""
        context_format = "empty context (no hotwords)"
        context_hotword_count = 0
    model_load_start = time.perf_counter()
    model, device, dtype_name, attention = resolve_runtime(args)
    model_load_seconds = time.perf_counter() - model_load_start
    language = args.language.strip() or None
    metrics_path = candidate_dir.parent / "inference_metrics.json"
    per_audio_metrics = {}
    if metrics_path.is_file():
        existing_metrics = read_json(metrics_path)
        existing_per_audio = existing_metrics.get("per_audio", {})
        if not isinstance(existing_per_audio, dict):
            raise ValueError(f"Invalid existing metrics file: {metrics_path}")
        per_audio_metrics.update(existing_per_audio)

    print(f"[stage 1] Audio files : {len(audio_ids)} ({len(pending_ids)} pending)")
    print(f"[stage 1] Context     : {args.context_mode}")
    print(f"[stage 1] Hotword set : {len(hotwords)} from {hotwords_file}")
    print(f"[stage 1] Model       : {args.model_path}")
    print(f"[stage 1] Runtime     : device={device}, dtype={dtype_name}, attention={attention}")
    print(f"[stage 1] Model load  : {model_load_seconds:.2f} seconds")

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
        "audio_count": len(audio_ids),
        "context_mode": args.context_mode,
        "context_hotword_count": context_hotword_count,
        "context_hotwords_source": (
            "all_hotwords.json (exact file contents)"
            if args.context_mode == "global"
            else "none"
        ),
        "hotwords_file": str(hotwords_file),
        "hotword_count": len(hotwords),
        "hotwords_sha256": file_sha256(hotwords_file),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "ground_truth_sha256": file_sha256(benchmark_dir / "hotwords.json"),
        "pseudo_transcripts_sha256": file_sha256(
            benchmark_dir / "pseudo_transcripts.json"
        ),
        "context_format": context_format,
        "evaluation_protocol": {
            "asr_hotword_recall": (
                "presence of each unique target hotword per audio; repeated "
                "occurrences are not counted separately"
            ),
            "mer_reference": (
                "pseudo_transcripts.json generated by GPT transcription with "
                "oracle keywords; MER is auxiliary"
            ),
        },
    }
    write_json_atomic(candidate_dir.parent / "run_config.json", config)

    for index, audio_id in enumerate(pending_ids, start=1):
        audio_path = benchmark_dir / "audio" / f"{audio_id}.wav"
        transcription_path = candidate_dir / audio_id / "transcription.json"
        print(f"[{index}/{len(pending_ids)}] Transcribing {audio_id}: {audio_path}")

        audio_seconds = wav_duration_seconds(audio_path)
        prepare_gpu_measurement(device)
        inference_start = time.perf_counter()
        results = model.transcribe(
            audio=str(audio_path),
            context=context,
            language=language,
            return_time_stamps=False,
        )
        peak_gpu_allocated, peak_gpu_reserved = finish_gpu_measurement(device)
        inference_seconds = time.perf_counter() - inference_start
        if len(results) != 1:
            raise RuntimeError(
                f"Expected one transcription for audio {audio_id}, got {len(results)}"
            )

        # Match the uploaded evaluate.py contract: a list of ASR segments, each
        # containing a string-valued ``text`` field.
        write_json_atomic(
            transcription_path,
            [{"text": results[0].text or ""}],
        )
        per_audio_metrics[audio_id] = {
            "audio_seconds": round(audio_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "rtf": round(inference_seconds / audio_seconds, 6),
            "peak_gpu_allocated_mib": (
                round(peak_gpu_allocated, 3)
                if peak_gpu_allocated is not None
                else None
            ),
            "peak_gpu_reserved_mib": (
                round(peak_gpu_reserved, 3)
                if peak_gpu_reserved is not None
                else None
            ),
            "peak_process_rss_mib": round(peak_process_rss_mib(), 3),
        }
        efficiency_metrics = build_efficiency_metrics(
            per_audio=per_audio_metrics,
            benchmark_audio_count=len(audio_ids),
            model_load_seconds=model_load_seconds,
        )
        write_json_atomic(metrics_path, efficiency_metrics)
        print(
            f"[{index}/{len(pending_ids)}] Done {audio_id}: "
            f"audio={audio_seconds:.2f}s, inference={inference_seconds:.2f}s, "
            f"RTF={inference_seconds / audio_seconds:.4f}"
        )


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

    for audio_id in audio_ids:
        transcription_path = candidate_dir / audio_id / "transcription.json"
        data = read_json(transcription_path)
        if isinstance(data, dict):
            valid = isinstance(data.get("text"), str)
        elif isinstance(data, list):
            valid = all(
                isinstance(segment, dict)
                and isinstance(segment.get("text"), str)
                for segment in data
            )
        else:
            valid = False
        if not valid:
            raise ValueError(
                f"{transcription_path} is incompatible with the uploaded "
                "evaluate.py. Expected {\"text\": \"...\"} or "
                "[{\"text\": \"...\"}, ...]."
            )


def run_evaluation(
    evaluation_script: Path,
    candidate_dir: Path,
    report_path: Path,
    predicted_keywords: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(evaluation_script),
        "--candidate",
        str(candidate_dir),
        "--output",
        str(report_path),
    ]
    if predicted_keywords is not None:
        command.extend(["--predicted-keywords", str(predicted_keywords)])

    print(
        "[stage 2] ASR hotword Recall is presence-based per unique hotword and "
        "audio; repeated occurrences are not counted separately."
    )
    print(
        "[stage 2] MER uses GPT-generated pseudo transcripts with oracle-keyword "
        "prompting, so treat MER as an auxiliary comparison only."
    )
    print("[stage 2] Running benchmark evaluator:")
    print(f"[stage 2] Evaluator SHA-256: {file_sha256(evaluation_script)}")
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
    predicted_keywords = (
        args.predicted_keywords.expanduser().resolve()
        if args.predicted_keywords
        else None
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            Path("exp")
            / "hotword_benchmark"
            / model_slug(args.model_path)
            / ("global_hotwords" if args.context_mode == "global" else "no_hotwords")
        ).resolve()
    )
    candidate_dir = output_dir / "candidate"
    report_path = output_dir / args.report_name

    audio_ids, hotwords = validate_benchmark(
        benchmark_dir=benchmark_dir,
        hotwords_file=hotwords_file,
        evaluation_script=evaluation_script,
    )
    if predicted_keywords is not None:
        if not predicted_keywords.is_file():
            raise FileNotFoundError(
                f"--predicted_keywords does not exist: {predicted_keywords}"
            )
        validate_predicted_keywords(predicted_keywords, audio_ids)
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
            predicted_keywords=predicted_keywords,
        )


if __name__ == "__main__":
    main()
