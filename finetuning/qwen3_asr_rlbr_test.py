#!/usr/bin/env python3
"""Decode direct-ASR or local-biasing JSONL and report WER/BWER/UWER."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from qwen3_asr_test import infer_one, load_asr_wrapper, resolve_dtype
from rlbr_utils import compute_corpus_error_rates, resolve_evaluation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate contextual Qwen3-ASR checkpoints")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--evaluation_name",
        type=str,
        default="",
        help="Label recorded in prediction and metric outputs.",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["none", "biasing"],
        default="biasing",
        help=(
            "none discards every row prompt for a direct-ASR baseline; biasing "
            "uses `prompt` or builds one from `bias_list`"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = resolve_dtype(args.dtype, args.device)
    wrapper = load_asr_wrapper(args.model_path, dtype=dtype, device=args.device)

    source_rows = load_jsonl(args.input_jsonl)
    evaluated_rows: List[Dict[str, Any]] = []
    failures = 0
    prediction_path = os.path.join(args.output_dir, "predictions.jsonl")

    with open(prediction_path, "w", encoding="utf-8") as output:
        for index, row in enumerate(source_rows, start=1):
            error = ""
            prompt = resolve_evaluation_prompt(row, args.prompt_mode)
            try:
                hypothesis = infer_one(
                    asr_wrapper=wrapper,
                    audio_path=row["audio"],
                    prompt=prompt,
                    sr=args.sr,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            except Exception as exc:  # Preserve the failure and continue corpus evaluation.
                failures += 1
                hypothesis = ""
                error = f"{type(exc).__name__}: {exc}"
                print(f"[decode-error] line={index} id={row.get('text_id', index)} {error}")

            evaluated = {
                "text_id": row.get("text_id", f"line{index}"),
                "reference": row.get("reference", row.get("text", "")),
                "hypothesis": hypothesis,
                "bias_words": row.get("bias_words", []),
                "bias_list": row.get("bias_list", []),
                "bias_list_size": row.get(
                    "bias_list_size", len(row.get("bias_list", []))
                ),
                "distractor_count": row.get("distractor_count"),
                "data_condition": row.get("condition", ""),
                "evaluation_name": args.evaluation_name,
                "prompt_mode": args.prompt_mode,
                "prompt": prompt,
                "decode_error": error,
            }
            evaluated_rows.append(evaluated)
            output.write(json.dumps(evaluated, ensure_ascii=False) + "\n")

    metrics = compute_corpus_error_rates(evaluated_rows)
    metrics.update(
        {
            "model_path": args.model_path,
            "input_jsonl": args.input_jsonl,
            "evaluation_name": args.evaluation_name,
            "prompt_mode": args.prompt_mode,
            "decode_failures": failures,
            "decode_failure_rate": failures / len(source_rows) if source_rows else 0.0,
        }
    )
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[info] predictions: {prediction_path}")
    print(f"[info] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
