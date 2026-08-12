#!/usr/bin/env bash

set -euo pipefail

# Stages:
#   1: Qwen3-ASR inference without hotwords
#   2: Qwen3-ASR inference with the global all_hotwords.json list
#   3: Evaluate both conditions with hotword_benchmark/evaluate.py
stage=1
stop_stage=3

benchmark_dir=hotword_benchmark
model_path=Qwen/Qwen3-ASR-1.7B
output_root=
gpuid=0
language=Chinese
device=cuda
dtype=auto
attn_implementation=auto
max_new_tokens=4096
max_inference_batch_size=1
overwrite=false

help_message="Usage: $0 [options]

Options:
  --stage INT                       First stage to run (default: 1)
  --stop-stage INT                  Last stage to run (default: 3)
  --benchmark-dir PATH              Extracted hotword_benchmark directory
  --model-path PATH_OR_ID           Qwen3-ASR checkpoint
  --output-root PATH                Experiment root (default: exp/hotword_benchmark/<model>)
  --gpuid INT                       CUDA_VISIBLE_DEVICES value (default: 0)
  --language NAME                   Forced ASR language (default: Chinese)
  --device {cuda,cpu,auto}          Inference device (default: cuda)
  --dtype NAME                      auto, float32, float16, or bfloat16
  --attn-implementation NAME        auto, flash_attention_2, sdpa, or eager
  --max-new-tokens INT              Maximum generated tokens (default: 4096)
  --max-inference-batch-size INT    Qwen chunk batch size (default: 1)
  --overwrite {true,false}          Re-run existing transcriptions (default: false)"

. ./local/parse_options.sh
. ./path.sh

if [ "$stage" -gt "$stop_stage" ]; then
    echo "--stage cannot be greater than --stop-stage" >&2
    exit 1
fi

if [ -z "$output_root" ]; then
    model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')
    output_root=exp/hotword_benchmark/$model_name
fi

no_hotword_dir=$output_root/no_hotwords
hotword_dir=$output_root/global_hotwords

common_args=(
    --benchmark_dir "$benchmark_dir"
    --model_path "$model_path"
    --language "$language"
    --device "$device"
    --dtype "$dtype"
    --attn_implementation "$attn_implementation"
    --max_new_tokens "$max_new_tokens"
    --max_inference_batch_size "$max_inference_batch_size"
)

overwrite_args=()
if $overwrite; then
    overwrite_args+=(--overwrite)
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: inference without hotwords -> $no_hotword_dir"
    CUDA_VISIBLE_DEVICES=$gpuid \
        python local/run_hotword_benchmark.py \
            "${common_args[@]}" \
            --context_mode none \
            --output_dir "$no_hotword_dir" \
            --stage 1 \
            --stop_stage 1 \
            "${overwrite_args[@]}"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: inference with global hotwords -> $hotword_dir"
    CUDA_VISIBLE_DEVICES=$gpuid \
        python local/run_hotword_benchmark.py \
            "${common_args[@]}" \
            --context_mode global \
            --output_dir "$hotword_dir" \
            --stage 1 \
            --stop_stage 1 \
            "${overwrite_args[@]}"
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Stage 3: evaluate inference without hotwords"
    python local/run_hotword_benchmark.py \
        "${common_args[@]}" \
        --context_mode none \
        --output_dir "$no_hotword_dir" \
        --stage 2 \
        --stop_stage 2

    echo "Stage 3: evaluate inference with global hotwords"
    python local/run_hotword_benchmark.py \
        "${common_args[@]}" \
        --context_mode global \
        --output_dir "$hotword_dir" \
        --stage 2 \
        --stop_stage 2
fi
