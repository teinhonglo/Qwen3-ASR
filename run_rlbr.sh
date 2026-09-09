#!/bin/bash

set -euo pipefail

stage=0
stop_stage=100
gpuid=0
seed=66
resume=0

librispeech_root=
data_dir=data/rlbr_librispeech
exp_root=exp/rlbr/qwen3_asr_06b
sft_conf=conf/rlbr_qwen3_asr_06b_sft.json
rlbr_conf=conf/rlbr_qwen3_asr_06b_grpo.json
base_model_path=Qwen/Qwen3-ASR-0.6B
biasing_benchmark_root=
train_num_positive=3
train_distractors="100 500 1000"
eval_bias_sizes="100 500 1000"
eval_distractors=
sft_eval_bias_size=100
sft_eval_distractor=
eval_models="baseline local global rlbr"

. ./local/parse_options.sh
. ./path.sh

if [ -n "$eval_distractors" ]; then
    echo "[warning] --eval_distractors is deprecated; use --eval_bias_sizes" >&2
    eval_bias_sizes=
    for size in $eval_distractors; do
        if [ "$size" -gt 0 ]; then
            eval_bias_sizes="$eval_bias_sizes $size"
        fi
    done
fi
if [ -n "$sft_eval_distractor" ]; then
    echo "[warning] --sft_eval_distractor is deprecated; use --sft_eval_bias_size" >&2
    sft_eval_bias_size=$sft_eval_distractor
fi

sft_dir=$exp_root/sft
rlbr_dir=$exp_root/rlbr

IFS=',' read -r -a gpu_array <<< "$gpuid"
nproc_per_node=${#gpu_array[@]}

if [[ " $eval_bias_sizes " != *" $sft_eval_bias_size "* ]]; then
    echo "[error] --eval_bias_sizes must include --sft_eval_bias_size $sft_eval_bias_size" >&2
    exit 1
fi

run_training() {
    local script=$1
    shift
    if [ "$nproc_per_node" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$gpuid torchrun \
            --standalone \
            --nproc_per_node=$nproc_per_node \
            "$script" "$@"
    else
        CUDA_VISIBLE_DEVICES=$gpuid python "$script" "$@"
    fi
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    data_ready=true
    for required_file in manifest.json train.jsonl; do
        if [ ! -s "$data_dir/$required_file" ]; then
            data_ready=false
            break
        fi
    done
    for split in test_clean test_other; do
        if [ ! -s "$data_dir/${split}_local.jsonl" ]; then
            data_ready=false
            break
        fi
        for bias_list_size in $eval_bias_sizes; do
            if [ ! -s "$data_dir/${split}_n${bias_list_size}.jsonl" ]; then
                data_ready=false
                break 2
            fi
        done
    done
    if [ ! -s "$data_dir/dev_clean_n${sft_eval_bias_size}.jsonl" ]; then
        data_ready=false
    fi
    if [ -n "$biasing_benchmark_root" ] && \
            ! grep -q '"biasing_list_source": "official_is21_deep_bias"' \
                "$data_dir/manifest.json" 2>/dev/null; then
        data_ready=false
    fi
    if [ "$data_ready" = true ]; then
        echo "[skip] RLBR data already prepared: $data_dir"
    else
        if [ -z "$librispeech_root" ]; then
            echo "[error] stage 0 requires --librispeech_root /path/to/LibriSpeech" >&2
            exit 1
        fi
        benchmark_args=()
        if [ -n "$biasing_benchmark_root" ]; then
            benchmark_args+=(--biasing_benchmark_root "$biasing_benchmark_root")
        fi
        python local/prepare_rlbr_librispeech.py \
            --librispeech_root "$librispeech_root" \
            --output_dir "$data_dir" \
            --train_num_positive "$train_num_positive" \
            --train_distractors $train_distractors \
            --eval_bias_sizes $eval_bias_sizes \
            "${benchmark_args[@]}" \
            --seed "$seed"
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    if [ -s "$sft_dir/adapter_config.json" ] && [ "$resume" -eq 0 ]; then
        echo "[skip] contextual SFT already completed: $sft_dir"
    else
        resume_args=()
        if [ "$resume" -eq 1 ]; then
            resume_args+=(--resume 1)
        fi
        run_training finetuning/qwen3_asr_sft.py \
            --train_conf "$sft_conf" \
            --train_file "$data_dir/train.jsonl" \
            --eval_file "$data_dir/dev_clean_n${sft_eval_bias_size}.jsonl" \
            --output_dir "$sft_dir" \
            --seed "$seed" \
            "${resume_args[@]}"
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    if [ -s "$rlbr_dir/adapter_config.json" ] && [ "$resume" -eq 0 ]; then
        echo "[skip] RLBR already completed: $rlbr_dir"
    else
        if [ ! -s "$sft_dir/adapter_config.json" ]; then
            echo "[error] RLBR requires the contextual-SFT adapter at $sft_dir" >&2
            exit 1
        fi
        resume_args=()
        if [ "$resume" -eq 1 ]; then
            resume_args+=(--resume 1)
        fi
        run_training finetuning/qwen3_asr_rlbr.py \
            --train_conf "$rlbr_conf" \
            --train_file "$data_dir/train.jsonl" \
            --seed_model_path "$sft_dir" \
            --output_dir "$rlbr_dir" \
            --seed "$seed" \
            "${resume_args[@]}"
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    for model_name in $eval_models; do
        case $model_name in
            baseline)
                model_path=$base_model_path
                prompt_mode=none
                model_conditions=local
                ;;
            local)
                model_path=$base_model_path
                prompt_mode=biasing
                model_conditions=local
                ;;
            global)
                model_path=$base_model_path
                prompt_mode=biasing
                model_conditions=$eval_bias_sizes
                ;;
            base)
                echo "[warning] eval model 'base' is the legacy alias for 'global'" >&2
                model_path=$base_model_path
                prompt_mode=biasing
                model_conditions=$eval_bias_sizes
                ;;
            sft)
                model_path=$sft_dir
                prompt_mode=biasing
                model_conditions=$eval_bias_sizes
                ;;
            rlbr)
                model_path=$rlbr_dir
                prompt_mode=biasing
                model_conditions=$eval_bias_sizes
                ;;
            *)
                echo "[error] unsupported eval model: $model_name" >&2
                exit 1
                ;;
        esac

        for split in test_clean test_other; do
            for condition in $model_conditions; do
                if [ "$condition" = local ]; then
                    input_jsonl=$data_dir/${split}_local.jsonl
                    output_dir=$exp_root/eval/$model_name/$split
                else
                    input_jsonl=$data_dir/${split}_n${condition}.jsonl
                    output_dir=$exp_root/eval/$model_name/${split}_n${condition}
                fi
                if [ -s "$output_dir/metrics.json" ] && \
                        grep -Eq "\"prompt_mode\"[[:space:]]*:[[:space:]]*\"$prompt_mode\"" \
                            "$output_dir/metrics.json" && \
                        grep -Eq "\"evaluation_name\"[[:space:]]*:[[:space:]]*\"$model_name\"" \
                            "$output_dir/metrics.json"; then
                    echo "[skip] evaluation already completed: $output_dir"
                    continue
                fi
                CUDA_VISIBLE_DEVICES=$gpuid python finetuning/qwen3_asr_rlbr_test.py \
                    --model_path "$model_path" \
                    --input_jsonl "$input_jsonl" \
                    --output_dir "$output_dir" \
                    --evaluation_name "$model_name" \
                    --prompt_mode "$prompt_mode" \
                    --device cuda:0
            done
        done
    done
fi
