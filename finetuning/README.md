## Fine-tuning Qwen3-ASR

This script fine-tunes **Qwen3-ASR** using JSONL audio-text pairs. It supports multi-GPU training via `torchrun`.

## RLBR reproduction with Qwen3-ASR

`run_rlbr.sh` reproduces the method in [**RLBR: Reinforcement Learning with Biasing Rewards for Contextual Speech Large Language Models**](https://arxiv.org/abs/2601.13409) (Ren et al., ICASSP 2026) with Qwen3-ASR-0.6B as the backbone. It runs four stages:

1. Construct contextual LibriSpeech data.
2. Train the contextual supervised fine-tuning (SFT) seed.
3. Continue the SFT adapter with reference-aware Group Relative Policy Optimization (GRPO) and the RLBR reward.
4. Decode four explicit systems and report WER, BWER, UWER, and the decode-failure rate.

| Evaluation system | Checkpoint | Prompt | Purpose |
|---|---|---|---|
| `baseline` | Qwen3-ASR base | none | direct speech recognition |
| `local` | Qwen3-ASR base | only rare words in this reference | positive-only contextual biasing |
| `global` | Qwen3-ASR base | fixed Rare5k condition N list | contextual biasing with distractors |
| `rlbr` | RLBR adapter | the same fixed Rare5k list | the proposed method |

`sft` remains available as an optional ablation. The legacy `base` name remains
an alias for the base-model global-biasing behavior; use `baseline` when the
model must not see any biasing words.

Assuming `qwen-asr` is already installed, install the additional dependencies with:

```bash
pip install -U datasets peft
```

LibriSpeech itself does not ship contextual lists. For the paper-compatible
test conditions, clone the public Rare5k benchmark files released for the
protocol followed by RLBR:

```bash
git clone --depth 1 https://github.com/facebookresearch/fbai-speech.git /path/to/fbai-speech
```

Run the complete experiment from an existing official LibriSpeech directory:

```bash
./run_rlbr.sh \
  --librispeech_root /path/to/LibriSpeech \
  --biasing_benchmark_root /path/to/fbai-speech/is21_deep_bias \
  --gpuid 0,1,2,3,4,5,6,7
```

The script follows the repository's `stage`/`stop_stage` convention and skips completed artifacts. For example, to resume only RLBR training:

```bash
./run_rlbr.sh --stage 2 --stop_stage 2 --gpuid 0,1 --resume 1
```

To run only the four-way comparison after training:

```bash
./run_rlbr.sh \
  --stage 3 \
  --stop_stage 3 \
  --eval_models "baseline local global rlbr" \
  --gpuid 0
```

`baseline` and `local` are each decoded once per test split. `global` and
`rlbr` are decoded at each `--eval_bias_sizes` value using identical JSONL rows
and bias lists. The evaluator's
`--prompt_mode none` always discards the row prompt; `--prompt_mode biasing`
uses the row's `prompt`, or constructs the same marked prompt from `bias_list`
when `prompt` is absent. Every prediction and metric file records the selected
mode.

The default `--eval_bias_sizes "100 500 1000"` therefore produces three
`global` conditions and the same three `rlbr` conditions. After Stage 3 finishes,
the script automatically writes the following consolidated report files:

```text
exp/rlbr/qwen3_asr_06b/eval/report/summary.md
exp/rlbr/qwen3_asr_06b/eval/report/summary.csv
exp/rlbr/qwen3_asr_06b/eval/report/summary.json
```

The report contains WER, BWER, UWER, and decode failures for every requested
system and test split. It also reports the absolute change and relative error
reduction of `rlbr` against `global` at each matching bias condition N.

The prepared JSONL keeps the same record usable by both stages. `prompt`
contains the marked contextual list, `text` is the marked Qwen3-ASR target,
`reference` is the unmarked transcript, `bias_words` contains positive terms,
and `bias_list` contains the actual prompt list. The public Rare5k protocol
defines the top 5,000 training words as common and the remaining 209.2K words
as rare. Its fixed test TSVs provide the rare reference words and the complete
lists for conditions N=100, 500, and 1,000. `local` uses only the former, while
`global` and `rlbr` preserve each released list exactly. Because local positives
are selected from the reference, `local` is an oracle-style analysis condition
rather than a deployable retrieval method.

The RLBR paper describes `N` as the distractor condition, while the public
benchmark README calls it the biasing-list size. The released TSV contents are
the authority for evaluation: positive reference words are guaranteed to be in
the list, and the resulting list length can differ slightly from `N` after list
construction and deduplication. Therefore `bias_list_size` and
`distractor_count` record the actual per-utterance values instead of assuming
that every row has exactly `N` entries.

If Stage 0 is interrupted after some JSONLs finish, rerun it with `--resume 1`.
Completed files are reused only when their row count matches the source split;
an incomplete file is rewritten.

If `--biasing_benchmark_root` is omitted, the script deterministically
reconstructs equivalent lists from the available LibriSpeech transcripts and
records `biasing_list_source: generated` in `manifest.json`. Those generated
lists are useful for development but are not the fixed public benchmark lists.

The following paper-reported settings are preserved in the supplied configs:

- contextual SFT before RLBR
- LoRA rank 320 on attention and feed-forward projections
- cosine learning-rate schedule with peak rates `1e-5` for SFT and `5e-6` for RLBR
- eight categorical samples per input at temperature 1.2
- character-level edit distance
- biasing weight `lambda = 5`
- marked bias words in prompts, targets, and reward computation
- reference transcription appended as the ninth trajectory for group reward normalization and policy optimization
- GRPO clipping `epsilon = 0.28` and KL weight `beta = 0`

Several values required by executable code are not reported in the five-page paper. They remain explicit rather than being presented as paper settings: training positive-word count and distractor distribution in `run_rlbr.sh`, and epoch count, batch accumulation, warmup, LoRA alpha/dropout, and maximum completion length in the two `conf/rlbr_*.json` files. The supplied values are starting points for a controlled reproduction and should be reported as implementation choices.

The trainer can keep a frozen copy of the contextual-SFT LoRA adapter without
loading a second Qwen3-ASR backbone when a nonzero KL weight is requested. The
paper-compatible default uses `beta = 0`, so no reference-policy adapter is
needed. The gold reference trajectory receives reward zero and participates in
both group mean/standard-deviation calculation and policy loss, matching the
paper's reference-aware formulation. The current implementation performs one
on-policy update for each sampled group. The paper does not report repeated
PPO/GRPO inner iterations.

With the default layout, outputs are written below `exp/rlbr/qwen3_asr_06b/`. Each evaluation condition contains `predictions.jsonl` and `metrics.json`.

### 1) Setup

First, please install the two Python packages `qwen-asr` and `datasets` using the command below.

```bash
pip install -U qwen-asr datasets
```

Then, to reduce GPU memory usage and speed up training, it is recommended to install FlashAttention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

### 2) Input JSONL format

Prepare your training file as JSONL (one JSON per line). Each line must contain:

- `audio`: path to a WAV file
- `text`: transcript text (you can include a language prefix)

Example:
```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language English<asr_text>Another example."}
{"audio":"/data/wavs/utt0003.wav","text":"language English<asr_text>Fine-tuning data line."}
```

Language prefix recommendation:

- If you **have** language info, use:
  - `language English<asr_text>...`
  - `language Chinese<asr_text>...`
- If you **do not have** language info, use:
  - `language None<asr_text>...`

Note:
- If you set `language None`, the model will not learn language detection from that prefix.

### 3) Fine-tune (single GPU)

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

Checkpoints will be written to:
- `./qwen3-asr-finetuning-out/checkpoint-<global_step>`

### 4) Fine-tune (multi GPU with torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200
```

### 5) Resume training

Option A: explicitly set a checkpoint path:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume_from ./qwen3-asr-finetuning-out/checkpoint-200
```

Option B: automatically resume from the latest checkpoint under `output_dir`:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume 1
```

### 6) Quick inference test

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
)

print(results[0].language)
print(results[0].text)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
TRAIN_FILE="./train.jsonl"
EVAL_FILE="./eval.jsonl"
OUTPUT_DIR="./qwen3-asr-finetuning-out"

torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```
