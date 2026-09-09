#!/usr/bin/env python3
"""Reference-aware GRPO with RLBR rewards for Qwen3-ASR.

This is a narrow multimodal extension of ``transformers.Trainer``.  A standard
Trainer cannot generate grouped audio-conditioned trajectories inside
``compute_loss``, and text-only GRPO trainers do not preserve Qwen3-ASR's audio
placeholder/input-feature alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, Trainer, TrainingArguments

from qwen3_asr_sft import (
    MakeEveryCheckpointInferableCallback,
    build_prefix_messages,
    find_latest_checkpoint,
    load_audio,
    patch_outer_forward,
    save_prompt_txt,
)
from rlbr_utils import compute_rlbr_reward


POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-ASR RLBR fine-tuning")
    parser.add_argument("--train_conf", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument(
        "--seed_model_path",
        type=str,
        required=True,
        help=(
            "Contextual-SFT LoRA checkpoint used to initialize policy and fixed "
            "reference adapters."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--resume", type=int, default=0)
    return parser.parse_args()


def load_train_conf(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        not isinstance(config, list)
        or len(config) != 2
        or not all(isinstance(item, dict) for item in config)
    ):
        raise ValueError("train_conf must be [training_args, rlbr_args]")
    return config[0], config[1]


def build_prefix_text(processor, prompt: str) -> str:
    rendered = processor.apply_chat_template(
        [build_prefix_messages(prompt, None)],
        add_generation_prompt=True,
        tokenize=False,
    )
    return rendered[0] if isinstance(rendered, list) else rendered


def make_preprocess_fn(processor):
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        if "text" not in example:
            raise KeyError("RLBR JSONL requires a formatted `text` completion")
        reference = example.get("reference", example["text"])
        bias_words = example.get("bias_words", [])
        if isinstance(bias_words, str):
            bias_words = [bias_words]
        return {
            "audio": example["audio"],
            "prompt": example.get("prompt", ""),
            "prefix_text": build_prefix_text(processor, example.get("prompt", "")),
            "reference": reference,
            "reference_completion": example["text"],
            "bias_words": list(bias_words),
        }

    return preprocess


@dataclass
class RLBRDataCollator:
    """Keep raw paths/text until the online-generation training step."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        keys = (
            "audio",
            "prompt",
            "prefix_text",
            "reference",
            "reference_completion",
            "bias_words",
        )
        return {key: [feature[key] for feature in features] for key in keys}


def unwrap_sequences(generation_output) -> torch.Tensor:
    if hasattr(generation_output, "sequences"):
        return generation_output.sequences
    if isinstance(generation_output, dict) and "sequences" in generation_output:
        return generation_output["sequences"]
    if torch.is_tensor(generation_output):
        return generation_output
    raise TypeError(f"Unsupported generate() output: {type(generation_output)}")


def terminal_token_ids(tokenizer) -> List[int]:
    token_ids = tokenizer.eos_token_id
    if token_ids is None:
        result = []
    elif isinstance(token_ids, int):
        result = [token_ids]
    else:
        result = list(token_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is not None and pad_id not in result:
        result.append(pad_id)
    return result


def completion_mask(
    input_ids: torch.Tensor, eos_ids: Sequence[int], pad_id: Optional[int]
) -> torch.Tensor:
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    if not eos_ids:
        if pad_id is not None:
            mask &= input_ids.ne(pad_id)
        return mask.long()

    eos = torch.zeros_like(mask, dtype=torch.bool)
    for token_id in eos_ids:
        eos |= input_ids.eq(token_id)
    # Include the first EOS and mask everything after it.
    eos_seen_before = eos.long().cumsum(dim=1) - eos.long()
    mask &= eos_seen_before.eq(0)
    # Some decoder-only tokenizers use EOS as PAD. In that case the EOS logic
    # above distinguishes the first terminal token from subsequent padding.
    if pad_id is not None and pad_id not in eos_ids:
        mask &= input_ids.ne(pad_id)
    return mask.long()


def pad_sequences(
    sequences: Iterable[Sequence[int]], pad_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences = [list(sequence) for sequence in sequences]
    max_length = max(len(sequence) for sequence in sequences)
    ids = torch.full(
        (len(sequences), max_length), pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        if sequence:
            length = len(sequence)
            ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[index, :length] = 1
    return ids, mask


class Qwen3ASRRLBRTrainer(Trainer):
    """Online, reference-aware GRPO for Qwen3-ASR audio-text batches."""

    def __init__(self, *args, processor, rlbr_args: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.rlbr_args = rlbr_args
        self.num_generations = int(rlbr_args.get("num_generations", 8))
        self.temperature = float(rlbr_args.get("temperature", 1.2))
        self.top_p = float(rlbr_args.get("top_p", 1.0))
        self.max_completion_length = int(rlbr_args.get("max_completion_length", 256))
        self.bias_weight = float(rlbr_args.get("bias_weight", 5.0))
        self.edit_level = str(rlbr_args.get("edit_level", "char"))
        self.reference_aware = bool(rlbr_args.get("reference_aware", True))
        self.beta = float(rlbr_args.get("beta", 0.0))
        self.epsilon = float(rlbr_args.get("epsilon", 0.28))
        self.logprob_micro_batch_size = int(
            rlbr_args.get("logprob_micro_batch_size", 1)
        )
        self.sampling_rate = int(rlbr_args.get("sr", 16000))
        self._last_rlbr_log_step = -1

        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2 for group-relative advantages")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive for categorical sampling")
        if self.edit_level not in {"char", "word"}:
            raise ValueError("edit_level must be 'char' or 'word'")
        if self.logprob_micro_batch_size <= 0:
            raise ValueError("logprob_micro_batch_size must be positive")

    def _peft_model(self, model):
        return self.accelerator.unwrap_model(model)

    def _activate_adapter(self, model, adapter_name: str, train_policy: bool) -> None:
        peft_model = self._peft_model(model)
        peft_model.set_adapter(adapter_name)
        for name, parameter in peft_model.named_parameters():
            parameter.requires_grad = train_policy and f".{POLICY_ADAPTER}." in name

    def _move_batch(
        self, inputs: Dict[str, torch.Tensor], model
    ) -> Dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        model_dtype = getattr(self._peft_model(model), "dtype", None)
        moved = {}
        for key, value in inputs.items():
            value = value.to(device)
            if model_dtype is not None and value.is_floating_point():
                value = value.to(model_dtype)
            moved[key] = value
        return moved

    def _prepare_prompt_inputs(self, raw_inputs: Dict[str, List[Any]], model):
        audios = [load_audio(path, sr=self.sampling_rate) for path in raw_inputs["audio"]]
        processed = self.processor(
            text=raw_inputs["prefix_text"],
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        return self._move_batch(dict(processed), model)

    def _generate(
        self, model, prompt_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unwrapped = self._peft_model(model)
        tokenizer = self.processor.tokenizer
        terminal_ids = terminal_token_ids(tokenizer)
        was_training = unwrapped.training
        unwrapped.eval()
        with torch.no_grad():
            generated = unwrapped.generate(
                **prompt_inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.num_generations,
                max_new_tokens=self.max_completion_length,
                eos_token_id=terminal_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        if was_training:
            unwrapped.train()

        sequences = unwrap_sequences(generated)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        completion_ids = sequences[:, prompt_length:]
        pad_id = tokenizer.pad_token_id
        masks = completion_mask(
            completion_ids,
            terminal_ids,
            pad_id,
        )
        return completion_ids, masks

    def _append_reference_trajectories(
        self,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        reference_completions: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

        all_sequences: List[List[int]] = []
        batch_size = len(reference_completions)
        for batch_index in range(batch_size):
            offset = batch_index * self.num_generations
            for generation_index in range(self.num_generations):
                row = offset + generation_index
                length = int(generated_mask[row].sum().item())
                all_sequences.append(generated_ids[row, :length].tolist())

            if self.reference_aware:
                reference_ids = tokenizer.encode(
                    reference_completions[batch_index], add_special_tokens=False
                )[: self.max_completion_length]
                if eos_id is not None:
                    if len(reference_ids) == self.max_completion_length:
                        reference_ids[-1] = eos_id
                    elif not reference_ids or reference_ids[-1] != eos_id:
                        reference_ids.append(eos_id)
                all_sequences.append(reference_ids)

        completion_ids, masks = pad_sequences(
            all_sequences, pad_id=pad_id, device=generated_ids.device
        )
        group_size = self.num_generations + int(self.reference_aware)
        return completion_ids, masks, group_size

    def _expand_model_inputs(
        self,
        prompt_inputs: Dict[str, torch.Tensor],
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
        group_size: int,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        prompt_ids = prompt_inputs["input_ids"].repeat_interleave(group_size, dim=0)
        prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(group_size, dim=0)
        model_inputs = {
            key: value.repeat_interleave(group_size, dim=0)
            for key, value in prompt_inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }
        model_inputs["input_ids"] = torch.cat([prompt_ids, completion_ids], dim=1)
        model_inputs["attention_mask"] = torch.cat(
            [prompt_mask, completion_attention_mask], dim=1
        )
        return model_inputs, prompt_ids.shape[1]

    def _completion_logps(
        self,
        model,
        model_inputs: Dict[str, torch.Tensor],
        prompt_length: int,
    ) -> torch.Tensor:
        total = model_inputs["input_ids"].shape[0]
        chunks = []
        for start in range(0, total, self.logprob_micro_batch_size):
            end = min(start + self.logprob_micro_batch_size, total)
            chunk = {key: value[start:end] for key, value in model_inputs.items()}
            outputs = model(**chunk, use_cache=False)
            completion_targets = chunk["input_ids"][:, prompt_length:]
            completion_logits = outputs.logits[
                :, prompt_length - 1 : -1, :
            ].float()
            selected_logits = torch.gather(
                completion_logits,
                dim=-1,
                index=completion_targets.unsqueeze(-1),
            ).squeeze(-1)
            chunks.append(selected_logits - torch.logsumexp(completion_logits, dim=-1))
        return torch.cat(chunks, dim=0)

    def _decode_generated(self, ids: torch.Tensor, mask: torch.Tensor) -> List[str]:
        sequences = [
            row[: int(row_mask.sum().item())].tolist()
            for row, row_mask in zip(ids, mask)
        ]
        return self.processor.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _rewards_and_advantages(
        self,
        generated_text: Sequence[str],
        raw_inputs: Dict[str, List[Any]],
        group_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        rewards: List[float] = []
        overall_distances: List[int] = []
        bias_distances: List[int] = []

        for batch_index, reference_completion in enumerate(
            raw_inputs["reference_completion"]
        ):
            offset = batch_index * group_size
            for group_index in range(group_size):
                is_reference = self.reference_aware and group_index == group_size - 1
                hypothesis = (
                    reference_completion
                    if is_reference
                    else generated_text[offset + group_index]
                )
                result = compute_rlbr_reward(
                    reference=reference_completion,
                    hypothesis=hypothesis,
                    bias_words=raw_inputs["bias_words"][batch_index],
                    bias_weight=self.bias_weight,
                    edit_level=self.edit_level,
                )
                rewards.append(result.reward)
                overall_distances.append(result.edit_distance)
                bias_distances.append(result.bias_edit_distance)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        grouped = reward_tensor.view(-1, group_size)
        means = grouped.mean(dim=1, keepdim=True)
        stds = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = torch.where(
            stds > 1e-6,
            (grouped - means) / (stds + 1e-6),
            torch.zeros_like(grouped),
        ).reshape(-1)
        stats = {
            "reward_mean": float(reward_tensor.mean().item()),
            "reward_std": float(reward_tensor.std(unbiased=False).item()),
            "edit_distance_mean": float(np.mean(overall_distances)),
            "bias_edit_distance_mean": float(np.mean(bias_distances)),
            "zero_group_std_fraction": float((stds <= 1e-6).float().mean().item()),
        }
        return reward_tensor, advantages, stats

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        self._activate_adapter(model, POLICY_ADAPTER, train_policy=True)
        prompt_inputs = self._prepare_prompt_inputs(inputs, model)
        generated_ids, generated_mask = self._generate(model, prompt_inputs)
        completion_ids, completion_attention_mask, group_size = (
            self._append_reference_trajectories(
                generated_ids,
                generated_mask,
                inputs["reference_completion"],
            )
        )
        generated_text = self._decode_generated(
            completion_ids, completion_attention_mask
        )
        model_inputs, prompt_length = self._expand_model_inputs(
            prompt_inputs,
            completion_ids,
            completion_attention_mask,
            group_size,
        )
        _, advantages, reward_stats = self._rewards_and_advantages(
            generated_text,
            inputs,
            group_size,
            completion_ids.device,
        )

        reference_logps = None
        if self.beta > 0:
            self._activate_adapter(model, REFERENCE_ADAPTER, train_policy=False)
            reference_model = self._peft_model(model)
            was_training = reference_model.training
            reference_model.eval()
            with torch.no_grad():
                reference_logps = self._completion_logps(
                    reference_model, model_inputs, prompt_length
                )
            if was_training:
                reference_model.train()

        self._activate_adapter(model, POLICY_ADAPTER, train_policy=True)
        policy_logps = self._completion_logps(model, model_inputs, prompt_length)
        old_policy_logps = policy_logps.detach()

        log_ratio = policy_logps - old_policy_logps
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantages.unsqueeze(1)
        clipped = torch.clamp(
            ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        ) * advantages.unsqueeze(1)
        per_token_loss = -torch.minimum(unclipped, clipped)

        if reference_logps is not None:
            log_ref_ratio = reference_logps - policy_logps
            per_token_kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_attention_mask.to(per_token_loss.dtype)
        per_sequence_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        loss = per_sequence_loss.mean()

        if (
            self.is_world_process_zero()
            and self.state.global_step != self._last_rlbr_log_step
            and self.state.global_step % max(1, self.args.logging_steps) == 0
        ):
            self._last_rlbr_log_step = self.state.global_step
            printable = " ".join(
                f"{key}={value:.4f}" for key, value in reward_stats.items()
            )
            print(f"[rlbr step={self.state.global_step}] {printable}")

        if return_outputs:
            return loss, {"loss": loss.detach(), **reward_stats}
        return loss


def load_seed_policy(
    seed_model_path: str,
    use_bf16: bool,
    need_reference_adapter: bool,
):
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise ImportError(
            "RLBR requires a LoRA SFT seed and PEFT. Install with `pip install -U peft`."
        ) from exc

    adapter_config_path = os.path.join(seed_model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(
            "RLBR expects the contextual-SFT seed to be a LoRA adapter, but "
            f"adapter_config.json is missing under {seed_model_path}"
        )

    peft_config = PeftConfig.from_pretrained(seed_model_path)
    wrapper = Qwen3ASRModel.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    base_model = wrapper.model
    patch_outer_forward(base_model)
    base_model.generation_config = GenerationConfig.from_model_config(base_model.config)

    model = PeftModel.from_pretrained(
        base_model,
        seed_model_path,
        adapter_name=POLICY_ADAPTER,
        is_trainable=True,
    )
    if need_reference_adapter:
        model.load_adapter(
            seed_model_path,
            adapter_name=REFERENCE_ADAPTER,
            is_trainable=False,
        )
    model.set_adapter(POLICY_ADAPTER)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = f".{POLICY_ADAPTER}." in name
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(
            f"No trainable parameters found for PEFT adapter {POLICY_ADAPTER!r}"
        )
    model.print_trainable_parameters()
    wrapper.model = model
    return wrapper


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    cli_args = parse_args()
    set_seed(cli_args.seed)
    training_args_conf, rlbr_args = load_train_conf(cli_args.train_conf)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = load_seed_policy(
        cli_args.seed_model_path,
        use_bf16=use_bf16,
        need_reference_adapter=float(rlbr_args.get("beta", 0.0)) > 0,
    )
    model = wrapper.model
    processor = wrapper.processor
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        terminal_ids = terminal_token_ids(processor.tokenizer)
        if not terminal_ids:
            raise ValueError("The tokenizer must define an EOS or PAD token")
        processor.tokenizer.pad_token_id = terminal_ids[0]

    if training_args_conf.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    raw_dataset = load_dataset("json", data_files={"train": cli_args.train_file})[
        "train"
    ]
    dataset = raw_dataset.map(make_preprocess_fn(processor), num_proc=1)
    keep = {
        "audio",
        "prompt",
        "prefix_text",
        "reference",
        "reference_completion",
        "bias_words",
    }
    drop = [column for column in dataset.column_names if column not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)

    training_args = TrainingArguments(
        output_dir=cli_args.output_dir,
        bf16=use_bf16,
        fp16=not use_bf16,
        **training_args_conf,
    )
    trainer = Qwen3ASRRLBRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=RLBRDataCollator(),
        tokenizer=processor.tokenizer,
        processor=processor,
        rlbr_args=rlbr_args,
        callbacks=[
            MakeEveryCheckpointInferableCallback(
                processor=processor,
                model=model,
                default_prompt="",
            )
        ],
    )

    os.makedirs(cli_args.output_dir, exist_ok=True)
    if trainer.args.process_index == 0:
        with open(
            os.path.join(cli_args.output_dir, "train_conf.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump([training_args_conf, rlbr_args], handle, ensure_ascii=False, indent=4)
        with open(
            os.path.join(cli_args.output_dir, "seed_model_path.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(cli_args.seed_model_path + "\n")
        processor.save_pretrained(cli_args.output_dir)
        save_prompt_txt(cli_args.output_dir, "")

    resume_from = cli_args.resume_from.strip()
    if not resume_from and cli_args.resume == 1:
        resume_from = find_latest_checkpoint(cli_args.output_dir) or ""
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(cli_args.output_dir)
    if trainer.args.process_index == 0:
        processor.save_pretrained(cli_args.output_dir)


if __name__ == "__main__":
    main()
