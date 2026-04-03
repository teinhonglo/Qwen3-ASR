#!/usr/bin/env python3
# coding=utf-8
"""Generate word-level forced-alignment visualizations for TinyStress-15K.

This script loads `slprl/TinyStress-15K` (train/test), performs word-level forced
alignment with `Qwen/Qwen3-ForcedAligner-0.6B`, and saves one figure per sample.

Figure layout (3 rows):
1) Mel-spectrogram
2) Word spans (stressed words are highlighted)
3) Prosody annotation row + timeline axis
"""

from __future__ import annotations

import argparse
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from qwen_asr import Qwen3ForcedAligner

try:
    from datasets import Audio, Dataset, load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Please install `datasets` first: pip install datasets"
    ) from exc


PROSODY_LABEL = 'prosody rate="52%" volume="+3.0dB" pitch="+3.0st"'
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)


@dataclass
class SsmlToken:
    text: str
    stressed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="slprl/TinyStress-15K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--output-dir", type=str, default="runs/tinystress_alignment_viz")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--ssml-field", type=str, default="ssml")
    parser.add_argument("--audio-field", type=str, default="audio")
    parser.add_argument("--max-samples", type=int, default=-1, help="Per split; -1 means all")
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def normalize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9']+", "", token).lower()


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def _attrs_match_stress(attrs: Dict[str, str]) -> bool:
    rate = str(attrs.get("rate", "")).strip().lower()
    volume = str(attrs.get("volume", "")).strip().lower()
    pitch = str(attrs.get("pitch", "")).strip().lower()
    return rate == "52%" and volume == "+3.0db" and pitch == "+3.0st"


def parse_ssml_tokens(ssml: str) -> List[SsmlToken]:
    wrapped = f"<root>{ssml}</root>"
    root = ET.fromstring(wrapped)
    out: List[SsmlToken] = []

    def add_text(content: Optional[str], stressed: bool) -> None:
        if not content:
            return
        for tok in tokenize_text(content):
            out.append(SsmlToken(text=tok, stressed=stressed and bool(normalize_token(tok))))

    def walk(node: ET.Element, inherited_stress: bool = False) -> None:
        tag = node.tag.split("}")[-1].lower()
        now_stress = inherited_stress
        if tag == "prosody" and _attrs_match_stress(node.attrib):
            now_stress = True

        add_text(node.text, now_stress)
        for child in node:
            walk(child, now_stress)
            add_text(child.tail, now_stress)

    for child in root:
        walk(child, False)
        if child.tail:
            add_text(child.tail, False)

    return out


def align_stress_flags(aligned_words: Sequence[str], ssml_tokens: Sequence[SsmlToken]) -> List[bool]:
    ssml_norm = [normalize_token(t.text) for t in ssml_tokens]
    ssml_stress = [t.stressed for t in ssml_tokens]

    flags: List[bool] = []
    pos = 0
    for w in aligned_words:
        wn = normalize_token(w)
        if not wn:
            flags.append(False)
            continue

        matched = False
        while pos < len(ssml_norm):
            if ssml_norm[pos] == wn:
                flags.append(ssml_stress[pos])
                pos += 1
                matched = True
                break
            pos += 1

        if not matched:
            flags.append(False)

    return flags


def resolve_text(sample: Dict, text_field: str, ssml_field: str) -> Tuple[str, str]:
    text = str(sample.get(text_field, "")).strip()
    ssml = str(sample.get(ssml_field, "")).strip()
    if not text and ssml:
        text = " ".join(tok.text for tok in parse_ssml_tokens(ssml))
    return text, ssml


def resolve_audio(sample: Dict, audio_field: str) -> Tuple[np.ndarray, int]:
    audio_obj = sample[audio_field]
    if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
        wav = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        return wav, sr
    raise ValueError(f"Unsupported audio format for field={audio_field}: {type(audio_obj)}")


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def render_figure(
    wav: np.ndarray,
    sr: int,
    aligned_words: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    stress_flags: Sequence[bool],
    output_path: str,
    hop_length: int,
    n_mels: int,
    dpi: int,
) -> None:
    duration = float(len(wav)) / float(sr) if len(wav) > 0 else 0.0
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2, 1.2]},
    )

    ax_spec, ax_word, ax_prosody = axes

    ax_spec.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, n_mels],
        cmap="magma",
    )
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_title("Word-level Forced Alignment")

    for w, st, et, is_stress in zip(aligned_words, starts, ends, stress_flags):
        color = "#ef4444" if is_stress else "#60a5fa"
        alpha = 0.25 if is_stress else 0.10
        for ax in axes:
            ax.axvspan(st, et, color=color, alpha=alpha, linewidth=0)
        ax_spec.vlines([st, et], 0, n_mels, colors=color, alpha=0.35, linewidth=0.6)

    ax_word.set_ylim(0, 1)
    ax_word.set_yticks([])
    ax_word.set_ylabel("Word")
    for w, st, et, is_stress in zip(aligned_words, starts, ends, stress_flags):
        mid = (st + et) / 2.0
        ax_word.text(
            mid,
            0.5,
            w,
            ha="center",
            va="center",
            fontsize=9,
            color="#dc2626" if is_stress else "#111827",
            fontweight="bold" if is_stress else "normal",
        )

    ax_prosody.set_ylim(0, 1)
    ax_prosody.set_yticks([])
    ax_prosody.set_ylabel("Prosody")
    for st, et, is_stress in zip(starts, ends, stress_flags):
        if not is_stress:
            continue
        mid = (st + et) / 2.0
        ax_prosody.text(
            mid,
            0.5,
            PROSODY_LABEL,
            ha="center",
            va="center",
            fontsize=8,
            color="#991b1b",
        )

    ax_prosody.set_xlim(0, duration)
    ax_prosody.set_xlabel("Time (s)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def process_split(
    ds: Dataset,
    split: str,
    aligner: Qwen3ForcedAligner,
    args: argparse.Namespace,
) -> None:
    out_dir = os.path.join(args.output_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    limit = len(ds) if args.max_samples < 0 else min(args.max_samples, len(ds))

    for i in range(limit):
        sample = ds[i]
        text, ssml = resolve_text(sample, args.text_field, args.ssml_field)
        if not text:
            print(f"[WARN] skip split={split} idx={i}, missing text/ssml")
            continue

        wav, sr = resolve_audio(sample, args.audio_field)
        fa_result = aligner.align(audio=(wav, sr), text=text, language=args.language)[0]

        aligned_words = [it.text for it in fa_result]
        starts = [float(it.start_time) for it in fa_result]
        ends = [float(it.end_time) for it in fa_result]

        ssml_tokens = parse_ssml_tokens(ssml) if ssml else []
        stress_flags = align_stress_flags(aligned_words, ssml_tokens) if ssml_tokens else [False] * len(aligned_words)

        output_path = os.path.join(out_dir, f"{split}_{i:05d}.png")
        render_figure(
            wav=wav,
            sr=sr,
            aligned_words=aligned_words,
            starts=starts,
            ends=ends,
            stress_flags=stress_flags,
            output_path=output_path,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            dpi=args.dpi,
        )
        print(f"[INFO] saved {output_path}")


def main() -> None:
    args = parse_args()

    dtype = _dtype_from_name(args.dtype)
    aligner = Qwen3ForcedAligner.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
    )

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = load_dataset(args.dataset, split=split)
        if args.audio_field in ds.column_names:
            ds = ds.cast_column(args.audio_field, Audio())
        process_split(ds=ds, split=split, aligner=aligner, args=args)


if __name__ == "__main__":
    main()
