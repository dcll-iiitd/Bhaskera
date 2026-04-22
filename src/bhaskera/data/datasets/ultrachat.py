"""UltraChat SFT dataset (HuggingFaceH4/ultrachat_200k)."""
from __future__ import annotations

import ray.data

from bhaskera.data.registry import register
from bhaskera.data.tokenize import tokenize_dataset


@register("ultrachat")
def build(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ray.data.from_huggingface(hf_ds)
    return tokenize_dataset(ds, cfg, text_col="prompt")