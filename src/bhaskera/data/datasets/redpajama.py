"""RedPajama sample dataset."""
from __future__ import annotations

import ray.data

from bhaskera.data.registry import register
from bhaskera.data.tokenize import tokenize_dataset


@register("redpajama")
def build(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    ds = ray.data.from_huggingface(hf_ds)
    return tokenize_dataset(ds, cfg, text_col="text")