"""
bhaskera.data
=============
Ray Data–native dataset pipeline.

Adding a new dataset: create a function with signature
    build(cfg) -> ray.data.Dataset
and register it with @register('name'). That's it.
"""
from __future__ import annotations
from typing import Callable
import logging

import ray.data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a dataset builder."""
    def _wrap(fn):
        REGISTRY[name] = fn
        return fn
    return _wrap


def build_ray_dataset(cfg) -> ray.data.Dataset:
    name = cfg.data.name
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(REGISTRY)}. "
            "Register yours with @register('name')."
        )
    logger.info(f"Building Ray dataset: {name}")
    return REGISTRY[name](cfg)


# ---------------------------------------------------------------------------
# Built-in datasets
# ---------------------------------------------------------------------------

@register("ultrachat")
def _ultrachat(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ray.data.from_huggingface(hf_ds)
    return _tokenize_dataset(ds, cfg, text_col="prompt")


@register("openassistant")
def _openassistant(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("OpenAssistant/oasst1", split="train")
    ds = ray.data.from_huggingface(hf_ds)
    return _tokenize_dataset(ds, cfg, text_col="text")


@register("redpajama")
def _redpajama(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    ds = ray.data.from_huggingface(hf_ds)
    return _tokenize_dataset(ds, cfg, text_col="text")


# ---------------------------------------------------------------------------
# Tokenisation — class-based actor for efficiency
# ---------------------------------------------------------------------------
# BEFORE: tokenizer was instantiated INSIDE the map function, meaning it was
#         loaded from disk for EVERY batch on EVERY worker.
# AFTER:  TokenizerActor loads the tokenizer ONCE in __init__, reuses across
#         all batches assigned to that actor. Much faster for large tokenizers
#         (like Param2's Indic language tokenizer).
# ---------------------------------------------------------------------------

class TokenizerActor:
    """
    Stateful batch tokenizer for Ray Data.
    Loads the tokenizer once; processes many batches.
    """

    def __init__(self, model_name: str, seq_len: int, trust_remote_code: bool = False):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self._text_col: str = "text"  # set by caller via _TokenizerActorFactory

    def __call__(self, batch: dict) -> dict:
        import numpy as np

        texts = batch[self._text_col]
        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        out = self.tokenizer(
            texts,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return {
            "input_ids":      out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels":         out["input_ids"].copy(),
        }


class _TokenizerActorFactory:
    """
    Factory that creates TokenizerActor instances with the right text_col.
    This is needed because Ray Data's class-based map_batches expects a callable
    class, and we need to pass the text_col dynamically.
    """

    def __init__(self, model_name: str, seq_len: int, text_col: str,
                 trust_remote_code: bool = False):
        self.model_name = model_name
        self.seq_len = seq_len
        self.text_col = text_col
        self.trust_remote_code = trust_remote_code

    def __call__(self, batch: dict) -> dict:
        # Lazy init — loads tokenizer on first call, then reuses
        if not hasattr(self, "_actor"):
            self._actor = TokenizerActor(
                self.model_name, self.seq_len, self.trust_remote_code
            )
            self._actor._text_col = self.text_col
        return self._actor(batch)


def _tokenize_dataset(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
) -> ray.data.Dataset:
    model_name = cfg.model.name
    seq_len = cfg.data.seq_len
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)
    num_workers = getattr(cfg.data, "num_workers", 4)

    factory = _TokenizerActorFactory(
        model_name=model_name,
        seq_len=seq_len,
        text_col=text_col,
        trust_remote_code=trust_remote_code,
    )
    ds = ds.repartition(num_workers * 2)
    return ds.map_batches(
        factory,
        batch_format="numpy",
        batch_size=256,
        num_cpus=1,
        concurrency=num_workers,
    )
