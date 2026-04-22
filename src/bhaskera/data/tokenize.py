"""
bhaskera.data.tokenize
======================
Stateful Ray-Data tokeniser.

CRITICAL FIX vs v1:
    Old `labels = input_ids.copy()` caused the model to be trained to predict
    pad tokens everywhere attention_mask == 0. HuggingFace CausalLM uses
    CrossEntropyLoss(ignore_index=-100), so the correct label for pad
    positions is -100.  We now apply that mask before returning.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import ray.data

logger = logging.getLogger(__name__)


class TokenizerActor:
    """Loads the tokeniser once, tokenises many batches."""

    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str = "text",
        trust_remote_code: bool = False,
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            # Falcon, GPT-2, Llama-2 all need this.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.text_col = text_col

    def __call__(self, batch: dict) -> dict:
        texts = batch[self.text_col]
        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        out = self.tokenizer(
            texts,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]

        # ── CRITICAL ─────────────────────────────────────────────────
        # Mask pad positions so the LM loss ignores them.
        # Without this, CausalLM loss is diluted across every padded
        # position (often 60–80 % of the sequence).
        labels = input_ids.copy()
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


class _TokenizerActorFactory:
    """Lazy per-process initialiser for Ray-Data map_batches."""

    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.seq_len = seq_len
        self.text_col = text_col
        self.trust_remote_code = trust_remote_code
        self._actor: Optional[TokenizerActor] = None

    def __call__(self, batch: dict) -> dict:
        if self._actor is None:
            self._actor = TokenizerActor(
                model_name=self.model_name,
                seq_len=self.seq_len,
                text_col=self.text_col,
                trust_remote_code=self.trust_remote_code,
            )
        return self._actor(batch)


def tokenize_dataset(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
) -> ray.data.Dataset:
    """Apply tokenisation to a Ray dataset using the given config."""
    model_name         = cfg.model.name
    seq_len            = cfg.data.seq_len
    trust_remote_code  = getattr(cfg.model, "trust_remote_code", False)
    num_workers        = getattr(cfg.data, "num_workers", 4)

    factory = _TokenizerActorFactory(
        model_name=model_name,
        seq_len=seq_len,
        text_col=text_col,
        trust_remote_code=trust_remote_code,
    )

    ds = ds.repartition(max(num_workers * 2, 1))
    return ds.map_batches(
        factory,
        batch_format="numpy",
        batch_size=256,
        num_cpus=1,
        concurrency=num_workers,
    )