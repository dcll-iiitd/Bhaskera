"""
bhaskera.data.tokenize
======================
Stateful Ray-Data tokeniser with persistent caching.


"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray.data
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Bhaskera version written into metadata.json
_BHASKERA_VERSION = "2.3.0"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_version_hash(
    model_name: str,
    seq_len: int,
    dataset_name: str,
    format_name: Optional[str] = None,
    format_options: Optional[dict] = None,
) -> str:
    """
    Deterministic 16-char hex hash.

    NEVER use Python's hash() — it is randomised per-process since Python 3.3
    (PEP 456). hashlib.sha256 is stable across runs and machines.

    Phase 2: format_name + format_options are mixed into the key so a config
    change like ``format: chatml -> alpaca`` invalidates the cache without
    the user having to clear it manually. Old configs (no format set) hash
    to the same value as before, so existing caches keep working.
    """
    parts = [model_name, str(seq_len), dataset_name]
    if format_name:
        parts.append(f"fmt:{format_name}")
    if format_options:
        parts.append(f"opts:{json.dumps(format_options, sort_keys=True, default=str)}")
    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _write_metadata(
    cache_path: str,
    model_name: str,
    seq_len: int,
    dataset_name: str,
    num_rows: int,
    format_name: Optional[str] = None,
    format_options: Optional[dict] = None,
) -> None:
    """
    Write metadata.json alongside the parquet files.
    Must only be called from rank 0.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    meta = {
        "model_name":       model_name,
        "seq_len":          seq_len,
        "dataset_name":     dataset_name,
        "num_rows":         num_rows,
        "schema":           ["input_ids", "attention_mask", "labels"],
        "created_at":       datetime.datetime.utcnow().isoformat() + "Z",
        "bhaskera_version": _BHASKERA_VERSION,
        "format_name":      format_name,
        "format_options":   format_options or {},
    }
    meta_path = os.path.join(cache_path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Tokenizer cache metadata written → {meta_path}")


def _verify_cache(
    cache_path: str,
    model_name: str,
    seq_len: int,
    dataset_name: str,
    format_name: Optional[str] = None,
) -> bool:
    """
    Returns True only if ALL of:
      - metadata.json exists at cache_path
      - model_name, seq_len, dataset_name match
      - (if provided) format_name matches
      - at least one .parquet file exists in cache_path
    """
    meta_path = os.path.join(cache_path, "metadata.json")
    if not os.path.isfile(meta_path):
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Cache metadata corrupt or unreadable at {meta_path}")
        return False

    for key, expected in (
        ("model_name",   model_name),
        ("seq_len",      seq_len),
        ("dataset_name", dataset_name),
    ):
        if meta.get(key) != expected:
            logger.info(
                f"Cache miss: {key} mismatch "
                f"(cached={meta.get(key)!r}, requested={expected!r})"
            )
            return False

    # format_name is only checked if the caller supplied one. This means caches
    # written by older Bhaskera versions (no format_name in metadata) still
    # validate when the new code is asked for "no format" (format_name=None).
    if format_name is not None and meta.get("format_name") != format_name:
        logger.info(
            f"Cache miss: format_name mismatch "
            f"(cached={meta.get('format_name')!r}, requested={format_name!r})"
        )
        return False

    parquet_files = list(Path(cache_path).glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"Cache dir exists but contains no .parquet files: {cache_path}")
        return False

    return True


# ---------------------------------------------------------------------------
# Partition calculation (fix #10)
# ---------------------------------------------------------------------------

def _compute_num_partitions(cfg, world_size: int) -> int:
    """Round up to the nearest multiple of world_size; minimum 16."""
    base = max(world_size * 4, cfg.data.num_workers * 4, 16)
    return ((base + world_size - 1) // world_size) * world_size


# ---------------------------------------------------------------------------
# Persistent cache (fix #2)
# ---------------------------------------------------------------------------

def persist_tokenized(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
    dataset_name: str,
) -> str:
    """
    Tokenize *ds* once and write to disk as snappy/zstd/uncompressed parquet.

    Reads format settings from cfg.data:
      * cfg.data.format          — name of a registered format (or None)
      * cfg.data.format_options  — free-form dict, hashed into cache key

    Returns the absolute path to the cache directory.
    """
    if not cfg.data.cache_dir:
        raise ValueError(
            "cfg.data.cache_dir must be set to use persist_tokenized(). "
            "Set data.cache_dir in your config or pass --cache-dir."
        )

    model_name     = cfg.model.name
    seq_len        = cfg.data.seq_len
    format_name    = getattr(cfg.data, "format", None)
    format_options = dict(getattr(cfg.data, "format_options", None) or {})

    version    = _cache_version_hash(model_name, seq_len, dataset_name,
                                     format_name, format_options)
    cache_path = os.path.join(cfg.data.cache_dir, f"{dataset_name}_{version}")

    if (_verify_cache(cache_path, model_name, seq_len, dataset_name, format_name)
            and not cfg.data.overwrite_cache):
        logger.info(
            f"Tokenizer cache hit → {cache_path} "
            f"(model={model_name!r}, seq_len={seq_len}, "
            f"dataset={dataset_name!r}, format={format_name!r})"
        )
        return cache_path

    if cfg.data.overwrite_cache and os.path.exists(cache_path):
        logger.info(f"overwrite_cache=True — removing existing cache: {cache_path}")
        import shutil
        shutil.rmtree(cache_path)

    logger.info(
        f"Tokenizing dataset '{dataset_name}' → {cache_path} "
        f"(model={model_name!r}, seq_len={seq_len}, "
        f"format={format_name!r}, compression={cfg.data.tokenize_compression!r})"
    )

    tokenized_ds = _apply_map_batches(ds, cfg, text_col)

    Path(cache_path).mkdir(parents=True, exist_ok=True)
    tokenized_ds.write_parquet(
        cache_path,
        compression=cfg.data.tokenize_compression,
        num_rows_per_file=50_000,
    )

    try:
        num_rows = tokenized_ds.count()
    except Exception:
        num_rows = -1

    _write_metadata(cache_path, model_name, seq_len, dataset_name, num_rows,
                    format_name, format_options)

    logger.info(f"Tokenization complete → {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# Load pre-tokenized cache (called from driver before TorchTrainer)
# ---------------------------------------------------------------------------

def load_tokenized(tokenized_path: str, cfg, world_size: int) -> ray.data.Dataset:
    if not os.path.isdir(tokenized_path):
        raise FileNotFoundError(
            f"tokenized_path='{tokenized_path}' does not exist or is not a directory. "
            "Run: bhaskera-tokenize --config <config.yaml>"
        )

    parquet_files = list(Path(tokenized_path).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"tokenized_path='{tokenized_path}' exists but contains no .parquet files."
        )

    logger.info(
        f"Loading pre-tokenized dataset from {tokenized_path} "
        f"({len(parquet_files)} parquet file(s))"
    )

    ds = ray.data.read_parquet(tokenized_path)
    num_partitions = _compute_num_partitions(cfg, world_size)
    ds = ds.repartition(num_partitions)
    logger.info(
        f"Dataset repartitioned to {num_partitions} partitions "
        f"for world_size={world_size}"
    )
    return ds


# ---------------------------------------------------------------------------
# Public tokenize_dataset (dispatch)
# ---------------------------------------------------------------------------

def tokenize_dataset(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
    world_size: int = 1,
) -> ray.data.Dataset:
    """
    If cfg.data.tokenized_path is set, load from cache.
    Otherwise tokenize on the fly (re-tokenizes every run — use the CLI for prod).
    """
    if cfg.data.tokenized_path:
        return load_tokenized(cfg.data.tokenized_path, cfg, world_size)
    return _apply_map_batches(ds, cfg, text_col)


# ---------------------------------------------------------------------------
# Internal: TokenizerActor (now format-aware)
# ---------------------------------------------------------------------------

class TokenizerActor:
    """
    Loads the tokenizer once per Ray worker, tokenizes many batches.

    When ``format_name`` is set, each row is rendered to a string by the
    format registry before tokenisation, so the same actor handles raw
    pretrain text *and* chat data without an extra .map() pass.
    """

    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str = "text",
        trust_remote_code: bool = False,
        format_name: Optional[str] = None,
        format_options: Optional[dict] = None,
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            # Falcon, GPT-2, Llama-2, Param2 all need this.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len        = seq_len
        self.text_col       = text_col
        self.format_name    = format_name
        self.format_options = format_options or {}

        # Pre-import the format registry on this worker so the lookup is hot.
        if self.format_name:
            from bhaskera.data.formats import _ensure_builtins_loaded
            _ensure_builtins_loaded()

    # ------------------------------------------------------------------
    # Format rendering
    # ------------------------------------------------------------------
    def _render_batch(self, batch: dict) -> list[str]:
        """Render every row in the batch to a string using the chosen format."""
        from bhaskera.data.formats import render_with_format

        # Find the batch length from any column.
        any_col = next(iter(batch.values()))
        n = len(any_col)

        # Build per-row dicts. batch values arrive as numpy arrays / object arrays.
        texts: list[str] = []
        for i in range(n):
            row = {k: v[i] for k, v in batch.items()}
            texts.append(
                render_with_format(
                    self.format_name, row, self.tokenizer, self.format_options,
                )
            )
        return texts

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def __call__(self, batch: dict) -> dict:
        if self.format_name:
            texts = self._render_batch(batch)
        else:
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

        input_ids      = out["input_ids"]
        attention_mask = out["attention_mask"]

        # CRITICAL: mask pad positions so the LM loss ignores them.
        labels = input_ids.copy()
        labels[attention_mask == 0] = -100

        # fix #27: filter empty rows
        valid_mask = attention_mask.sum(axis=1) > 0
        if not valid_mask.all():
            n_bad = int((~valid_mask).sum())
            logger.warning(
                f"TokenizerActor: filtered {n_bad} empty row(s) "
                f"(text was empty or rendered to empty string)"
            )
            input_ids      = input_ids[valid_mask]
            attention_mask = attention_mask[valid_mask]
            labels         = labels[valid_mask]

        if len(input_ids) == 0:
            dummy = np.zeros((1, self.seq_len), dtype=np.int32)
            return {
                "input_ids":      dummy,
                "attention_mask": dummy,
                "labels":         np.full_like(dummy, -100),
            }

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


class _TokenizerActorFactory:
    """
    Lazy per-process initialiser for Ray-Data map_batches.

    fix #6: tokenizer is initialised lazily so Ray pickles a tiny stateless
    object instead of a live tokenizer. Phase 2 just adds format fields.
    """

    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str,
        trust_remote_code: bool = False,
        format_name: Optional[str] = None,
        format_options: Optional[dict] = None,
    ):
        self.model_name        = model_name
        self.seq_len           = seq_len
        self.text_col          = text_col
        self.trust_remote_code = trust_remote_code
        self.format_name       = format_name
        self.format_options    = format_options
        self._actor: Optional[TokenizerActor] = None

    def __call__(self, batch: dict) -> dict:
        if self._actor is None:
            self._actor = TokenizerActor(
                model_name=self.model_name,
                seq_len=self.seq_len,
                text_col=self.text_col,
                trust_remote_code=self.trust_remote_code,
                format_name=self.format_name,
                format_options=self.format_options,
            )
        return self._actor(batch)


def _apply_map_batches(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
) -> ray.data.Dataset:
    """
    Apply tokenisation via Ray Data map_batches.

    fix #27: batch_size from cfg.data.tokenize_batch_size.
    Phase 2: format_name / format_options pulled from cfg.data and threaded
    through the factory.
    """
    model_name        = cfg.model.name
    seq_len           = cfg.data.seq_len
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)
    num_workers       = getattr(cfg.data, "num_workers", 4)
    batch_size        = getattr(cfg.data, "tokenize_batch_size", 128)
    format_name       = getattr(cfg.data, "format", None)
    format_options    = dict(getattr(cfg.data, "format_options", None) or {})

    factory = _TokenizerActorFactory(
        model_name=model_name,
        seq_len=seq_len,
        text_col=text_col,
        trust_remote_code=trust_remote_code,
        format_name=format_name,
        format_options=format_options,
    )

    ds = ds.repartition(max(num_workers * 2, 1))

    return ds.map_batches(
        factory,
        batch_format="numpy",
        batch_size=batch_size,
        num_cpus=1,
        concurrency=num_workers,
    )
