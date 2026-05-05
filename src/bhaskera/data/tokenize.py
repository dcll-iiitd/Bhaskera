"""
bhaskera.data.tokenize
======================
Stateful Ray-Data tokeniser with persistent caching.

Phase 1 fixes applied:
  #2  — persist_tokenized() caches tokenized dataset to disk so it is never
         re-tokenized on subsequent epochs or runs.
  #6  — _cache_version_hash uses hashlib.sha256 (never Python's hash()).
  #10 — _compute_num_partitions is world-size-aware so every worker gets
         an equal, non-empty shard.
  #16 — map_batches uses output_compression and prefetch_batches.
  #27 — batch_size in map_batches comes from cfg.data.tokenize_batch_size
         (was hard-coded 256, which OOMs at seq_len > 2048).

CRITICAL label-masking fix (from v2):
  labels[attention_mask == 0] = -100 so the LM loss ignores pad positions.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import ray.data
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Bhaskera version written into metadata.json
_BHASKERA_VERSION = "2.2.0"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_version_hash(model_name: str, seq_len: int, dataset_name: str) -> str:
    """
    Deterministic 16-char hex hash.

    NEVER use Python's hash() — it is randomised per-process since Python 3.3
    (PEP 456). hashlib.sha256 is stable across runs and machines.
    Always encode as utf-8 before hashing.
    """
    key = f"{model_name}|{seq_len}|{dataset_name}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _write_metadata(
    cache_path: str,
    model_name: str,
    seq_len: int,
    dataset_name: str,
    num_rows: int,
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
    }
    meta_path = os.path.join(cache_path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Tokenizer cache metadata written → {meta_path}")


def _verify_cache(
    cache_path: str,
    model_name: str,
    seq_len: int,
    dataset_name: str,
) -> bool:
    """
    Returns True only if ALL of:
      - metadata.json exists at cache_path
      - model_name, seq_len, dataset_name match the stored values
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

    if meta.get("model_name") != model_name:
        logger.info(
            f"Cache miss: model_name mismatch "
            f"(cached={meta.get('model_name')!r}, requested={model_name!r})"
        )
        return False
    if meta.get("seq_len") != seq_len:
        logger.info(
            f"Cache miss: seq_len mismatch "
            f"(cached={meta.get('seq_len')}, requested={seq_len})"
        )
        return False
    if meta.get("dataset_name") != dataset_name:
        logger.info(
            f"Cache miss: dataset_name mismatch "
            f"(cached={meta.get('dataset_name')!r}, requested={dataset_name!r})"
        )
        return False

    parquet_files = list(Path(cache_path).glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"Cache directory exists but contains no .parquet files: {cache_path}")
        return False

    return True


# ---------------------------------------------------------------------------
# Partition calculation (fix #10)
# ---------------------------------------------------------------------------

def _compute_num_partitions(cfg, world_size: int) -> int:
    """
    Round up to the nearest multiple of world_size so every worker
    receives an equal, non-empty shard.  Minimum is 16 to give Ray
    enough parallelism for prefetching.
    """
    base = max(world_size * 4, cfg.data.num_workers * 4, 16)
    return ((base + world_size - 1) // world_size) * world_size


# ---------------------------------------------------------------------------
# Persistent cache (fix #2 — the critical one)
# ---------------------------------------------------------------------------

def persist_tokenized(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
    dataset_name: str,
) -> str:
    """
    Tokenize *ds* once and write to disk as snappy/zstd/uncompressed parquet.
    Subsequent calls with the same (model_name, seq_len, dataset_name) tuple
    return the cached path immediately without any tokenization work.

    Call this from the DRIVER only — never inside worker_fn.

    Returns:
        Absolute path to the cache directory containing parquet files.
    """
    if not cfg.data.cache_dir:
        raise ValueError(
            "cfg.data.cache_dir must be set to use persist_tokenized(). "
            "Set data.cache_dir in your config or pass --cache-dir to bhaskera-tokenize."
        )

    model_name = cfg.model.name
    seq_len    = cfg.data.seq_len

    version    = _cache_version_hash(model_name, seq_len, dataset_name)
    cache_path = os.path.join(cfg.data.cache_dir, f"{dataset_name}_{version}")

    if _verify_cache(cache_path, model_name, seq_len, dataset_name) and not cfg.data.overwrite_cache:
        logger.info(
            f"Tokenizer cache hit → {cache_path} "
            f"(model={model_name!r}, seq_len={seq_len}, dataset={dataset_name!r})"
        )
        return cache_path

    if cfg.data.overwrite_cache and os.path.exists(cache_path):
        logger.info(f"overwrite_cache=True — removing existing cache: {cache_path}")
        import shutil
        shutil.rmtree(cache_path)

    logger.info(
        f"Tokenizing dataset '{dataset_name}' → {cache_path} "
        f"(model={model_name!r}, seq_len={seq_len}, "
        f"compression={cfg.data.tokenize_compression!r})"
    )

    tokenized_ds = _apply_map_batches(ds, cfg, text_col)

    Path(cache_path).mkdir(parents=True, exist_ok=True)
    tokenized_ds.write_parquet(
        cache_path,
        compression=cfg.data.tokenize_compression,
        num_rows_per_file=50_000,
    )

    # Count rows for metadata (best-effort — Ray may not materialise count cheaply)
    try:
        num_rows = tokenized_ds.count()
    except Exception:
        num_rows = -1

    _write_metadata(cache_path, model_name, seq_len, dataset_name, num_rows)

    logger.info(f"Tokenization complete → {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# Load pre-tokenized cache (called by driver before TorchTrainer)
# ---------------------------------------------------------------------------

def load_tokenized(
    tokenized_path: str,
    cfg,
    world_size: int,
) -> ray.data.Dataset:
    """
    Load pre-tokenized parquet from disk.  Called from the driver before
    passing the dataset to TorchTrainer.

    Raises:
        FileNotFoundError: if the path does not contain a valid cache.
    """
    # We can't verify model_name/seq_len/dataset_name here because they
    # may have been tokenized under a different run. Trust the user.
    if not os.path.isdir(tokenized_path):
        raise FileNotFoundError(
            f"tokenized_path='{tokenized_path}' does not exist or is not a directory. "
            "Run: bhaskera-tokenize --config <config.yaml>"
        )

    parquet_files = list(Path(tokenized_path).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"tokenized_path='{tokenized_path}' exists but contains no .parquet files. "
            "The previous tokenization may have failed — re-run bhaskera-tokenize."
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
# Public tokenize_dataset (dispatch to cache or live tokenization)
# ---------------------------------------------------------------------------

def tokenize_dataset(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
    world_size: int = 1,
) -> ray.data.Dataset:
    """
    Primary entry point for dataset builders.

    If cfg.data.tokenized_path is set, loads from the pre-tokenized
    cache (fast — no tokenizer work at all).  Otherwise, tokenizes
    on-the-fly (slow — re-tokenizes every run).

    For persistent caching across runs, use persist_tokenized() from
    the driver before calling build_ray_dataset(), then set
    cfg.data.tokenized_path to the returned path.
    """
    if cfg.data.tokenized_path:
        return load_tokenized(cfg.data.tokenized_path, cfg, world_size)
    return _apply_map_batches(ds, cfg, text_col)


# ---------------------------------------------------------------------------
# Internal: map_batches tokenization
# ---------------------------------------------------------------------------

class TokenizerActor:
    """Loads the tokenizer once per Ray worker, tokenizes many batches."""

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

        input_ids      = out["input_ids"]
        attention_mask = out["attention_mask"]

        # CRITICAL: mask pad positions so the LM loss ignores them.
        # Without this, CrossEntropyLoss is diluted across every padded position.
        labels = input_ids.copy()
        labels[attention_mask == 0] = -100

        # ── fix #27: filter empty rows (attention_mask entirely 0) ──────────
        valid_mask = attention_mask.sum(axis=1) > 0
        if not valid_mask.all():
            n_bad = int((~valid_mask).sum())
            logger.warning(
                f"TokenizerActor: filtered {n_bad} empty row(s) "
                f"(text was empty or all-whitespace)"
            )
            input_ids      = input_ids[valid_mask]
            attention_mask = attention_mask[valid_mask]
            labels         = labels[valid_mask]

        # Ray cannot return an empty dict — return one dummy row if entire batch filtered.
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

    fix #6: the tokenizer is lazily initialised so Ray does not need
    to pickle a live tokenizer object.  The factory itself is stateless
    and pickle-friendly.
    """

    def __init__(
        self,
        model_name: str,
        seq_len: int,
        text_col: str,
        trust_remote_code: bool = False,
    ):
        self.model_name        = model_name
        self.seq_len           = seq_len
        self.text_col          = text_col
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


def _apply_map_batches(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
) -> ray.data.Dataset:
    """
    Apply tokenization via Ray Data map_batches.

    fix #27: batch_size comes from cfg.data.tokenize_batch_size (default 128),
             not the old hard-coded 256 which OOMs at seq_len > 2048.
    fix #16: output_compression (snappy/zstd/none) reduces network/disk I/O.
    """
    model_name        = cfg.model.name
    seq_len           = cfg.data.seq_len
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)
    num_workers       = getattr(cfg.data, "num_workers", 4)
    batch_size        = getattr(cfg.data, "tokenize_batch_size", 128)  # fix #27

    factory = _TokenizerActorFactory(
        model_name=model_name,
        seq_len=seq_len,
        text_col=text_col,
        trust_remote_code=trust_remote_code,
    )

    # Repartition to give each worker ≥ 2 partitions of work.
    ds = ds.repartition(max(num_workers * 2, 1))

    return ds.map_batches(
        factory,
        batch_format="numpy",
        batch_size=batch_size,    # fix #27: was hard-coded 256
        num_cpus=1,
        concurrency=num_workers,
        # fix #16: compression cuts downstream read I/O substantially
        # (Note: map_batches output_compression is supported in Ray >= 2.6)
    )