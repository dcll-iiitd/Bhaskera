"""
bhaskera.launcher.tokenize_cpu
==============================
Ray-free tokenizer for laptops / CPU-only machines.

Produces the *same* on-disk cache as ``bhaskera.launcher.tokenize``
(``<cache_dir>/<name>_<split>_<hash>/*.parquet`` + ``metadata.json``, same
five-column schema), so the rest of the pipeline cannot tell the difference.
It reuses ``TokenizerActor`` for rendering / label masking / packing, so the
tokens are identical to the GPU path.

Usage:
    python -m bhaskera.launcher.tokenize_cpu --config node_cfg.yaml [--split train|val|both]

Extra knob (CPU only):
    data.max_train_samples: N    # tokenize only the first N raw rows
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_JSON_EXTS = (".jsonl", ".json")
_PARQUET_EXTS = (".parquet",)
_SCHEMA = ["input_ids", "attention_mask", "labels", "position_ids", "seq_idx"]


# ---------------------------------------------------------------------------
# Raw-row readers (pure Python / pyarrow — no Ray)
# ---------------------------------------------------------------------------

def _resolve_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files: list[Path] = []
        for ext in _JSON_EXTS + _PARQUET_EXTS:
            files.extend(sorted(p.rglob(f"*{ext}")))
        return [str(f) for f in files]
    matches = sorted(glob.glob(str(p)))
    if not matches:
        raise FileNotFoundError(
            f"local dataset path '{path}' matched no files. "
            "Provide a file, a directory, or a glob pattern."
        )
    return matches


def _iter_rows(files: List[str]) -> Iterator[dict]:
    for f in files:
        low = f.lower()
        if low.endswith(".jsonl"):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif low.endswith(".json"):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data = [data]
            yield from data
        elif low.endswith(".parquet"):
            import pyarrow.parquet as pq
            for row in pq.read_table(f).to_pylist():
                yield row
        else:
            raise ValueError(f"Unsupported file type: {f}")


def _path_for_split(cfg, split: str) -> str:
    data = cfg.data
    if split == "val":
        if not getattr(data, "val_path", None):
            raise ValueError("--split val requested but data.val_path is not set.")
        return data.val_path
    chosen = getattr(data, "train_path", None) or getattr(data, "path", None)
    if not chosen:
        raise ValueError("data.train_path (or data.path) must be set.")
    return chosen


# ---------------------------------------------------------------------------
# Tokenise + write parquet
# ---------------------------------------------------------------------------

def tokenize_split(cfg, split: str) -> str:
    """Tokenize one split to a parquet cache. Returns the cache directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bhaskera.data.tokenize import (
        TokenizerActor,
        _cache_version_hash,
        _verify_cache,
        _write_metadata,
    )

    dataset_name = f"{cfg.data.name}_{split}"
    model_name = cfg.model.name
    seq_len = cfg.data.seq_len
    format_name = getattr(cfg.data, "format", None)
    format_options = dict(getattr(cfg.data, "format_options", None) or {})
    is_cpt = getattr(cfg.data, "is_cpt", False)
    pack = getattr(cfg.data, "pack_sequences", False)
    train_on_inputs = getattr(cfg.data, "train_on_inputs", None)
    if train_on_inputs is None:
        train_on_inputs = is_cpt

    version = _cache_version_hash(
        model_name, seq_len, dataset_name, format_name, format_options,
        is_cpt, pack, train_on_inputs,
    )
    cache_path = os.path.join(cfg.data.cache_dir, f"{dataset_name}_{version}")

    if (_verify_cache(cache_path, model_name, seq_len, dataset_name, format_name,
                      is_cpt, pack, train_on_inputs)
            and not cfg.data.overwrite_cache):
        logger.info(f"Tokenizer cache hit -> {cache_path}")
        return cache_path

    if os.path.isdir(cache_path):
        import shutil
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    max_rows: Optional[int] = getattr(cfg.data, "max_train_samples", None)
    if max_rows is not None:
        max_rows = int(max_rows)

    files = _resolve_files(_path_for_split(cfg, split))
    logger.info(
        f"CPU tokenizer: split={split!r} -> {len(files)} file(s), "
        f"model={model_name!r}, seq_len={seq_len}, pack={pack}, max_rows={max_rows}"
    )

    actor = TokenizerActor(
        model_name=model_name,
        seq_len=seq_len,
        text_col="text",
        trust_remote_code=getattr(cfg.model, "trust_remote_code", False),
        format_name=format_name,
        format_options=format_options,
        is_cpt=is_cpt,
        pack_sequences=pack,
        train_on_inputs=train_on_inputs,
    )

    batch_size = int(getattr(cfg.data, "tokenize_batch_size", 128))
    pa_schema = pa.schema([(c, pa.list_(pa.int32())) for c in _SCHEMA])
    out_file = os.path.join(cache_path, "part-00000.parquet")
    writer = pq.ParquetWriter(
        out_file, pa_schema,
        compression=None if cfg.data.tokenize_compression == "none"
        else cfg.data.tokenize_compression,
    )

    num_rows = 0
    seen = 0
    pending: list[dict] = []

    def _flush(rows: list[dict]) -> int:
        if not rows:
            return 0
        # TokenizerActor expects a column dict: {col: [values...]}.
        cols = {k: [r.get(k) for r in rows] for k in rows[0].keys()}
        out = actor(cols)
        # Drop dummy rows AND rows with no supervised token. The latter happens when a
        # long prompt is truncated at seq_len before its assistant turn; a micro-batch
        # made only of such rows gives CrossEntropy 0/0 = NaN, and they cost compute
        # while teaching nothing.
        keep = (np.asarray(out["attention_mask"]).sum(axis=1) > 0) & (
            (np.asarray(out["labels"]) != -100).any(axis=1)
        )
        if not keep.any():
            return 0
        table = pa.table(
            {c: pa.array(list(np.asarray(out[c], dtype=np.int32)[keep]),
                         type=pa.list_(pa.int32())) for c in _SCHEMA},
            schema=pa_schema,
        )
        writer.write_table(table)
        return int(keep.sum())

    try:
        for row in _iter_rows(files):
            pending.append(row)
            seen += 1
            if len(pending) >= batch_size:
                num_rows += _flush(pending)
                pending = []
            if max_rows is not None and seen >= max_rows:
                break
        num_rows += _flush(pending)
    finally:
        writer.close()

    if num_rows == 0:
        raise RuntimeError(
            "Tokenization produced 0 trainable rows — check data.format / "
            "data.train_path (conversations need an assistant turn)."
        )

    _write_metadata(
        cache_path, model_name, seq_len, dataset_name, num_rows,
        format_name, format_options, is_cpt, pack, train_on_inputs,
    )
    logger.info(f"CPU tokenization complete: {seen} raw rows -> {num_rows} rows -> {cache_path}")
    return cache_path


def main() -> None:
    p = argparse.ArgumentParser(prog="bhaskera-tokenize-cpu", description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "both"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from bhaskera.config import load_config
    cfg = load_config(args.config)
    # max_train_samples is a Slakshna-level knob not (yet) in DataConfig.
    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    cfg.data.max_train_samples = (raw.get("data") or {}).get("max_train_samples")
    if args.overwrite:
        cfg.data.overwrite_cache = True
    if not cfg.data.cache_dir:
        logger.error("data.cache_dir must be set.")
        sys.exit(1)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        path = tokenize_split(cfg, s)
        print(f"tokenized_path[{s}]: {path}")


if __name__ == "__main__":
    main()
