"""
bhaskera.launcher.tokenize
==========================
One-shot CLI that tokenizes a dataset, writes to persistent storage, and
prints the config snippet to paste into your training config.

Usage:
    bhaskera-tokenize --config configs/config.yaml
    bhaskera-tokenize --config configs/config.yaml --dataset ultrachat
    bhaskera-tokenize --config configs/config.yaml --storage-path /scratch/cache
    bhaskera-tokenize --config configs/config.yaml --overwrite --num-workers 16

After running, paste the printed YAML snippet into your config:

    data:
      tokenized_path: "/scratch/cache/ultrachat_<hash>"

Then training will skip tokenization entirely and load directly from disk.

fix #28: adds bhaskera-tokenize CLI entrypoint.
fix #2:  uses persist_tokenized() so subsequent runs hit the cache.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_BOX_WIDTH = 72


def main() -> None:
    args = _parse_args()

    # ── Load config ─────────────────────────────────────────────────
    from bhaskera.config import load_config
    cfg = load_config(args.config)

    # CLI overrides
    if args.dataset:
        cfg.data.name = args.dataset
    if args.storage_path:
        cfg.data.cache_dir = args.storage_path
    if args.overwrite:
        cfg.data.overwrite_cache = True
    if args.num_workers:
        cfg.data.num_workers = args.num_workers

    # Validate that cache_dir is set
    if not cfg.data.cache_dir:
        logger.error(
            "cfg.data.cache_dir is not set. "
            "Pass --storage-path <dir> or set data.cache_dir in your config."
        )
        sys.exit(1)

    dataset_name = cfg.data.name
    logger.info(f"bhaskera-tokenize | dataset={dataset_name!r} | config={args.config!r}")

    # ── Init Ray (no monitoring — headless tokenization job) ────────
    import ray
    ray.init(
        num_cpus=os.cpu_count(),
        include_dashboard=False,
        ignore_reinit_error=True,
    )
    logger.info("Ray initialized for tokenization.")

    # ── Import dataset builders (triggers @register_raw decorators) ─
    import bhaskera.data.datasets  # noqa: F401 — side-effect: populates RAW_REGISTRY

    from bhaskera.data.registry import RAW_REGISTRY, TEXT_COL
    from bhaskera.data.tokenize import persist_tokenized

    if dataset_name not in RAW_REGISTRY:
        logger.error(
            f"Dataset '{dataset_name}' is not registered in RAW_REGISTRY. "
            f"Available: {sorted(RAW_REGISTRY)}. "
            "Register it with @register_raw('name', text_col='...')."
        )
        ray.shutdown()
        sys.exit(1)

    if dataset_name not in TEXT_COL:
        logger.error(
            f"No text_col registered for dataset '{dataset_name}'. "
            "Use @register_raw('name', text_col='column_name')."
        )
        ray.shutdown()
        sys.exit(1)

    # ── Build raw dataset ───────────────────────────────────────────
    logger.info(f"Loading raw dataset '{dataset_name}' from HuggingFace...")
    raw_ds   = RAW_REGISTRY[dataset_name](cfg)
    text_col = TEXT_COL[dataset_name]

    # ── Tokenize and persist ────────────────────────────────────────
    cache_path = persist_tokenized(
        ds=raw_ds,
        cfg=cfg,
        text_col=text_col,
        dataset_name=dataset_name,
    )

    ray.shutdown()

    # ── Print result banner ─────────────────────────────────────────
    _print_result_box(cache_path, dataset_name, cfg)


def _print_result_box(cache_path: str, dataset_name: str, cfg) -> None:
    """Print a copy-pasteable config snippet inside a box."""
    snippet = (
        f"data:\n"
        f"  name: \"{dataset_name}\"\n"
        f"  tokenized_path: \"{cache_path}\"\n"
        f"  seq_len: {cfg.data.seq_len}"
    )

    lines = [
        "Tokenization complete!",
        "",
        f"Cache path: {cache_path}",
        "",
        "Paste this into your training config:",
        "",
    ] + snippet.splitlines()

    border = "═" * _BOX_WIDTH
    print(f"\n╔{border}╗")
    for line in lines:
        padding = _BOX_WIDTH - len(line) - 1
        print(f"║ {line}{' ' * max(padding, 0)}║")
    print(f"╚{border}╝\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bhaskera-tokenize",
        description=(
            "One-shot tokenization CLI. Tokenizes a dataset once and writes to disk. "
            "Subsequent runs with the same config return immediately (cache hit)."
        ),
    )
    p.add_argument(
        "--config", required=True,
        help="Path to Bhaskera YAML config (model.name and data.seq_len are read from here)",
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name to tokenize (overrides config.data.name)",
    )
    p.add_argument(
        "--storage-path", type=str, default=None,
        help="Root directory for tokenized parquet cache (overrides config.data.cache_dir)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Force re-tokenization even if a valid cache exists",
    )
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of Ray CPU workers for tokenization (overrides config.data.num_workers)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()