"""
bhaskera.launcher.train
=======================
Unified CLI + Ray Train driver.

Local (1–N GPUs):
    python -m bhaskera.launcher.train --config configs/config.yaml

SLURM (called by slurm/submit.sh after Ray cluster is bootstrapped):
    python -m bhaskera.launcher.train --config configs/config.yaml --num-workers 8
"""
from __future__ import annotations
import argparse
import logging
import os
import subprocess

import ray
import ray.data
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from bhaskera.config import load_config
from bhaskera.data import build_ray_dataset
from bhaskera.launcher.worker import worker_fn

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)

    _init_ray()

    # Dataset — built once on the driver, Ray distributes shards to workers
    ray_dataset = build_ray_dataset(cfg)

    num_workers = args.num_workers or _count_gpus()
    logger.info(f"Launching with {num_workers} GPU worker(s)")

    trainer = TorchTrainer(
        train_loop_per_worker=worker_fn,
        train_loop_config=cfg.as_dict(),
        datasets={"train": ray_dataset},
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
        run_config=RunConfig(
            name=cfg.logging.run_name,
            storage_path=os.path.abspath(args.storage_path or cfg.checkpoint.save_dir),
            checkpoint_config=CheckpointConfig(num_to_keep=cfg.checkpoint.keep_last_n),
            failure_config=ray.train.FailureConfig(max_failures=args.max_failures),
        ),
    )

    result = trainer.fit()
    logger.info(f"Training finished | best checkpoint: {result.best_checkpoints}")


# ---------------------------------------------------------------------------
# Ray init
# ---------------------------------------------------------------------------

def _init_ray() -> None:
    if ray.is_initialized():
        return

    slurm_address = os.environ.get("RAY_ADDRESS")  # set by slurm/submit.sh

    if slurm_address:
        # SLURM: submit.sh already started the cluster and exported RAY_ADDRESS
        logger.info(f"Connecting to Ray cluster at {slurm_address}")
        ray.init(address=slurm_address)
    else:
        # Local: kill any stale Ray session from a previous job on this node,
        # then start a clean single-node cluster using all local GPUs.
        logger.info("Stopping any stale Ray session...")
        subprocess.run(["ray", "stop", "--force"], capture_output=True)

        # Strip any leftover Ray env vars the shell may have inherited
        for var in ("RAY_ADDRESS", "RAY_HEAD_SERVICE_HOST", "RAY_HEAD_SERVICE_PORT"):
            os.environ.pop(var, None)

        n_gpus = _count_gpus()
        logger.info(f"Starting local Ray cluster ({n_gpus} GPU(s))")
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=n_gpus,
            include_dashboard=False,   # skip dashboard in interactive sessions
        )

    logger.info(f"Ray resources: {ray.available_resources()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bhaskera training launcher")
    p.add_argument("--config",        required=True,          help="Path to YAML config")
    p.add_argument("--num-workers",   type=int, default=None, help="Number of GPU workers (default: all visible GPUs)")
    p.add_argument("--max-failures",  type=int, default=2,    help="Ray fault tolerance — worker restart limit")
    p.add_argument("--storage-path",  type=str, default=None, help="Ray Train storage path (overrides config)")
    return p.parse_args()


def _count_gpus() -> int:
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No GPUs found. Check your CUDA installation.")
    return n


if __name__ == "__main__":
    main()
