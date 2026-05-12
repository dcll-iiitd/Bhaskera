"""
bhaskera.launcher.train
=======================
Unified CLI + Ray Train driver.

Local (1–N GPUs):
    python -m bhaskera.launcher.train --config configs/config.yaml

SLURM (called by scripts/submit.sh after Ray cluster is bootstrapped):
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
from bhaskera.launcher.monitoring import setup_monitoring
from bhaskera.launcher.worker import worker_fn

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)

    if args.no_dashboard:
        cfg.monitoring.dashboard = False
    if args.dashboard_port:
        cfg.monitoring.dashboard_port = args.dashboard_port

    monitoring = setup_monitoring(cfg)

    _init_ray(monitoring)

    logger.info(monitoring.banner())

    # fix #26: resolve world_size before building the dataset so
    # partitioning reflects the actual cluster size (not just head-node GPUs)
    num_workers = args.num_workers or _count_gpus()
    logger.info(f"Launching with {num_workers} GPU worker(s)")

    # fix #10: pass world_size so build_ray_dataset can partition correctly
    ray_dataset = build_ray_dataset(cfg, world_size=num_workers)

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

def _init_ray(monitoring) -> None:
    if ray.is_initialized():
        return

    slurm_address = os.environ.get("RAY_ADDRESS")

    if slurm_address:
        logger.info(f"Connecting to Ray cluster at {slurm_address}")
        ray.init(address=slurm_address)
    else:
        logger.info("Stopping any stale Ray session...")
        subprocess.run(["ray", "stop", "--force"], capture_output=True)

        for var in ("RAY_ADDRESS", "RAY_HEAD_SERVICE_HOST", "RAY_HEAD_SERVICE_PORT"):
            os.environ.pop(var, None)

        n_gpus = _count_gpus()
        logger.info(f"Starting local Ray cluster ({n_gpus} GPU(s))")

        init_kwargs = {
            "num_cpus": os.cpu_count(),
            "num_gpus": n_gpus,
        }
        init_kwargs.update(monitoring.ray_init_kwargs())
        ray.init(**init_kwargs)

    logger.info(f"Ray resources: {ray.available_resources()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bhaskera training launcher")
    p.add_argument("--config",         required=True,          help="Path to YAML config")
    p.add_argument("--num-workers",    type=int, default=None, help="Number of GPU workers (default: all visible GPUs)")
    p.add_argument("--max-failures",   type=int, default=2,    help="Ray fault tolerance — worker restart limit")
    p.add_argument("--storage-path",   type=str, default=None, help="Ray Train storage path (overrides config)")
    p.add_argument("--no-dashboard",   action="store_true",    help="Disable Ray Dashboard for this run (overrides config)")
    p.add_argument("--dashboard-port", type=int, default=None, help="Ray Dashboard port (overrides config)")
    return p.parse_args()


def _count_gpus() -> int:
    """
    fix #26: returns the true total GPU count across the SLURM job.

    On SLURM, torch.cuda.device_count() only sees GPUs on the head node
    (typically 0 or 1 on login nodes). The correct count comes from
    SLURM_NNODES × SLURM_GPUS_PER_NODE when both are set.

    Priority:
      1. SLURM_NNODES × SLURM_GPUS_PER_NODE (multi-node SLURM job)
      2. torch.cuda.device_count()            (local / single-node)
    """
    import torch

    slurm_nodes = int(os.environ.get("SLURM_NNODES", 0))
    slurm_gpus  = int(os.environ.get("SLURM_GPUS_PER_NODE", 0))

    if slurm_nodes > 0 and slurm_gpus > 0:
        total = slurm_nodes * slurm_gpus
        logger.info(
            f"SLURM GPU count: {slurm_nodes} nodes × {slurm_gpus} GPUs/node = {total} total"
        )
        return total

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError(
            "No GPUs found. Check your CUDA installation. "
            "If running on SLURM, ensure SLURM_NNODES and SLURM_GPUS_PER_NODE are set."
        )
    return n


if __name__ == "__main__":
    main()