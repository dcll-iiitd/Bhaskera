"""
bhaskera.launcher.worker
========================
Per-GPU entry point.  Called by Ray Train's TorchTrainer for each
actor and also directly by raw SLURM workers.

Changes vs v2:
    * **Every rank now gets a logger**, not just rank 0.  Rank-0
      logger fans out to W&B / MLflow as before; non-rank-0 loggers
      contain only the Ray Prometheus child so per-GPU stats from
      every worker land on the dashboard with a ``rank`` tag.
    * Passes ``rank`` and ``world_size`` into ``build_logger`` so the
      backends can tag samples appropriately.
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import ray.train
import torch

from bhaskera.config import Config
from bhaskera.distributed import wrap_model
from bhaskera.models import build_model
from bhaskera.trainer import train
from bhaskera.utils import build_logger

logger = logging.getLogger(__name__)


def worker_fn(cfg_dict: dict) -> None:
    """Entry point for a single GPU worker."""
    cfg = Config.from_dict(cfg_dict)

    ray_ctx    = ray.train.get_context()
    local_rank = ray_ctx.get_local_rank()
    rank       = ray_ctx.get_world_rank()
    world_size = ray_ctx.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    _seed_everything(cfg.training.seed, rank, cfg.training.deterministic)

    logger.info(f"[rank {rank}/{world_size}] GPU {local_rank} ready")

    dataset = ray.train.get_dataset_shard("train")

    strategy    = cfg.training.distributed.strategy.lower()
    load_device = torch.device("cpu") if strategy == "fsdp" else device

    model, profile = build_model(cfg, load_device)

    if rank == 0:
        logger.info(
            f"Model profile: moe={profile.is_moe} | "
            f"decoder={profile.decoder_layer_cls.__name__ if profile.decoder_layer_cls else 'None'} | "
            f"experts={len(profile.expert_modules)} | "
            f"dtype={profile.model_dtype} | "
            f"has_aux_loss={profile.has_aux_loss}"
        )

    model = wrap_model(model, cfg, local_rank, profile)

    # Every rank builds a logger.  build_logger() decides internally
    # which backends to enable for non-rank-0 (Ray-only).
    tracker = build_logger(cfg, rank=rank, world_size=world_size)

    train(
        model=model,
        dataset=dataset,
        cfg=cfg,
        profile=profile,
        rank=rank,
        local_rank=local_rank,
        tracker=tracker,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_everything(base_seed: int, rank: int, deterministic: bool) -> None:
    """
    Seed PyTorch, NumPy and Python RNGs with a rank-offset seed so each
    rank shuffles its data differently but every run is reproducible.
    """
    seed = int(base_seed) + int(rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not enable deterministic mode: {e}")
