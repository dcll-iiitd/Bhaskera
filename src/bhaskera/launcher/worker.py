"""
bhaskera.launcher.worker
========================
The per-GPU training function.
Called by Ray Train workers (via TorchTrainer) AND by raw SLURM workers.
This is the ONLY place that wires model + data + trainer together.

The ModelProfile from introspect.py flows through here to distributed/ and trainer/.
"""
from __future__ import annotations

import logging

import torch
import ray.train
from omegaconf import OmegaConf, DictConfig

from bhaskera.models import build_model
from bhaskera.distributed import wrap_model
from bhaskera.trainer import train
from bhaskera.utils import build_logger

logger = logging.getLogger(__name__)


def worker_fn(cfg_dict: dict) -> None:
    """
    Entry point for a single GPU worker.
    Ray Train calls this inside each actor.

    cfg_dict: plain dict (JSON-serialisable) so Ray can ship it across the cluster.
    """
    # Convert to OmegaConf DictConfig for attribute access
    if isinstance(cfg_dict, DictConfig):
        cfg = cfg_dict
    elif isinstance(cfg_dict, dict):
        cfg = OmegaConf.create(cfg_dict)
    else:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(cfg_dict):
            cfg = OmegaConf.create(asdict(cfg_dict))
        else:
            cfg = OmegaConf.create(dict(cfg_dict))

    ray_ctx    = ray.train.get_context()
    local_rank = ray_ctx.get_local_rank()
    rank       = ray_ctx.get_world_rank()
    world_size = ray_ctx.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"[rank {rank}/{world_size}] GPU {local_rank} ready")

    # Data — Ray Dataset shard is injected by TorchTrainer
    dataset = ray.train.get_dataset_shard("train")

    # Model — load on CPU for FSDP2 (shards to GPU during wrap), GPU for DDP
    strategy = cfg.training.distributed.strategy.lower()
    load_device = torch.device("cpu") if strategy == "fsdp" else device

    # build_model now returns (model, profile) — profile contains all
    # auto-detected info (MoE topology, layer classes, dtype, etc.)
    model, profile = build_model(cfg, load_device)

    if rank == 0:
        logger.info(
            f"Model profile: moe={profile.is_moe} | "
            f"decoder={profile.decoder_layer_cls.__name__ if profile.decoder_layer_cls else 'None'} | "
            f"experts={len(profile.expert_modules)} | "
            f"dtype={profile.model_dtype} | "
            f"has_aux_loss={profile.has_aux_loss}"
        )

    # Distributed wrap (FSDP2 or DDP) — uses profile for auto layer detection
    model = wrap_model(model, cfg, local_rank, profile)

    # Logger (rank 0 only to avoid duplicate logging)
    tracker = build_logger(cfg) if rank == 0 else None

    # Train — profile tells the loop about MoE aux loss, autocast dtype, etc.
    train(
        model=model,
        dataset=dataset,
        cfg=cfg,
        profile=profile,
        rank=rank,
        local_rank=local_rank,
        tracker=tracker,
    )
