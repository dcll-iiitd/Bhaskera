"""
bhaskera.trainer
================
Pure training loop — no distributed logic, no Ray, no SLURM.
Called identically from Ray Train workers and raw SLURM workers.

MoE-aware: captures auxiliary loss, logs expert utilization,
auto-detects autocast dtype.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from bhaskera.distributed import save_checkpoint
from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(
    *,
    model: torch.nn.Module,
    dataset: "ray.data.Dataset",  # noqa: F821 — Ray Data, passed in
    cfg,
    profile: ModelProfile,
    rank: int,
    local_rank: int,
    tracker=None,
) -> None:
    """
    Run the training loop.

    Args:
        model:      Distributed-wrapped model (FSDP2 or DDP).
        dataset:    Ray Dataset pre-tokenised by bhaskera.data.
        cfg:        Bhaskera Config object.
        profile:    ModelProfile from introspection.
        rank:       Global rank of this worker.
        local_rank: Local GPU index.
        tracker:    Optional logger (WandbLogger / MLflowLogger / None).
    """
    device     = torch.device(f"cuda:{local_rank}")
    train_cfg  = cfg.training
    ckpt_cfg   = cfg.checkpoint

    optimizer  = _build_optimizer(model, train_cfg)
    scheduler  = _build_scheduler(optimizer, train_cfg)

    step       = 0
    best_ckpts: list[tuple[float, str]] = []

    # Resume from checkpoint if available
    if ckpt_cfg.enabled:
        step = _maybe_resume(model, optimizer, ckpt_cfg.save_dir, device)

    for epoch in range(train_cfg.num_epochs):
        step, best_ckpts = _run_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            profile=profile,
            rank=rank,
            local_rank=local_rank,
            device=device,
            epoch=epoch,
            step=step,
            tracker=tracker,
            best_ckpts=best_ckpts,
        )
        if step >= train_cfg.max_steps:
            break

    if tracker and rank == 0:
        tracker.finish()
    if rank == 0:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    *, model, dataset, optimizer, scheduler, cfg, profile, rank, local_rank,
    device, epoch, step, tracker, best_ckpts,
):
    train_cfg = cfg.training
    ckpt_cfg  = cfg.checkpoint
    grad_accum = train_cfg.grad_accum
    micro  = 0
    epoch_loss = 0.0
    epoch_aux_loss = 0.0
    epoch_steps = 0

    # ── Resolve autocast dtype from profile (no hardcoding) ─────────
    autocast_dtype = _resolve_autocast_dtype(cfg, profile)

    # ── MoE config ──────────────────────────────────────────────────
    moe_cfg = getattr(cfg, "moe", None)
    aux_loss_weight = getattr(moe_cfg, "aux_loss_weight", 0.01) if moe_cfg else 0.01
    log_expert_util = (
        profile.is_moe
        and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_interval = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    # Ray Data: iter_torch_batches handles sharding across workers automatically
    loader = dataset.iter_torch_batches(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=1000,
        prefetch_batches=2,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        if step >= train_cfg.max_steps:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # ── Forward pass ────────────────────────────────────────────
        with torch.autocast("cuda", dtype=autocast_dtype):
            # For MoE: pass output_router_logits=True if the model supports it
            forward_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            if profile.is_moe and profile.has_aux_loss:
                forward_kwargs["output_router_logits"] = True

            out = model(**forward_kwargs)

            # ── Compute total loss (main + MoE auxiliary) ───────────
            main_loss = out.loss
            aux_loss = _extract_aux_loss(out, profile)

            if aux_loss is not None:
                total_loss = (main_loss + aux_loss_weight * aux_loss) / grad_accum
            else:
                total_loss = main_loss / grad_accum

        total_loss.backward()
        micro += 1

        if micro < grad_accum:
            continue

        # ── Gradient step ───────────────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.max_grad_norm
        ).item()

        if not math.isfinite(grad_norm):
            logger.warning(
                f"[rank {rank}][epoch {epoch}][step {step}] "
                f"Non-finite grad norm — skipping step"
            )
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        micro = 0

        actual_loss = main_loss.item()
        actual_aux = aux_loss.item() if aux_loss is not None else 0.0
        lr = scheduler.get_last_lr()[0]
        epoch_loss += actual_loss
        epoch_aux_loss += actual_aux
        epoch_steps += 1
        step += 1

        # ── Logging ─────────────────────────────────────────────────
        if rank == 0:
            log_msg = (
                f"[epoch {epoch}][step {step}] "
                f"loss={actual_loss:.4f} lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            metrics = {
                "loss": actual_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "epoch": epoch,
            }

            if profile.is_moe:
                log_msg += f" aux_loss={actual_aux:.4f}"
                metrics["aux_loss"] = actual_aux
                metrics["total_loss"] = actual_loss + aux_loss_weight * actual_aux

            logger.info(log_msg)

            if tracker:
                # Expert utilization logging (periodic, not every step)
                if log_expert_util and step % expert_log_interval == 0:
                    util_metrics = _compute_expert_utilization(out, profile)
                    metrics.update(util_metrics)

                tracker.log(metrics, step=step)

    if epoch_steps == 0:
        return step, best_ckpts

    avg_loss = epoch_loss / epoch_steps
    if rank == 0:
        epoch_msg = f"[epoch {epoch}] avg_loss={avg_loss:.4f}"
        epoch_metrics = {"epoch_avg_loss": avg_loss, "epoch": epoch}
        if profile.is_moe:
            avg_aux = epoch_aux_loss / epoch_steps
            epoch_msg += f" avg_aux_loss={avg_aux:.4f}"
            epoch_metrics["epoch_avg_aux_loss"] = avg_aux
        logger.info(epoch_msg)
        if tracker:
            tracker.log(epoch_metrics, step=step)

    # Checkpoint — ALL ranks participate (FSDP2 collective)
    if ckpt_cfg.enabled and (epoch + 1) % ckpt_cfg.save_interval == 0:
        best_ckpts = _checkpoint(
            model=model, optimizer=optimizer, step=step,
            avg_loss=avg_loss, ckpt_cfg=ckpt_cfg,
            rank=rank, best_ckpts=best_ckpts,
        )

    return step, best_ckpts


# ---------------------------------------------------------------------------
# MoE helpers
# ---------------------------------------------------------------------------

def _extract_aux_loss(out, profile: ModelProfile) -> Optional[torch.Tensor]:
    """
    Extract auxiliary loss from model output.
    Handles different MoE implementations:
      - out.aux_loss (Mixtral, Qwen2MoE, most HF MoE models)
      - out.router_logits → compute load balancing loss
    Returns None for dense models.
    """
    if not profile.is_moe:
        return None

    # Direct aux_loss attribute (most common)
    aux = getattr(out, "aux_loss", None)
    if aux is not None and isinstance(aux, torch.Tensor):
        return aux

    # Some models put it in a different attribute
    for attr in ("router_aux_loss", "moe_loss", "load_balancing_loss"):
        aux = getattr(out, attr, None)
        if aux is not None and isinstance(aux, torch.Tensor):
            return aux

    router_logits = getattr(out, "router_logits", None)
    if router_logits is not None:
        # Some custom MoE models return a tuple of tuples: (gate_logits, ...)
        extracted_logits = []
        for l in router_logits:
            if isinstance(l, tuple) or isinstance(l, list):
                # Ensure we hit the tensor
                extracted_logits.append(l[0])
            else:
                extracted_logits.append(l)
        router_logits = tuple(extracted_logits)
        
        return _load_balancing_loss_from_logits(router_logits, profile)

    return None


def _load_balancing_loss_from_logits(
    router_logits: tuple,
    profile: ModelProfile,
) -> Optional[torch.Tensor]:
    """
    Compute load balancing loss from router logits (HuggingFace format).
    router_logits is a tuple of (logits_tensor, ...) per layer.
    """
    try:
        from transformers.models.mixtral.modeling_mixtral import (
            load_balancing_loss_func,
        )
        return load_balancing_loss_func(
            router_logits,
            profile.num_experts,
            profile.experts_per_token,
        )
    except ImportError:
        pass

    # Manual fallback: simple load balancing
    try:
        total_loss = torch.tensor(0.0, device=router_logits[0].device)
        for logits in router_logits:
            if not isinstance(logits, torch.Tensor):
                continue
            # logits shape: (batch * seq_len, num_experts)
            probs = torch.softmax(logits, dim=-1)
            # Fraction of tokens assigned to each expert
            expert_mask = torch.zeros_like(probs)
            top_k_indices = probs.topk(profile.experts_per_token, dim=-1).indices
            expert_mask.scatter_(1, top_k_indices, 1.0)
            tokens_per_expert = expert_mask.mean(dim=0)
            avg_probs = probs.mean(dim=0)
            # Load balancing: wants uniform distribution
            total_loss += (tokens_per_expert * avg_probs).sum() * profile.num_experts
        return total_loss / len(router_logits)
    except Exception as e:
        logger.warning(f"Failed to compute load balancing loss: {e}")
        return None


def _compute_expert_utilization(out, profile: ModelProfile) -> dict:
    """Compute expert utilization metrics from router logits for logging."""
    metrics = {}
    router_logits = getattr(out, "router_logits", None)
    if router_logits is None:
        return metrics

    try:
        all_loads = []
        for layer_idx, logits in enumerate(router_logits):
            if not isinstance(logits, torch.Tensor):
                continue
            probs = torch.softmax(logits.float(), dim=-1)
            # Top-k routing
            top_k_indices = probs.topk(
                min(profile.experts_per_token, probs.shape[-1]), dim=-1
            ).indices
            expert_mask = torch.zeros_like(probs)
            expert_mask.scatter_(1, top_k_indices, 1.0)
            load = expert_mask.sum(dim=0)  # tokens per expert
            all_loads.append(load)

        if all_loads:
            avg_load = torch.stack(all_loads).mean(dim=0)
            total_tokens = avg_load.sum()
            if total_tokens > 0:
                utilization = avg_load / total_tokens
                metrics["expert/load_max"] = utilization.max().item()
                metrics["expert/load_min"] = utilization.min().item()
                metrics["expert/load_std"] = utilization.std().item()
                # Imbalance ratio: max / min (1.0 = perfect balance)
                min_val = utilization.min().item()
                if min_val > 0:
                    metrics["expert/imbalance_ratio"] = utilization.max().item() / min_val
    except Exception as e:
        logger.debug(f"Expert utilization computation failed: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Precision helper
# ---------------------------------------------------------------------------

def _resolve_autocast_dtype(cfg, profile: ModelProfile) -> torch.dtype:
    """
    Determine autocast dtype. No hardcoding to bfloat16.
    Priority: explicit config dtype > profile.model_dtype > bfloat16 fallback.
    """
    raw_dtype = getattr(cfg.model, "dtype", "bfloat16")
    if raw_dtype == "auto":
        return profile.model_dtype
    return _DTYPE_MAP.get(raw_dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model, train_cfg) -> AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(
        params, lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95),
    )


def _build_scheduler(optimizer, train_cfg) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0,
        total_iters=train_cfg.warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg.max_steps - train_cfg.warmup_steps),
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[train_cfg.warmup_steps],
    )


def _checkpoint(*, model, optimizer, step, avg_loss, ckpt_cfg, rank, best_ckpts):
    Path(ckpt_cfg.save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_cfg.save_dir, f"step_{step:07d}_loss_{avg_loss:.4f}.pt")

    # ALL ranks call save_checkpoint (FSDP2 all-gather), only rank 0 writes
    save_checkpoint(model, optimizer, step, path)

    if rank == 0:
        best_ckpts.append((avg_loss, path))
        best_ckpts.sort(key=lambda x: x[0])
        while len(best_ckpts) > ckpt_cfg.keep_last_n:
            _, old_path = best_ckpts.pop()
            try:
                os.remove(old_path)
            except FileNotFoundError:
                pass
        logger.info(f"Checkpoint → {path} | best kept: {[p for _, p in best_ckpts]}")

    return best_ckpts


def _maybe_resume(model, optimizer, save_dir: str, device) -> int:
    from bhaskera.distributed import load_checkpoint
    ckpts = sorted(Path(save_dir).glob("*.pt")) if Path(save_dir).exists() else []
    if not ckpts:
        return 0
    latest = str(ckpts[-1])
    logger.info(f"Resuming from {latest}")
    return load_checkpoint(model, optimizer, latest, device)
