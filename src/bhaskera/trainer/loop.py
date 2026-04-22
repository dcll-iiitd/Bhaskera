"""
bhaskera.trainer.loop
=====================
Pure training loop.

Key correctness fixes vs v1:

  * Loss averaging:
      v1 reported `main_loss.item()` which was only the LAST micro-batch
      in a grad-accum window, giving noisy and misleading metrics.
      v2 accumulates per-micro-batch losses as detached tensors and
      reports the mean over the window when the optimizer actually steps.

  * Autocast redundancy:
      Under FSDP2, MixedPrecisionPolicy already casts params and inputs.
      Wrapping forward in torch.autocast on top of it is redundant and
      can cause dtype surprises at edge boundaries (fp32 config +
      bf16 FSDP policy).  We now use autocast ONLY for DDP.

  * Barrier before checkpoint:
      DCP save is a collective.  If one rank exits the epoch early due
      to data-shard imbalance, the slow rank deadlocks on the all-gather.
      We barrier before save_and_prune and also on the max-steps exit.

  * Redundant device moves:
      Ray's iter_torch_batches already places tensors on `device`.  The
      prior `.to(device)` calls were no-ops that still synced CUDA.

  * Router logits passed per-forward:
      Set `output_router_logits=True` on the forward call when the
      profile reports aux-loss support.  The load-time config mutation
      in v1 has been removed.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Optional

import torch
import torch.distributed as dist

from bhaskera.introspect import ModelProfile

from .checkpointing import maybe_resume, save_and_prune
from .moe import compute_expert_utilization, extract_aux_loss
from .optim import build_optimizer, build_scheduler
from .precision import resolve_autocast_dtype

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(
    *,
    model: torch.nn.Module,
    dataset,
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
        local_rank: Local GPU index on this host.
        tracker:    Optional logger (Wandb/MLflow/None).  Only populated
                    on rank 0 by convention.
    """
    device    = torch.device(f"cuda:{local_rank}")
    train_cfg = cfg.training
    ckpt_cfg  = cfg.checkpoint

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    step = 0
    if ckpt_cfg.enabled:
        step = maybe_resume(model, optimizer, ckpt_cfg.save_dir)

    best_ckpts: list[tuple[float, str]] = []

    for epoch in range(train_cfg.num_epochs):
        step, best_ckpts = _run_epoch(
            model=model, dataset=dataset, optimizer=optimizer,
            scheduler=scheduler, cfg=cfg, profile=profile,
            rank=rank, local_rank=local_rank, device=device,
            epoch=epoch, step=step, tracker=tracker,
            best_ckpts=best_ckpts,
        )
        if step >= train_cfg.max_steps:
            break

    if tracker and rank == 0:
        tracker.finish()
    if rank == 0:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    *, model, dataset, optimizer, scheduler, cfg, profile,
    rank, local_rank, device, epoch, step, tracker, best_ckpts,
):
    train_cfg = cfg.training
    ckpt_cfg  = cfg.checkpoint
    grad_accum = train_cfg.grad_accum

    # ── Mixed precision strategy ────────────────────────────────────
    # FSDP2 already handles casts via MixedPrecisionPolicy; enabling
    # torch.autocast on top is redundant and sometimes harmful.  Use
    # autocast ONLY for DDP.
    strategy = cfg.training.distributed.strategy.lower()
    autocast_dtype = resolve_autocast_dtype(cfg, profile)
    use_autocast = (strategy == "ddp" and device.type == "cuda")

    # ── MoE config ──────────────────────────────────────────────────
    moe_cfg = getattr(cfg, "moe", None)
    aux_loss_weight    = getattr(moe_cfg, "aux_loss_weight",        0.01) if moe_cfg else 0.01
    log_expert_util    = (
        profile.is_moe
        and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_every   = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    # ── Data loader ─────────────────────────────────────────────────
    # Ray already places tensors on `device`, so do NOT re-.to() them.
    loader = dataset.iter_torch_batches(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=1000,
        local_shuffle_seed=train_cfg.seed + rank,
        prefetch_batches=2,
        device=device,
    )

    # ── Per-accum-window accumulators ───────────────────────────────
    micro_losses:     list[torch.Tensor] = []
    micro_aux_losses: list[torch.Tensor] = []
    epoch_loss     = 0.0
    epoch_aux_loss = 0.0
    epoch_steps    = 0
    micro          = 0

    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        if step >= train_cfg.max_steps:
            break

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = batch["labels"]

        # Forward
        ctx = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with ctx:
            forward_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            if profile.is_moe and profile.has_aux_loss:
                forward_kwargs["output_router_logits"] = True

            out       = model(**forward_kwargs)
            main_loss = out.loss
            aux_loss  = extract_aux_loss(out, profile)

            if aux_loss is not None:
                total_loss = (main_loss + aux_loss_weight * aux_loss) / grad_accum
            else:
                total_loss = main_loss / grad_accum

        # Backward — outside autocast
        total_loss.backward()

        micro_losses.append(main_loss.detach())
        if aux_loss is not None:
            micro_aux_losses.append(aux_loss.detach())
        micro += 1

        if micro < grad_accum:
            continue

        # ── Optimizer step ──────────────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            train_cfg.max_grad_norm,
        ).item()

        if not math.isfinite(grad_norm):
            logger.warning(
                f"[rank {rank}][epoch {epoch}][step {step}] "
                f"Non-finite grad_norm={grad_norm} — skipping optimizer step"
            )
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            micro_losses.clear()
            micro_aux_losses.clear()
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Loss aggregation across the accum window ────────────────
        window_loss = torch.stack(micro_losses).mean().item()
        window_aux  = (
            torch.stack(micro_aux_losses).mean().item()
            if micro_aux_losses else 0.0
        )
        micro_losses.clear()
        micro_aux_losses.clear()
        micro = 0

        lr = scheduler.get_last_lr()[0]
        epoch_loss     += window_loss
        epoch_aux_loss += window_aux
        epoch_steps    += 1
        step           += 1

        # ── Logging (rank 0 only) ───────────────────────────────────
        if rank == 0:
            msg = (
                f"[epoch {epoch}][step {step}] loss={window_loss:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            metrics = {
                "loss":      window_loss,
                "lr":        lr,
                "grad_norm": grad_norm,
                "epoch":     epoch,
            }
            if profile.is_moe:
                msg += f" aux_loss={window_aux:.4f}"
                metrics["aux_loss"]   = window_aux
                metrics["total_loss"] = window_loss + aux_loss_weight * window_aux
            logger.info(msg)

            if tracker:
                if log_expert_util and step % expert_log_every == 0:
                    metrics.update(compute_expert_utilization(out, profile))
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

    # Barrier before checkpoint — other ranks may still be exiting the loop.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if ckpt_cfg.enabled and (epoch + 1) % ckpt_cfg.save_interval == 0:
        best_ckpts = save_and_prune(
            model=model, optimizer=optimizer, step=step,
            avg_loss=avg_loss, ckpt_cfg=ckpt_cfg,
            rank=rank, best_ckpts=best_ckpts,
        )

    return step, best_ckpts