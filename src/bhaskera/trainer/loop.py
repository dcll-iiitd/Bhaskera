"""
bhaskera.trainer.loop
=====================
Pure training loop.


"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.introspect import ModelProfile
from bhaskera.utils import ThroughputTracker
from bhaskera.utils.system_stats import system_stats, cuda_memory_stats

from .checkpointing import maybe_resume, save_and_prune
from .moe import compute_expert_utilization, extract_aux_loss
from .optim import build_optimizer, build_scheduler
from .precision import resolve_autocast_dtype

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSDP2 + DDP gradient-sync helper
# ---------------------------------------------------------------------------

def _set_grad_sync(model: torch.nn.Module, enabled: bool) -> None:
    """
    Toggle gradient all-reduce for the wrapped model.

    Dispatches by wrapper type:

        FSDP2  → set_requires_gradient_sync(model, enabled)
                 Walks every sharded submodule regardless of any PEFT /
                 wrapper layers sitting on top.  set_requires_gradient_sync
                 bypasses wrapper __getattr__ chains and operates directly
                 on the FSDP2 submodule state.

        DDP    → model.require_backward_grad_sync = enabled
                 This is the underlying flag that DDP.no_sync() toggles.
                 Setting it directly avoids the no_sync() context-manager
                 dance (and works even when DDP is wrapping a PEFT model
                 that doesn't delegate the no_sync attribute).
                 SKIPPED when static_graph=True — DDP's static_graph mode
                 caches the reduction order on the first iteration and is
                 incompatible with mid-run sync toggling.

        Other  → no-op (single-GPU / CPU / mixed contexts).

    Call with enabled=False before every micro-step except the last,
    and enabled=True on the last micro-step so the all-reduce fires
    exactly once per optimizer step.
    """
    # ── FSDP2 path ──────────────────────────────────────────────────
    try:
        from torch.distributed._composable.fsdp import (
            FSDPModule,
            set_requires_gradient_sync,
        )
        if isinstance(model, FSDPModule) or any(
            isinstance(m, FSDPModule) for m in model.modules()
        ):
            set_requires_gradient_sync(model, enabled)
            return
    except ImportError:
        # torch < 2.4 — FSDP2 unavailable, fall through to DDP / no-op.
        pass

    # ── DDP path ────────────────────────────────────────────────────
    if isinstance(model, DDP):
        # static_graph=True bakes the reduction graph into DDP after the
        # first iteration; toggling require_backward_grad_sync after that
        # raises "Your training graph has changed in this iteration".
        if getattr(model, "_bhaskera_static_graph", False) or getattr(
            model, "static_graph", False
        ):
            return
        model.require_backward_grad_sync = enabled
        return

    # ── Non-distributed: nothing to do ──────────────────────────────


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
    world_size: int = 1,
) -> None:
    """
    Run the training loop.

    Args:
        model:       Distributed-wrapped model (FSDP2 or DDP).
        dataset:     Ray Dataset pre-tokenized by bhaskera.data.
        cfg:         Bhaskera Config object.
        profile:     ModelProfile from introspection.
        rank:        Global rank of this worker.
        local_rank:  Local GPU index on this host.
        tracker:     Optional logger (MultiLogger from build_logger).
        world_size:  Total number of training ranks.
    """
    device    = torch.device(f"cuda:{local_rank}")
    train_cfg = cfg.training
    ckpt_cfg  = cfg.checkpoint

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    model.train()
    step = 0
    if ckpt_cfg.enabled:
        step = maybe_resume(model, optimizer, ckpt_cfg.save_dir)
        model.train()
    best_ckpts: list[tuple[float, str]] = []

    # ── Throughput / MFU tracker ────────────────────────────────────
    metrics_cfg    = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    throughput_on  = bool(getattr(metrics_cfg, "throughput", True)) if metrics_cfg else True
    peak_tflops    = float(getattr(metrics_cfg, "peak_tflops_per_gpu", 312.0)) if metrics_cfg else 312.0

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_for_flops = total_params or trainable_params

    throughput = ThroughputTracker(
        params_for_flops=params_for_flops,
        world_size=max(1, int(world_size)),
        peak_flops_per_gpu=peak_tflops * 1e12,
        window=int(getattr(metrics_cfg, "throughput_window", 50)) if metrics_cfg else 50,
        warmup_steps=int(getattr(metrics_cfg, "throughput_warmup", 5)) if metrics_cfg else 5,
    ) if throughput_on else None

    if tracker:
        tracker.log({
            "model/total_params":     float(total_params),
            "model/trainable_params": float(trainable_params),
            "model/world_size":       float(world_size),
        }, step=0)

    for epoch in range(train_cfg.num_epochs):
        step, best_ckpts = _run_epoch(
            model=model, dataset=dataset, optimizer=optimizer,
            scheduler=scheduler, cfg=cfg, profile=profile,
            rank=rank, local_rank=local_rank, device=device,
            epoch=epoch, step=step, tracker=tracker,
            best_ckpts=best_ckpts, throughput=throughput,
            world_size=world_size,
        )
        if step >= train_cfg.max_steps:
            break

    if tracker:
        tracker.finish()
    if rank == 0:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    *, model, dataset, optimizer, scheduler, cfg, profile,
    rank, local_rank, device, epoch, step, tracker, best_ckpts,
    throughput: Optional[ThroughputTracker], world_size: int,
):
    train_cfg  = cfg.training
    ckpt_cfg   = cfg.checkpoint
    grad_accum = train_cfg.grad_accum

    strategy       = cfg.training.distributed.strategy.lower()
    autocast_dtype = resolve_autocast_dtype(cfg, profile)
    use_autocast   = (strategy == "ddp" and device.type == "cuda")

    moe_cfg         = getattr(cfg, "moe", None)
    aux_loss_weight = getattr(moe_cfg, "aux_loss_weight", 0.01) if moe_cfg else 0.01
    log_expert_util = (
        profile.is_moe and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_every = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    sys_every   = int(getattr(metrics_cfg, "system_every_n_steps", 10)) if metrics_cfg else 10
    cuda_every  = int(getattr(metrics_cfg, "cuda_every_n_steps", 10))   if metrics_cfg else 10
    sys_on      = bool(getattr(metrics_cfg, "enabled", True))           if metrics_cfg else True

    # ── Data loader ─────────────────────────────────────────────────
    # fix #16: use prefetch_batches from config
    # fix loop: drop_last=True prevents shape mismatches in the last batch
    # fix loop: explicit dtypes — int64 is required for CUDA embedding lookup
    loader = dataset.iter_torch_batches(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=max(
            train_cfg.batch_size * cfg.data.local_shuffle_buffer_multiplier, 1000
        ),
        local_shuffle_seed=train_cfg.seed + rank,
        prefetch_batches=cfg.data.prefetch_batches,   # fix #16
        drop_last=True,                               # fix loop: no shape mismatch
        dtypes={                                      # fix loop: int64 for embeddings
            "input_ids":      torch.long,
            "attention_mask": torch.long,
            "labels":         torch.long,
        },
        device=device,
    )

    # Per-step accumulators
    epoch_loss     = 0.0
    epoch_aux_loss = 0.0
    epoch_steps    = 0

    # Loss EMA (for spike detection)
    loss_ema: Optional[float] = None
    loss_ema_alpha = 0.05

    # Tokens/samples per accum window — for throughput
    window_tokens  = 0
    window_samples = 0
    window_seq_len = 0

    optimizer.zero_grad(set_to_none=True)

    if throughput is not None:
        throughput.reset_step_clock()

    # ── Training loop ───────────────────────────────────────────────
    # Gradient-sync invariant (non-negotiable):
    #   All-reduce must fire exactly ONCE per optimizer step — on the last
    #   micro-step of every accumulation window.  For FSDP2 we use
    #   set_requires_gradient_sync(); for DDP we toggle
    #   require_backward_grad_sync directly.  Both are routed through
    #   _set_grad_sync() above, which dispatches by wrapper type and
    #   safely falls back when neither applies (single-GPU, static_graph DDP).
    loader_iter = iter(loader)

    while step < train_cfg.max_steps:
        micro_losses:     list[torch.Tensor] = []
        micro_aux_losses: list[torch.Tensor] = []
        window_tokens  = 0
        window_samples = 0

        # ── Gradient accumulation loop ───────────────────────────────
        for micro_step in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Epoch exhausted mid-accumulation window — stop cleanly
                loader_iter = None  # type: ignore[assignment]
                break

            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels         = batch["labels"]

            try:
                window_tokens  += int(attention_mask.sum().item())
                window_samples += int(input_ids.size(0))
                window_seq_len  = int(input_ids.size(1))
            except Exception:
                pass

            is_last = (micro_step == grad_accum - 1)

            # Gradient-sync toggle:
            #   False  → suppress all-reduce on this micro-step
            #   True   → allow all-reduce on the final micro-step
            # Dispatches to FSDP2's set_requires_gradient_sync,
            # DDP's require_backward_grad_sync, or no-op as appropriate.
            _set_grad_sync(model, enabled=is_last)

            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )

            with autocast_ctx:
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

                total_loss.backward()

            micro_losses.append(main_loss.detach())
            if aux_loss is not None:
                micro_aux_losses.append(aux_loss.detach())

        # If loader exhausted mid-window, break the outer loop too
        if loader_iter is None:
            break

        # Ensure sync is re-enabled after the accumulation window in case
        # something above exited early (e.g. StopIteration on last micro-step).
        _set_grad_sync(model, enabled=True)

        # ── Optimizer step ──────────────────────────────────────────
        # Use model.clip_grad_norm_ for FSDP2 if available (handles sharded grads).
        # DDP doesn't expose clip_grad_norm_; falls through to the manual clip.
        grad_clip = getattr(train_cfg, "grad_clip", None) or getattr(train_cfg, "max_grad_norm", 1.0)
        if hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(grad_clip).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                grad_clip,
            ).item()

        if not math.isfinite(grad_norm):
            logger.warning(
                f"[rank {rank}][epoch {epoch}][step {step}] "
                f"Non-finite grad_norm={grad_norm} — skipping optimizer step"
            )
            optimizer.zero_grad(set_to_none=True)
            if tracker:
                tracker.log({"train/non_finite_grad": 1.0}, step=step)
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Loss aggregation ────────────────────────────────────────
        window_loss = torch.stack(micro_losses).mean().item()
        window_aux  = (
            torch.stack(micro_aux_losses).mean().item()
            if micro_aux_losses else 0.0
        )

        # Loss EMA + spike ratio
        if loss_ema is None:
            loss_ema = window_loss
        else:
            loss_ema = (1 - loss_ema_alpha) * loss_ema + loss_ema_alpha * window_loss
        loss_spike = (window_loss / loss_ema) if loss_ema > 0 else 1.0

        lr = scheduler.get_last_lr()[0]
        epoch_loss     += window_loss
        epoch_aux_loss += window_aux
        epoch_steps    += 1
        step           += 1

        # ── Throughput ──────────────────────────────────────────────
        throughput_metrics: dict[str, float] = {}
        if throughput is not None:
            throughput_metrics = throughput.step(
                tokens_in_step=window_tokens,
                samples_in_step=window_samples,
                seq_len=window_seq_len,
            )

        # ── Logging ────────────────────────────────────────────────
        if rank == 0:
            msg = (
                f"[epoch {epoch}][step {step}] loss={window_loss:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            if "throughput/tokens_per_sec" in throughput_metrics:
                msg += f" tok/s={throughput_metrics['throughput/tokens_per_sec']:.0f}"
            if "throughput/mfu_pct" in throughput_metrics:
                msg += f" MFU={throughput_metrics['throughput/mfu_pct']:.1f}%"
            logger.info(msg)

            metrics: dict[str, float] = {
                "loss":             window_loss,
                "lr":               lr,
                "grad_norm":        grad_norm,
                "epoch":            float(epoch),
                "loss_running_avg": loss_ema,
                "loss_spike_ratio": loss_spike,
            }
            if profile.is_moe:
                metrics["aux_loss"]   = window_aux
                metrics["total_loss"] = window_loss + aux_loss_weight * window_aux
            metrics.update(throughput_metrics)

            if tracker:
                if log_expert_util and step % expert_log_every == 0:
                    metrics.update(compute_expert_utilization(out, profile))
                tracker.log(metrics, step=step)

        if tracker and sys_on and sys_every > 0 and step % sys_every == 0:
            sysm: dict[str, float] = {}
            sysm.update(system_stats(
                gpu=bool(getattr(metrics_cfg, "gpu", True)) if metrics_cfg else True,
                cpu=bool(getattr(metrics_cfg, "cpu", True)) if metrics_cfg else True,
            ))
            if cuda_every > 0 and step % cuda_every == 0:
                if not metrics_cfg or getattr(metrics_cfg, "cuda_memory", True):
                    sysm.update(cuda_memory_stats(device))
            if sysm:
                tracker.log(sysm, step=step)

    if epoch_steps == 0:
        return step, best_ckpts

    avg_loss = epoch_loss / epoch_steps
    if rank == 0:
        epoch_msg     = f"[epoch {epoch}] avg_loss={avg_loss:.4f}"
        epoch_metrics = {"epoch_avg_loss": avg_loss, "epoch": epoch}
        if profile.is_moe:
            avg_aux = epoch_aux_loss / epoch_steps
            epoch_msg += f" avg_aux_loss={avg_aux:.4f}"
            epoch_metrics["epoch_avg_aux_loss"] = avg_aux
        logger.info(epoch_msg)
        if tracker:
            tracker.log(epoch_metrics, step=step)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if ckpt_cfg.enabled and (epoch + 1) % ckpt_cfg.save_interval == 0:
        best_ckpts = save_and_prune(
            model=model, optimizer=optimizer, step=step,
            avg_loss=avg_loss, ckpt_cfg=ckpt_cfg,
            rank=rank, best_ckpts=best_ckpts,
        )

    return step, best_ckpts
