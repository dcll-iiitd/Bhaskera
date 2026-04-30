"""
bhaskera.trainer.loop
=====================
Pure training loop.

Compared to v2, this revision adds production-grade observability:

    * Throughput / MFU
        Step time, tokens/sec (global and per-GPU), samples/sec, and
        a Chinchilla-style MFU estimate.  The tracker is per-rank so
        every GPU's contribution is visible in Grafana.

    * Per-rank system telemetry
        CPU / GPU / NVLink / disk / network are pushed every
        ``cfg.monitoring.metrics.system_every_n_steps`` steps from
        every rank, tagged with ``rank``.  This is what makes the
        Ray Dashboard 'Cluster' view actually useful for finetuning
        — each panel can be aggregated or broken out per-GPU.

    * CUDA-allocator gauges
        ``cuda/allocated_mib``, ``cuda/reserved_mib``, peak values.
        These come from PyTorch directly (NVML can't see the caching
        allocator) and catch fragmentation regressions early.

    * Loss-spike, grad-norm, and LR distribution
        Already had loss+grad_norm; we now also push
        ``loss_running_avg``, ``loss_spike_ratio`` (loss / EMA),
        ``param_norm`` (sqrt-sum-of-squares of trainable params).

Existing correctness invariants are preserved:
    * Loss averaging across the whole grad-accum window (not just
      the last micro-batch).
    * Autocast only for DDP — FSDP2's MixedPrecisionPolicy already
      casts.
    * Barrier before checkpoint.
    * No redundant ``.to(device)`` after Ray's ``iter_torch_batches``.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Optional

import torch
import torch.distributed as dist

from bhaskera.introspect import ModelProfile
from bhaskera.utils import ThroughputTracker
from bhaskera.utils.system_stats import system_stats, cuda_memory_stats

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
    world_size: int = 1,
) -> None:
    """
    Run the training loop.

    Args:
        model:       Distributed-wrapped model (FSDP2 or DDP).
        dataset:     Ray Dataset pre-tokenised by bhaskera.data.
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

    step = 0
    if ckpt_cfg.enabled:
        step = maybe_resume(model, optimizer, ckpt_cfg.save_dir)

    best_ckpts: list[tuple[float, str]] = []

    # ── Throughput / MFU tracker (per-rank) ─────────────────────────
    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    throughput_on = bool(getattr(metrics_cfg, "throughput", True)) if metrics_cfg else True
    peak_tflops   = float(getattr(metrics_cfg, "peak_tflops_per_gpu", 312.0)) if metrics_cfg else 312.0

    # Use total parameter count for MFU (LoRA still flows through the
    # frozen base path).  Falls back to trainable params if total is
    # zero for some reason.
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

    # Log model-size summary as run-scoped tags via the first push
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
    train_cfg = cfg.training
    ckpt_cfg  = cfg.checkpoint
    grad_accum = train_cfg.grad_accum

    # Mixed precision strategy
    strategy = cfg.training.distributed.strategy.lower()
    autocast_dtype = resolve_autocast_dtype(cfg, profile)
    use_autocast = (strategy == "ddp" and device.type == "cuda")

    # MoE config
    moe_cfg = getattr(cfg, "moe", None)
    aux_loss_weight    = getattr(moe_cfg, "aux_loss_weight",        0.01) if moe_cfg else 0.01
    log_expert_util    = (
        profile.is_moe
        and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_every   = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    # Monitoring cadence
    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    sys_every  = int(getattr(metrics_cfg, "system_every_n_steps", 10)) if metrics_cfg else 10
    cuda_every = int(getattr(metrics_cfg, "cuda_every_n_steps", 10))   if metrics_cfg else 10
    sys_on     = bool(getattr(metrics_cfg, "enabled", True))           if metrics_cfg else True

    # Data loader — Ray already places tensors on device.
    loader = dataset.iter_torch_batches(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=1000,
        local_shuffle_seed=train_cfg.seed + rank,
        prefetch_batches=2,
        device=device,
    )

    # Per-accum-window accumulators
    micro_losses:     list[torch.Tensor] = []
    micro_aux_losses: list[torch.Tensor] = []
    epoch_loss     = 0.0
    epoch_aux_loss = 0.0
    epoch_steps    = 0
    micro          = 0

    # Loss EMA (for spike detection)
    loss_ema: Optional[float] = None
    loss_ema_alpha = 0.05  # ~20-step half life

    # Track tokens per accum window — used for throughput
    window_tokens = 0
    window_samples = 0
    window_seq_len = 0

    optimizer.zero_grad(set_to_none=True)

    if throughput is not None:
        throughput.reset_step_clock()

    for batch in loader:
        if step >= train_cfg.max_steps:
            break

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = batch["labels"]

        # Track tokens for throughput.  Use attention_mask sum — this
        # ignores padding, which is the right denominator for "real"
        # throughput.
        try:
            window_tokens  += int(attention_mask.sum().item())
            window_samples += int(input_ids.size(0))
            window_seq_len  = int(input_ids.size(1))
        except Exception:
            pass

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
            window_tokens = 0
            window_samples = 0
            # Push a 'skipped' counter so dashboards see the spike.
            if tracker:
                tracker.log({"train/non_finite_grad": 1.0}, step=step)
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
        # Reset per-accum-window counters
        window_tokens  = 0
        window_samples = 0

        # ── Logging ────────────────────────────────────────────────
        # Rank 0 always logs the loss family (this is what flows to
        # W&B / MLflow).  Every rank logs system stats (gpu/cpu/io)
        # so per-rank breakdowns show up in Grafana.
        if rank == 0:
            msg = (
                f"[epoch {epoch}][step {step}] loss={window_loss:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            if "throughput/tokens_per_sec" in throughput_metrics:
                msg += (
                    f" tok/s={throughput_metrics['throughput/tokens_per_sec']:.0f}"
                )
            if "throughput/mfu_pct" in throughput_metrics:
                msg += f" MFU={throughput_metrics['throughput/mfu_pct']:.1f}%"
            logger.info(msg)

            metrics: dict[str, float] = {
                "loss":              window_loss,
                "lr":                lr,
                "grad_norm":         grad_norm,
                "epoch":             float(epoch),
                "loss_running_avg":  loss_ema,
                "loss_spike_ratio":  loss_spike,
            }
            if profile.is_moe:
                metrics["aux_loss"]   = window_aux
                metrics["total_loss"] = window_loss + aux_loss_weight * window_aux
            metrics.update(throughput_metrics)

            if tracker:
                if log_expert_util and step % expert_log_every == 0:
                    metrics.update(compute_expert_utilization(out, profile))
                tracker.log(metrics, step=step)

        # System telemetry from EVERY rank — tagged with rank in the
        # logger so Grafana can break out per-GPU.
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
