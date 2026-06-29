"""
bhaskera.trainer.loop
=====================
Pure training loop with pluggable evaluation and throughput clock protection.

Changes vs previous version (all relate to ThroughputTracker API update)
--------------------------------------------------------------------------
1. ``window_hardware_tokens`` reset moved INSIDE the grad-accum loop reset
   block so it is always fresh per optimizer step (was fine before but
   the dual-reset at top + inside was confusing and fragile).

2. ``local_tokens_in_step`` now passes ``window_hardware_tokens`` — unchanged,
   this was already correct.

3. Print block now also shows ``mfu_ema_pct`` (new key from tracker) and
   uses the EMA tok/s for the display line so the console is less jittery,
   while MFU still shows the instantaneous value (true utilisation).

4. ``tok/s`` reported to tracker now uses the instantaneous global key
   ``throughput/tokens_per_sec_global`` — previously aliased to ``tok/s``
   by hand, now read directly from throughput_metrics to stay in sync.

5. ``MFU`` reported to Ray/tracker now reads ``throughput/mfu_pct`` from the
   dict rather than re-keying manually — removes the stale-key risk when
   the tracker dict changes.

6. Dead ``step_num`` variable (was ``throughput/total_steps``) replaced by
   the actual ``step`` counter for the print line — ``total_steps`` resets
   per-tracker-instance, not per epoch, so using ``step`` is cleaner.

7. Non-finite grad_norm ``continue`` now also resets the throughput clock
   so skipped steps don't inflate the next step's dt.
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
from bhaskera.evaluation import Evaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FSDP2 + DDP gradient-sync helper
# ---------------------------------------------------------------------------

def _set_grad_sync(model: torch.nn.Module, enabled: bool) -> None:
    """
    Toggle gradient all-reduce for the wrapped model.
    Dispatches by wrapper type: FSDP2 → set_requires_gradient_sync(model, enabled)
    DDP → model.require_backward_grad_sync = enabled
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
        pass

    # ── DDP path ────────────────────────────────────────────────────
    if isinstance(model, DDP):
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
    val_dataset=None,
    cfg,
    profile: ModelProfile,
    rank: int,
    local_rank: int,
    tracker=None,
    world_size: int = 1,
) -> None:
    """
    Run the training loop.
    """
    device = torch.device(f"cuda:{local_rank}")
    train_cfg = cfg.training
    ckpt_cfg = cfg.checkpoint

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    evaluator = Evaluator(cfg, model, profile, rank, world_size)

    model.train()

    step = 0
    if ckpt_cfg.enabled:
        step = maybe_resume(model, optimizer, ckpt_cfg.save_dir)
        model.train()

    best_ckpts: list[tuple[float, str]] = []

    # ── Throughput / MFU tracker ────────────────────────────────────
    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    throughput_on = bool(getattr(metrics_cfg, "throughput", True)) if metrics_cfg else True
    peak_tflops = float(getattr(metrics_cfg, "peak_tflops_per_gpu", 312.0)) if metrics_cfg else 312.0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_for_flops = total_params or trainable_params

    throughput = ThroughputTracker(
        params_for_flops=params_for_flops,
        world_size=max(1, int(world_size)),
        peak_flops_per_gpu=peak_tflops * 1e12,
        window=int(getattr(metrics_cfg, "throughput_window", 50)) if metrics_cfg else 50,
        warmup_steps=int(getattr(metrics_cfg, "throughput_warmup", 5)) if metrics_cfg else 5,
        is_peft=getattr(cfg.lora, "enabled", False),          # no-op in new tracker
        activation_checkpointing=getattr(train_cfg, "gradient_checkpointing", False),
        num_layers=getattr(profile, "num_hidden_layers", 0),
        hidden_size=getattr(profile, "hidden_size", 0),
    ) if throughput_on else None

    if tracker:
        tracker.log({
            "model/total_params": float(total_params),
            "model/trainable_params": float(trainable_params),
            "model/world_size": float(world_size),
        }, step=0)

    for epoch in range(train_cfg.num_epochs):
        step, best_ckpts = _run_epoch(
            model=model,
            dataset=dataset,
            val_dataset=val_dataset,
            evaluator=evaluator,
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
            throughput=throughput,
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
    *,
    model,
    dataset,
    val_dataset,
    evaluator,
    optimizer,
    scheduler,
    cfg,
    profile,
    rank,
    local_rank,
    device,
    epoch,
    step,
    tracker,
    best_ckpts,
    throughput: Optional[ThroughputTracker],
    world_size: int,
):
    train_cfg = cfg.training
    ckpt_cfg = cfg.checkpoint
    grad_accum = train_cfg.grad_accum
    strategy = cfg.training.distributed.strategy.lower()

    autocast_dtype = resolve_autocast_dtype(cfg, profile)
    use_autocast = (strategy == "ddp" and device.type == "cuda")

    moe_cfg = getattr(cfg, "moe", None)
    aux_loss_weight = getattr(moe_cfg, "aux_loss_weight", 0.01) if moe_cfg else 0.01
    log_expert_util = (
        profile.is_moe
        and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_every = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    sys_every = int(getattr(metrics_cfg, "system_every_n_steps", 10)) if metrics_cfg else 10
    cuda_every = int(getattr(metrics_cfg, "cuda_every_n_steps", 10)) if metrics_cfg else 10
    sys_on = bool(getattr(metrics_cfg, "enabled", True)) if metrics_cfg else True

    # ── Data loader ─────────────────────────────────────────────────
    loader = dataset.iter_torch_batches(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=max(
            train_cfg.batch_size * cfg.data.local_shuffle_buffer_multiplier,
            1000
        ),
        local_shuffle_seed=train_cfg.seed + rank,
        prefetch_batches=cfg.data.prefetch_batches,
        drop_last=True,
        dtypes={
            "input_ids": torch.long,
            "attention_mask": torch.long,
            "labels": torch.long,
        },
        device=device,
    )

    epoch_loss = 0.0
    epoch_aux_loss = 0.0
    epoch_steps = 0

    loss_ema: Optional[float] = None
    loss_ema_alpha = 0.05

    optimizer.zero_grad(set_to_none=True)

    if throughput is not None:
        throughput.reset_step_clock()

    loader_iter = iter(loader)
    while step < train_cfg.max_steps:
        micro_losses: list[torch.Tensor] = []
        micro_aux_losses: list[torch.Tensor] = []

        # FIX 1: Accumulator variables are declared fresh here, once per
        # optimizer step, not split across the outer loop top + inner reset.
        window_hardware_tokens = 0   # ALL tokens incl. padding (for MFU / HW tok/s)
        window_tokens = 0            # Non-padding tokens only (for useful tok/s)
        window_samples = 0
        window_seq_len = 0

        # ── Gradient accumulation loop ───────────────────────────────
        for micro_step in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = None  # type: ignore[assignment]
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            try:
                window_hardware_tokens += int(input_ids.numel())
                window_tokens += int(attention_mask.sum().item())
                window_samples += int(input_ids.size(0))
                window_seq_len = int(input_ids.size(1))
            except Exception:
                pass

            is_last = (micro_step == grad_accum - 1)
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

                out = model(**forward_kwargs)
                main_loss = out.loss
                aux_loss = extract_aux_loss(out, profile)

                if aux_loss is not None:
                    total_loss = (main_loss + aux_loss_weight * aux_loss) / grad_accum
                else:
                    total_loss = main_loss / grad_accum

                total_loss.backward()

            micro_losses.append(main_loss.detach())
            if aux_loss is not None:
                micro_aux_losses.append(aux_loss.detach())

        if loader_iter is None:
            break

        _set_grad_sync(model, enabled=True)

        # ── Optimizer step ──────────────────────────────────────────
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
            # FIX 7: Reset the clock so this skipped step's wall time
            # (which may include the full backward) does not get charged
            # to the next valid step, inflating its dt and crashing MFU.
            if throughput is not None:
                throughput.reset_step_clock()
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        window_loss = torch.stack(micro_losses).mean().item()
        window_aux = (
            torch.stack(micro_aux_losses).mean().item()
            if micro_aux_losses
            else 0.0
        )

        if loss_ema is None:
            loss_ema = window_loss
        else:
            loss_ema = (1 - loss_ema_alpha) * loss_ema + loss_ema_alpha * window_loss

        loss_spike = (window_loss / loss_ema) if loss_ema > 0 else 1.0
        lr = scheduler.get_last_lr()[0]

        epoch_loss += window_loss
        epoch_aux_loss += window_aux
        epoch_steps += 1
        step += 1

        # ── Throughput ──────────────────────────────────────────────
        throughput_metrics: dict[str, float] = {}
        if throughput is not None:
            throughput_metrics = throughput.step(
                local_tokens_in_step=window_hardware_tokens,
                local_samples_in_step=window_samples,
                seq_len=window_seq_len,
            )

            # FIX 3, 4, 5, 6: Use keys directly from the tracker dict.
            # - global_tps_inst: instantaneous (new key _global, not EMA)
            # - global_tps_ema: EMA-smoothed for stable console display
            # - mfu_inst: true instantaneous MFU (throughput/mfu_pct)
            # - mfu_ema: smoothed MFU (throughput/mfu_ema_pct)
            if "throughput/tokens_per_sec_global" in throughput_metrics:
                global_tps_inst = throughput_metrics["throughput/tokens_per_sec_global"]
                # EMA global = per_gpu_ema * world_size
                global_tps_ema = (
                    throughput_metrics.get("throughput/tokens_per_sec_per_gpu_ema", 0.0)
                    * max(1, int(world_size))
                )
                mfu_inst = throughput_metrics.get("throughput/mfu_pct", 0.0)
                mfu_ema  = throughput_metrics.get("throughput/mfu_ema_pct", 0.0)

                ratio = window_tokens / window_hardware_tokens if window_hardware_tokens > 0 else 1.0
                useful_global_tps = global_tps_inst * ratio
                padding_pct = (1.0 - ratio) * 100.0

                # FIX 6: Use ``step`` (global optimizer step counter),
                # not ``throughput/total_steps`` (tracker-instance counter).
                # Console shows EMA tok/s (stable) but instantaneous MFU (honest).
                print(
                    f"Step {step} | "
                    f"HW Tok/s: {global_tps_inst:,.0f} (ema: {global_tps_ema:,.0f}) | "
                    f"Useful Tok/s: {useful_global_tps:,.0f} | "
                    f"Pad Waste: {padding_pct:.1f}% | "
                    f"HW MFU: {mfu_inst:.2f}% (ema: {mfu_ema:.2f}%)"
                )

        # ── Logging ────────────────────────────────────────────────
        if rank == 0:
            msg = (
                f"[epoch {epoch}][step {step}] loss={window_loss:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            if "throughput/tokens_per_sec_global" in throughput_metrics:
                msg += f" tok/s={throughput_metrics['throughput/tokens_per_sec_global']:.0f}"
            if "throughput/mfu_pct" in throughput_metrics:
                msg += f" MFU={throughput_metrics['throughput/mfu_pct']:.1f}%"
            logger.info(msg)

        # FIX 4 & 5: Build the metrics dict using keys from throughput_metrics
        # directly rather than hand-aliasing tok/s and MFU.  The tracker owns
        # the naming; callers should read from it, not re-key it manually.
        metrics: dict[str, float] = {
            "step": float(step),
            "loss": window_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "epoch": float(epoch),
            "loss_running_avg": loss_ema,
            "loss_spike_ratio": loss_spike,
        }
        if profile.is_moe:
            metrics["aux_loss"] = window_aux
            metrics["total_loss"] = window_loss + aux_loss_weight * window_aux

        # Merge all throughput keys (tok/s, mfu_pct, mfu_ema_pct, step_time…)
        metrics.update(throughput_metrics)

        # Convenience aliases for dashboards / Ray that expect short names
        if "throughput/tokens_per_sec_global" in throughput_metrics:
            metrics["tok/s"] = throughput_metrics["throughput/tokens_per_sec_global"]
        if "throughput/mfu_pct" in throughput_metrics:
            metrics["MFU"] = throughput_metrics["throughput/mfu_pct"]
        if "throughput/mfu_ema_pct" in throughput_metrics:
            metrics["MFU_ema"] = throughput_metrics["throughput/mfu_ema_pct"]

        if tracker:
            if log_expert_util and step % expert_log_every == 0:
                metrics.update(compute_expert_utilization(out, profile))
            tracker.log(metrics, step=step)

            if sys_on and sys_every > 0 and step % sys_every == 0:
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

        # ── Evaluation & Benchmarking ──────────────────────────────
        ran_eval = False

        if evaluator.should_run_validation(step):
            val_metrics = evaluator.run_validation(val_dataset)
            if tracker and val_metrics:
                tracker.log(val_metrics, step=step)
            if rank == 0 and val_metrics:
                logger.info(f"\033[1;32m[Validation @ Step {step}] {val_metrics}\033[0m")
            ran_eval = True

        if evaluator.should_run_benchmark(step):
            bench_metrics = evaluator.run_benchmarks(tokenizer=None)
            if tracker and bench_metrics:
                tracker.log(bench_metrics, step=step)
            if rank == 0 and bench_metrics:
                logger.info(f"\033[1;34m[Benchmarks @ Step {step}] {bench_metrics}\033[0m")
            ran_eval = True

        # Clock protection: reset throughput clock after eval stalls
        if ran_eval:
            if throughput is not None:
                throughput.reset_step_clock()
            torch.cuda.empty_cache()

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

    # ── Checkpoint ──────────────────────────────────────────────────
    if ckpt_cfg.enabled and (epoch + 1) % ckpt_cfg.save_interval == 0:
        best_ckpts = save_and_prune(
            model=model,
            optimizer=optimizer,
            step=step,
            avg_loss=avg_loss,
            ckpt_cfg=ckpt_cfg,
            rank=rank,
            best_ckpts=best_ckpts,
        )

    return step, best_ckpts
