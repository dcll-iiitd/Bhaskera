"""
bhaskera.utils.throughput
=========================
Step-time, tokens/sec, samples/sec, and an MFU estimate for LLM
fine-tuning loops.

The MFU calculation uses the Chinchilla / PaLM convention:
    flops_per_token ≈ 6 * trainable_params (forward + backward)
    + 12 * num_layers * seq_len * hidden  (attention, exact term)

For LoRA fine-tuning the dominant FLOP is still through the frozen
base weights (the LoRA update is multiplied into the base path during
forward+backward), so we use the *full* model parameter count, not the
trainable count.  Pass ``params_for_flops`` explicitly to override.

Notes:
    * The first ``warmup_steps`` step times are dropped from the
      moving averages — they include compile / cache warmup and would
      otherwise drag the EMA down for the rest of the run.
    * ``peak_flops_per_gpu`` defaults to A100-bf16 (312 TFLOPS).
      Override per-GPU-type via the ``monitoring.peak_tflops_per_gpu``
      config field.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class ThroughputTracker:
    """Lightweight tracker — call ``step()`` once per optimizer step."""

    def __init__(
        self,
        *,
        params_for_flops: int,
        world_size: int,
        peak_flops_per_gpu: float = 312e12,  # A100 bf16
        window: int = 50,
        warmup_steps: int = 5,
        is_peft: bool = False,               # NEW: Tracks if we are using LoRA
    ) -> None:
        self._params = max(1, int(params_for_flops))
        self._world  = max(1, int(world_size))
        self._peak   = max(1.0, float(peak_flops_per_gpu))
        self._window = max(1, int(window))
        self._warmup = max(0, int(warmup_steps))
        self._is_peft = is_peft

        self._step_times: deque[float] = deque(maxlen=self._window)
        self._last_t: Optional[float] = None
        self._steps_seen = 0

    def reset_step_clock(self) -> None:
        """Call right before the first forward of a new step."""
        self._last_t = time.perf_counter()

    def step(
        self,
        *,
        local_tokens_in_step: int,  # Tokens processed by ONE GPU this step
        local_samples_in_step: int, # Samples processed by ONE GPU this step
        seq_len: int,               
    ) -> dict[str, float]:
        """
        Close out one optimizer step and emit derived metrics.

        Returns a dict like::

            {
                "throughput/step_time_s":            0.412,
                "throughput/step_time_ema_s":        0.418,
                "throughput/tokens_per_sec_global":  19880.0,
                "throughput/tokens_per_sec_per_gpu": 2485.0,
                "throughput/samples_per_sec_global": 9.7,
                "throughput/mfu_pct":                41.2,
                "throughput/total_steps":            137.0,
            }
        """
        now = time.perf_counter()
        out: dict[str, float] = {}
        self._steps_seen += 1
        out["throughput/total_steps"] = float(self._steps_seen)

        if self._last_t is None:
            self._last_t = now
            return out
        dt = now - self._last_t
        self._last_t = now

        if dt <= 0:
            return out

        out["throughput/step_time_s"] = dt
        if self._steps_seen > self._warmup:
            self._step_times.append(dt)
            
        if self._step_times:
            ema = sum(self._step_times) / len(self._step_times)
            out["throughput/step_time_ema_s"] = ema
            ref_dt = ema
        else:
            ref_dt = dt

        # ---------------------------------------------------------
        # Throughput Calculations (Using local per-GPU inputs)
        # ---------------------------------------------------------
        if local_tokens_in_step > 0:
            local_tps = local_tokens_in_step / ref_dt
            out["throughput/tokens_per_sec_per_gpu"] = local_tps
            # Scale up for whole-system throughput
            out["throughput/tokens_per_sec_global"] = local_tps * self._world
            
        if local_samples_in_step > 0:
            local_sps = local_samples_in_step / ref_dt
            out["throughput/samples_per_sec_global"] = local_sps * self._world

        # ---------------------------------------------------------
        # MFU Calculation (Strictly Per-GPU)
        # ---------------------------------------------------------
        # Standard FT: 6 FLOPs per param (2 fwd, 2 bwd_act, 2 bwd_weight)
        # LoRA/PEFT: ~4 FLOPs per param (2 fwd, 2 bwd_act, 0 bwd_weight for base)
        flops_multiplier = 4.0 if self._is_peft else 6.0 
        flops_per_token = flops_multiplier * self._params
        
        if local_tokens_in_step > 0:
            # How many FLOPs this specific GPU achieved per second
            achieved_flops_per_sec_per_gpu = flops_per_token * local_tps
            out["throughput/mfu_pct"] = 100.0 * (achieved_flops_per_sec_per_gpu / self._peak)

        return out
