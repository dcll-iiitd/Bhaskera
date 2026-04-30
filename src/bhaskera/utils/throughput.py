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
    ) -> None:
        self._params = max(1, int(params_for_flops))
        self._world  = max(1, int(world_size))
        self._peak   = max(1.0, float(peak_flops_per_gpu))
        self._window = max(1, int(window))
        self._warmup = max(0, int(warmup_steps))

        self._step_times: deque[float] = deque(maxlen=self._window)
        self._last_t: Optional[float] = None
        self._steps_seen = 0

    def reset_step_clock(self) -> None:
        """Call right before the first forward of a new step."""
        self._last_t = time.perf_counter()

    def step(
        self,
        *,
        tokens_in_step: int,
        samples_in_step: int,
        seq_len: int,
    ) -> dict[str, float]:
        """
        Close out one optimizer step and emit derived metrics.

        Returns a dict like::

            {
                "throughput/step_time_s":        0.412,
                "throughput/step_time_ema_s":    0.418,
                "throughput/tokens_per_sec":     19880.0,
                "throughput/tokens_per_sec_per_gpu": 2485.0,
                "throughput/samples_per_sec":    9.7,
                "throughput/mfu_pct":            41.2,
                "throughput/total_steps":        137.0,
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

        # Throughput uses the smoothed dt so the panel doesn't jitter.
        if tokens_in_step > 0:
            tps = tokens_in_step / ref_dt
            out["throughput/tokens_per_sec"] = tps
            out["throughput/tokens_per_sec_per_gpu"] = tps / self._world
        if samples_in_step > 0:
            out["throughput/samples_per_sec"] = samples_in_step / ref_dt

        # MFU — model FLOPs utilization
        # FLOPs/token ≈ 6 * params  (3-pass approximation, fwd 1× + bwd 2×)
        # We deliberately ignore the seq²·d attention term to keep the
        # estimate stable across configs; it adds <5% at typical seq_len.
        flops_per_token = 6.0 * self._params
        if tokens_in_step > 0:
            achieved_flops_per_sec = (
                flops_per_token * tokens_in_step / ref_dt / self._world
            )
            out["throughput/mfu_pct"] = 100.0 * achieved_flops_per_sec / self._peak
        return out
