"""
bhaskera.launcher.train_cpu
===========================
Single-process, Ray-free, CPU-only LoRA/SFT trainer — the laptop counterpart
of ``bhaskera.launcher.train`` (Ray Train + FSDP2/DDP + NCCL).

It is a drop-in replacement from the point of view of ``ml_engine.py``:

  * same CLI flags (``--config --num-workers --resume-samples --resume-tokens``)
  * same stdout contract, one line per optimizer step:
        [epoch 0][step 3] loss=2.2678 lr=... grad_norm=... samples=24 tokens=3072
  * same checkpoint layout: ``<save_dir>/step_XXXXXXX/{model/,meta.json,.complete}``
    where ``model/`` is a torch.distributed.checkpoint of the LoRA tensors
    (+ ``adapter_model.safetensors`` fallback), so the delta-extraction code in
    ml_engine.py works unchanged.

What is different (and why):
  * No Ray / FSDP / NCCL — one process, ``torch.set_num_threads(N)``.
  * Model is loaded in float32 (bf16 matmuls are slow or unsupported on most
    laptop CPUs, and fp32 keeps AdamW stable).
  * The packed block-diagonal attention mask is built exactly like the GPU path.
  * Dataset position is deterministic (seeded permutation per pass) so
    ``--resume-samples`` resumes exactly, and the data cycles when exhausted
    instead of ending the round.

Extra YAML section understood only by this launcher:

    cpu:
      threads: 4              # torch intra-op threads (default: $BHASKERA_CPU_THREADS or all cores)
      max_train_seconds: 240  # stop the round early once this wall-clock budget is spent
      data_shard: "0/2"       # this node trains on rows[0::2] (also: $SLAKSHNA_DATA_SHARD)
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("bhaskera.cpu")

_BATCH_KEYS = ("input_ids", "attention_mask", "labels", "position_ids", "seq_idx")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(tokenized_path: str) -> dict[str, list[np.ndarray]]:
    """Load every parquet shard under ``tokenized_path`` into memory."""
    import pyarrow.parquet as pq

    files = sorted(Path(tokenized_path).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under tokenized_path={tokenized_path!r}")
    cols: dict[str, list[np.ndarray]] = {k: [] for k in _BATCH_KEYS}
    for f in files:
        table = pq.read_table(str(f))
        for k in _BATCH_KEYS:
            if k in table.column_names:
                cols[k].extend(
                    np.asarray(v, dtype=np.int64) for v in table.column(k).to_pylist()
                )
    n = len(cols["input_ids"])
    if n == 0:
        raise RuntimeError(f"Tokenized cache {tokenized_path!r} is empty.")
    for k in _BATCH_KEYS:  # tolerate caches written before position_ids/seq_idx existed
        if not cols[k]:
            if k == "seq_idx":
                cols[k] = [np.ones_like(x) for x in cols["input_ids"]]
            elif k == "position_ids":
                cols[k] = [np.arange(len(x), dtype=np.int64) for x in cols["input_ids"]]
    return cols


def apply_shard(cols: dict, shard: str | None) -> dict:
    """``"i/n"`` -> keep rows[i::n]. Lets several simulated nodes see different data."""
    if not shard:
        return cols
    try:
        i, n = (int(x) for x in str(shard).split("/"))
        assert n >= 1 and 0 <= i < n
    except Exception:
        raise ValueError(f"Bad data shard {shard!r}; expected 'i/n' with 0 <= i < n")
    if n == 1:
        return cols
    out = {k: v[i::n] for k, v in cols.items()}
    if not out["input_ids"]:  # fewer rows than shards -> fall back to the whole set
        logger.warning(f"Shard {shard} is empty (dataset too small) — using all rows.")
        return cols
    return out


class DeterministicBatcher:
    """Serves batches by *global sample index* so a resume lands on the same row."""

    def __init__(self, cols: dict, batch_size: int, seed: int, start: int = 0):
        self.cols = cols
        self.n = len(cols["input_ids"])
        self.bs = batch_size
        self.seed = seed
        self.pos = start
        self._perm_pass = -1
        self._perm = None

    def _row(self, g: int) -> int:
        p, off = divmod(g, self.n)
        if p != self._perm_pass:
            self._perm = np.random.RandomState(self.seed + p).permutation(self.n)
            self._perm_pass = p
        return int(self._perm[off])

    def next(self) -> dict[str, torch.Tensor]:
        idx = [self._row(self.pos + j) for j in range(self.bs)]
        self.pos += self.bs
        return {
            k: torch.from_numpy(np.stack([self.cols[k][i] for i in idx]))
            for k in _BATCH_KEYS
        }


# ---------------------------------------------------------------------------
# Checkpoint (LoRA-only, ml_engine-compatible)
# ---------------------------------------------------------------------------

def save_lora_checkpoint(model, step: int, save_dir: str, avg_loss: float,
                         samples: int, tokens: int, keep_last_n: int) -> str:
    import json
    import re
    import shutil

    from safetensors.torch import save_file

    from bhaskera.distributed.checkpoint import _cleanup_old_checkpoints, _dcp_save

    path = os.path.join(save_dir, f"step_{step:07d}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    # PEFT's own key naming (…lora_A.default.weight) — identical to what
    # ml_engine.py reconstructs on a meta-device PEFT model for round 1.
    lora_sd = {
        k: v.detach().cpu().float().contiguous()
        for k, v in model.state_dict().items() if "lora" in k
    }
    if not lora_sd:
        raise RuntimeError("No LoRA tensors found in the model state_dict.")
    _dcp_save({"model": lora_sd}, os.path.join(tmp, "model"))

    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"step": int(step), "avg_loss": float(avg_loss),
                   "samples_consumed": int(samples), "tokens_consumed": int(tokens)}, f)

    adapter = {
        re.sub(r"(lora_[AB])\.[^.]+\.(weight)", r"\1.\2", k): v
        for k, v in lora_sd.items()
    }
    save_file(adapter, os.path.join(path, "adapter_model.safetensors"))

    open(os.path.join(path, ".complete"), "w").close()  # written last
    _cleanup_old_checkpoints(save_dir, keep_last_n)
    logger.info(f"Checkpoint saved -> {path} ({len(lora_sd)} LoRA tensors)")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera CPU (Ray-free) trainer")
    p.add_argument("--config", required=True)
    p.add_argument("--num-workers", type=int, default=1, help="Ignored (CPU = 1 process).")
    p.add_argument("--resume-samples", type=int, default=0)
    p.add_argument("--resume-tokens", type=int, default=0)
    p.add_argument("--max-failures", type=int, default=0, help="Ignored.")
    p.add_argument("--storage-path", type=str, default=None, help="Ignored.")
    p.add_argument("--no-dashboard", action="store_true", help="Ignored.")
    p.add_argument("--dashboard-port", type=int, default=None, help="Ignored.")
    args = p.parse_args()

    from bhaskera.config import load_config
    from bhaskera.plugins.loader import load_plugins

    cfg = load_config(args.config)
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    cpu_cfg = raw.get("cpu") or {}

    # ── CPU-specific overrides ───────────────────────────────────────────
    cfg.model.dtype = "float32"
    cfg.model.use_liger_kernel = False          # Triton kernels: GPU only
    if cfg.model.quantization not in (None, "none"):
        logger.warning("model.quantization needs bitsandbytes/CUDA — disabled on CPU.")
        cfg.model.quantization = "none"
    if cfg.model.attn_impl in ("flash_attention_2", "flash_attention_3"):
        cfg.model.attn_impl = "sdpa"
    cfg.training.distributed.strategy = "ddp"   # only used by lora.py's dtype branch
    load_plugins(cfg)

    threads = int(cpu_cfg.get("threads") or os.environ.get("BHASKERA_CPU_THREADS") or os.cpu_count() or 1)
    torch.set_num_threads(max(1, threads))
    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_seconds = float(cpu_cfg.get("max_train_seconds") or os.environ.get("SLAKSHNA_CPU_MAX_SECONDS") or 0)
    shard = cpu_cfg.get("data_shard") or os.environ.get("SLAKSHNA_DATA_SHARD")

    logger.info(
        f"CPU trainer | threads={torch.get_num_threads()} | model={cfg.model.name} | "
        f"batch={cfg.training.batch_size}x{cfg.training.grad_accum} | seq_len={cfg.data.seq_len} | "
        f"max_steps={cfg.training.max_steps} | shard={shard or 'all'} | "
        f"budget={max_seconds or 'none'}s"
    )

    # ── Data ─────────────────────────────────────────────────────────────
    cols = apply_shard(load_rows(cfg.data.tokenized_path), shard)
    n_rows = len(cols["input_ids"])
    logger.info(f"Loaded {n_rows} packed rows from {cfg.data.tokenized_path}")

    # ── Model (+ LoRA) ───────────────────────────────────────────────────
    from bhaskera.models import build_model
    model, profile = build_model(cfg, torch.device("cpu"))
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model ready | trainable={trainable:,}/{total:,} params")

    # ── Optimiser / scheduler (same factories as the GPU path) ───────────
    from bhaskera.trainer.optim import build_optimizer, build_scheduler
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training)

    from bhaskera.trainer.packing import build_4d_attention_mask

    bs = cfg.training.batch_size
    accum = cfg.training.grad_accum
    max_steps = cfg.training.max_steps
    grad_clip = cfg.training.grad_clip or cfg.training.max_grad_norm or 1.0
    is_packed = bool(cfg.data.pack_sequences)

    samples = int(args.resume_samples)
    tokens = int(args.resume_tokens)
    batcher = DeterministicBatcher(cols, bs, seed, start=samples)

    t0 = time.monotonic()
    step = 0
    step_losses: list[float] = []
    non_finite = 0

    while step < max_steps:
        micro_losses = []
        for _ in range(accum):
            batch = batcher.next()
            input_ids = batch["input_ids"]
            kwargs = dict(
                input_ids=input_ids,
                labels=batch["labels"],
                position_ids=batch["position_ids"],
                use_cache=False,
            )
            if is_packed:
                kwargs["attention_mask"] = build_4d_attention_mask(batch["seq_idx"], torch.float32)
            else:
                kwargs["attention_mask"] = batch["attention_mask"]

            samples += int(input_ids.size(0))
            tokens += int(input_ids.numel())
            out = model(**kwargs)
            if not torch.isfinite(out.loss):
                # e.g. a micro-batch with zero supervised tokens (0/0). Skip it instead
                # of poisoning the whole step; the cursor still advances past it.
                logger.warning("non-finite micro-batch loss — skipping this micro-batch")
                continue
            (out.loss / accum).backward()
            micro_losses.append(out.loss.detach())

        if not micro_losses:
            optimizer.zero_grad(set_to_none=True)
            non_finite += 1
            if non_finite >= 5:
                raise RuntimeError("Too many empty/non-finite micro-batches — check the tokenized cache.")
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        ).item()

        if not math.isfinite(grad_norm):
            non_finite += 1
            logger.warning(f"[step {step}] non-finite grad_norm={grad_norm} — skipping update")
            optimizer.zero_grad(set_to_none=True)
            if non_finite >= cfg.training.max_grad_skip_steps or non_finite >= 5:
                raise RuntimeError("Too many non-finite gradient steps — aborting round.")
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        loss = float(torch.stack(micro_losses).mean())
        step_losses.append(loss)
        elapsed = time.monotonic() - t0
        # Exact format parsed by ml_engine.py (regex on [epoch N][step N] loss=… samples=… tokens=…)
        print(
            f"[epoch 0][step {step}] loss={loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
            f"grad_norm={grad_norm:.4f} samples={samples} tokens={tokens} "
            f"tok/s={(tokens - args.resume_tokens) / max(elapsed, 1e-9):.0f}",
            flush=True,
        )

        if max_seconds and elapsed >= max_seconds and step < max_steps:
            logger.warning(
                f"Wall-clock budget {max_seconds:.0f}s reached after {step}/{max_steps} steps — "
                "ending this round early."
            )
            break

    if step == 0:
        raise RuntimeError("No optimizer step completed.")

    avg_loss = float(np.mean(step_losses))
    print(f"[epoch 0] avg_loss={avg_loss:.4f}", flush=True)

    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
        save_lora_checkpoint(
            model, step, cfg.checkpoint.save_dir, avg_loss, samples, tokens,
            keep_last_n=max(1, cfg.checkpoint.keep_last_n),
        )
    logger.info(f"Training complete in {time.monotonic() - t0:.0f}s ({step} steps).")


if __name__ == "__main__":
    main()
