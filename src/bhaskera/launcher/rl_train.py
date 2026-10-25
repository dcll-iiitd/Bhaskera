"""
bhaskera.launcher.rl_train
==========================
CLI for post-training RL (GRPO) on a single GPU.

    bhaskera-rl --config configs/rl_math.yaml

Reads the same YAML as ``bhaskera-train`` plus an ``rl:`` section. It writes LoRA
checkpoints in the same layout (``<save_dir>/step_NNNNNNN/model`` + ``.complete``)
and prints progress lines in the same format, so tools that consume SFT runs,
such as the Slakshna federated engine, work unchanged.

Single-GPU only: ``--num-workers`` is accepted for CLI compatibility but must be 1.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

from bhaskera.config import Config, load_config

logger = logging.getLogger(__name__)


def progress_line(epoch: int, step: int, m: dict, samples: int, tokens: int) -> str:
    """One stdout line per step, in the same prefix format as SFT logs:
    ``[epoch E][step S] loss=... samples=... tokens=...``."""
    return (
        f"[epoch {epoch}][step {step}] loss={m['loss']:.6f} "
        f"reward={m['reward_mean']:.4f} kl={m['kl']:.5f} clip_frac={m['clip_frac']:.4f} "
        f"grad_norm={m['grad_norm']:.4f} completion_len={m['completion_len']:.1f} "
        f"samples={samples} tokens={tokens}"
    )


def validate(cfg: Config) -> None:
    if not cfg.lora.enabled:
        raise ValueError(
            "RL training needs lora.enabled: true (the reference policy is the base model)"
        )
    if not cfg.rl.train_path:
        raise ValueError("rl.train_path is required")
    if cfg.rl.group_size < 2:
        raise ValueError("rl.group_size must be >= 2")
    if cfg.rl.epochs_per_rollout < 1 or cfg.rl.prompts_per_step < 1 or cfg.rl.micro_batch_size < 1:
        raise ValueError(
            "rl.epochs_per_rollout, prompts_per_step and micro_batch_size must be >= 1"
        )


def run(cfg: Config, resume_samples: int = 0) -> dict:
    """Train according to ``cfg``; returns the final metrics. Importable for tests and notebooks."""
    from transformers import AutoTokenizer

    from bhaskera.distributed.checkpoint import _checkpoint_path, save_checkpoint
    from bhaskera.models import build_model
    from bhaskera.plugins.loader import load_plugins
    from bhaskera.rl.data import PromptSampler, load_examples
    from bhaskera.rl.rewards import get_reward
    from bhaskera.rl.trainer import GRPOTrainer, TrainerSettings
    from bhaskera.trainer.optim import build_optimizer, build_scheduler

    validate(cfg)
    load_plugins(cfg)   # custom optimizers (e.g. muon) register themselves on import
    rl, train = cfg.rl, cfg.training

    # Round 1 (offset 0) uses the plain seed, so every node starts from the same LoRA init and
    # their deltas are comparable. Later rounds shift it so sampling noise is not replayed.
    seed = train.seed + resume_samples
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # RL runs on one device without FSDP/DDP wrapping. Selecting "ddp" here keeps the
    # LoRA weights in fp32 (see models/lora.py), which AdamW needs at small learning rates.
    train.distributed.strategy = "ddp"

    model, _ = build_model(cfg, device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code
    )

    examples = load_examples(rl.train_path)
    eval_examples = load_examples(rl.eval_path, rl.eval_max_prompts) if rl.eval_path else None
    # Resume the prompt stream where earlier rounds stopped: `samples` counts completions.
    sampler = PromptSampler(examples, seed=train.seed, position=resume_samples // rl.group_size)

    optimizer = build_optimizer(model, train)
    scheduler = build_scheduler(optimizer, train)
    trainer = GRPOTrainer(
        model, tokenizer,
        TrainerSettings(
            group_size=rl.group_size, prompts_per_step=rl.prompts_per_step,
            max_new_tokens=rl.max_new_tokens, temperature=rl.temperature,
            kl_coef=rl.kl_coef, clip_eps=rl.clip_eps,
            epochs_per_rollout=rl.epochs_per_rollout, micro_batch_size=rl.micro_batch_size,
            grad_clip=train.grad_clip, system_prompt=rl.system_prompt,
        ),
        get_reward(rl.reward), sampler, optimizer, scheduler, samples_offset=resume_samples,
    )

    def evaluate_and_print(step: int) -> dict:
        result = trainer.evaluate(eval_examples, rl.eval_batch_size)
        print(f"[eval][step {step}] eval_reward={result['eval_reward']:.4f} "
              f"examples={result['eval_examples']}", flush=True)
        return result

    def checkpoint(step: int) -> None:
        if cfg.checkpoint.enabled:
            save_checkpoint(
                model, optimizer, step, _checkpoint_path(cfg.checkpoint.save_dir, step),
                extra={"samples": trainer.samples_seen}, keep_last_n=cfg.checkpoint.keep_last_n,
                lora_only=True,
            )

    final = {}
    if eval_examples is not None:
        final["initial_eval_reward"] = evaluate_and_print(0)["eval_reward"]

    metrics = {}
    for step in range(1, train.max_steps + 1):
        metrics = trainer.step()
        line = progress_line(
            sampler.epoch, step, metrics, trainer.samples_seen, trainer.tokens_seen
        )
        print(line, flush=True)
        # the final step is evaluated once, after the loop
        if (eval_examples is not None and rl.eval_every and step % rl.eval_every == 0
                and step != train.max_steps):
            evaluate_and_print(step)
        if cfg.checkpoint.save_interval and step % cfg.checkpoint.save_interval == 0 \
                and step != train.max_steps:
            checkpoint(step)

    checkpoint(train.max_steps)   # the last step is always saved
    if eval_examples is not None:
        final["final_eval_reward"] = evaluate_and_print(train.max_steps)["eval_reward"]
    final.update(last_step_metrics=metrics)
    return final


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[%(asctime)s][%(name)s] %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Bhaskera post-training RL (GRPO)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--num-workers", type=int, default=1, help="Must be 1 (single-GPU RL)")
    p.add_argument("--resume-samples", type=int, default=0,
                   help="Completions generated in earlier rounds")
    p.add_argument("--resume-tokens", type=int, default=0,
                   help="Accepted for CLI compatibility; unused")
    args = p.parse_args()

    if args.num_workers != 1:
        logger.warning("RL training is single-GPU; ignoring --num-workers=%d and using one GPU",
                       args.num_workers)
    if os.environ.get("RAY_ADDRESS"):
        logger.info("RL training does not use Ray; ignoring RAY_ADDRESS")

    cfg = load_config(args.config)
    run(cfg, resume_samples=args.resume_samples)


if __name__ == "__main__":
    main()
