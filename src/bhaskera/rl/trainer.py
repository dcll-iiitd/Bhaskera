"""
bhaskera.rl.trainer
===================
GRPO training of a LoRA-adapted causal LM on a single device.

One training step:

    1. take ``prompts_per_step`` prompts
    2. sample ``group_size`` completions for each          (rollout.py)
    3. score each completion with the reward function      (rewards.py)
    4. advantage = reward relative to its group            (grpo.py)
    5. take ``epochs_per_rollout`` optimizer steps on the clipped GRPO loss

The reference policy for the KL term is the same network with the LoRA adapter
switched off, so it costs one extra forward pass and no extra memory.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from bhaskera.rl.data import Example, PromptSampler, format_prompt
from bhaskera.rl.grpo import completion_logprobs, group_advantages, grpo_sequence_loss
from bhaskera.rl.rewards import RewardFn
from bhaskera.rl.rollout import Rollouts, generate_rollouts

logger = logging.getLogger(__name__)


@dataclass
class TrainerSettings:
    group_size: int = 8
    prompts_per_step: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    epochs_per_rollout: int = 1
    micro_batch_size: int = 8
    grad_clip: Optional[float] = 1.0
    system_prompt: str = ""


class GRPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        settings: TrainerSettings,
        reward_fn: RewardFn,
        sampler: PromptSampler,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        samples_offset: int = 0,
    ):
        if not hasattr(model, "disable_adapter"):
            raise ValueError(
                "GRPOTrainer needs a PEFT (LoRA) model: the reference policy is the base model"
            )
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError("model has no trainable parameters")
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = settings
        self.reward_fn = reward_fn
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_count = 0
        self.samples_seen = samples_offset   # completions generated so far, across rounds
        self.tokens_seen = 0

    # ------------------------------------------------------------------
    def _micro_batches(self, n: int):
        size = self.cfg.micro_batch_size
        for start in range(0, n, size):
            yield slice(start, min(start + size, n))

    @torch.no_grad()
    def _logprobs(self, r: Rollouts, adapter: bool) -> torch.Tensor:
        """Completion logprobs without gradients; ``adapter=False`` gives the reference policy."""
        chunks = []
        for s in self._micro_batches(r.input_ids.shape[0]):
            args = (self.model, r.input_ids[s], r.attention_mask[s], r.prompt_len,
                    self.cfg.temperature)
            if adapter:
                chunks.append(completion_logprobs(*args))
            else:
                with self.model.disable_adapter():
                    chunks.append(completion_logprobs(*args))
        return torch.cat(chunks)

    def _score(self, texts: List[str], examples: List[Example]) -> torch.Tensor:
        g = self.cfg.group_size
        return torch.tensor(
            [self.reward_fn(t, examples[i // g].answer) for i, t in enumerate(texts)],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, float]:
        """Run one GRPO step and return its metrics."""
        cfg = self.cfg
        t0 = time.time()
        examples = self.sampler.next_batch(cfg.prompts_per_step)
        prompts = [format_prompt(self.tokenizer, e.prompt, cfg.system_prompt) for e in examples]

        rollouts = generate_rollouts(
            self.model, self.tokenizer, prompts, cfg.group_size, cfg.max_new_tokens, cfg.temperature
        )
        device = rollouts.input_ids.device
        rewards = self._score(rollouts.texts, examples)
        advantages = group_advantages(rewards, cfg.group_size).to(device)
        n = rewards.numel()
        n_completion_tokens = int(rollouts.completion_mask.sum().item())
        group_has_signal = rewards.view(-1, cfg.group_size).std(dim=1) > 0

        metrics = {
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item() if n > 1 else 0.0,
            "completion_len": n_completion_tokens / n,
            "frac_groups_with_signal": group_has_signal.float().mean().item(),
            "loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
        }
        self.samples_seen += n
        self.tokens_seen += n_completion_tokens
        self.step_count += 1

        # If every group got identical rewards there is no learning signal. Skip the
        # update (it would only apply the KL term), but still count the step.
        if (advantages != 0).any():
            metrics.update(self._update(rollouts, advantages, n))
        else:
            logger.info("step %d: all groups had identical rewards; skipping update",
                        self.step_count)

        metrics["step_seconds"] = time.time() - t0
        return metrics

    def _update(self, r: Rollouts, advantages: torch.Tensor, n: int) -> Dict[str, float]:
        cfg = self.cfg
        # With a single optimizer step per rollout the policy has not moved yet, so
        # old == current and the ratio is 1; otherwise fix old logprobs up front.
        old_logp = self._logprobs(r, adapter=True) if cfg.epochs_per_rollout > 1 else None
        ref_logp = self._logprobs(r, adapter=False) if cfg.kl_coef > 0 else None

        self.model.eval()  # dropout off; gradients still flow
        stats = {}
        for _ in range(cfg.epochs_per_rollout):
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum, kl_sum, clipped, tokens = 0.0, 0.0, 0.0, 0.0
            for s in self._micro_batches(n):
                logp = completion_logprobs(
                    self.model, r.input_ids[s], r.attention_mask[s], r.prompt_len, cfg.temperature
                )
                old = logp.detach() if old_logp is None else old_logp[s]
                ref = None if ref_logp is None else ref_logp[s]
                mask = r.completion_mask[s]
                per_seq = grpo_sequence_loss(
                    logp, old, ref, advantages[s], mask, cfg.clip_eps, cfg.kl_coef
                )
                (per_seq.sum() / n).backward()   # mean over all n sequences, across micro-batches

                with torch.no_grad():
                    m = mask.float()
                    loss_sum += per_seq.sum().item()
                    tokens += m.sum().item()
                    if ref is not None:
                        d = ref - logp
                        kl_sum += ((torch.exp(d) - d - 1) * m).sum().item()
                    ratio = torch.exp(logp - old)
                    adv = advantages[s].unsqueeze(1)
                    too_high = (ratio > 1 + cfg.clip_eps) & (adv > 0)
                    too_low = (ratio < 1 - cfg.clip_eps) & (adv < 0)
                    outside = too_high | too_low
                    clipped += (outside.float() * m).sum().item()

            params = [p for p in self.model.parameters() if p.requires_grad]
            if cfg.grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
            else:
                squares = [(p.grad.float() ** 2).sum() for p in params if p.grad is not None]
                grad_norm = torch.sqrt(sum(squares)).item()
            if not torch.isfinite(torch.tensor(grad_norm)):
                logger.warning("non-finite gradient norm; skipping optimizer step")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            stats = {
                "loss": loss_sum / n,
                "kl": kl_sum / max(tokens, 1.0),
                "clip_frac": clipped / max(tokens, 1.0),
                "grad_norm": grad_norm,
            }
        return stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict[str, float]:
        """Greedy-decode held-out prompts and report mean reward (accuracy for 0/1 rewards)."""
        total, count = 0.0, 0
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            prompts = [format_prompt(self.tokenizer, e.prompt, self.cfg.system_prompt)
                       for e in chunk]
            r = generate_rollouts(
                self.model, self.tokenizer, prompts, 1, self.cfg.max_new_tokens, do_sample=False
            )
            for text, e in zip(r.texts, chunk):
                total += self.reward_fn(text, e.answer)
                count += 1
        return {"eval_reward": total / count, "eval_examples": count}
