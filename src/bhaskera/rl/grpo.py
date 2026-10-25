"""
bhaskera.rl.grpo
================
The GRPO math (DeepSeekMath / DeepSeek-R1 recipe), kept as small pure functions.

For each prompt we sample a group of G completions and score them. A completion's
advantage is how much better its reward is than the group average:

    A_i = (r_i - mean(r)) / (std(r) + eps)

The loss for one completion, averaged over its tokens, is the PPO clipped objective
plus a KL penalty that keeps the policy close to the reference (the base model):

    -min(ratio * A, clip(ratio, 1-e, 1+e) * A)  +  beta * KL(policy || reference)

    ratio = exp(logp_new - logp_old)
    KL    = exp(d) - d - 1,  d = logp_ref - logp_new        (the k3 estimator)

No value network is needed, which is why GRPO is cheap enough to run with LoRA.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """Normalise rewards within each group. ``rewards`` is flat: G consecutive entries per prompt.

    A group whose rewards are all equal gets advantage 0 (nothing to learn from it).
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2: advantages are relative to the group")
    if rewards.numel() % group_size != 0:
        raise ValueError(f"{rewards.numel()} rewards is not a multiple of group_size={group_size}")
    grouped = rewards.float().view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)  # unbiased, as in TRL
    return ((grouped - mean) / (std + eps)).view(-1)


def completion_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log-probability of every completion token under ``model``: shape ``[N, C]``.

    Prompts are left-padded, so position ids must come from the attention mask;
    without them padded rows would see shifted positions and disagree with what
    ``generate`` used when it sampled the tokens.
    """
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    # logits at position t predict token t+1, so slice [P-1, T-1) for completion tokens
    logits = logits[:, prompt_len - 1:-1, :].float() / temperature
    targets = input_ids[:, prompt_len:]
    return torch.gather(F.log_softmax(logits, dim=-1), 2, targets.unsqueeze(-1)).squeeze(-1)


def grpo_sequence_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: Optional[torch.Tensor],
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    kl_coef: float,
) -> torch.Tensor:
    """Per-sequence GRPO loss, shape ``[N]`` (token mean within each sequence).

    ``logp``/``old_logp``/``ref_logp`` are ``[N, C]``; ``advantages`` is ``[N]``;
    ``mask`` is ``[N, C]`` with 1 on real completion tokens.
    """
    mask = mask.float()
    adv = advantages.unsqueeze(1)
    ratio = torch.exp(logp - old_logp)
    per_token = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    if kl_coef > 0:
        if ref_logp is None:
            raise ValueError("kl_coef > 0 requires ref_logp")
        d = ref_logp - logp
        per_token = per_token + kl_coef * (torch.exp(d) - d - 1)
    return (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
