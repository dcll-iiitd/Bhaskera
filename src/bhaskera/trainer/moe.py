"""
bhaskera.trainer.moe
====================
MoE auxiliary loss extraction, load balancing, and expert utilisation
metrics.

Changes vs v1:
  * _load_balancing_loss_from_logits now accumulates into a list and
    torch.stacks at the end instead of `total_loss += ...` on a fresh
    leaf tensor — cleaner graph, one allocation.
  * Guards against a mix of Tensor / non-Tensor router outputs (some HF
    MoE models emit tuples of (logits, indices)).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aux loss extraction
# ---------------------------------------------------------------------------

def extract_aux_loss(out, profile: ModelProfile) -> Optional[torch.Tensor]:
    """
    Pull the auxiliary loss out of the model output.  Supports:
      * out.aux_loss             (Mixtral, Qwen2MoE, most HF MoE models)
      * out.router_aux_loss      (alternative attribute)
      * out.moe_loss             (some custom models)
      * out.load_balancing_loss  (some custom models)
      * out.router_logits        → computed manually
    Returns None for dense models.
    """
    if not profile.is_moe:
        return None

    for attr in ("aux_loss", "router_aux_loss", "moe_loss", "load_balancing_loss"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val

    router_logits = getattr(out, "router_logits", None)
    if router_logits is not None:
        return _load_balancing_loss_from_logits(router_logits, profile)

    return None


def _load_balancing_loss_from_logits(
    router_logits,
    profile: ModelProfile,
) -> Optional[torch.Tensor]:
    """
    Switch-Transformer-style load balancing loss: N * sum_i (f_i * P_i),
    averaged across layers.  `f_i` is the fraction of tokens routed to
    expert i; `P_i` is the average routing probability for expert i.
    """
    # Prefer HF's reference implementation when available.
    try:
        from transformers.models.mixtral.modeling_mixtral import (
            load_balancing_loss_func,
        )
        return load_balancing_loss_func(
            router_logits,
            profile.num_experts,
            profile.experts_per_token,
        )
    except ImportError:
        pass

    try:
        per_layer: list[torch.Tensor] = []
        for logits in router_logits:
            if not isinstance(logits, torch.Tensor):
                continue
            probs = torch.softmax(logits, dim=-1)
            top_k = probs.topk(profile.experts_per_token, dim=-1).indices
            mask = torch.zeros_like(probs)
            mask.scatter_(1, top_k, 1.0)
            tokens_per_expert = mask.mean(dim=0)        # no grad — hard selection
            avg_probs         = probs.mean(dim=0)       # grad flows through here
            per_layer.append(
                (tokens_per_expert * avg_probs).sum() * profile.num_experts
            )
        if not per_layer:
            return None
        return torch.stack(per_layer).mean()
    except Exception as e:
        logger.warning(f"Failed to compute fallback load-balancing loss: {e}")
        return None


# ---------------------------------------------------------------------------
# Expert utilisation
# ---------------------------------------------------------------------------

def compute_expert_utilization(out, profile: ModelProfile) -> dict:
    """Compute expert utilisation metrics for logging."""
    metrics: dict = {}
    router_logits = getattr(out, "router_logits", None)
    if router_logits is None:
        return metrics

    try:
        all_loads = []
        for logits in router_logits:
            if not isinstance(logits, torch.Tensor):
                continue
            probs = torch.softmax(logits.float(), dim=-1)
            top_k = probs.topk(
                min(profile.experts_per_token, probs.shape[-1]), dim=-1
            ).indices
            mask = torch.zeros_like(probs)
            mask.scatter_(1, top_k, 1.0)
            all_loads.append(mask.sum(dim=0))

        if not all_loads:
            return metrics

        avg_load = torch.stack(all_loads).mean(dim=0)
        total = avg_load.sum()
        if total <= 0:
            return metrics

        util = avg_load / total
        metrics["expert/load_max"] = util.max().item()
        metrics["expert/load_min"] = util.min().item()
        metrics["expert/load_std"] = util.std().item()
        min_val = util.min().item()
        if min_val > 0:
            metrics["expert/imbalance_ratio"] = util.max().item() / min_val
    except Exception as e:
        logger.debug(f"Expert utilisation computation failed: {e}")

    return metrics