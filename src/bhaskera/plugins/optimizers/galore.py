"""
GaLore Optimizer Plugin for Bhaskera.
=====================================
Gradient Low-Rank Projection (GaLore) — full-parameter fine-tuning with
Adam-level performance at a fraction of the optimizer-state memory.
Reference: https://arxiv.org/abs/2403.03507 (Zhao et al., 2024)

Self-contained: implements the projector + GaLoreAdamW directly, no
`galore-torch` package required.

Usage: run with `lora.enabled: false` (GaLore replaces LoRA/QLoRA; it does
not compose with them — QLoRA's Params4bit base weights have no gradient
for GaLore to project, and LoRA's adapters make full-param projection moot).
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer

from bhaskera.trainer.optimizer_registry import register_optimizer

logger = logging.getLogger(__name__)

# Falls back to these common attention/MLP projection names if the config
# doesn't specify target_modules.
_DEFAULT_GALORE_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "w1", "w2", "w3", "fc1", "fc2",
)


# ---------------------------------------------------------------------------
# Core GaLore algorithm
# ---------------------------------------------------------------------------

class GaLoreProjector:
    """
    Maintains a low-rank orthogonal projection matrix (refreshed via SVD
    every `update_proj_gap` steps) that projects a full-size gradient down
    to `rank` dimensions and back. Adam's exp_avg / exp_avg_sq are kept in
    the projected (small) space — that's where the memory savings come
    from, not from the weights themselves (which stay full-size and dense).
    """

    def __init__(self, rank, update_proj_gap=200, scale=0.25, proj_type="std"):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix = None

    @staticmethod
    def _orthogonal_basis(tensor: torch.Tensor, rank: int, side: str) -> torch.Tensor:
        orig_dtype = tensor.dtype
        orig_device = tensor.device
        # Compute SVD in float32 for stability
        matrix = tensor.float() if orig_dtype != torch.float32 else tensor

        U, _, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if side == "right":
            basis = Vh[:rank, :]
        elif side == "left":
            basis = U[:, :rank]
        else:
            raise ValueError("side must be 'left' or 'right'")

        return basis.to(device=orig_device, dtype=orig_dtype)

    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        if self.proj_type != "std":
            raise NotImplementedError(
                f"proj_type='{self.proj_type}' not implemented in this "
                "inlined version — only 'std' is provided. Add reverse_std/"
                "left/right/full variants here if you need them (see the "
                "GaLore paper appendix)."
            )
        
        # To maximize memory savings, we project away the larger dimension.
        # Wide matrix (M < N): project columns (right) -> state size M x Rank
        # Tall matrix (M > N): project rows (left) -> state size Rank x N
        wide = full_rank_grad.shape[0] < full_rank_grad.shape[1]
        side = "right" if wide else "left"

        if self.ortho_matrix is None or step % self.update_proj_gap == 0:
            self.ortho_matrix = self._orthogonal_basis(full_rank_grad, self.rank, side)

        if side == "right":
            return full_rank_grad @ self.ortho_matrix.t()
        return self.ortho_matrix.t() @ full_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        # Use the same side decision the projector made when building ortho_matrix:
        # ortho_matrix is (rank, d) for 'right' side, (d, rank) for 'left' side.
        if self.ortho_matrix.shape[1] != self.rank:  # (rank, d) -> was 'right'
            full = low_rank_grad @ self.ortho_matrix
        else:  # (d, rank) -> was 'left'
            full = self.ortho_matrix @ low_rank_grad
        return full * self.scale


class GaLoreAdamW(Optimizer):
    """
    AdamW with GaLore gradient projection applied to any param group that
    carries a 'rank' key. Param groups without 'rank' behave like plain
    AdamW (used for embeddings/norms/biases in the framework's split).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            use_galore = "rank" in group

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GaLore does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                if use_galore:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            rank=group["rank"],
                            update_proj_gap=group.get("update_proj_gap", 200),
                            scale=group.get("scale", 0.25),
                            proj_type=group.get("proj_type", "std"),
                        )
                    grad = state["projector"].project(grad, state["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Standard PyTorch AdamW bias correction
                bc1 = 1.0 - beta1 ** state["step"] if group["correct_bias"] else 1.0
                bc2 = 1.0 - beta2 ** state["step"] if group["correct_bias"] else 1.0
                
                # Epsilon must be added AFTER variance bias correction
                denom = (exp_avg_sq / bc2).sqrt().add_(group["eps"])
                step_size = group["lr"] / bc1

                update = exp_avg / denom
                if use_galore:
                    update = state["projector"].project_back(update)

                # Weight decay must be applied BEFORE subtracting the gradient update
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                p.add_(update, alpha=-step_size)

        return loss


# ---------------------------------------------------------------------------
# Bhaskera plugin wiring
# ---------------------------------------------------------------------------

def _split_galore_params(model: nn.Module, target_suffixes: set[str]):
    """
    Route trainable params into two buckets:
      - galore_params: 2-D Linear weights inside target modules
      - regular_params: everything else trainable (embeddings, norms,
        biases, lm_head) — plain AdamW behavior in the same optimizer
    """
    galore_params, regular_params = [], []
    galore_ids = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = module_name.split(".")[-1]
        if target_suffixes and short_name not in target_suffixes:
            continue
        if module.weight.requires_grad:
            galore_params.append(module.weight)
            galore_ids.add(id(module.weight))

    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in galore_ids:
            regular_params.append(p)

    return galore_params, regular_params


@register_optimizer("galore")
def build_galore(model, train_cfg):
    opt_cfg = train_cfg.optimizer
    kwargs = dict(opt_cfg.kwargs)  # copy — we're about to pop from it

    lr = kwargs.pop("lr", train_cfg.lr)
    weight_decay = kwargs.pop("weight_decay", train_cfg.weight_decay)
    rank = kwargs.pop("rank", 128)
    update_proj_gap = kwargs.pop("update_proj_gap", 200)
    scale = kwargs.pop("scale", 0.25)
    proj_type = kwargs.pop("proj_type", "std")
    target_suffixes = set(kwargs.pop("target_modules", _DEFAULT_GALORE_TARGETS))
    kwargs.pop("use_8bit", None)  # not implemented in this inlined version

    galore_params, regular_params = _split_galore_params(model, target_suffixes)

    if not galore_params:
        raise ValueError(
            "GaLore found no matching Linear weights to project. Check "
            "optimizer.kwargs.target_modules against your model's actual "
            "module names (run `bhaskera-introspect` to list them)."
        )

    param_groups = [
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "weight_decay": weight_decay,
        },
        {
            "params": regular_params,
            "weight_decay": weight_decay,
        },
    ]

    logger.info(
        f"GaLore: {len(galore_params)} projected tensor(s) "
        f"(rank={rank}, update_proj_gap={update_proj_gap}, proj_type={proj_type}) "
        f"+ {len(regular_params)} regular tensor(s)"
    )

    return GaLoreAdamW(param_groups, lr=lr, betas=(0.9, 0.999), **kwargs)
