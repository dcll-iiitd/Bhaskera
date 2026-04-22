"""
bhaskera.introspect
===================
Auto-detect model architecture properties — MoE topology, layer classes,
LoRA targets, dtype, aux-loss support — by walking the module tree once.

Returns a ModelProfile consumed by models/, distributed/, and trainer/.
Zero hardcoded class names. Works for any HuggingFace CausalLM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Profile dataclass — everything the pipeline needs to know
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    # Architecture type
    is_moe: bool = False
    num_experts: int = 0
    num_shared_experts: int = 0
    experts_per_token: int = 0

    # Layer classes (resolved types, not strings)
    decoder_layer_cls: Optional[type] = None
    expert_module_cls: Optional[type] = None

    # Actual module references for FSDP per-expert sharding
    expert_modules: list[nn.Module] = field(default_factory=list)

    # Router / gate module names for freezing
    router_module_names: list[str] = field(default_factory=list)

    # Loss
    has_aux_loss: bool = False
    aux_loss_attr: str = "aux_loss"          # "aux_loss" or "router_logits" style

    # LoRA
    lora_targets: list[str] = field(default_factory=list)

    # Precision
    model_dtype: torch.dtype = torch.bfloat16

    # Metadata
    num_hidden_layers: int = 0
    model_type: str = ""


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def introspect_model(model: nn.Module) -> ModelProfile:
    """
    Walk a loaded model ONCE and populate a ModelProfile.
    No hardcoded class names — pure structural detection.
    """
    profile = ModelProfile()

    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Model has no .config — introspection will be limited")
        profile.model_dtype = _detect_dtype_from_params(model)
        return profile

    # ── Basic metadata ──────────────────────────────────────────────
    profile.model_type = getattr(config, "model_type", "")
    profile.num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    profile.model_dtype = _resolve_dtype(model, config)

    # ── MoE detection from config ───────────────────────────────────
    profile.is_moe = _detect_moe_from_config(config)
    if profile.is_moe:
        profile.num_experts = _read_num_experts(config)
        profile.num_shared_experts = _read_num_shared_experts(config)
        profile.experts_per_token = _read_experts_per_token(config)

    # ── Structural detection from module tree ───────────────────────
    decoder_cls = _find_decoder_layer_cls(model, profile.num_hidden_layers)
    profile.decoder_layer_cls = decoder_cls

    # MoE structural detection (validates / supplements config detection)
    expert_cls, expert_modules, router_names = _find_moe_components(model)
    if expert_modules:
        profile.is_moe = True  # structural override — always trust the tree
        profile.expert_module_cls = expert_cls
        profile.expert_modules = expert_modules
        profile.router_module_names = router_names
        if profile.num_experts == 0:
            profile.num_experts = len(expert_modules) // max(profile.num_hidden_layers, 1)

    # ── Aux loss detection ──────────────────────────────────────────
    profile.has_aux_loss = _detect_aux_loss(config)
    profile.aux_loss_attr = _detect_aux_loss_attr(config)

    # ── LoRA targets ────────────────────────────────────────────────
    profile.lora_targets = _find_lora_targets(model, decoder_cls, profile.router_module_names)

    _log_profile(profile)
    return profile


# ──────────────────────────────────────────────────────────────────────
# MoE config detection
# ──────────────────────────────────────────────────────────────────────

_MOE_CONFIG_KEYS = [
    "num_local_experts",  # Mixtral, Qwen2MoE, Param2
    "num_experts",        # some custom models
    "n_routed_experts",   # DeepSeek
    "num_experts_per_tok",
    "num_moe_experts",
]


def _detect_moe_from_config(config) -> bool:
    """Check if HF config has any MoE-related attributes."""
    for key in _MOE_CONFIG_KEYS:
        val = getattr(config, key, None)
        if val is not None and val > 0:
            return True
    # Some models just set model_type with "moe" in it
    model_type = getattr(config, "model_type", "")
    if "moe" in model_type.lower():
        return True
    return False


def _read_num_experts(config) -> int:
    """Read total number of experts from config — tries multiple attribute names."""
    for key in ["num_local_experts", "num_experts", "n_routed_experts", "num_moe_experts"]:
        val = getattr(config, key, None)
        if val is not None and val > 0:
            return int(val)
    return 0


def _read_num_shared_experts(config) -> int:
    for key in ["num_shared_experts", "n_shared_experts", "num_shared_expert"]:
        val = getattr(config, key, None)
        if val is not None and val > 0:
            return int(val)
    return 0


def _read_experts_per_token(config) -> int:
    for key in ["num_experts_per_tok", "top_k", "num_selected_experts", "experts_per_token"]:
        val = getattr(config, key, None)
        if val is not None and val > 0:
            return int(val)
    return 2  # default for most MoE models


# ──────────────────────────────────────────────────────────────────────
# Structural detection — decoder layers
# ──────────────────────────────────────────────────────────────────────

# Common attribute names for the main layer container across architectures
_LAYER_CONTAINER_ATTRS = [
    ("model", "layers"),       # LLaMA, Mistral, Mixtral, Qwen, Param2
    ("transformer", "h"),      # GPT-2, Falcon
    ("gpt_neox", "layers"),    # GPT-NeoX, Pythia
    ("model", "decoder", "layers"),  # BART / OPT-style
]


def _find_decoder_layer_cls(model: nn.Module, expected_count: int) -> Optional[type]:
    """
    Find the decoder layer class by locating the container that holds
    exactly `expected_count` children of the same type.
    Tries known attribute paths first, then brute-force walks.
    """
    # Fast path: try common attribute paths
    for attr_path in _LAYER_CONTAINER_ATTRS:
        container = model
        for attr in attr_path:
            container = getattr(container, attr, None)
            if container is None:
                break
        if container is not None and isinstance(container, nn.ModuleList):
            children = list(container.children())
            if len(children) >= 1:
                cls = type(children[0])
                # Verify all are the same class
                if all(type(c) == cls for c in children):
                    if expected_count == 0 or len(children) == expected_count:
                        logger.info(
                            f"Decoder layer class: {cls.__name__} "
                            f"({len(children)} layers via fast path)"
                        )
                        return cls

    # Slow path: brute-force walk — find any ModuleList whose children
    # all share a type and whose count matches expected_count
    if expected_count > 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList):
                children = list(module.children())
                if len(children) == expected_count:
                    cls = type(children[0])
                    if all(type(c) == cls for c in children):
                        logger.info(
                            f"Decoder layer class: {cls.__name__} "
                            f"({len(children)} layers via brute-force)"
                        )
                        return cls

    logger.warning("Could not auto-detect decoder layer class")
    return None


# ──────────────────────────────────────────────────────────────────────
# Structural detection — MoE experts and routers
# ──────────────────────────────────────────────────────────────────────

_EXPERT_KEYWORDS = {"experts", "local_experts", "routed_experts"}
_ROUTER_KEYWORDS = {"gate", "router", "switch", "gating"}


def _find_moe_components(
    model: nn.Module,
) -> tuple[Optional[type], list[nn.Module], list[str]]:
    """
    Walk the model tree looking for expert modules and router/gate modules.

    Returns:
        expert_cls:      type of individual expert (or None)
        expert_modules:  flat list of all expert nn.Module instances
        router_names:    list of full names for gate/router modules
    """
    expert_cls: Optional[type] = None
    expert_modules: list[nn.Module] = []
    router_names: list[str] = []

    for name, module in model.named_modules():
        name_lower = name.split(".")[-1].lower()

        # Detect expert containers (nn.ModuleList named "experts" etc.)
        if name_lower in _EXPERT_KEYWORDS and isinstance(module, nn.ModuleList):
            children = list(module.children())
            if len(children) >= 2:
                cls = type(children[0])
                if expert_cls is None:
                    expert_cls = cls
                expert_modules.extend(children)

        # Detect routers / gates
        if any(kw in name_lower for kw in _ROUTER_KEYWORDS):
            # Only count actual parameter-bearing modules, not containers
            if list(module.parameters()):
                router_names.append(name)

    # Deduplicate shared experts if they're also in the experts list
    # (some architectures put shared experts separately)
    for name, module in model.named_modules():
        name_lower = name.split(".")[-1].lower()
        if "shared" in name_lower and "expert" in name_lower:
            if isinstance(module, nn.ModuleList):
                for child in module.children():
                    if child not in expert_modules:
                        expert_modules.append(child)
            elif list(module.parameters()) and module not in expert_modules:
                expert_modules.append(module)

    if expert_modules:
        logger.info(
            f"MoE detected: {len(expert_modules)} expert modules, "
            f"expert class={expert_cls.__name__ if expert_cls else 'N/A'}, "
            f"{len(router_names)} router modules"
        )

    return expert_cls, expert_modules, router_names


# ──────────────────────────────────────────────────────────────────────
# Aux loss detection
# ──────────────────────────────────────────────────────────────────────

def _detect_aux_loss(config) -> bool:
    """Check if the model config indicates auxiliary loss support."""
    # HuggingFace standard: router_aux_loss_coef > 0
    coef = getattr(config, "router_aux_loss_coef", None)
    if coef is not None and coef > 0:
        return True
    # Some models use these
    for attr in ["aux_loss_alpha", "moe_aux_loss_coeff", "output_router_logits"]:
        val = getattr(config, attr, None)
        if val is not None and val:
            return True
    # If it's MoE, assume aux loss is available
    if _detect_moe_from_config(config):
        return True
    return False

def _detect_aux_loss_attr(config) -> str:
    """Determine how the model exposes auxiliary loss."""
    # Most HF MoE models put it in output.aux_loss when output_router_logits=True
    if hasattr(config, "output_router_logits"):
        return "aux_loss"
    # DeepSeek-style: separate attribute
    if hasattr(config, "aux_loss_alpha"):
        return "aux_loss"
    return "aux_loss"  # safe default


# ──────────────────────────────────────────────────────────────────────
# LoRA target discovery
# ──────────────────────────────────────────────────────────────────────

def _find_lora_targets(
    model: nn.Module,
    decoder_cls: Optional[type],
    router_names: list[str],
) -> list[str]:
    """
    Inspect one decoder layer to find all nn.Linear module names.
    Excludes router/gate modules.
    Returns short names like ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', ...].
    """
    if decoder_cls is None:
        return []

    # Grab one decoder layer instance
    sample_layer = None
    for module in model.modules():
        if isinstance(module, decoder_cls):
            sample_layer = module
            break
    if sample_layer is None:
        return []

    # Collect router keywords from full paths (just the last segment)
    router_short = set()
    for rn in router_names:
        parts = rn.split(".")
        router_short.update(parts)

    targets = set()
    for name, mod in sample_layer.named_modules():
        if isinstance(mod, nn.Linear):
            short_name = name.split(".")[-1]
            # Skip router / gate linear layers
            if any(kw in short_name.lower() for kw in _ROUTER_KEYWORDS):
                continue
            if short_name in router_short:
                continue
            targets.add(short_name)

    result = sorted(targets)
    logger.info(f"Auto-detected LoRA targets: {result}")
    return result


# ──────────────────────────────────────────────────────────────────────
# Dtype resolution
# ──────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(model: nn.Module, config) -> torch.dtype:
    """
    Determine the model's native dtype.
    Priority: config.torch_dtype > first parameter dtype > bfloat16 fallback.
    """
    # HF stores this as a string or torch.dtype
    cfg_dtype = getattr(config, "torch_dtype", None)
    if cfg_dtype is not None:
        if isinstance(cfg_dtype, torch.dtype):
            return cfg_dtype
        if isinstance(cfg_dtype, str) and cfg_dtype in _DTYPE_MAP:
            return _DTYPE_MAP[cfg_dtype]

    return _detect_dtype_from_params(model)


def _detect_dtype_from_params(model: nn.Module) -> torch.dtype:
    """Read dtype from the first parameter."""
    for p in model.parameters():
        return p.dtype
    return torch.bfloat16


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

def _log_profile(profile: ModelProfile) -> None:
    lines = [
        f"  model_type       = {profile.model_type}",
        f"  is_moe           = {profile.is_moe}",
        f"  num_experts      = {profile.num_experts}",
        f"  shared_experts   = {profile.num_shared_experts}",
        f"  experts_per_tok  = {profile.experts_per_token}",
        f"  decoder_cls      = {profile.decoder_layer_cls.__name__ if profile.decoder_layer_cls else 'None'}",
        f"  expert_cls       = {profile.expert_module_cls.__name__ if profile.expert_module_cls else 'None'}",
        f"  expert_modules   = {len(profile.expert_modules)}",
        f"  router_modules   = {len(profile.router_module_names)}",
        f"  has_aux_loss     = {profile.has_aux_loss}",
        f"  lora_targets     = {profile.lora_targets}",
        f"  model_dtype      = {profile.model_dtype}",
        f"  hidden_layers    = {profile.num_hidden_layers}",
    ]
    logger.info("ModelProfile:\n" + "\n".join(lines))
