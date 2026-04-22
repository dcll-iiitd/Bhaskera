"""
Bhaskera config — single source of truth.
All fields have sane defaults; override via YAML.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str = "tiiuae/falcon-7b"
    dtype: str = "bfloat16"                  # bfloat16 | float16 | float32 | auto
    attn_impl: Optional[str] = None          # flash_attention_2 | None
    trust_remote_code: bool = False           # required for custom-code models (Param2, etc.)


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # "auto" → introspect.py discovers targets; explicit list → use as-is
    target_modules: list[str] = field(default_factory=lambda: ["auto"])
    include_experts: bool = False             # also LoRA expert FFN layers in MoE
    freeze_router: bool = True                # freeze gate/router after LoRA
    modules_to_save: list[str] = field(default_factory=list)  # fully train embed/lm_head


@dataclass
class MoEConfig:
    """MoE-specific training params. Only used when the model is detected as MoE."""
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    freeze_router: bool = True
    log_expert_utilization: bool = True
    log_every_n_steps: int = 10


# ---------------------------------------------------------------------------
# Inference sub-configs
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantConfig:
    """TurboQuant KV-cache quantization (Google, ICLR 2026 — arXiv:2504.19874).

    Uses MSE-only V3 (community-validated): random rotation + Lloyd-Max
    quantization without QJL residual correction, which hurts in practice
    because softmax amplifies QJL variance (confirmed by 6+ independent teams).

    K4/V2 asymmetric allocation (same avg 3-bit budget as uniform) gives
    ~4.4× memory reduction at 99.5%+ attention cosine similarity.
    """
    enabled: bool = False
    key_bits: int = 4          # bits for keys — higher = better attention quality
    value_bits: int = 2        # bits for values — errors cancel naturally
    residual_window: int = 128  # last N tokens kept in fp16 (critical for gen quality)
    protected_layers: int = 2   # first + last N layers use full key_bits+2/value_bits+2


@dataclass
class SpeculativeConfig:
    """Speculative decoding (Leviathan et al. 2023). Lossless 2–3× decode speedup.

    Requires a small draft model that shares the vocabulary with the target.
    Draft generates `num_draft_tokens` speculatively; target verifies them
    in a single forward pass using rejection sampling.
    """
    enabled: bool = False
    draft_model_name: str = ""   # HuggingFace model id for the draft model
    num_draft_tokens: int = 5    # tokens generated per speculative step


@dataclass
class InferenceConfig:
    """Inference engine configuration."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    batch_size: int = 1
    kv_cache: str = "static"          # "static" | "turboquant" | "none"
    device: str = "auto"               # "auto" | "cuda" | "cpu" | "mps"
    torch_compile: bool = False        # torch.compile the model for decode speed
    turboquant: TurboQuantConfig = field(default_factory=TurboQuantConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)


@dataclass
class DataConfig:
    source: str = "registry"                  # registry | huggingface | local
    name: str = "ultrachat"                  # dataset registry key
    hf_dataset: str = ""                      # used when source=huggingface
    hf_split: str = "train"

    local_path: str = ""                      # file, dir, or glob when source=local
    local_format: str = "jsonl"               # jsonl | json | csv | parquet | text

    text_column: str = "text"                 # single text column for hf/local tabular data
    prompt_column: str = ""                   # optional prompt column
    completion_column: str = ""               # optional completion column
    prompt_completion_template: str = "{prompt}\n{completion}"

    seq_len: int = 2048
    num_workers: int = 4                     # Ray Data parallelism


@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"   # FULL_SHARD | SHARD_GRAD_OP | HYBRID_SHARD | NO_SHARD
    # Empty list → auto-detect from introspect.py; explicit names → manual override
    transformer_layer_cls: list[str] = field(default_factory=list)
    param_dtype: str = "bfloat16"            # "auto" → read from model
    reduce_dtype: str = "bfloat16"           # "auto" → float32 for safer grad reduce
    buffer_dtype: str = "bfloat16"
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    shard_experts_individually: bool = True   # MoE: per-expert FSDP sharding


@dataclass
class DDPConfig:
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = False


@dataclass
class DistributedConfig:
    strategy: str = "fsdp"                  # fsdp | ddp
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)


@dataclass
class TrainingConfig:
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    num_epochs: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_dir: str = "./checkpoints"
    save_interval: int = 1               # every N epochs
    keep_last_n: int = 2


@dataclass
class LoggingConfig:
    tracker: Optional[str] = None        # wandb | mlflow | None
    project: str = "bhaskera"
    run_name: str = "run"
    mlflow_tracking_uri: Optional[str] = None
    log_gpu_every_n_steps: int = 10


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _dict_to_config(d: dict) -> Config:
    """Build Config from a plain dict. Manual approach — simpler and zero-clutter."""
    raw = d

    def get(d, *keys, default=None):
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
            if d is default:
                return default
        return d

    fsdp_raw    = get(raw, 'training', 'distributed', 'fsdp', default={})
    ddp_raw     = get(raw, 'training', 'distributed', 'ddp', default={})
    dist_raw    = get(raw, 'training', 'distributed', default={})
    train_raw   = get(raw, 'training', default={})
    log_raw     = get(raw, 'logging', default={})
    ckpt_raw    = get(raw, 'checkpoint', default={})
    model_raw   = get(raw, 'model', default={})
    data_raw    = get(raw, 'data', default={})
    lora_raw    = get(raw, 'lora', default={})
    moe_raw     = get(raw, 'moe', default={})
    infer_raw   = get(raw, 'inference', default={})
    tq_raw      = get(raw, 'inference', 'turboquant', default={})
    spec_raw    = get(raw, 'inference', 'speculative', default={})

    data_source = data_raw.get('source')
    if not data_source:
        if data_raw.get('local_path'):
            data_source = 'local'
        elif data_raw.get('hf_dataset'):
            data_source = 'huggingface'
        else:
            data_source = 'registry'

    return Config(
        model=ModelConfig(
            name=model_raw.get('name', 'tiiuae/falcon-7b'),
            dtype=model_raw.get('dtype', 'bfloat16'),
            attn_impl=model_raw.get('attn_impl'),
            trust_remote_code=model_raw.get('trust_remote_code', False),
        ),
        data=DataConfig(
            source=data_source,
            name=data_raw.get('name', 'ultrachat'),
            hf_dataset=data_raw.get('hf_dataset', ''),
            hf_split=data_raw.get('hf_split', 'train'),
            local_path=data_raw.get('local_path', ''),
            local_format=data_raw.get('local_format', 'jsonl'),
            text_column=data_raw.get('text_column', 'text'),
            prompt_column=data_raw.get('prompt_column', ''),
            completion_column=data_raw.get('completion_column', ''),
            prompt_completion_template=data_raw.get(
                'prompt_completion_template', '{prompt}\n{completion}'
            ),
            seq_len=data_raw.get('seq_len', 2048),
            num_workers=data_raw.get('num_workers', 4),
        ),
        lora=LoraConfig(
            enabled=lora_raw.get('enabled', False),
            r=lora_raw.get('r', 16),
            alpha=lora_raw.get('alpha', 32),
            dropout=lora_raw.get('dropout', 0.05),
            target_modules=lora_raw.get('target_modules', ['auto']),
            include_experts=lora_raw.get('include_experts', False),
            freeze_router=lora_raw.get('freeze_router', True),
            modules_to_save=lora_raw.get('modules_to_save', []),
        ),
        moe=MoEConfig(
            aux_loss_weight=float(moe_raw.get('aux_loss_weight', 0.01)),
            router_z_loss_weight=float(moe_raw.get('router_z_loss_weight', 0.001)),
            freeze_router=moe_raw.get('freeze_router', True),
            log_expert_utilization=moe_raw.get('log_expert_utilization', True),
            log_every_n_steps=moe_raw.get('log_every_n_steps', 10),
        ),
        training=TrainingConfig(
            batch_size=train_raw.get('batch_size', 2),
            grad_accum=train_raw.get('grad_accum', 4),
            lr=float(train_raw.get('lr', 2e-4)),
            weight_decay=float(train_raw.get('weight_decay', 0.01)),
            max_steps=train_raw.get('max_steps', 1000),
            num_epochs=train_raw.get('num_epochs', 1),
            warmup_steps=train_raw.get('warmup_steps', 100),
            max_grad_norm=float(train_raw.get('max_grad_norm', 1.0)),
            distributed=DistributedConfig(
                strategy=dist_raw.get('strategy', 'fsdp'),
                fsdp=FSDPConfig(
                    sharding_strategy=fsdp_raw.get('sharding_strategy', 'FULL_SHARD'),
                    transformer_layer_cls=fsdp_raw.get('transformer_layer_cls', []),
                    param_dtype=fsdp_raw.get('param_dtype', 'bfloat16'),
                    reduce_dtype=fsdp_raw.get('reduce_dtype', 'bfloat16'),
                    buffer_dtype=fsdp_raw.get('buffer_dtype', 'bfloat16'),
                    activation_checkpointing=fsdp_raw.get('activation_checkpointing', True),
                    cpu_offload=fsdp_raw.get('cpu_offload', False),
                    shard_experts_individually=fsdp_raw.get('shard_experts_individually', True),
                ),
                ddp=DDPConfig(
                    find_unused_parameters=ddp_raw.get('find_unused_parameters', False),
                    gradient_as_bucket_view=ddp_raw.get('gradient_as_bucket_view', True),
                    broadcast_buffers=ddp_raw.get('broadcast_buffers', False),
                ),
            ),
        ),
        checkpoint=CheckpointConfig(
            enabled=ckpt_raw.get('enabled', True),
            save_dir=ckpt_raw.get('save_dir', './checkpoints'),
            save_interval=ckpt_raw.get('save_interval', 1),
            keep_last_n=ckpt_raw.get('keep_last_n', 2),
        ),
        logging=LoggingConfig(
            tracker=log_raw.get('tracker'),
            project=log_raw.get('project', 'bhaskera'),
            run_name=log_raw.get('run_name', 'run'),
            mlflow_tracking_uri=log_raw.get('mlflow_tracking_uri'),
            log_gpu_every_n_steps=log_raw.get('log_gpu_every_n_steps', 10),
        ),
        inference=InferenceConfig(
            max_new_tokens=infer_raw.get('max_new_tokens', 512),
            temperature=float(infer_raw.get('temperature', 1.0)),
            top_p=float(infer_raw.get('top_p', 0.9)),
            top_k=int(infer_raw.get('top_k', 50)),
            do_sample=infer_raw.get('do_sample', True),
            batch_size=int(infer_raw.get('batch_size', 1)),
            kv_cache=infer_raw.get('kv_cache', 'static'),
            device=infer_raw.get('device', 'auto'),
            torch_compile=infer_raw.get('torch_compile', False),
            turboquant=TurboQuantConfig(
                enabled=tq_raw.get('enabled', False),
                key_bits=int(tq_raw.get('key_bits', 4)),
                value_bits=int(tq_raw.get('value_bits', 2)),
                residual_window=int(tq_raw.get('residual_window', 128)),
                protected_layers=int(tq_raw.get('protected_layers', 2)),
            ),
            speculative=SpeculativeConfig(
                enabled=spec_raw.get('enabled', False),
                draft_model_name=spec_raw.get('draft_model_name', ''),
                num_draft_tokens=int(spec_raw.get('num_draft_tokens', 5)),
            ),
        ),
    )


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)
