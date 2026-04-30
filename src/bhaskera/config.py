"""
Bhaskera config — single source of truth.

Changes vs v2:
  * ``logging.tracker`` may now be a list:
        tracker: ["wandb", "ray"]    # both
        tracker: "wandb"              # legacy single-string still OK
        tracker: null                  # → Ray Dashboard only (default)
  * ``logging.tags`` and ``logging.group`` for cohort filtering in
    W&B / MLflow.
  * NEW ``monitoring`` block — Ray Dashboard, Prometheus, Grafana,
    and per-step custom-metric toggles.  See ``MonitoringConfig``.
  * MFU / throughput config: ``monitoring.peak_tflops_per_gpu`` plus
    ``monitoring.metrics_every_n_steps``.

The ``tracker`` field type is widened to ``Optional[Any]`` because
YAML can deliver either a string, a list, or null and we want to
defer validation to the loggers package (where the error message is
informative).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str = "tiiuae/falcon-7b"
    dtype: str = "bfloat16"
    attn_impl: Optional[str] = None
    trust_remote_code: bool = False
    use_liger_kernel: bool = True


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["auto"])
    include_experts: bool = False
    freeze_router: bool = True
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class MoEConfig:
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    freeze_router: bool = True
    log_expert_utilization: bool = True
    log_every_n_steps: int = 10


@dataclass
class TurboQuantConfig:
    enabled: bool = False
    key_bits: int = 4
    value_bits: int = 2
    residual_window: int = 128
    protected_layers: int = 2


@dataclass
class SpeculativeConfig:
    enabled: bool = False
    draft_model_name: str = ""
    num_draft_tokens: int = 5


@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    batch_size: int = 1
    kv_cache: str = "static"
    device: str = "auto"
    torch_compile: bool = False
    turboquant: TurboQuantConfig = field(default_factory=TurboQuantConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)


@dataclass
class DataConfig:
    name: str = "ultrachat"
    seq_len: int = 2048
    num_workers: int = 4


@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"
    transformer_layer_cls: list[str] = field(default_factory=list)
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "bfloat16"
    buffer_dtype: str = "bfloat16"
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    shard_experts_individually: bool = True


@dataclass
class DDPConfig:
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = False


@dataclass
class DistributedConfig:
    strategy: str = "fsdp"
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
    seed: int = 42
    deterministic: bool = False
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_dir: str = "./checkpoints"
    save_interval: int = 1
    keep_last_n: int = 2


@dataclass
class LoggingConfig:
    # str | list[str] | None — see _normalize_trackers in loggers/__init__
    tracker: Optional[Union[str, list]] = None
    project: str = "bhaskera"
    run_name: str = "run"
    mlflow_tracking_uri: Optional[str] = None
    log_gpu_every_n_steps: int = 10
    tags: list[str] = field(default_factory=list)
    group: Optional[str] = None


# ── NEW: monitoring sub-configs ─────────────────────────────────────

@dataclass
class PrometheusConfig:
    """How Ray should reach a Prometheus server (for embedded panels)."""
    enabled: bool = False
    host: str = "http://localhost:9090"      # → RAY_PROMETHEUS_HOST
    name: str = "Prometheus"                 # → RAY_PROMETHEUS_NAME
    # Authentication headers (rare, but supported by Ray).  The dict
    # is JSON-encoded into RAY_PROMETHEUS_HEADERS.
    headers: dict = field(default_factory=dict)


@dataclass
class GrafanaConfig:
    """How Ray should embed Grafana panels into the dashboard."""
    enabled: bool = False
    host: str = "http://localhost:3000"      # → RAY_GRAFANA_HOST
    # Optional public-facing URL the user's browser uses for the
    # iframe (cluster-internal vs external addresses often differ).
    iframe_host: Optional[str] = None        # → RAY_GRAFANA_IFRAME_HOST
    org_id: str = "1"                        # → RAY_GRAFANA_ORG_ID


@dataclass
class MetricsConfig:
    """Per-step custom-metric toggles."""
    enabled: bool = True
    # How often (in optimizer steps) to push system stats from every rank.
    system_every_n_steps: int = 10
    # How often (in optimizer steps) to push CUDA-allocator stats.
    cuda_every_n_steps: int = 10
    # Sub-collectors
    gpu: bool = True
    cpu: bool = True
    cuda_memory: bool = True
    throughput: bool = True
    # Throughput / MFU
    peak_tflops_per_gpu: float = 312.0       # A100 bf16; override per cluster.
    throughput_window: int = 50              # steps to average for tokens/sec EMA
    throughput_warmup: int = 5               # drop these many steps from EMA


@dataclass
class MonitoringConfig:
    """
    Ray Dashboard / Prometheus / Grafana / per-step metrics.

    Default-constructed values give you a working Ray Dashboard with
    no external services — exactly the "tracker is none → Ray
    Dashboard by default" behaviour the user asked for.
    """
    dashboard: bool                = True
    dashboard_host: str            = "0.0.0.0"
    dashboard_port: int            = 8265
    metrics_export_port: int       = 8080
    prometheus: PrometheusConfig   = field(default_factory=PrometheusConfig)
    grafana: GrafanaConfig         = field(default_factory=GrafanaConfig)
    metrics: MetricsConfig         = field(default_factory=MetricsConfig)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig            = field(default_factory=ModelConfig)
    data: DataConfig              = field(default_factory=DataConfig)
    lora: LoraConfig              = field(default_factory=LoraConfig)
    moe: MoEConfig                = field(default_factory=MoEConfig)
    training: TrainingConfig      = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig  = field(default_factory=CheckpointConfig)
    logging: LoggingConfig        = field(default_factory=LoggingConfig)
    inference: InferenceConfig    = field(default_factory=InferenceConfig)
    monitoring: MonitoringConfig  = field(default_factory=MonitoringConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _dict_to_config(d or {})


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _get(d: Any, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def _dict_to_config(raw: dict) -> Config:
    fsdp_raw    = _get(raw, "training", "distributed", "fsdp", default={}) or {}
    ddp_raw     = _get(raw, "training", "distributed", "ddp",  default={}) or {}
    dist_raw    = _get(raw, "training", "distributed", default={}) or {}
    train_raw   = _get(raw, "training", default={}) or {}
    log_raw     = _get(raw, "logging",  default={}) or {}
    ckpt_raw    = _get(raw, "checkpoint", default={}) or {}
    model_raw   = _get(raw, "model",    default={}) or {}
    data_raw    = _get(raw, "data",     default={}) or {}
    lora_raw    = _get(raw, "lora",     default={}) or {}
    moe_raw     = _get(raw, "moe",      default={}) or {}
    infer_raw   = _get(raw, "inference", default={}) or {}
    tq_raw      = _get(raw, "inference", "turboquant", default={}) or {}
    spec_raw    = _get(raw, "inference", "speculative", default={}) or {}

    mon_raw     = _get(raw, "monitoring", default={}) or {}
    prom_raw    = _get(raw, "monitoring", "prometheus", default={}) or {}
    graf_raw    = _get(raw, "monitoring", "grafana",    default={}) or {}
    metrics_raw = _get(raw, "monitoring", "metrics",    default={}) or {}

    return Config(
        model=ModelConfig(
            name=model_raw.get("name", "tiiuae/falcon-7b"),
            dtype=model_raw.get("dtype", "bfloat16"),
            attn_impl=model_raw.get("attn_impl"),
            trust_remote_code=model_raw.get("trust_remote_code", False),
            use_liger_kernel=bool(model_raw.get("use_liger_kernel", True)),
        ),
        data=DataConfig(
            name=data_raw.get("name", "ultrachat"),
            seq_len=int(data_raw.get("seq_len", 2048)),
            num_workers=int(data_raw.get("num_workers", 4)),
        ),
        lora=LoraConfig(
            enabled=bool(lora_raw.get("enabled", False)),
            r=int(lora_raw.get("r", 16)),
            alpha=int(lora_raw.get("alpha", 32)),
            dropout=float(lora_raw.get("dropout", 0.05)),
            target_modules=list(lora_raw.get("target_modules", ["auto"])),
            include_experts=bool(lora_raw.get("include_experts", False)),
            freeze_router=bool(lora_raw.get("freeze_router", True)),
            modules_to_save=list(lora_raw.get("modules_to_save", [])),
        ),
        moe=MoEConfig(
            aux_loss_weight=float(moe_raw.get("aux_loss_weight", 0.01)),
            router_z_loss_weight=float(moe_raw.get("router_z_loss_weight", 0.001)),
            freeze_router=bool(moe_raw.get("freeze_router", True)),
            log_expert_utilization=bool(moe_raw.get("log_expert_utilization", True)),
            log_every_n_steps=int(moe_raw.get("log_every_n_steps", 10)),
        ),
        training=TrainingConfig(
            batch_size=int(train_raw.get("batch_size", 2)),
            grad_accum=int(train_raw.get("grad_accum", 4)),
            lr=float(train_raw.get("lr", 2e-4)),
            weight_decay=float(train_raw.get("weight_decay", 0.01)),
            max_steps=int(train_raw.get("max_steps", 1000)),
            num_epochs=int(train_raw.get("num_epochs", 1)),
            warmup_steps=int(train_raw.get("warmup_steps", 100)),
            max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
            seed=int(train_raw.get("seed", 42)),
            deterministic=bool(train_raw.get("deterministic", False)),
            distributed=DistributedConfig(
                strategy=str(dist_raw.get("strategy", "fsdp")),
                fsdp=FSDPConfig(
                    sharding_strategy=fsdp_raw.get("sharding_strategy", "FULL_SHARD"),
                    transformer_layer_cls=list(fsdp_raw.get("transformer_layer_cls", [])),
                    param_dtype=fsdp_raw.get("param_dtype", "bfloat16"),
                    reduce_dtype=fsdp_raw.get("reduce_dtype", "bfloat16"),
                    buffer_dtype=fsdp_raw.get("buffer_dtype", "bfloat16"),
                    activation_checkpointing=bool(fsdp_raw.get("activation_checkpointing", True)),
                    cpu_offload=bool(fsdp_raw.get("cpu_offload", False)),
                    shard_experts_individually=bool(
                        fsdp_raw.get("shard_experts_individually", True)
                    ),
                ),
                ddp=DDPConfig(
                    find_unused_parameters=bool(ddp_raw.get("find_unused_parameters", False)),
                    gradient_as_bucket_view=bool(ddp_raw.get("gradient_as_bucket_view", True)),
                    broadcast_buffers=bool(ddp_raw.get("broadcast_buffers", False)),
                ),
            ),
        ),
        checkpoint=CheckpointConfig(
            enabled=bool(ckpt_raw.get("enabled", True)),
            save_dir=str(ckpt_raw.get("save_dir", "./checkpoints")),
            save_interval=int(ckpt_raw.get("save_interval", 1)),
            keep_last_n=int(ckpt_raw.get("keep_last_n", 2)),
        ),
        logging=LoggingConfig(
            tracker=log_raw.get("tracker"),
            project=str(log_raw.get("project", "bhaskera")),
            run_name=str(log_raw.get("run_name", "run")),
            mlflow_tracking_uri=log_raw.get("mlflow_tracking_uri"),
            log_gpu_every_n_steps=int(log_raw.get("log_gpu_every_n_steps", 10)),
            tags=list(log_raw.get("tags", []) or []),
            group=log_raw.get("group"),
        ),
        inference=InferenceConfig(
            max_new_tokens=int(infer_raw.get("max_new_tokens", 512)),
            temperature=float(infer_raw.get("temperature", 1.0)),
            top_p=float(infer_raw.get("top_p", 0.9)),
            top_k=int(infer_raw.get("top_k", 50)),
            do_sample=bool(infer_raw.get("do_sample", True)),
            batch_size=int(infer_raw.get("batch_size", 1)),
            kv_cache=str(infer_raw.get("kv_cache", "static")),
            device=str(infer_raw.get("device", "auto")),
            torch_compile=bool(infer_raw.get("torch_compile", False)),
            turboquant=TurboQuantConfig(
                enabled=bool(tq_raw.get("enabled", False)),
                key_bits=int(tq_raw.get("key_bits", 4)),
                value_bits=int(tq_raw.get("value_bits", 2)),
                residual_window=int(tq_raw.get("residual_window", 128)),
                protected_layers=int(tq_raw.get("protected_layers", 2)),
            ),
            speculative=SpeculativeConfig(
                enabled=bool(spec_raw.get("enabled", False)),
                draft_model_name=str(spec_raw.get("draft_model_name", "")),
                num_draft_tokens=int(spec_raw.get("num_draft_tokens", 5)),
            ),
        ),
        monitoring=MonitoringConfig(
            dashboard=bool(mon_raw.get("dashboard", True)),
            dashboard_host=str(mon_raw.get("dashboard_host", "0.0.0.0")),
            dashboard_port=int(mon_raw.get("dashboard_port", 8265)),
            metrics_export_port=int(mon_raw.get("metrics_export_port", 8080)),
            prometheus=PrometheusConfig(
                enabled=bool(prom_raw.get("enabled", False)),
                host=str(prom_raw.get("host", "http://localhost:9090")),
                name=str(prom_raw.get("name", "Prometheus")),
                headers=dict(prom_raw.get("headers", {}) or {}),
            ),
            grafana=GrafanaConfig(
                enabled=bool(graf_raw.get("enabled", False)),
                host=str(graf_raw.get("host", "http://localhost:3000")),
                iframe_host=graf_raw.get("iframe_host"),
                org_id=str(graf_raw.get("org_id", "1") or "1"),
            ),
            metrics=MetricsConfig(
                enabled=bool(metrics_raw.get("enabled", True)),
                system_every_n_steps=int(metrics_raw.get("system_every_n_steps", 10)),
                cuda_every_n_steps=int(metrics_raw.get("cuda_every_n_steps", 10)),
                gpu=bool(metrics_raw.get("gpu", True)),
                cpu=bool(metrics_raw.get("cpu", True)),
                cuda_memory=bool(metrics_raw.get("cuda_memory", True)),
                throughput=bool(metrics_raw.get("throughput", True)),
                peak_tflops_per_gpu=float(metrics_raw.get("peak_tflops_per_gpu", 312.0)),
                throughput_window=int(metrics_raw.get("throughput_window", 50)),
                throughput_warmup=int(metrics_raw.get("throughput_warmup", 5)),
            ),
        ),
    )


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)
