# Bhaskera Code Flow and Options

This document maps the executable entry points in `pyproject.toml`, their downstream
code paths, and the configuration and registry choices currently implemented in the
repository. Paths are relative to the repository root.

## Entry-Point Tree

```text
Bhaskera
├── bash setup.sh
│   ├── detect CUDA: override → CUDA_HOME → nvcc → nvidia-smi → Spack → SLURM probe
│   ├── create Python 3.11 environment with uv
│   ├── install matching PyTorch wheel and editable package
│   ├── optionally build flash-attn
│   └── generate bhaskera-activate.sh
├── bhaskera-tokenize → bhaskera.launcher.tokenize:main
│   ├── load YAML → apply CLI overrides → resolve split(s)
│   ├── prefetch Hugging Face tokenizer → initialize local Ray
│   ├── RAW_REGISTRY[data.name] → raw Ray Dataset
│   ├── optional format renderer → tokenize/pack → Parquet cache
│   └── print tokenized_path / val_tokenized_path YAML snippet
├── bhaskera-train → bhaskera.launcher.train:main
│   ├── load YAML → load optimizer plugins → configure monitoring
│   ├── connect to RAY_ADDRESS or start a local Ray cluster
│   ├── REGISTRY[data.name] → cached/tokenized training Dataset
│   ├── optional validation Dataset
│   └── Ray TorchTrainer → one worker_fn per GPU
│       ├── rebuild Config → load plugins → seed rank
│       ├── build_model
│       │   ├── registered loader or Hugging Face AutoModelForCausalLM
│       │   ├── quantization: none | qlora (QLoRA requires DDP)
│       │   ├── introspect model → ModelProfile
│       │   ├── optional Liger kernels
│       │   └── optional LoRA
│       ├── wrap_model: fsdp | ddp
│       ├── logger fan-out: ray | mlflow | wandb | disabled
│       └── trainer.train
│           ├── optimizer: default AdamW | torch.optim class | plugin
│           ├── checkpoint resume → epoch/micro-batch loop
│           ├── forward → optional MoE aux loss → backward → grad sync/clip
│           ├── optimizer step → warmup/cosine scheduler → metrics
│           ├── optional validation/benchmarks with optimizer offload
│           └── DCP checkpoint save/prune → finish loggers
├── bhaskera-infer → bhaskera.launcher.infer:main
│   ├── prompt or prompt file → YAML/default Config → CLI overrides
│   ├── InferenceEngine.load
│   │   ├── vLLM on CUDA when installed and speculation is off
│   │   └── otherwise Hugging Face generate
│   │       ├── KV cache: static | turboquant | none
│   │       └── optional speculative decoder / torch.compile
│   ├── generate → decode → optionally hide thinking block
│   └── print throughput/cache stats → optional output file
├── bhaskera-diag → bhaskera.launcher.diagnostics:main
│   └── Ray TorchTrainer → per-GPU bf16 all-reduce and NCCL bandwidth test
├── bhaskera-dashboard → bhaskera.launcher.dashboard:main
│   ├── start → detached MLflow UI and saved PID/config
│   ├── stop → terminate saved PID
│   ├── status → report process/store/port
│   └── tunnel → print ssh -L command
└── sbatch scripts/submit.sh [training options]
    ├── activate environment → tune NCCL for IB/RoCE/TCP
    ├── derive Ray head address and worker count from SLURM
    └── ray symmetric-run → python -m bhaskera.launcher.train
```

Installed commands can also be run as modules, for example
`python -m bhaskera.launcher.train`.

## Command-Line Options

### `bhaskera-train`

| Option | Default | Effect |
| --- | --- | --- |
| `--config PATH` | required | YAML configuration file. |
| `--num-workers N` | detected GPUs | One Ray training worker per GPU. |
| `--max-failures N` | `2` | Ray Train restart tolerance. |
| `--storage-path PATH` | `checkpoint.save_dir` | Ray Train result/checkpoint root. |
| `--no-dashboard` | false | Disable Ray Dashboard and its automatic Ray logger. |
| `--dashboard-port PORT` | YAML / `8265` | Override the Ray Dashboard port. |

`RAY_ADDRESS` selects an existing cluster. Without it, the launcher stops stale local
Ray state and creates a local cluster. `SLURM_NNODES` and `SLURM_GPUS_PER_NODE` override
local GPU counting when both are present.

### `bhaskera-tokenize`

| Option | Default | Effect |
| --- | --- | --- |
| `--config PATH` | required | YAML configuration file. |
| `--dataset NAME` | `data.name` | Dataset registry key. |
| `--storage-path PATH` | `data.cache_dir` | Parquet cache root; one must be set. |
| `--overwrite` | false | Ignore an existing valid cache. |
| `--num-workers N` | `data.num_workers` | Ray CPU tokenizer workers. |
| `--split {train,val,both,none}` | `train` | Split handling; only `local` supports separate train/val. |
| `--format NAME` | `data.format` | `chatml`, `alpaca`, `sharegpt`, or a registered renderer. |
| `--train-path PATH` | `data.train_path` | Local training file, directory, or glob. |
| `--val-path PATH` | `data.val_path` | Local validation file, directory, or glob. |

### `bhaskera-infer`

Exactly one of `--prompt TEXT` and `--prompt-file FILE` is required.

| Group | Options |
| --- | --- |
| Model/config | `--config PATH`, `--model ID`, `--device auto\|cuda\|cpu\|mps` |
| Generation | `--max-new-tokens N`, `--temperature F`, `--top-p F`, `--top-k N`, `--no-sample` |
| KV cache | `--kv-cache static\|turboquant\|none`, `--key-bits N`, `--value-bits N`, `--residual-window N` |
| Speculation | `--speculative`, `--draft-model ID`, `--num-draft-tokens N` |
| Rendering | `--show-thinking`, `--system-prompt TEXT`, `--return-full` |
| Output/runtime | `--output-file FILE`, `--torch-compile`, `-v` / `--verbose` |

CLI generation values override YAML. `BHASKERA_BACKEND=hf` disables automatic vLLM
selection. Note: `--device` defaults to `auto`, so CLI assembly currently replaces
`inference.device` from YAML even when the flag is omitted. `--system-prompt` is parsed
but the CLI currently calls generic `generate()`, so it does not affect that path.

### `bhaskera-diag`

`--num-workers N` defaults to `torch.cuda.device_count()`. Each worker gets one GPU.

### `bhaskera-dashboard`

Usage: `bhaskera-dashboard [start|stop|status|tunnel] [options]`; action defaults to
`start`.

| Option | Default | Effect |
| --- | --- | --- |
| `--store URI` | `~/mlflow-runs` | File-backed MLflow store. |
| `--port N` | `5000` | UI and SSH tunnel port. |
| `--login-node HOST` | current hostname | Tunnel destination shown to the user. |
| `--user USER` | omitted | SSH username in the tunnel command. |
| `--no-save` | false | Do not save start settings under `~/.bhaskera/`. |
| `-v`, `--verbose` | false | Debug logging. |

`MLFLOW_TRACKING_URI=file://...` and `MLFLOW_PORT` are environment overrides; explicit
CLI values win. State is persisted in `~/.bhaskera/mlflow-ui.json`.

### Shell Entrypoints

- `setup.sh` takes no flags. Environment overrides are `BHASKERA_CUDA` (for example
  `12.4`) and `BHASKERA_VENV` (environment path).
- `scripts/submit.sh` forwards every argument to the training parser. Its checked-in
  SBATCH defaults are 5 nodes, 2 GPUs/node, 16 CPUs/task, `gpu` partition, and 16 hours.
  Edit SBATCH directives and `NUM_GPUS` together when adapting the cluster layout.

## YAML Configuration Options

Fields not supplied use these defaults from `src/bhaskera/config.py`.

### Model, Adapters, and MoE

| Path | Default / choices |
| --- | --- |
| `model.name` | `tiiuae/falcon-7b`; Hugging Face ID or registered model key |
| `model.dtype` | `bfloat16`; `float32`, `float16`, `bfloat16`, or `auto` where supported |
| `model.attn_impl` | `null`; forwarded as HF `attn_implementation` |
| `model.trust_remote_code` | `false` |
| `model.use_liger_kernel` | `true` |
| `model.quantization` | `none`; `qlora` is implemented and requires DDP/BitsAndBytes |
| `lora.enabled` | `false` |
| `lora.r`, `lora.alpha`, `lora.dropout` | `16`, `32`, `0.05` |
| `lora.target_modules` | `[auto]`; explicit module-name list also accepted |
| `lora.include_experts` | `false` |
| `lora.freeze_router` | `true` |
| `lora.modules_to_save` | `[]` |
| `moe.aux_loss_weight` | `0.01` |
| `moe.router_z_loss_weight` | `0.001` |
| `moe.freeze_router` | `true` |
| `moe.log_expert_utilization` | `true` |
| `moe.log_every_n_steps` | `10` |

### Data and Tokenization

| Path | Default / choices |
| --- | --- |
| `data.name` | `ultrachat`; built-ins listed under Registries below |
| `data.seq_len` | `2048` |
| `data.num_workers` | `4` |
| `data.is_cpt` | `false`; enable continual-pretraining packing |
| `data.tokenized_path` | `null`; training cache path |
| `data.val_tokenized_path` | `null`; validation cache path |
| `data.cache_dir` | `null`; required by tokenization CLI |
| `data.overwrite_cache` | `false` |
| `data.tokenize_batch_size` | `128` |
| `data.tokenize_compression` | `snappy`; `snappy`, `zstd`, or `none` |
| `data.prefetch_batches` | `2` |
| `data.local_shuffle_buffer_multiplier` | `10` |
| `data.pack_sequences` | `false`; compatibility alias for continuous packing |
| `data.format` | `null`; `chatml`, `alpaca`, `sharegpt`, or custom |
| `data.format_options` | `{}`; renderer-specific mapping |
| `data.path` | `null`; shorthand local training source |
| `data.train_path`, `data.val_path` | `null`; local split sources |

### Training and Distribution

| Path | Default / choices |
| --- | --- |
| `training.batch_size` | `2` per worker |
| `training.grad_accum` | `4` micro-batches per optimizer step |
| `training.lr`, `training.weight_decay` | `2e-4`, `0.01` |
| `training.max_steps`, `training.num_epochs` | `1000`, `1`; first limit reached stops training |
| `training.warmup_steps` | `100`; followed by cosine decay |
| `training.max_grad_norm`, `training.grad_clip` | `1.0`, `1.0` |
| `training.max_grad_skip_steps` | `100` |
| `training.seed`, `training.deterministic` | `42`, `false` |
| `training.optimizer.backend` | `default`; `default`, `torch`, or `plugin` |
| `training.optimizer.class_name` | `null`; a `torch.optim` class for `torch` backend |
| `training.optimizer.name` | `null`; registry key for `plugin` backend |
| `training.optimizer.kwargs` | `{}`; passed to selected optimizer |
| `training.distributed.strategy` | `fsdp`; `fsdp` or `ddp` |
| `training.distributed.fsdp.sharding_strategy` | `FULL_SHARD` |
| `training.distributed.fsdp.transformer_layer_cls` | `[]`; introspection is preferred |
| `training.distributed.fsdp.param_dtype` | `bfloat16` |
| `training.distributed.fsdp.reduce_dtype` | `bfloat16` |
| `training.distributed.fsdp.buffer_dtype` | `bfloat16` |
| `training.distributed.fsdp.activation_checkpointing` | `true` |
| `training.distributed.fsdp.cpu_offload` | `false` |
| `training.distributed.fsdp.shard_experts_individually` | `true` |
| `training.distributed.ddp.find_unused_parameters` | `false`; MoE may force it on |
| `training.distributed.ddp.gradient_as_bucket_view` | `true` |
| `training.distributed.ddp.broadcast_buffers` | `false` |
| `training.distributed.ddp.activation_checkpointing` | `false` |
| `training.distributed.ddp.static_graph` | `false`; incompatible paths may force it off |

### Checkpointing, Logging, and Monitoring

| Path | Default / choices |
| --- | --- |
| `checkpoint.enabled` | `true` |
| `checkpoint.save_dir` | `./checkpoints` |
| `checkpoint.save_interval` | `1` |
| `checkpoint.keep_last_n` | `2` |
| `logging.tracker` | `null`; string/list of `ray`, `mlflow`, `wandb`, or off aliases `none`, `off`, `false`, `""` |
| `logging.project`, `logging.run_name` | `bhaskera`, `run` |
| `logging.mlflow_tracking_uri` | `null` |
| `logging.log_gpu_every_n_steps` | `10` |
| `logging.tags`, `logging.group` | `[]`, `null` |
| `monitoring.dashboard` | `true`; automatically adds the Ray logger |
| `monitoring.dashboard_host`, `monitoring.dashboard_port` | `0.0.0.0`, `8265` |
| `monitoring.metrics_export_port` | `8080` |
| `monitoring.metrics.enabled` | `true` |
| `monitoring.metrics.system_every_n_steps` | `10` |
| `monitoring.metrics.cuda_every_n_steps` | `10` |
| `monitoring.metrics.gpu`, `.cpu`, `.cuda_memory`, `.throughput` | all `true` |
| `monitoring.metrics.peak_tflops_per_gpu` | `312.0` |
| `monitoring.metrics.throughput_window`, `.throughput_warmup` | `50`, `5` |

### Evaluation and Inference

| Path | Default / choices |
| --- | --- |
| `evaluation.enabled` | `false` |
| `evaluation.offload_optimizer` | declared default `true`; YAML loader currently does not apply an override |
| `evaluation.validation.dataset` | `validation` |
| `evaluation.validation.every_n_steps`, `.every_n_epochs` | `500`, `1` |
| `evaluation.validation.metrics` | `[loss, perplexity]` |
| `evaluation.benchmarks.every_n_steps`, `.every_n_epochs` | `2000`, `1` |
| `evaluation.benchmarks.tasks` | `[]` |
| `inference.max_new_tokens` | `512` |
| `inference.temperature`, `.top_p`, `.top_k` | `1.0`, `0.9`, `50` |
| `inference.do_sample`, `.batch_size` | `true`, `1` |
| `inference.kv_cache` | `static`; `static`, `turboquant`, or `none` |
| `inference.device` | `auto` |
| `inference.torch_compile` | `false` |
| `inference.turboquant.enabled` | `false`; selecting `turboquant` via CLI enables it |
| `inference.turboquant.key_bits`, `.value_bits` | `4`, `2` |
| `inference.turboquant.residual_window`, `.protected_layers` | `128`, `2` |
| `inference.speculative.enabled` | `false` |
| `inference.speculative.draft_model_name` | empty string |
| `inference.speculative.num_draft_tokens` | `5` |
| `plugins.optimizers` | `[]`; importable module paths loaded in driver and workers |

### Declared but Currently Inactive Fields

The loader accepts several compatibility/future-facing fields that do not presently
change the execution path. `training.distributed.fsdp.sharding_strategy` and
`.cpu_offload`, `moe.router_z_loss_weight`, `moe.freeze_router`, and
`training.max_grad_skip_steps` are stored but never consumed outside configuration.
Likewise, `monitoring.metrics.gpu` and `.cpu` are not checked individually, and
`inference.turboquant.enabled` is informational—the active cache is selected by
`inference.kv_cache`. Treat these as non-operational until their consumers are wired.
`evaluation.offload_optimizer` is operational at runtime, but `_dict_to_config()` omits
the YAML value and therefore always leaves its default `true`.

## Built-In Registries and Extension Options

```text
Datasets
├── ultrachat       (HF UltraChat; raw text column: prompt)
├── openassistant   (HF OpenAssistant Guanaco; text)
├── redpajama       (HF RedPajama sample; text)
├── local           (JSONL/JSON/Parquet train and validation files)
└── local_cpt       (local files with continual sequence packing)

Formats
├── chatml          options: messages_field
├── alpaca          option: use_chat_template
└── sharegpt        options: conversations_field, role_map

Optimizer plugins
├── lion            module: bhaskera.plugins.optimizers.lion
└── galore          module: bhaskera.plugins.optimizers.galore

Validation metrics
├── loss
├── perplexity
└── token_accuracy

Benchmarks
├── hellaswag
├── arc_easy
└── arc_challenge
```

The `torch` optimizer backend also accepts any optimizer class exposed by
`torch.optim`. Custom integrations use `@register`, `@register_raw`,
`@register_format`, `@register_optimizer`, `@register_metric`,
`@register_benchmark`, or `@register_model`. Dataset modules must be imported from
`src/bhaskera/data/datasets/__init__.py`; optimizer plugin modules must be listed in
`plugins.optimizers` so registration runs in both the Ray driver and workers.

## Example Config Catalog

`configs/` contains ready-made paths through the tree:

- Training/distribution: `config.yaml`, `config_falcon.yaml`, `2node.yaml`, `3gpu.yaml`,
  `qwen.yaml`, `qwen_ddp.yaml`, `qwen_hybrid_shard.yaml`, `test_param.yaml`,
  `finetune_param_local_data.yaml`, and `liger_with_dashboard.yaml`.
- QLoRA/CPT/evaluation/optimizers: `param_qlora.yaml`, `pre.yaml`, `fsdp_cpt.yaml`,
  `eval.yaml`, `adadelta.yaml`, `lion.yaml`, and `galore.yaml`.
- Tokenization: `tokenize.yaml`, `tokenize_qwen.yaml`, `token_falcon.yaml`,
  `cpt_token.yaml`, and `local_data_tokenize_chatml.yaml`.
- Inference: `inference_param2.yaml`, `inference_turboquant.yaml`, and
  `inference_speculative.yaml`.
