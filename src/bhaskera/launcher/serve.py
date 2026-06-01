"""
bhaskera-serve
==============
CLI entry point for the Bhaskera LLM serving stack.

Reads a YAML config, initialises Ray, starts Ray Serve's HTTP server,
deploys ``LLMDeployment``, then blocks until the process receives
SIGINT or SIGTERM.

Usage examples::

    # Serve using config defaults (HF backend)
    bhaskera-serve --config configs/serve.yaml

    # Override backend and port at the command line
    bhaskera-serve --config configs/serve.yaml --backend vllm --port 8080

    # Connect to an existing Ray cluster
    bhaskera-serve --config configs/serve.yaml --ray-address ray://head:10001

    # Local single-node cluster (useful for development)
    bhaskera-serve --config configs/serve.yaml --ray-address local

Minimal YAML (``configs/serve.yaml``)::

    model:
      name: "meta-llama/Llama-3.1-8B-Instruct"
      dtype: "bfloat16"
      attn_impl: "flash_attention_2"

    serve:
      backend: "vllm"
      host: "0.0.0.0"
      port: 8000
      num_replicas: 1
      ray_actor_options:
        num_gpus: 1
      vllm:
        tensor_parallel_size: 1
        gpu_memory_utilization: 0.90

    inference:
      max_new_tokens: 512
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhaskera-serve",
        description=(
            "Serve a Bhaskera LLM as an OpenAI-compatible HTTP API "
            "via Ray Serve (POST /v1/chat/completions)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        metavar="PATH",
        help="Path to the YAML configuration file.",
    )
    # ── CLI overrides (all optional) ────────────────────────────────────
    p.add_argument(
        "--host",
        default=None,
        metavar="ADDR",
        help="Override cfg.serve.host (e.g. '0.0.0.0').",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Override cfg.serve.port.",
    )
    p.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default=None,
        help="Override cfg.serve.backend.",
    )
    p.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        metavar="N",
        help="Override cfg.serve.num_replicas.",
    )
    # ── Ray cluster ─────────────────────────────────────────────────────
    p.add_argument(
        "--ray-address",
        default="auto",
        metavar="ADDR",
        help=(
            "Ray cluster address.  "
            "'auto' attaches to a running local cluster.  "
            "'local' starts a new single-node cluster.  "
            "'ray://host:port' connects to a remote cluster."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── Logging setup ───────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quieten noisy third-party loggers at WARNING unless the user asked
    # for DEBUG.
    if args.log_level != "DEBUG":
        for noisy in ("ray", "ray.serve", "urllib3", "filelock", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Load config ──────────────────────────────────────────────────────
    logger.info("Loading config from %s …", args.config)
    from bhaskera.config import load_config

    cfg = load_config(args.config)

    # Apply CLI overrides (take precedence over YAML values).
    if args.host is not None:
        cfg.serve.host = args.host
    if args.port is not None:
        cfg.serve.port = args.port
    if args.backend is not None:
        cfg.serve.backend = args.backend
    if args.num_replicas is not None:
        cfg.serve.num_replicas = args.num_replicas

    _log_startup_banner(cfg)

    # ── Validate ─────────────────────────────────────────────────────────
    if cfg.serve.backend not in ("vllm", "hf"):
        logger.error(
            "cfg.serve.backend must be 'vllm' or 'hf', got %r",
            cfg.serve.backend,
        )
        sys.exit(1)

    # ── Initialise Ray ───────────────────────────────────────────────────
    import ray

    ray_address: str | None = (
        None if args.ray_address == "local" else args.ray_address
    )

    logger.info(
        "Initialising Ray | address=%s",
        ray_address or "local (new cluster)",
    )
    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        logging_level=logging.WARNING,
        # Surface runtime errors immediately rather than silently retrying.
        runtime_env={"env_vars": {"RAY_SERVE_HTTP_PROXY_TIMEOUT_S": "600"}},
    )
    logger.info("Ray resources: %s", ray.available_resources())

    # ── Start Ray Serve ──────────────────────────────────────────────────
    from ray import serve

    # ``serve.start()`` is idempotent when detached=True — safe to call
    # even if another bhaskera-serve process already initialised Serve on
    # this cluster.
    serve.start(
        detached=True,
        http_options={
            "host": cfg.serve.host,
            "port": cfg.serve.port,
        },
    )

    # ── Build and deploy the application ─────────────────────────────────
    logger.info("Building application …")
    from bhaskera.serve.app import build_app

    app = build_app(cfg)

    logger.info("Deploying to Ray Serve (this may take a minute for large models) …")
    serve.run(
        app,
        route_prefix=cfg.serve.route_prefix,
        name="bhaskera_llm",
    )

    # ── Ready ─────────────────────────────────────────────────────────────
    base_url = f"http://{cfg.serve.host}:{cfg.serve.port}"
    logger.info("=" * 60)
    logger.info("  Bhaskera LLM API is live")
    logger.info("  Chat:    %s/v1/chat/completions", base_url)
    logger.info("  Models:  %s/v1/models", base_url)
    logger.info("  Health:  %s/health", base_url)
    logger.info("  Docs:    %s/docs", base_url)
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop.")

    # ── Block until signal ───────────────────────────────────────────────
    def _shutdown(sig: int, _frame) -> None:
        sig_name = signal.Signals(sig).name
        logger.info("Received %s — shutting down …", sig_name)
        try:
            serve.shutdown()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass
        logger.info("Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)


# ---------------------------------------------------------------------------
# Banner helper
# ---------------------------------------------------------------------------

def _log_startup_banner(cfg) -> None:
    logger.info(
        "bhaskera-serve | model=%s backend=%s replicas=%d "
        "http://%s:%d%s",
        cfg.model.name,
        cfg.serve.backend,
        cfg.serve.num_replicas,
        cfg.serve.host,
        cfg.serve.port,
        cfg.serve.route_prefix,
    )
    if cfg.serve.backend == "vllm":
        logger.info(
            "  vLLM | tp=%d gpu_util=%.0f%% max_model_len=%s",
            cfg.serve.vllm.tensor_parallel_size,
            cfg.serve.vllm.gpu_memory_utilization * 100,
            cfg.serve.vllm.max_model_len or "auto",
        )
    else:
        logger.info(
            "  HF   | device=%s max_concurrent_queries=%d",
            cfg.serve.hf.device,
            cfg.serve.hf.max_concurrent_queries,
        )
    if cfg.lora.enabled:
        logger.info("  LoRA enabled (r=%d alpha=%d)", cfg.lora.r, cfg.lora.alpha)


if __name__ == "__main__":
    main()
