"""
bhaskera-serve
==============
CLI entry point for the Bhaskera LLM serving stack.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import socket
import subprocess
import threading
import re
import os

logger = logging.getLogger(__name__)

_active_processes = []

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _shutdown_subprocesses():
    for p in _active_processes:
        if p.poll() is None:
            p.terminate()

def monitor_cloudflared(proc):
    for line in proc.stderr:
        match = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", line)
        if match:
            logger.info("=" * 60)
            logger.info("🌍 PUBLIC GATEWAY URL (Cloudflare):")
            logger.info("   %s", match.group(0))
            logger.info("=" * 60)

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

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.log_level != "DEBUG":
        for noisy in ("ray", "ray.serve", "urllib3", "filelock", "transformers", "uvicorn", "langfuse"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Loading config from %s …", args.config)
    from bhaskera.config import load_config

    cfg = load_config(args.config)

    if args.host is not None:
        cfg.serve.host = args.host
    if args.port is not None:
        cfg.serve.port = args.port
    if args.backend is not None:
        cfg.serve.backend = args.backend
    if args.num_replicas is not None:
        cfg.serve.num_replicas = args.num_replicas

    if cfg.serve.port == 0:
        cfg.serve.port = get_free_port()
        
    proxy_port = cfg.serve.gateway.proxy_port
    if cfg.serve.gateway.enabled and proxy_port == 0:
        proxy_port = get_free_port()

    _log_startup_banner(cfg)

    if cfg.serve.backend not in ("vllm", "hf"):
        logger.error(
            "cfg.serve.backend must be 'vllm' or 'hf', got %r",
            cfg.serve.backend,
        )
        sys.exit(1)

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
        runtime_env={"env_vars": {"RAY_SERVE_HTTP_PROXY_TIMEOUT_S": "600"}},
    )
    logger.info("Ray resources: %s", ray.available_resources())

    from ray import serve

    serve.start(
        detached=True,
        http_options={
            "host": cfg.serve.host,
            "port": cfg.serve.port,
        },
    )

    logger.info("Building application …")
    from bhaskera.serve.app import build_app

    app = build_app(cfg)

    logger.info("Deploying to Ray Serve (this may take a minute for large models) …")
    serve.run(
        app,
        route_prefix=cfg.serve.route_prefix,
        name="bhaskera_llm",
    )
    
    if cfg.serve.gateway.enabled:
        logger.info("Starting Custom Langfuse Gateway on port %d...", proxy_port)
        
        # Inject the internal Ray port into the environment so Uvicorn can find it
        env = os.environ.copy()
        env["RAY_PORT"] = str(cfg.serve.port)
        
        # Launch Uvicorn
        uvicorn_cmd = [
            sys.executable, "-m", "uvicorn", 
            "bhaskera.gateway:app", 
            "--host", "127.0.0.1",
            "--port", str(proxy_port)
        ]
        
        p_gateway = subprocess.Popen(uvicorn_cmd, env=env)
        _active_processes.append(p_gateway)
        
        if cfg.serve.gateway.cloudflared:
            logger.info("Starting Cloudflare Tunnel...")
            cf_cmd = [
                "./cloudflared", "tunnel", "--url", f"http://127.0.0.1:{proxy_port}"
            ]
            
            p_cf = subprocess.Popen(
                cf_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            _active_processes.append(p_cf)
            threading.Thread(target=monitor_cloudflared, args=(p_cf,), daemon=True).start()

    logger.info("=" * 60)
    logger.info("  Bhaskera Engine (Internal) is live")
    logger.info("  Port:    %d", cfg.serve.port)
    if cfg.serve.gateway.enabled:
        logger.info("  Langfuse Gateway is live on port %d", proxy_port)
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop.")

    def _shutdown(sig: int, _frame) -> None:
        sig_name = signal.Signals(sig).name
        logger.info("Received %s — shutting down …", sig_name)
        _shutdown_subprocesses()
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

def _log_startup_banner(cfg) -> None:
    logger.info(
        "bhaskera-serve | model=%s backend=%s replicas=%d http://%s:%d%s",
        cfg.model.name,
        cfg.serve.backend,
        cfg.serve.num_replicas,
        cfg.serve.host,
        cfg.serve.port,
        cfg.serve.route_prefix,
    )

if __name__ == "__main__":
    main()
