"""
bhaskera-decision-fetch
=======================
Download and verify what a decision config needs (GGUF weights and the pinned llama.cpp
runtime) without starting a server, and print where they are.

    bhaskera-decision-fetch --config configs/serve_jevos.yaml
"""
from __future__ import annotations

import argparse
import sys


def _progress(name: str, done: int, total: int) -> None:
    if sys.stderr.isatty():
        share = f"{100 * done / total:5.1f}%" if total else f"{done >> 20} MiB"
        sys.stderr.write(f"\r{name}: {share}")
        if total and done >= total:
            sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bhaskera-decision-fetch", description=__doc__)
    parser.add_argument("--config", "-c", required=True, metavar="PATH")
    args = parser.parse_args(argv)

    from bhaskera.config import load_config
    from bhaskera.serve.decision.loader import provision

    cfg = load_config(args.config)
    provision(cfg, progress=_progress)
    print(f"gguf: {cfg.model.gguf}")
    print(f"runtime_dir: {cfg.serve.decision.llama_cpp.runtime_dir}")


if __name__ == "__main__":
    main()
