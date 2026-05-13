"""
bhaskera-infer — command-line inference entry point (v3, Ray+vLLM)

Backend selection
-----------------
  Auto (default):  Ray+vLLM on CUDA, HF generate() everywhere else.
  BHASKERA_BACKEND=vllm  Force Ray+vLLM
  BHASKERA_BACKEND=hf    Force HF generate() (CPU / dev / no-vLLM)

SLURM / multi-node
------------------
  Set RAY_ADDRESS=<head>:<port> in your batch script, or pass
  --ray-address <addr> to this CLI.

  Minimal SLURM example::

      #!/bin/bash
      #SBATCH --nodes=1
      #SBATCH --gres=gpu:4
      ray start --head --num-gpus=4 --port=6379
      export RAY_ADDRESS="localhost:6379"
      bhaskera-infer --config configs/inference_param2.yaml \\
                     --prompt "Explain attention mechanisms."

Examples
--------
    # Standard generation
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "Explain attention mechanisms."

    # Param2-Thinking (structured output, <think> hidden by default)
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "What is 17 × 23?" --param2

    # Show chain-of-thought
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "What is 17 × 23?" --param2 --show-thinking

    # Multiple prompts from file + save output
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt-file prompts.txt --output-file results.txt

    # Force HF backend (no vLLM required)
    BHASKERA_BACKEND=hf bhaskera-infer --config configs/inference_turboquant.yaml \\
                        --prompt "Hello"

    # Explicit tensor-parallel size (4 GPUs)
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --tensor-parallel-size 4 \\
                   --prompt "Explain quantum entanglement."
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bhaskera.infer")

SEP = "─" * 72


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhaskera-infer",
        description="Bhaskera LLM inference CLI (Ray + vLLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config / model ───────────────────────────────────────────────
    p.add_argument("--config",  default=None, help="Path to Bhaskera YAML config")
    p.add_argument("--model",   default=None, help="HuggingFace model id (overrides config)")
    p.add_argument("--device",  default="auto", help="Device: auto | cuda | cpu | mps")

    # ── Input ────────────────────────────────────────────────────────
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--prompt",      default=None, help="Single prompt string")
    inp.add_argument("--prompt-file", default=None, metavar="FILE",
                     help="File with one prompt per line")

    # ── Generation ───────────────────────────────────────────────────
    p.add_argument("--max-new-tokens",  type=int,   default=None)
    p.add_argument("--temperature",     type=float, default=None)
    p.add_argument("--top-p",           type=float, default=None)
    p.add_argument("--top-k",           type=int,   default=None)
    p.add_argument("--no-sample",       action="store_true",
                   help="Greedy decoding (overrides do_sample=true in config)")
    p.add_argument("--return-full",     action="store_true",
                   help="Include the prompt in the output")

    # ── Param2 / thinking model ───────────────────────────────────────
    p.add_argument("--param2",         action="store_true",
                   help="Use Param2 structured output (reasoning + final_answer)")
    p.add_argument("--show-thinking",  action="store_true",
                   help="Print <think> reasoning block (Param2 / thinking models)")
    p.add_argument("--system-prompt",  default="You are a helpful assistant.",
                   help="System prompt for chat-template models")

    # ── Ray / infrastructure ──────────────────────────────────────────
    p.add_argument("--ray-address",
                   default=None, metavar="HOST:PORT",
                   help="Ray cluster address (sets RAY_ADDRESS; for SLURM head node)")
    p.add_argument("--tensor-parallel-size", type=int, default=None,
                   help="Number of GPUs for tensor parallelism (vLLM, default: all visible)")
    p.add_argument("--backend", default=None, choices=["vllm", "hf"],
                   help="Force a specific backend (overrides BHASKERA_BACKEND env var)")

    # ── Output ───────────────────────────────────────────────────────
    p.add_argument("--output-file", default=None, metavar="FILE",
                   help="Write raw outputs to file (one response per line)")
    p.add_argument("--verbose", "-v", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace):
    """Load YAML config (if given) then apply CLI overrides."""
    if args.config:
        from bhaskera.config import load_config
        cfg = load_config(args.config)
    else:
        from bhaskera.config import Config
        cfg = Config()

    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.inference.device = args.device

    infer = cfg.inference
    if args.max_new_tokens is not None: infer.max_new_tokens = args.max_new_tokens
    if args.temperature    is not None: infer.temperature    = args.temperature
    if args.top_p          is not None: infer.top_p          = args.top_p
    if args.top_k          is not None: infer.top_k          = args.top_k
    if args.no_sample:                  infer.do_sample       = False

    # Tensor-parallel size: store on InferenceConfig for backend to pick up
    if args.tensor_parallel_size is not None:
        infer.tensor_parallel_size = args.tensor_parallel_size

    return cfg


# ---------------------------------------------------------------------------
# Output rendering helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_RE  = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    clean = _THINK_RE.sub("", text)
    clean = _TOOL_RE.sub("", clean)
    return clean.strip()


def _count_tokens(text: str) -> int:
    """Rough word-based token estimate (fallback when tokenizer unavailable)."""
    return max(1, int(len(text.split()) * 0.9))


def _render_output(
    idx: int,
    total: int,
    prompt: str,
    output_text: str,
    is_thinking_model: bool,
    show_thinking: bool,
) -> str:
    lines = []
    if total > 1:
        lines.append(f"\n{SEP}")
        lines.append(f"[{idx + 1}/{total}] {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
        lines.append(SEP)

    if is_thinking_model and not show_thinking:
        lines.append(_strip_thinking(output_text))
    else:
        lines.append(output_text)

    return "\n".join(lines)


def _render_param2_output(
    idx: int,
    total: int,
    prompt: str,
    out,            # Param2Output
    show_thinking: bool,
) -> str:
    lines = []
    if total > 1:
        lines.append(f"\n{SEP}")
        lines.append(f"[{idx + 1}/{total}] {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
        lines.append(SEP)

    if show_thinking and out.reasoning:
        lines.append("── Reasoning ──")
        lines.append(out.reasoning)
        lines.append("── Answer ──")

    lines.append(out.final_answer)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Config ───────────────────────────────────────────────────────
    cfg = _build_config(args)

    # ── Logger ───────────────────────────────────────────────────────
    from bhaskera.utils import build_logger
    tracker = build_logger(cfg)

    # ── Apply backend override ────────────────────────────────────────
    if args.backend:
        os.environ["BHASKERA_BACKEND"] = args.backend
        logger.info(f"Backend forced to: {args.backend}")

    # ── Apply Ray address override ────────────────────────────────────
    if args.ray_address:
        os.environ["RAY_ADDRESS"] = args.ray_address
        logger.info(f"RAY_ADDRESS set to: {args.ray_address}")

    # ── Load prompts ──────────────────────────────────────────────────
    if args.prompt:
        prompts = [args.prompt]
    else:
        path = Path(args.prompt_file)
        if not path.exists():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        with open(path) as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
        if not prompts:
            parser.error(f"No prompts found in {args.prompt_file}")
        logger.info(f"Loaded {len(prompts)} prompt(s) from {args.prompt_file}")

    # ── Engine ───────────────────────────────────────────────────────
    from bhaskera.inference import InferenceEngine
    engine = InferenceEngine(cfg)
    engine.load()

    logger.info(f"Engine: {engine}")

    # ── Detect thinking model ─────────────────────────────────────────
    is_thinking = (
        args.param2
        or "param2" in cfg.model.name.lower()
        or "thinking" in cfg.model.name.lower()
    )

    # ── Generate ─────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.param2:
        # Structured Param2 output
        outputs_raw = engine.generate_param2(
            prompts,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens or cfg.inference.max_new_tokens,
            temperature=args.temperature or cfg.inference.temperature,
            top_p=args.top_p or cfg.inference.top_p,
            top_k=args.top_k or cfg.inference.top_k,
            do_sample=not args.no_sample and cfg.inference.do_sample,
        )
        elapsed = time.perf_counter() - t0

        # Render
        for i, (prompt, out) in enumerate(zip(prompts, outputs_raw)):
            rendered = _render_param2_output(i, len(prompts), prompt, out, args.show_thinking)
            print(rendered)

        # Stats
        total_answer_tokens = sum(_count_tokens(o.final_answer) for o in outputs_raw)
        total_think_tokens  = sum(_count_tokens(o.reasoning)    for o in outputs_raw)
        total_tokens        = total_answer_tokens + total_think_tokens
        _print_stats(elapsed, len(prompts), total_tokens, total_answer_tokens, total_think_tokens, tracker)

        # File output
        if args.output_file:
            _write_output_file(args.output_file, [o.raw for o in outputs_raw])

    else:
        # Plain text output
        outputs = engine.generate(
            prompts,
            max_new_tokens=args.max_new_tokens or cfg.inference.max_new_tokens,
            temperature=args.temperature or cfg.inference.temperature,
            top_p=args.top_p or cfg.inference.top_p,
            top_k=args.top_k or cfg.inference.top_k,
            do_sample=not args.no_sample and cfg.inference.do_sample,
            return_full_text=args.return_full,
        )
        elapsed = time.perf_counter() - t0

        for i, (prompt, text) in enumerate(zip(prompts, outputs)):
            rendered = _render_output(
                i, len(prompts), prompt, text,
                is_thinking_model=is_thinking,
                show_thinking=args.show_thinking,
            )
            print(rendered)

        total_tokens = sum(_count_tokens(t) for t in outputs)
        _print_stats(elapsed, len(prompts), total_tokens, tracker=tracker)

        if args.output_file:
            _write_output_file(args.output_file, outputs)

    # ── KV cache stats ────────────────────────────────────────────────
    stats = engine.kv_cache_stats()
    if stats and stats.get("compression_ratio", 0) > 0:
        print(
            f"TurboQuant KV cache: {stats['tq_mb']:.1f} MB "
            f"(bf16 baseline: {stats['bf16_mb']:.1f} MB, "
            f"ratio: {stats['compression_ratio']:.1f}×)"
        )
        if tracker:
            tracker.log({
                "inference/kv_tq_mb": stats['tq_mb'],
                "inference/kv_bf16_mb": stats['bf16_mb'],
                "inference/kv_compression_ratio": stats['compression_ratio'],
            }, step=1)

    # Thinking model note
    if is_thinking and not args.show_thinking and not args.param2:
        print("(Thinking/reasoning block hidden — use --show-thinking to display)")

    if tracker:
        tracker.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_stats(
    elapsed: float,
    n_prompts: int,
    total_tokens: int,
    answer_tokens: Optional[int] = None,
    thinking_tokens: Optional[int] = None,
    tracker = None
) -> None:
    print(f"\n{SEP}")
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    
    metrics = {
        "inference/elapsed_time": elapsed,
        "inference/total_tokens": total_tokens,
        "inference/tokens_per_sec": tps,
    }

    if thinking_tokens and thinking_tokens > 0:
        ans_tps = (answer_tokens or 0) / elapsed if elapsed > 0 else 0.0
        print(
            f"Generated {n_prompts} response(s) | "
            f"{answer_tokens} answer tokens + {thinking_tokens} thinking tokens | "
            f"{elapsed:.2f}s | "
            f"\033[1;32m{ans_tps:.1f} answer tok/s\033[0m "
            f"({tps:.1f} total tok/s)"
        )
        metrics.update({
            "inference/answer_tokens": answer_tokens,
            "inference/think_tokens": thinking_tokens,
            "inference/answer_tokens_per_sec": ans_tps,
        })
    else:
        print(
            f"Generated {n_prompts} response(s) | "
            f"{total_tokens} tokens | "
            f"{elapsed:.2f}s | "
            f"\033[1;32m{tps:.1f} tok/s\033[0m"
        )
        
    if tracker:
        tracker.log(metrics, step=1)


def _write_output_file(path: str, texts: List[str]) -> None:
    out_path = Path(path)
    with open(out_path, "w") as f:
        for t in texts:
            f.write(t.replace("\n", "\\n") + "\n")
    logger.info(f"Outputs written to {out_path}")


if __name__ == "__main__":
    main()