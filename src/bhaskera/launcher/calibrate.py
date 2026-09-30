"""
bhaskera-calibrate
==================
Fit per-question-type temperatures for a decision model from labelled logits.

    bhaskera-calibrate --rows rows.jsonl --fingerprint <engine.fingerprint from /health> --out cal.json

Each row: {"type": "boolean"|"choice"|"score", "logits": [...], "label_index": i}. From an
uncalibrated server, log-probabilities work as logits. Serve the result with
serve.decision.calibration: cal.json (the engine checks the fingerprint).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bhaskera-calibrate", description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; refusing to overwrite")

    from bhaskera.serve.decision.calibration import fit_temperature

    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    calibration = fit_temperature(rows, args.fingerprint)
    args.out.write_text(calibration.model_dump_json(indent=2) + "\n")
    for key, metrics in sorted(calibration.fit_metrics.items()):
        print(
            f"{key}: T={calibration.temperatures[key]:.3f}  nll {metrics['fit_nll_before']:.4f}"
            f" -> {metrics['fit_nll_after']:.4f}  ({metrics['rows']} rows)"
        )
    print(f"wrote {args.out}. Fitted only: check it on held-out data the fit never saw.")


if __name__ == "__main__":
    main()
