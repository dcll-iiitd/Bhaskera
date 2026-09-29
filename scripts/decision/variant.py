"""Write a copy of a YAML config with dotted keys overridden; values are parsed as YAML.

    python scripts/decision/variant.py configs/serve_jevos.yaml out.yaml \
        model.gguf=jevos-q4_k_m serve.decision.batching.enabled=true
"""
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit(__doc__)
    base, out, *pairs = argv
    data = yaml.safe_load(Path(base).read_text()) or {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {pair!r}")
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
