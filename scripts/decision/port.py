"""Copy one file from the pinned upstream jev checkout into Bhaskera, with exact, counted edits.

    python scripts/decision/port.py engine/schema.py src/bhaskera/serve/decision/schema.py \
        --replace 'from ..engine.schema import' 'from .schema import'

Every --replace must match exactly once, so an upstream change can never be ported silently.
`\\n` in OLD/NEW means a newline.
"""
import argparse
import subprocess
from pathlib import Path

UPSTREAM = Path(".upstream/jev")
COMMIT = "5be02a3db3b0b32f01c176d8cd122f8c2a8e4db1"
HEADER = f"# Ported from feder-cr/jev@{COMMIT[:7]} (MIT); see NOTICE in this directory.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="path under jev's src/jev/")
    parser.add_argument("dest")
    parser.add_argument("--replace", nargs=2, action="append", default=[], metavar=("OLD", "NEW"))
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"{UPSTREAM} is at {head}, expected {COMMIT}")
    text = (UPSTREAM / "src" / "jev" / args.source).read_text()
    for old, new in args.replace:
        old, new = old.replace("\\n", "\n"), new.replace("\\n", "\n")
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{args.source}: {old!r} occurs {count} times, expected 1")
        text = text.replace(old, new)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(HEADER + text)
    print(f"{args.source} -> {dest} ({len(args.replace)} edits)")


if __name__ == "__main__":
    main()
