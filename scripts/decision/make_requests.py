"""Write the deterministic request sets.

    python scripts/decision/make_requests.py parity --out benchmarks/decision/parity/parity_requests.jsonl
    python scripts/decision/make_requests.py bench --out benchmarks/decision/requests
"""
import argparse
from pathlib import Path

from bhaskera.serve.decision import benchtools as bt

BENCH_COUNTS = (1, 3, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("parity", "bench"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.kind == "parity":
        bt.write_jsonl(args.out, bt.parity_requests())
        print(args.out)
        return
    for length in bt.LENGTHS:
        for count in BENCH_COUNTS:
            path = args.out / f"bench_{length}_q{count}.jsonl"
            bt.write_jsonl(path, bt.bench_requests(length, count))
            print(path)


if __name__ == "__main__":
    main()
