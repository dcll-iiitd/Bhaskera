"""Markdown tables from benchmark result lines, one table per request set.

    python scripts/decision/bench_report.py benchmarks/decision/results/results.jsonl > SUMMARY.md
"""
import sys
from collections import defaultdict

from bhaskera.serve.decision import benchtools as bt


def main() -> None:
    rows = bt.read_jsonl(sys.argv[1])
    by_set = defaultdict(list)
    for row in rows:
        by_set[row["requests"]].append(row)
    for name in sorted(by_set):
        print(f"## {name}\n")
        print(bt.table(sorted(by_set[name], key=lambda r: (r["label"], r["concurrency"]))))
        print()


if __name__ == "__main__":
    main()
