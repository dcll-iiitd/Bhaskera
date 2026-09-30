"""Record /v1/systemone answers for a request set, or compare two recordings.

    python scripts/decision/parity.py record --url http://127.0.0.1:8017 \
        --requests benchmarks/decision/parity/parity_requests.jsonl --out ref.jsonl [--concurrency 8]
    python scripts/decision/parity.py compare ref.jsonl cand.jsonl [--tolerance 0.01] [--report r.json]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx

from bhaskera.serve.decision import benchtools as bt


def record(url: str, requests: list[dict], concurrency: int, api_key: str | None) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(base_url=url, timeout=600) as client:
        def one(item):
            response = client.post("/v1/systemone", json=item["request"], headers=headers)
            return {"id": item["id"], "status": response.status_code, "response": response.json()}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = []
            for index, row in enumerate(pool.map(one, requests), 1):
                rows.append(row)
                if index % 25 == 0:
                    print(f"{index}/{len(requests)}", file=sys.stderr)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record")
    rec.add_argument("--url", required=True)
    rec.add_argument("--requests", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--concurrency", type=int, default=1)
    rec.add_argument("--api-key")
    cmp_ = commands.add_parser("compare")
    cmp_.add_argument("reference")
    cmp_.add_argument("candidate")
    cmp_.add_argument("--tolerance", type=float, default=0.01)
    cmp_.add_argument("--report")
    args = parser.parse_args()
    if args.command == "record":
        rows = record(args.url, bt.read_jsonl(args.requests), args.concurrency, args.api_key)
        bt.write_jsonl(args.out, rows)
        print(f"{len(rows)} recorded -> {args.out}")
        return
    report = bt.compare(bt.read_jsonl(args.reference), bt.read_jsonl(args.candidate), args.tolerance)
    text = json.dumps(report, indent=2)
    if args.report:
        with open(args.report, "w") as out:
            out.write(text + "\n")
    print(text)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
