"""Closed-loop load on /v1/systemone at several concurrency levels; one JSON line per level.

    python scripts/decision/bench.py --url http://127.0.0.1:8100 \
        --requests benchmarks/decision/requests/bench_short_q1.jsonl \
        --levels 1,4,16,64 --duration 15 --warmup 3 --label bhaskera-r4-q8_0 --out results.jsonl
"""
import argparse
import asyncio
import itertools
import json
import subprocess
import threading
import time
from pathlib import Path

import httpx

from bhaskera.serve.decision import benchtools as bt


class GpuSampler(threading.Thread):
    """nvidia-smi every `every` seconds: peak memory and mean utilisation of one GPU."""

    def __init__(self, gpu: int, every: float = 0.5):
        super().__init__(daemon=True)
        self.gpu, self.every = gpu, every
        self.memory: list[float] = []
        self.util: list[float] = []
        self._halt = threading.Event()

    def run(self):
        query = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits", "-i", str(self.gpu)]
        while not self._halt.is_set():
            try:
                out = subprocess.run(query, capture_output=True, text=True, timeout=5).stdout
                memory, util = (float(x) for x in out.strip().split(","))
                self.memory.append(memory)
                self.util.append(util)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._halt.wait(self.every)

    def finish(self) -> dict:
        self._halt.set()
        self.join()
        return {
            "max_memory_mib": max(self.memory) if self.memory else None,
            "mean_util_pct": sum(self.util) / len(self.util) if self.util else None,
        }


async def level(url, records, concurrency, duration, warmup, headers) -> dict:
    latencies: list[float] = []
    counts = {"errors": 0, "questions": 0}
    measure_from = time.perf_counter() + warmup
    stop_at = measure_from + duration
    order = itertools.count()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=600, limits=limits) as client:
        async def worker():
            while time.perf_counter() < stop_at:
                item = records[next(order) % len(records)]
                start = time.perf_counter()
                try:
                    response = await client.post("/v1/systemone", json=item["request"], headers=headers)
                    ok = response.status_code == 200
                except httpx.HTTPError:
                    ok = False
                if start < measure_from:
                    continue
                if ok:
                    latencies.append(time.perf_counter() - start)
                    counts["questions"] += len(item["request"]["questions"])
                else:
                    counts["errors"] += 1

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    summary = bt.summarize(latencies, counts["errors"], counts["questions"], duration)
    summary["completed"] = summary.pop("requests")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--levels", default="1,4,16,64")
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    records = bt.read_jsonl(args.requests)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for concurrency in (int(x) for x in args.levels.split(",")):
        sampler = GpuSampler(args.gpu)
        sampler.start()
        summary = asyncio.run(level(args.url, records, concurrency, args.duration, args.warmup, headers))
        row = {"label": args.label, "url": args.url, "requests": args.requests.stem,
               "concurrency": concurrency, **summary, **sampler.finish()}
        with args.out.open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row))


if __name__ == "__main__":
    main()
