import argparse
import json
import logging
import time
import requests
from statistics import mean

from bhaskera.config import load_config
from bhaskera.inference import InferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("benchmark")

def count_tokens(text: str) -> int:
    """Rough word-based token estimate."""
    return max(1, int(len(text.split()) * 0.9))

def main():
    parser = argparse.ArgumentParser(description="System load benchmark using Latency-Aware-PR-Benchmark")
    parser.add_argument("--config", default="configs/param2.yaml", help="Path to Bhaskera config")
    parser.add_argument("--url", default="https://raw.githubusercontent.com/Varun-Gambhir/Latency-Aware-PR-Benchmarking/refs/heads/main/hft_benchmark.json", help="URL to the benchmark JSON")
    parser.add_argument("--max-samples", type=int, default=10, help="Max number of requests to evaluate")
    parser.add_argument("--batch-size", type=int, default=1, help="Requests per batch")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens per generation")
    args = parser.parse_args()

    # 1. Fetch benchmark dataset
    logger.info(f"Fetching dataset from {args.url} ...")
    try:
        resp = requests.get(args.url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch dataset: {e}")
        return
    
    samples = data[:args.max_samples]
    prompts = [item["clean_prompt"] for item in samples]
    logger.info(f"Loaded {len(prompts)} prompts for benchmarking.")

    # 2. Init model engine
    logger.info(f"Loading engine with config: {args.config} ...")
    cfg = load_config(args.config)
    engine = InferenceEngine(cfg)
    try:
        engine.load()
    except Exception as e:
        logger.error(f"Engine failed to load: {e}")
        return

    logger.info("Starting generation benchmark...")
    
    latencies = []
    total_tokens = 0
    t_start_total = time.perf_counter()

    # 3. Execution Loop
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        
        t0 = time.perf_counter()
        
        # Uses standard generation for stress testing load 
        # (you could swap to generate_param2() if testing internal Param2 features)
        outputs = engine.generate(
            batch, 
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )
        t1 = time.perf_counter()
        
        latency = t1 - t0
        latencies.append(latency)
        
        batch_tokens = sum(count_tokens(out) for out in outputs)
        total_tokens += batch_tokens
            
        logger.info(f"Batch {i//args.batch_size + 1}/{(len(prompts)+args.batch_size-1)//args.batch_size} processed in {latency:.2f}s ({batch_tokens} tokens)")
        
    t_end_total = time.perf_counter()
    total_time = t_end_total - t_start_total
    overall_throughput = total_tokens / total_time if total_time > 0 else 0

    # 4. Report
    logger.info("=" * 60)
    logger.info("Benchmark Complete")
    logger.info("=" * 60)
    logger.info(f"Total Samples    : {len(prompts)}")
    logger.info(f"Total Time       : {total_time:.2f} s")
    logger.info(f"Avg Latency/Batch: {mean(latencies):.2f} s")
    logger.info(f"Generated Tokens : {total_tokens}")
    logger.info(f"Avg Throughput   : {overall_throughput:.2f} tokens/sec")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
