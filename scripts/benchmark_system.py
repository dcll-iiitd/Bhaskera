import argparse
import json
import logging
import time
import requests
import threading
import concurrent.futures
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
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests")
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
    
    # Init logger
    from bhaskera.utils import build_logger
    tracker = build_logger(cfg)

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
    
    # Tracking concurrency
    active_requests = 0
    lock = threading.Lock()

    # Worker function for parallel execution
    def run_inference(prompt):
        nonlocal active_requests
        with lock:
            active_requests += 1
            current_active = active_requests
            
        t0 = time.perf_counter()
        outputs = engine.generate(
            [prompt], 
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )
        t1 = time.perf_counter()
        
        with lock:
            active_requests -= 1
            
        latency = t1 - t0
        tokens = count_tokens(outputs[0])
        return latency, tokens, current_active

    # 3. Execution Loop (Parallel)
    logger.info(f"Running {len(prompts)} prompts with {args.concurrency} concurrent threads...")
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Map prompts to futures
        future_to_prompt = {executor.submit(run_inference, p): p for p in prompts}
        
        for future in concurrent.futures.as_completed(future_to_prompt):
            try:
                latency, tokens, concurrent_active = future.result()
                latencies.append(latency)
                total_tokens += tokens
                
                completed += 1
                
                logger.info(f"Completed {completed}/{len(prompts)} prompts... [Active threads: {active_requests}]")
                    
                if tracker:
                    tracker.log({
                        "benchmark/request_latency": latency,
                        "benchmark/request_tokens": tokens,
                        "benchmark/request_throughput": tokens / latency if latency > 0 else 0,
                        "benchmark/active_requests": concurrent_active,
                    }, step=completed)
                    
            except Exception as exc:
                logger.error(f"Generation generated an exception: {exc}")

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

    if tracker:
        tracker.log({
            "benchmark/total_time": total_time,
            "benchmark/avg_latency": mean(latencies) if latencies else 0,
            "benchmark/overall_throughput": overall_throughput,
            "benchmark/total_tokens": total_tokens
        }, step=len(prompts))
        tracker.finish()

if __name__ == "__main__":
    main()
