"""Convert GSM8K into the JSONL format used by bhaskera-rl.

    python scripts/prepare_gsm8k_rl.py --out-dir data/gsm8k

Writes train.jsonl (7,473 problems) and test.jsonl (1,319). Each line is
{"prompt": <question>, "answer": <full solution ending in "#### <number>">}; the
reward function reads the number after "####".
"""
from __future__ import annotations

import argparse
import json
import os


def convert(parquet_path: str, out_path: str) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path).to_pydict()
    with open(out_path, "w", encoding="utf-8") as f:
        for question, answer in zip(table["question"], table["answer"]):
            f.write(json.dumps({"prompt": question, "answer": answer}) + "\n")
    return len(table["question"])


def shard(path: str, num_shards: int) -> list:
    """Split a JSONL file round-robin into ``train_0.jsonl`` ... so each node gets its own data."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    stem = path[: -len(".jsonl")]
    outputs = []
    for k in range(num_shards):
        out = f"{stem}_{k}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(lines[k::num_shards])
        outputs.append(out)
    return outputs


def main() -> None:
    from huggingface_hub import hf_hub_download

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-shards", type=int, default=1,
                   help="also split train.jsonl into this many disjoint files (one per node)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "test"):
        parquet = hf_hub_download(
            "openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset"
        )
        out = os.path.join(args.out_dir, f"{split}.jsonl")
        print(f"{split}: {convert(parquet, out)} examples -> {out}")
        if split == "train" and args.num_shards > 1:
            print("shards:", ", ".join(shard(out, args.num_shards)))


if __name__ == "__main__":
    main()
