import torch
import torch.distributed as dist
import logging
from tqdm import tqdm
from bhaskera.evaluation.registry import register_benchmark

logger = logging.getLogger(__name__)

@register_benchmark("arc_easy")
class ARCEasyBenchmark:
    def __init__(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        self.load_dataset = load_dataset

    def run(self, model, tokenizer, cfg) -> dict:
        if tokenizer is None:
            logger.warning("ARC-Easy requires a tokenizer. Skipping.")
            return {}

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        ds = self.load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        local_ds = ds.shard(num_shards=world_size, index=rank)
        
        local_correct = 0
        local_total = len(local_ds)

        logger.info(f"[Rank {rank}] Running ARC-Easy on {local_total} examples...")

        device = model.device
        for item in tqdm(local_ds, disable=(rank != 0), desc="ARC-Easy Eval"):
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]
            
            try:
                target_idx = labels.index(answer_key)
            except ValueError:
                local_total -= 1 # Skip malformed examples
                continue

            choice_logprobs = []
            
            for choice in choices:
                # Format: "Question: {q}\nAnswer: {c}"
                ctx = f"Question: {question}\nAnswer:"
                full_text = f"{ctx} {choice}"
                
                ctx_tokens = tokenizer(ctx, return_tensors="pt", add_special_tokens=True).input_ids
                full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
                ctx_len = ctx_tokens.shape[1]
                
                inputs = full_tokens.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    logits = outputs.logits
                
                shift_logits = logits[0, :-1, :]
                shift_labels = inputs[0, 1:]
                
                ending_logits = shift_logits[ctx_len - 1 :]
                ending_labels = shift_labels[ctx_len - 1 :]
                
                log_probs = torch.nn.functional.log_softmax(ending_logits, dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=ending_labels.unsqueeze(-1)).squeeze(-1)
                
                choice_score = token_log_probs.sum().item() / max(1, len(ending_labels))
                choice_logprobs.append(choice_score)

            prediction = choice_logprobs.index(max(choice_logprobs))
            if prediction == target_idx:
                local_correct += 1

        if dist.is_initialized():
            local_stats = {"correct": local_correct, "total": local_total}
            gathered_stats = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stats, local_stats)
            
            if rank == 0:
                global_correct = sum(stat["correct"] for stat in gathered_stats)
                global_total = sum(stat["total"] for stat in gathered_stats)
                accuracy = global_correct / global_total if global_total > 0 else 0.0
                return {"benchmark/arc_easy_accuracy": accuracy}
        else:
            accuracy = local_correct / local_total if local_total > 0 else 0.0
            return {"benchmark/arc_easy_accuracy": accuracy}
            
        return {}
