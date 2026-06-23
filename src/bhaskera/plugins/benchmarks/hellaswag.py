import torch
import torch.distributed as dist
import logging
from tqdm import tqdm
from bhaskera.evaluation.registry import register_benchmark

logger = logging.getLogger(__name__)

@register_benchmark("hellaswag")
class HellaSwagBenchmark:
    def __init__(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        self.load_dataset = load_dataset

    def run(self, model, tokenizer, cfg) -> dict:
        if tokenizer is None:
            logger.warning("HellaSwag requires a tokenizer. Skipping.")
            return {}

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 1. Load dataset (All ranks call this, HF handles lockfiles)
        # We use the validation split which contains 10,042 examples
        ds = self.load_dataset("Rowan/hellaswag", split="validation")
        
        # 2. Shard dataset across FSDP ranks to speed up evaluation
        # e.g., 4 GPUs = ~2510 examples per GPU
        local_ds = ds.shard(num_shards=world_size, index=rank)
        
        local_correct = 0
        local_total = len(local_ds)

        logger.info(f"[Rank {rank}] Running HellaSwag on {local_total} examples...")

        # 3. Local Evaluation Loop
        device = model.device
        for item in tqdm(local_ds, disable=(rank != 0), desc="HellaSwag Eval"):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])

            choice_logprobs = []
            
            for ending in endings:
                # Format: Context + " " + Ending
                full_text = f"{ctx} {ending}"
                ctx_tokens = tokenizer(ctx, return_tensors="pt", add_special_tokens=True).input_ids
                full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
                
                ctx_len = ctx_tokens.shape[1]
                
                inputs = full_tokens.to(device)
                
                with torch.no_grad():
                    # Autocast policy is handled by FSDP implicitly
                    outputs = model(inputs)
                    logits = outputs.logits  # (1, seq_len, vocab_size)
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[0, :-1, :]
                shift_labels = inputs[0, 1:]
                
                # We only care about the logprobs of the *ending* tokens
                # So we slice starting from ctx_len - 1
                ending_logits = shift_logits[ctx_len - 1 :]
                ending_labels = shift_labels[ctx_len - 1 :]
                
                # Compute log likelihood of this choice
                log_probs = torch.nn.functional.log_softmax(ending_logits, dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=ending_labels.unsqueeze(-1)).squeeze(-1)
                
                # Normalize by length to prevent bias towards shorter endings
                choice_score = token_log_probs.sum().item() / max(1, len(ending_labels))
                choice_logprobs.append(choice_score)

            # Prediction is the ending with the highest normalized log-probability
            prediction = choice_logprobs.index(max(choice_logprobs))
            if prediction == label:
                local_correct += 1

        # 4. Distributed Aggregation
        if dist.is_initialized():
            # Gather local results from all ranks
            local_stats = {"correct": local_correct, "total": local_total}
            gathered_stats = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stats, local_stats)
            
            if rank == 0:
                global_correct = sum(stat["correct"] for stat in gathered_stats)
                global_total = sum(stat["total"] for stat in gathered_stats)
                accuracy = global_correct / global_total if global_total > 0 else 0.0
                return {"benchmark/hellaswag_accuracy": accuracy}
        else:
            # Single GPU fallback
            accuracy = local_correct / local_total if local_total > 0 else 0.0
            return {"benchmark/hellaswag_accuracy": accuracy}
            
        return {}
