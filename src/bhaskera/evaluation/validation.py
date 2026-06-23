import logging
import torch
import torch.distributed as dist
from bhaskera.evaluation.registry import get_metric

logger = logging.getLogger(__name__)

def run_distributed_validation(cfg, model, val_dataset, profile, rank: int, world_size: int) -> dict:
    was_training = model.training
    model.eval()
    
    metric_names = cfg.evaluation.validation.metrics
    active_metrics = []
    for m_name in metric_names:
        m_cls = get_metric(m_name)
        if m_cls:
            active_metrics.append(m_cls())
        else:
            logger.warning(f"Metric '{m_name}' not found.")
    
    local_losses, local_preds, local_labels = [], [], []

    loader = val_dataset.iter_torch_batches(
        batch_size=cfg.training.batch_size,
        dtypes={"input_ids": torch.long, "attention_mask": torch.long, "labels": torch.long},
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
    ) if hasattr(val_dataset, "iter_torch_batches") else val_dataset

    with torch.no_grad():
        for batch in loader:
            forward_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "use_cache": False,
            }
            out = model(**forward_kwargs)
            local_losses.append(out.loss.item())
            
            if any(m_name in ["token_accuracy"] for m_name in metric_names):
                local_preds.append(out.logits.argmax(dim=-1).cpu())
                local_labels.append(batch["labels"].cpu())
    
    if dist.is_available() and dist.is_initialized():
        sum_loss = torch.tensor([sum(local_losses), len(local_losses)], dtype=torch.float64, device=model.device)
        dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
        global_loss_sum, global_count = sum_loss.tolist()
        global_losses = [global_loss_sum / max(1, global_count)] * int(global_count)
    else:
        global_losses = local_losses
        
    results = {}
    if rank == 0:
        for metric in active_metrics:
            results.update(metric.compute(local_preds, local_labels, global_losses))

    if dist.is_available() and dist.is_initialized():
        obj_list = [results]
        dist.broadcast_object_list(obj_list, src=0)
        results = obj_list[0]

    if was_training:
        model.train()
        
    return results
