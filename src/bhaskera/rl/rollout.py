"""
bhaskera.rl.rollout
===================
Sampling completions from the current policy with plain HuggingFace ``generate``.

We sample from the *untruncated* softmax at the given temperature (top_p=1,
top_k=0). The training loss evaluates exactly that distribution; truncating at
sampling time would make the sampled tokens inconsistent with the logprobs we
optimise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Rollouts:
    input_ids: torch.Tensor        # [N, P + C]  left-padded prompt + completion
    attention_mask: torch.Tensor   # [N, P + C]  0 on padding and on tokens after EOS
    completion_mask: torch.Tensor  # [N, C]      1 on completion tokens up to and incl. EOS
    prompt_len: int                # P, same for every row (prompts are left-padded)
    texts: List[str]               # decoded completions, N entries


def eos_token_ids(model, tokenizer) -> List[int]:
    """All token ids that end a completion (chat models often have more than one)."""
    ids = set()
    for source in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if source is None:
            continue
        ids.update(source if isinstance(source, (list, tuple)) else [source])
    if not ids:
        raise ValueError("tokenizer and model define no EOS token; cannot end completions")
    return sorted(ids)


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompts: List[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> Rollouts:
    """Sample ``group_size`` completions per prompt.

    Rows ``[i*G, (i+1)*G)`` of the result belong to prompt ``i``.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = next(model.parameters()).device
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = batch["input_ids"].shape[1]
    eos_ids = eos_token_ids(model, tokenizer)

    model.eval()  # no LoRA dropout while sampling or scoring
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=group_size if do_sample else 1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    else:  # greedy: clear sampling defaults from the model's generation_config
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)
    sequences = model.generate(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs
    )

    rows = sequences.shape[0]
    prompt_mask = batch["attention_mask"].repeat_interleave(rows // len(prompts), dim=0)
    completion = sequences[:, prompt_len:]

    # Keep tokens up to and including the first EOS; everything after is padding.
    is_eos = torch.isin(completion, torch.tensor(eos_ids, device=device))
    seen_eos_before = is_eos.long().cumsum(dim=1) - is_eos.long() > 0
    completion_mask = (~seen_eos_before).long()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    texts = tokenizer.batch_decode(completion, skip_special_tokens=True)
    return Rollouts(sequences, attention_mask, completion_mask, prompt_len, texts)
