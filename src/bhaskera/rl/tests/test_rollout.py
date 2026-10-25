from __future__ import annotations

import torch

from bhaskera.rl.grpo import completion_logprobs
from bhaskera.rl.rollout import eos_token_ids, generate_rollouts

PROMPTS = ["Q 1", "Q 2 3 4 5", "Q 6 7"]   # different lengths -> left padding is exercised
G, NEW = 3, 8


def _rollouts(model, tok, **kw):
    torch.manual_seed(1)
    return generate_rollouts(model, tok, PROMPTS, group_size=G, max_new_tokens=NEW, **kw)


def test_shapes_and_grouping(lora_model, tokenizer):
    r = _rollouts(lora_model, tokenizer)
    n = len(PROMPTS) * G
    assert r.input_ids.shape[0] == n and len(r.texts) == n
    assert r.attention_mask.shape == r.input_ids.shape
    assert r.completion_mask.shape == (n, r.input_ids.shape[1] - r.prompt_len)
    # rows [i*G, (i+1)*G) share prompt i: their (unpadded) prompt tokens are identical
    for i, prompt in enumerate(PROMPTS):
        expected = tokenizer(prompt)["input_ids"]
        for j in range(G):
            row = i * G + j
            real = r.attention_mask[row, :r.prompt_len].bool()
            prompt_ids = r.input_ids[row, :r.prompt_len][real]
            assert prompt_ids.tolist() == expected


def test_completion_mask_stops_at_first_eos(lora_model, tokenizer):
    r = _rollouts(lora_model, tokenizer)
    eos = set(eos_token_ids(lora_model, tokenizer))
    comp = r.input_ids[:, r.prompt_len:]
    for row in range(comp.shape[0]):
        m = r.completion_mask[row].tolist()
        # mask is a prefix of ones followed by zeros
        assert m == sorted(m, reverse=True)
        length = sum(m)
        assert length >= 1
        eos_positions = [t for t in range(comp.shape[1]) if comp[row, t].item() in eos]
        if eos_positions:
            assert length == eos_positions[0] + 1    # includes the EOS token itself
        else:
            assert length == comp.shape[1]           # ran to max_new_tokens


def test_logprobs_with_left_padding_match_unpadded_forward(lora_model, tokenizer):
    """The key correctness check: scoring a left-padded batch must equal scoring each
    sequence alone with no padding, otherwise old/new ratios would be garbage."""
    r = _rollouts(lora_model, tokenizer)
    batched = completion_logprobs(lora_model, r.input_ids, r.attention_mask, r.prompt_len)
    for row in range(r.input_ids.shape[0]):
        real = r.attention_mask[row].bool()
        seq = r.input_ids[row][real].unsqueeze(0)          # padding removed
        n_prompt = int(r.attention_mask[row, :r.prompt_len].sum())
        alone = completion_logprobs(lora_model, seq, torch.ones_like(seq), n_prompt)
        n_comp = alone.shape[1]
        assert torch.allclose(batched[row, :n_comp], alone[0], atol=1e-4), f"row {row}"


def test_greedy_tokens_are_the_argmax_of_the_scoring_forward_pass(lora_model, tokenizer):
    """generate() and our scoring pass must agree about the model's distribution."""
    r = generate_rollouts(lora_model, tokenizer, PROMPTS, 1, NEW, do_sample=False)
    position_ids = (r.attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = lora_model(input_ids=r.input_ids, attention_mask=r.attention_mask,
                            position_ids=position_ids).logits
    predicted = logits[:, r.prompt_len - 1:-1].argmax(-1)
    comp = r.input_ids[:, r.prompt_len:]
    valid = r.completion_mask.bool()
    assert torch.equal(predicted[valid], comp[valid])


def test_greedy_is_deterministic_and_sampling_is_seeded(lora_model, tokenizer):
    a = generate_rollouts(lora_model, tokenizer, PROMPTS, 1, NEW, do_sample=False)
    b = generate_rollouts(lora_model, tokenizer, PROMPTS, 1, NEW, do_sample=False)
    assert a.texts == b.texts
    assert _rollouts(lora_model, tokenizer).texts == _rollouts(lora_model, tokenizer).texts


def test_sampling_is_diverse(lora_model, tokenizer):
    r = _rollouts(lora_model, tokenizer)
    assert len(set(r.texts)) > 1


def test_eos_ids_union_of_tokenizer_and_generation_config(lora_model, tokenizer):
    lora_model.generation_config.eos_token_id = [tokenizer.eos_token_id, 4]
    assert eos_token_ids(lora_model, tokenizer) == sorted({tokenizer.eos_token_id, 4})
