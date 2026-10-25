from __future__ import annotations

import math

import pytest
import torch

from bhaskera.rl.data import Example, PromptSampler
from bhaskera.rl.trainer import GRPOTrainer, TrainerSettings

from .helpers import make_lora_model

EXAMPLES = [Example("Q 1", "x"), Example("Q 2 3", "x")]


def count_sevens(text: str, answer: str) -> float:
    """A genuine verifiable task on the toy vocabulary: reward = share of '7' tokens."""
    return text.split().count("7") / 6.0


def build(tokenizer, *, reward=count_sevens, lr=3e-3, seed=0, **overrides):
    model = make_lora_model(tokenizer, seed)
    settings = dict(group_size=8, prompts_per_step=2, max_new_tokens=6, kl_coef=0.04,
                    micro_batch_size=8)
    settings.update(overrides)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    trainer = GRPOTrainer(model, tokenizer, TrainerSettings(**settings), reward,
                          PromptSampler(EXAMPLES, seed=0), opt)
    return model, trainer


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def base_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}


def test_policy_learns_the_verifiable_task(tokenizer):
    torch.manual_seed(1)
    _, trainer = build(tokenizer)
    rewards = [trainer.step()["reward_mean"] for _ in range(80)]
    start, end = sum(rewards[:10]) / 10, sum(rewards[-10:]) / 10
    assert start < 0.25, f"task should start hard, got {start:.3f}"
    assert end > 0.8, f"reward should rise from {start:.3f}, got {end:.3f}"


def test_held_out_evaluation_improves_after_training(tokenizer):
    torch.manual_seed(1)
    _, trainer = build(tokenizer)
    held_out = [Example("Q 4 5", "x"), Example("Q 9", "x"), Example("Q 2", "x")]
    before = trainer.evaluate(held_out)
    for _ in range(80):
        trainer.step()
    after = trainer.evaluate(held_out)
    assert before["eval_examples"] == 3
    assert after["eval_reward"] > before["eval_reward"] + 0.3


def test_only_lora_weights_change(tokenizer):
    torch.manual_seed(1)
    model, trainer = build(tokenizer)
    base0, lora0 = base_state(model), lora_state(model)
    for _ in range(5):
        trainer.step()
    assert all(torch.equal(base0[n], p) for n, p in base_state(model).items())
    assert any(not torch.equal(lora0[n], p) for n, p in lora_state(model).items())


def test_reference_policy_is_the_frozen_base_model(tokenizer):
    """KL is measured against the adapter-disabled model, which must not move during training."""
    torch.manual_seed(1)
    _, trainer = build(tokenizer)
    torch.manual_seed(5)
    from bhaskera.rl.rollout import generate_rollouts
    r = generate_rollouts(trainer.model, tokenizer, ["Q 1"], 4, 5)
    ref_before = trainer._logprobs(r, adapter=False)
    policy_before = trainer._logprobs(r, adapter=True)
    assert torch.allclose(ref_before, policy_before, atol=1e-6)      # LoRA B starts at zero
    for _ in range(10):
        trainer.step()
    assert torch.equal(trainer._logprobs(r, adapter=False), ref_before)
    assert not torch.allclose(trainer._logprobs(r, adapter=True), ref_before, atol=1e-4)


def test_kl_penalty_keeps_policy_closer_to_reference(tokenizer):
    """Train with and without a strong KL penalty, then measure drift from the base model
    the same way for both (the trainer itself logs kl=0 when kl_coef=0, as it skips that pass)."""
    from bhaskera.rl.rollout import generate_rollouts

    def drift(beta):
        torch.manual_seed(1)
        _, trainer = build(tokenizer, kl_coef=beta)
        for _ in range(40):
            trainer.step()
        torch.manual_seed(99)
        r = generate_rollouts(trainer.model, tokenizer, ["Q 1", "Q 2 3"], 16, 6)
        d = trainer._logprobs(r, adapter=False) - trainer._logprobs(r, adapter=True)
        return (((torch.exp(d) - d - 1) * r.completion_mask).sum() / r.completion_mask.sum()).item()

    free, held = drift(0.0), drift(5.0)
    assert held < free, f"beta=5 should drift less than beta=0: {held:.3f} vs {free:.3f}"


def test_update_is_skipped_when_every_group_has_identical_rewards(tokenizer):
    torch.manual_seed(1)
    model, trainer = build(tokenizer, reward=lambda t, a: 1.0)
    before = lora_state(model)
    metrics = trainer.step()
    assert all(torch.equal(before[n], p) for n, p in lora_state(model).items())
    assert metrics["frac_groups_with_signal"] == 0.0 and metrics["grad_norm"] == 0.0
    assert trainer.step_count == 1 and trainer.samples_seen == 16   # counted, just not trained on


def test_counters_and_offset(tokenizer):
    torch.manual_seed(1)
    model, _ = build(tokenizer)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    trainer = GRPOTrainer(model, tokenizer, TrainerSettings(group_size=4, prompts_per_step=2,
                          max_new_tokens=4), count_sevens, PromptSampler(EXAMPLES, 0), opt,
                          samples_offset=100)
    trainer.step()
    trainer.step()
    assert trainer.samples_seen == 100 + 2 * 2 * 4
    assert trainer.tokens_seen > 0


def test_micro_batching_does_not_change_the_update(tokenizer):
    """Splitting the batch into micro-batches must give the same gradient step."""
    states = []
    for micro in (3, 16):
        torch.manual_seed(7)
        model, trainer = build(tokenizer, micro_batch_size=micro)
        trainer.step()
        states.append(lora_state(model))
    for name in states[0]:
        assert torch.allclose(states[0][name], states[1][name], atol=1e-5), name


def test_multiple_epochs_per_rollout_use_old_logprobs(tokenizer):
    torch.manual_seed(1)
    model, trainer = build(tokenizer, epochs_per_rollout=3, lr=1e-2)
    metrics = trainer.step()
    assert 0.0 <= metrics["clip_frac"] <= 1.0
    for v in metrics.values():
        assert math.isfinite(v)   # no NaN / inf
    # three optimizer steps were taken on one rollout batch
    torch.manual_seed(1)
    _, single = build(tokenizer, epochs_per_rollout=1, lr=1e-2)
    single.step()
    assert not all(torch.equal(a, b) for a, b in zip(lora_state(model).values(),
                                                     lora_state(single.model).values()))


def test_non_finite_gradients_do_not_update_weights(tokenizer):
    torch.manual_seed(1)
    model, trainer = build(tokenizer, reward=lambda t, a: float("nan"))
    before = lora_state(model)
    trainer.step()   # NaN rewards -> NaN advantages -> NaN gradients
    assert all(torch.equal(before[n], p) for n, p in lora_state(model).items())


def test_rejects_non_lora_model_and_frozen_model(tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM
    plain = LlamaForCausalLM(LlamaConfig(vocab_size=15, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2))
    opt = torch.optim.SGD(plain.parameters(), lr=0.1)
    args = (tokenizer, TrainerSettings(), count_sevens, PromptSampler(EXAMPLES, 0), opt)
    with pytest.raises(ValueError, match="PEFT"):
        GRPOTrainer(plain, *args)
    frozen = make_lora_model(tokenizer)
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        GRPOTrainer(frozen, *args)
