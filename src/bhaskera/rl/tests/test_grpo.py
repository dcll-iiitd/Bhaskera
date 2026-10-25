from __future__ import annotations

import math

import pytest
import torch

from bhaskera.rl.grpo import completion_logprobs, group_advantages, grpo_sequence_loss


class TestGroupAdvantages:
    def test_known_values(self):
        adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]), 4)
        g1, g2 = torch.tensor([1.0, 0, 0, 0]), torch.tensor([1.0, 1, 0, 0])

        def norm(g):
            return (g - g.mean()) / (g.std() + 1e-4)

        expected = torch.cat([norm(g1), norm(g2)])
        assert torch.allclose(adv, expected)

    def test_groups_are_normalised_independently_and_sum_to_zero(self):
        adv = group_advantages(torch.tensor([5.0, 1.0, 3.0, 100.0, 0.0, 50.0]), 3).view(2, 3)
        assert torch.allclose(adv.sum(dim=1), torch.zeros(2), atol=1e-5)

    def test_identical_rewards_give_zero_advantage(self):
        assert torch.equal(group_advantages(torch.ones(8), 4), torch.zeros(8))

    def test_better_sample_gets_higher_advantage(self):
        adv = group_advantages(torch.tensor([0.0, 1.0, 0.0, 0.0]), 4)
        assert adv[1] > 0 > adv[0] and adv[0] == adv[2] == adv[3]

    def test_validation(self):
        with pytest.raises(ValueError):
            group_advantages(torch.ones(4), 1)
        with pytest.raises(ValueError):
            group_advantages(torch.ones(5), 4)


def _loss(logp, old, ref, adv, mask, clip=0.2, kl=0.0):
    return grpo_sequence_loss(
        torch.tensor(logp), torch.tensor(old), None if ref is None else torch.tensor(ref),
        torch.tensor(adv), torch.tensor(mask), clip, kl,
    )


class TestGrpoLoss:
    def test_ratio_one_loss_is_minus_advantage(self):
        loss = _loss([[-1.0, -2.0]], [[-1.0, -2.0]], None, [0.7], [[1, 1]])
        assert loss.item() == pytest.approx(-0.7)

    def test_mask_excludes_padding_and_is_a_token_mean(self):
        # second token masked out: mean over the single real token
        loss = _loss([[-1.0, -9.0]], [[-1.0, -1.0]], None, [1.0], [[1, 0]])
        assert loss.item() == pytest.approx(-1.0)

    def test_positive_advantage_clips_when_ratio_too_high(self):
        ratio = math.exp(0.5)  # > 1.2
        loss = _loss([[0.5]], [[0.0]], None, [2.0], [[1]], clip=0.2)
        assert loss.item() == pytest.approx(-1.2 * 2.0)  # clipped, not ratio * A
        assert ratio * 2.0 > 1.2 * 2.0

    def test_negative_advantage_takes_the_pessimistic_branch(self):
        # ratio 1.5 with A=-1: ratio*A = -1.5 is below the clipped -1.2, so min keeps -1.5
        loss = _loss([[math.log(1.5)]], [[0.0]], None, [-1.0], [[1]], clip=0.2)
        assert loss.item() == pytest.approx(1.5)

    def test_clipped_branch_has_no_gradient(self):
        logp = torch.tensor([[0.5]], requires_grad=True)
        loss = grpo_sequence_loss(logp, torch.zeros(1, 1), None, torch.tensor([2.0]),
                                  torch.ones(1, 1), 0.2, 0.0)
        loss.sum().backward()
        assert logp.grad.item() == 0.0

    def test_kl_is_zero_when_policy_equals_reference(self):
        base = _loss([[-1.0, -2.0]], [[-1.0, -2.0]], None, [0.0], [[1, 1]])
        with_kl = _loss([[-1.0, -2.0]], [[-1.0, -2.0]], [[-1.0, -2.0]], [0.0], [[1, 1]], kl=0.5)
        assert with_kl.item() == pytest.approx(base.item())

    def test_kl_k3_value_and_nonnegativity(self):
        # d = ref - logp = 1.0 -> k3 = e^1 - 1 - 1
        loss = _loss([[0.0]], [[0.0]], [[1.0]], [0.0], [[1]], kl=0.1)
        assert loss.item() == pytest.approx(0.1 * (math.e - 2))
        for d in (-2.0, -0.5, 0.3, 2.0):
            assert math.exp(d) - d - 1 >= 0

    def test_kl_requires_reference(self):
        with pytest.raises(ValueError):
            _loss([[0.0]], [[0.0]], None, [0.0], [[1]], kl=0.1)

    def test_gradient_step_raises_logprob_of_positive_advantage_tokens(self):
        logits = torch.zeros(2, 1, 5, requires_grad=True)       # two sequences, one token each
        targets = torch.tensor([[1], [2]])
        adv = torch.tensor([1.0, -1.0])
        opt = torch.optim.SGD([logits], lr=0.5)
        before = torch.log_softmax(logits.detach(), -1).gather(2, targets.unsqueeze(-1)).squeeze()
        logp = torch.log_softmax(logits, -1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        loss = grpo_sequence_loss(logp, logp.detach(), None, adv, torch.ones(2, 1), 0.2, 0.0)
        loss.sum().backward()
        opt.step()
        after = torch.log_softmax(logits.detach(), -1).gather(2, targets.unsqueeze(-1)).squeeze()
        assert after[0] > before[0] and after[1] < before[1]


class TestCompletionLogprobs:
    def test_matches_manual_computation(self, lora_model):
        ids = torch.tensor([[3, 5, 6, 7, 8]])
        mask = torch.ones_like(ids)
        got = completion_logprobs(lora_model.eval(), ids, mask, prompt_len=2)
        with torch.no_grad():
            logits = lora_model(input_ids=ids).logits[0]
        expected = torch.log_softmax(logits.float(), -1)
        want = torch.stack([expected[t - 1, ids[0, t]] for t in range(2, 5)])
        assert torch.allclose(got[0], want, atol=1e-5)

    def test_temperature_scales_logits(self, lora_model):
        ids = torch.tensor([[3, 5, 6, 7]])
        mask = torch.ones_like(ids)
        t1 = completion_logprobs(lora_model.eval(), ids, mask, 2, temperature=1.0)
        t2 = completion_logprobs(lora_model.eval(), ids, mask, 2, temperature=2.0)
        assert not torch.allclose(t1, t2)

    def test_gradients_reach_only_lora_weights(self, lora_model):
        ids = torch.tensor([[3, 5, 6, 7]])
        mask = torch.ones_like(ids)
        completion_logprobs(lora_model.eval(), ids, mask, 2).sum().backward()
        graded = [n for n, p in lora_model.named_parameters() if p.grad is not None]
        assert graded and all("lora_" in n for n in graded)
