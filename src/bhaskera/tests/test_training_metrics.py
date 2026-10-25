"""Learning-rate schedule, token-weighted training loss, and exact validation sums."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from bhaskera.config import Config
# bhaskera.trainer must be imported before bhaskera.evaluation (an existing circular import)
from bhaskera.trainer.loop import token_weighted_loss  # isort: skip
from bhaskera.trainer.optim import build_scheduler  # isort: skip
from bhaskera.evaluation.validation import _chunked_causal_lm_loss_sum  # isort: skip


def lr_curve(total, warmup, offset, steps, min_ratio=0.1, peak=1.0):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=peak)
    cfg = SimpleNamespace(total_steps=total, warmup_steps=warmup, step_offset=offset,
                          min_lr_ratio=min_ratio, max_steps=steps)
    sched = build_scheduler(opt, cfg)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


class TestGlobalSchedule:
    def test_warmup_then_cosine_to_floor(self):
        lrs = lr_curve(total=100, warmup=10, offset=0, steps=100)
        assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)   # linear warmup to peak
        assert max(lrs) == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(0.1, abs=0.01)                         # decays to min_lr_ratio x peak
        assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))        # monotone decay after warmup

    def test_rounds_continue_one_curve_instead_of_restarting(self):
        """4 rounds of 25 steps with offsets must reproduce the single 100-step curve exactly."""
        whole = lr_curve(total=100, warmup=10, offset=0, steps=100)
        parts = sum((lr_curve(total=100, warmup=10, offset=25 * r, steps=25) for r in range(4)), [])
        assert parts == pytest.approx(whole)

    def test_without_total_steps_the_original_per_run_schedule_is_kept(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        p = torch.nn.Parameter(torch.zeros(1))
        sched = build_scheduler(torch.optim.SGD([p], lr=1.0),
                                SimpleNamespace(total_steps=0, warmup_steps=0, max_steps=10))
        assert isinstance(sched, CosineAnnealingLR)


class TestTokenWeightedLoss:
    def test_weights_micro_batches_by_supervised_tokens(self):
        loss, n = token_weighted_loss([torch.tensor(1.0), torch.tensor(3.0)], [10, 30])
        assert n == 40 and loss == pytest.approx((1 * 10 + 3 * 30) / 40)      # 2.5, not the plain mean 2.0

    def test_equals_the_loss_over_the_concatenated_tokens(self):
        torch.manual_seed(0)
        logits = [torch.randn(5, 7), torch.randn(12, 7)]
        labels = [torch.randint(0, 7, (5,)), torch.randint(0, 7, (12,))]
        means = [torch.nn.functional.cross_entropy(a, b) for a, b in zip(logits, labels)]
        joint = torch.nn.functional.cross_entropy(torch.cat(logits), torch.cat(labels))
        loss, _ = token_weighted_loss(means, [5, 12])
        assert loss == pytest.approx(joint.item(), abs=1e-6)

    def test_micro_batch_without_supervised_tokens_contributes_nothing(self):
        loss, n = token_weighted_loss([torch.tensor(2.0), torch.tensor(0.0)], [8, 0])
        assert n == 8 and loss == pytest.approx(2.0)
        assert math.isnan(token_weighted_loss([torch.tensor(0.0)], [0])[0])


class TestValidationSums:
    def test_sum_and_count_are_exact_and_ignore_masked_positions(self):
        torch.manual_seed(1)
        logits = torch.randn(3, 20, 11)
        labels = torch.randint(0, 11, (3, 20))
        labels[0, :7] = -100
        labels[2, 15:] = -100
        total, count = _chunked_causal_lm_loss_sum(lambda s, e: logits[:, s:e], labels, chunk_seq_len=6)
        ref = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1),
                                                ignore_index=-100, reduction="sum")
        assert total.item() == pytest.approx(ref.item(), rel=1e-5)
        assert int(count.item()) == int((labels != -100).sum())

    def test_chunking_does_not_change_the_result(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 33, 9)
        labels = torch.randint(0, 9, (2, 33))
        a = _chunked_causal_lm_loss_sum(lambda s, e: logits[:, s:e], labels, 4)
        b = _chunked_causal_lm_loss_sum(lambda s, e: logits[:, s:e], labels, 64)
        assert a[0].item() == pytest.approx(b[0].item(), rel=1e-6) and a[1] == b[1]


class TestConfigFields:
    def test_new_fields_round_trip_with_safe_defaults(self):
        c = Config.from_dict({})
        assert c.training.total_steps == 0 and c.training.step_offset == 0
        assert c.checkpoint.lora_only is False and c.lora.init_save_path is None
        c2 = Config.from_dict({"training": {"total_steps": 90, "step_offset": 30, "min_lr_ratio": 0.2},
                               "checkpoint": {"lora_only": True}, "lora": {"init_save_path": "/x/y.pth"}})
        assert (c2.training.total_steps, c2.training.step_offset, c2.training.min_lr_ratio) == (90, 30, 0.2)
        assert c2.checkpoint.lora_only and c2.lora.init_save_path == "/x/y.pth"
        assert Config.from_dict(c2.as_dict()).training.total_steps == 90
