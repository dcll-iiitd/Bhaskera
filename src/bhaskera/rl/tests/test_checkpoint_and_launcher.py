"""Checkpoint layout and the real `run()` entry point, end to end on a tiny saved model."""
from __future__ import annotations

import json
import os
import re

import pytest
import torch

from bhaskera.config import Config
from bhaskera.distributed.checkpoint import _STEP_RE, _dcp_load, save_checkpoint
from bhaskera.launcher import rl_train
from bhaskera.rl.rewards import register_reward

from .helpers import TARGETS, make_base_model, make_lora_model


def dir_size(path) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def ml_engine_style_load(model, ckpt_dir) -> dict:
    """What Slakshna's ml_engine.py does: allocate zeros for LoRA keys, DCP-load them."""
    sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items() if "lora" in k}
    _dcp_load({"model": sd}, os.path.join(ckpt_dir, "model"))
    return sd


class TestCheckpoint:
    def _trained(self, tokenizer):
        model = make_lora_model(tokenizer)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-1)
        ids = torch.tensor([[3, 5, 6, 7]])
        model(input_ids=ids).logits.sum().backward()
        opt.step()
        return model, opt

    def test_lora_only_layout_and_roundtrip(self, tokenizer, tmp_path):
        model, opt = self._trained(tokenizer)
        path = str(tmp_path / "step_0000001")
        save_checkpoint(model, opt, 1, path, lora_only=True)
        assert os.path.exists(os.path.join(path, ".complete"))
        assert os.path.isdir(os.path.join(path, "model"))
        assert os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        assert _STEP_RE.search(os.path.basename(path))

        fresh = make_lora_model(tokenizer, seed=3)            # different weights, same architecture
        loaded = ml_engine_style_load(fresh, path)
        trained = {k: v for k, v in model.state_dict().items() if "lora" in k}
        assert loaded.keys() == trained.keys() and loaded
        for k in trained:
            assert torch.equal(loaded[k], trained[k]), k

    def test_lora_only_is_smaller_and_excludes_base_weights(self, tokenizer, tmp_path):
        from torch.distributed.checkpoint import FileSystemReader

        model, opt = self._trained(tokenizer)
        save_checkpoint(model, opt, 1, str(tmp_path / "lora" / "step_0000001"), lora_only=True)
        save_checkpoint(model, opt, 1, str(tmp_path / "full" / "step_0000001"), lora_only=False)
        lora_dir = tmp_path / "lora" / "step_0000001" / "model"
        full_dir = tmp_path / "full" / "step_0000001" / "model"

        def keys(path):
            return set(FileSystemReader(str(path)).read_metadata().state_dict_metadata)

        assert keys(lora_dir) and all("lora_" in k for k in keys(lora_dir))
        assert any("lora_" not in k for k in keys(full_dir))          # full keeps base weights
        assert {k for k in keys(full_dir) if "lora_" in k} == keys(lora_dir)
        assert dir_size(lora_dir) < dir_size(full_dir)


# ---------------------------------------------------------------------------
# run() end to end
# ---------------------------------------------------------------------------

@register_reward("count_sevens_e2e")
def _count_sevens(completion: str, answer: str) -> float:
    return completion.split().count("7") / 6.0


@pytest.fixture()
def workspace(tokenizer, tmp_path):
    model_dir = tmp_path / "model"
    make_base_model(tokenizer).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    data = tmp_path / "prompts.jsonl"
    rows = [{"prompt": p, "answer": "x"} for p in ("Q 1", "Q 2 3", "Q 9")]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def make_cfg(ws, **rl_over) -> Config:
    raw = {
        "model": {"name": str(ws / "model"), "dtype": "float32", "use_liger_kernel": False},
        "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0, "target_modules": TARGETS},
        "training": {"max_steps": 40, "lr": 3e-3, "seed": 0, "warmup_steps": 0, "grad_clip": 1.0,
                     "distributed": {"strategy": "fsdp"}},   # RL must override this itself
        "checkpoint": {"enabled": True, "save_dir": str(ws / "ckpt"), "save_interval": 20,
                       "keep_last_n": 5},
        "rl": {"train_path": str(ws / "prompts.jsonl"), "eval_path": str(ws / "prompts.jsonl"),
               "reward": "count_sevens_e2e", "group_size": 8, "prompts_per_step": 2,
               "max_new_tokens": 6, "kl_coef": 0.04, "micro_batch_size": 8, **rl_over},
    }
    return Config.from_dict(raw)


STEP_LINE = re.compile(
    r"\[epoch\s+(\d+)\]\[step\s+(\d+)\]\s+loss=(-?[0-9.]+(?:e-?\d+)?)"
    r".*?samples=(\d+)\s+tokens=(\d+)"
)


def test_run_trains_checkpoints_and_reports(workspace, capsys):
    result = rl_train.run(make_cfg(workspace))
    out = capsys.readouterr().out
    steps = [STEP_LINE.search(line) for line in out.splitlines() if line.startswith("[epoch")]
    assert len(steps) == 40 and all(steps)
    assert [int(m.group(2)) for m in steps] == list(range(1, 41))
    samples = [int(m.group(4)) for m in steps]
    assert samples == [16 * i for i in range(1, 41)]                  # 2 prompts x 8 completions
    assert result["final_eval_reward"] > result["initial_eval_reward"] + 0.2
    ckpts = sorted(p.name for p in (workspace / "ckpt").iterdir())
    assert ckpts == ["step_0000020", "step_0000040"]
    for c in ckpts:
        assert (workspace / "ckpt" / c / ".complete").exists()


def test_resume_from_aggregated_adapter_continues_training(workspace, tokenizer, capsys):
    """The federated loop restarts every round from the aggregated adapter (lora.resume_path)."""
    first = rl_train.run(make_cfg(workspace))
    capsys.readouterr()
    final_ckpt = str(workspace / "ckpt" / "step_0000040")
    template = make_lora_model(tokenizer)
    # DCP key names (with the `.default.` adapter segment), exactly what ml_engine saves.
    resume = ml_engine_style_load(template, final_ckpt)
    resume_file = workspace / "base_lora.pth"
    torch.save(resume, resume_file)

    cfg = make_cfg(workspace)
    cfg.lora.resume_path = str(resume_file)
    cfg.training.max_steps = 0
    cfg.checkpoint.save_dir = str(workspace / "ckpt2")
    second = rl_train.run(cfg)
    assert second["initial_eval_reward"] == pytest.approx(first["final_eval_reward"], abs=1e-6)
    assert second["initial_eval_reward"] > first["initial_eval_reward"]   # resumed, not restarted


def test_resume_samples_continues_counters_and_prompt_stream(workspace, capsys):
    cfg = make_cfg(workspace)
    cfg.training.max_steps = 2
    rl_train.run(cfg, resume_samples=96)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[epoch")]
    first = STEP_LINE.search(lines[0])
    assert first.group(4) == str(96 + 16)           # sample counter continues from the offset
    # 96 completions / group size 8 = 12 prompts already used; the dataset has 3, so this
    # step draws prompts 12 and 13, both from pass 4 (13 // 3).
    assert first.group(1) == "4"


def test_validation_errors(workspace):
    cfg = make_cfg(workspace)
    cfg.lora.enabled = False
    with pytest.raises(ValueError, match="lora"):
        rl_train.run(cfg)
    cfg = make_cfg(workspace)
    cfg.rl.train_path = None
    with pytest.raises(ValueError, match="train_path"):
        rl_train.run(cfg)
    cfg = make_cfg(workspace)
    cfg.rl.group_size = 1
    with pytest.raises(ValueError, match="group_size"):
        rl_train.run(cfg)


def test_config_round_trips_through_dict(workspace):
    cfg = make_cfg(workspace, group_size=6)
    assert Config.from_dict(cfg.as_dict()).rl == cfg.rl
    assert cfg.rl.group_size == 6


def test_lora_rank_alias_is_honoured():
    """node_template.yaml spells it `rank`; Bhaskera's own configs use `r`. Both must work."""
    assert Config.from_dict({"lora": {"rank": 32}}).lora.r == 32
    assert Config.from_dict({"lora": {"r": 8, "rank": 32}}).lora.r == 8
    assert Config.from_dict({}).lora.r == 16


def test_plugin_optimizers_are_loaded(workspace, monkeypatch):
    """ml_engine can select a plugin optimizer (muon, galore); the RL launcher must load it."""
    (workspace / "rl_test_plugin_opt.py").write_text(
        "import torch\n"
        "from bhaskera.trainer.optimizer_registry import register_optimizer\n"
        "CALLS = []\n"
        "@register_optimizer('rl_test_sgd')\n"
        "def build(model, train_cfg):\n"
        "    CALLS.append(1)\n"
        "    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)\n"
    )
    monkeypatch.syspath_prepend(str(workspace))
    cfg = make_cfg(workspace)
    cfg.training.max_steps = 2
    cfg.plugins.optimizers = ["rl_test_plugin_opt"]
    cfg.training.optimizer.backend = "plugin"
    cfg.training.optimizer.name = "rl_test_sgd"
    rl_train.run(cfg)
    import rl_test_plugin_opt
    assert rl_test_plugin_opt.CALLS == [1]


def test_evaluation_cadence_and_batch_size(workspace, capsys):
    cfg = make_cfg(workspace, eval_every=10, eval_batch_size=2)
    cfg.training.max_steps = 20
    rl_train.run(cfg)
    evals = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[eval]")]
    steps = [int(re.search(r"step (\d+)", ln).group(1)) for ln in evals]
    assert steps == [0, 10, 20]      # start, every 10, and the end exactly once
