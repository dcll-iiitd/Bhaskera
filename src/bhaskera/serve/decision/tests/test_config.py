from pathlib import Path

from bhaskera.config import Config, _dict_to_config, load_config

ROOT = Path(__file__).resolve().parents[5]


def test_decision_config_defaults():
    cfg = Config()
    decision = cfg.serve.decision
    assert cfg.model.gguf is None
    assert decision.registry == "configs/models/gguf.yaml"
    assert decision.ctx == 8192 and decision.batch_size == 4 and decision.branch == "auto"
    assert decision.batching.enabled is False
    assert decision.llama_cpp.accelerator == "cuda" and decision.llama_cpp.runtime_dir is None


def test_decision_config_parses_and_round_trips():
    raw = {
        "model": {"name": "jevos", "gguf": "jevos-q8_0"},
        "serve": {
            "backend": "decision",
            "num_replicas": 4,
            "ray_actor_options": {"num_gpus": "auto"},
            "decision": {
                "ctx": 4096,
                "batch_size": 8,
                "branch": "seq-copy",
                "batching": {"enabled": True, "max_batch_size": 16},
                "llama_cpp": {"accelerator": "cuda12"},
            },
        },
    }
    cfg = _dict_to_config(raw)
    decision = cfg.serve.decision
    assert cfg.model.gguf == "jevos-q8_0"
    assert cfg.serve.ray_actor_options == {"num_gpus": "auto"}
    assert (decision.ctx, decision.batch_size, decision.branch) == (4096, 8, "seq-copy")
    assert decision.batching.enabled and decision.batching.max_batch_size == 16
    assert decision.batching.batch_wait_timeout_s == 0.005
    assert decision.llama_cpp.accelerator == "cuda12"
    assert Config.from_dict(cfg.as_dict()) == cfg


def test_shipped_decision_configs_load():
    jevos = load_config(str(ROOT / "configs" / "serve_jevos.yaml"))
    assert jevos.serve.backend == "decision" and jevos.model.gguf == "jevos-q8_0"
    assert jevos.serve.gateway.cloudflared is False
    gemma = load_config(str(ROOT / "configs" / "serve_gemma_decision.yaml"))
    assert gemma.model.gguf == "gemma-4-e4b-q8_0"


def test_shipped_decision_configs_bind_loopback():
    for name in ("serve_jevos", "serve_gemma_decision"):
        assert load_config(str(ROOT / "configs" / f"{name}.yaml")).serve.host == "127.0.0.1"
