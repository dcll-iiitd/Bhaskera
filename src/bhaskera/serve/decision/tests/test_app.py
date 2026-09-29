import pytest

from bhaskera.config import Config
from bhaskera.launcher.serve import _build_parser
from bhaskera.serve.app import decision_options


def _cfg(replicas, num_gpus):
    cfg = Config()
    cfg.serve.backend = "decision"
    cfg.serve.num_replicas = replicas
    cfg.serve.ray_actor_options = {"num_gpus": num_gpus}
    return cfg


def test_auto_splits_one_gpu_across_replicas():
    options = decision_options(_cfg(4, "auto"))
    assert options == {
        "num_replicas": 4,
        "ray_actor_options": {"num_gpus": 0.25},
        "max_ongoing_requests": 4,
    }


def test_explicit_gpu_share_is_kept():
    assert decision_options(_cfg(2, 0.5))["ray_actor_options"] == {"num_gpus": 0.5}
    assert decision_options(_cfg(1, 0))["ray_actor_options"] == {"num_gpus": 0}


def test_launcher_accepts_decision_backend_and_gguf():
    args = _build_parser().parse_args(
        ["-c", "x.yaml", "--backend", "decision", "--gguf", "jevos-q4_k_m"]
    )
    assert args.backend == "decision" and args.gguf == "jevos-q4_k_m"
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-c", "x.yaml", "--backend", "nope"])
