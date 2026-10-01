import asyncio

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


def test_systemone_route_reads_json_body_not_query():
    from bhaskera.serve import decision_deployment as dd
    from bhaskera.serve.decision import wire

    route = dd._decision_app.routes[-1].effective_candidates()[0]
    assert route.path == "/v1/systemone"
    assert [p.field_info.annotation for p in route.dependant.body_params] == [wire.SystemOneRequest]
    assert route.dependant.query_params == []


def test_route_annotations_survive_pickling_by_value():
    """Ray ships the class to replicas by value.  String annotations would then lose their
    globals (bytecode never references them), and FastAPI would read ``body``/``response``
    as query parameters."""
    import typing

    import ray.cloudpickle as cloudpickle

    from bhaskera.serve import decision_deployment as dd
    from bhaskera.serve.decision import wire

    cloudpickle.register_pickle_by_value(dd)
    try:
        clone = cloudpickle.loads(cloudpickle.dumps(dd.DecisionDeployment.func_or_class.systemone))
    finally:
        cloudpickle.unregister_pickle_by_value(dd)
    assert typing.get_type_hints(clone)["body"] is wire.SystemOneRequest


def test_batching_raises_max_ongoing_requests():
    cfg = _cfg(2, "auto")
    cfg.serve.decision.batching.enabled = True
    cfg.serve.decision.batching.max_batch_size = 16
    assert decision_options(cfg)["max_ongoing_requests"] == 16


def test_error_handlers_map_value_error_to_422_and_runtime_fault_to_500():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bhaskera.serve import decision_deployment as dd
    from bhaskera.serve.decision.runtime.llama_cpp import LlamaRuntimeError

    app = FastAPI()
    for exc, handler in dd._decision_app.exception_handlers.items():
        app.add_exception_handler(exc, handler)

    @app.get("/bad")
    async def bad():
        raise ValueError("prompt too long")

    @app.get("/fault")
    async def fault():
        raise LlamaRuntimeError("llama_decode returned -1")

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/bad").status_code == 422
    response = client.get("/fault")
    assert response.status_code == 500
    assert response.json() == {"detail": "decision backend error: llama_decode returned -1"}


class _Backend:
    def __init__(self, branch, delta):
        self.branch = branch
        self.metadata = {
            "model": "jevos", "precision": "q8_0", "branch_strategy": branch,
            "probe_branch_max_delta": delta,
        }


class _Engine:
    def __init__(self, branch, delta=0.0):
        self.backend = _Backend(branch, delta)


def _replica(monkeypatch, branch, delta):
    from bhaskera.serve import decision_deployment as dd
    from bhaskera.serve.decision import loader

    monkeypatch.setattr(loader, "load_engine", lambda cfg: _Engine(branch, delta))
    cfg = _cfg(1, 0)
    cfg.serve.decision.batching.enabled = True
    cls = dd.DecisionDeployment.func_or_class
    replica = object.__new__(cls)
    asyncio.run(cls.__init__(replica, cfg))  # serve.ingress wraps __init__ as a coroutine
    return replica


@pytest.mark.filterwarnings("ignore:coroutine .*never awaited:RuntimeWarning")
def test_batching_stays_on_for_a_verified_seq_copy_backend(monkeypatch):
    assert _replica(monkeypatch, "seq-copy", 0.0)._batching is True


@pytest.mark.filterwarnings("ignore:coroutine .*never awaited:RuntimeWarning")
def test_batching_is_disabled_when_the_probe_fell_back_to_state_restore(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        replica = _replica(monkeypatch, "state-restore", None)
    assert replica._batching is False
    assert "batching disabled" in caplog.text


@pytest.mark.filterwarnings("ignore:coroutine .*never awaited:RuntimeWarning")
def test_batching_is_disabled_when_the_branch_probe_drifts(monkeypatch):
    assert _replica(monkeypatch, "seq-copy", 0.2)._batching is False
