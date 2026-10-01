import math
import os
from pathlib import Path

import pytest

from bhaskera.config import Config
from bhaskera.serve.decision import loader
from bhaskera.serve.decision.calibration import Calibration
from bhaskera.serve.decision.engine import Engine
from bhaskera.serve.decision.runtime import llama_release
from bhaskera.serve.decision.tests.fakes import FakeBackend

REGISTRY = Path(__file__).resolve().parents[5] / "configs" / "models" / "gguf.yaml"


def _boolean(qid, instructions="Is this a billing problem?", state="I was charged twice."):
    return {"state": state, "questions": {qid: {"type": "boolean", "instructions": instructions}}}


def test_binary_engine_answers_yes_no():
    backend = FakeBackend({"billing": [0.0, math.log(9.0)]})
    result = Engine(backend, binary=True).decide(_boolean("billing"))
    assert result["answers"]["billing"]["noul"] == pytest.approx(0.9)
    assert backend.calls[0][1] == ["billing"]
    assert result["prefix_tokens"] > 0


def test_binary_engine_refuses_choice():
    engine = Engine(FakeBackend({}), binary=True)
    request = {
        "state": "s",
        "questions": {
            "c": {
                "type": "choice",
                "instructions": "Which?",
                "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
            }
        },
    }
    with pytest.raises(ValueError, match="only yes/no"):
        engine.decide(request)


def test_calibration_must_match_the_fingerprint():
    calibration = Calibration(
        fingerprint="other", dataset_sha256="x", temperatures={"boolean": 2.0}, fit_metrics={}
    )
    with pytest.raises(ValueError, match="different model"):
        Engine(FakeBackend({}), calibration=calibration, binary=True)


def test_calibration_temperature_is_applied():
    calibration = Calibration(
        fingerprint="fp", dataset_sha256="x", temperatures={"boolean": 2.0}, fit_metrics={}
    )
    engine = Engine(FakeBackend({"q": [0.0, 2.0]}), calibration=calibration, binary=True)
    noul = engine.decide(_boolean("q"))["answers"]["q"]["noul"]
    assert noul == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_provision_makes_every_path_absolute(monkeypatch, tmp_path):
    monkeypatch.delenv(llama_release.CACHE_ENV, raising=False)
    cfg = Config()
    cfg.serve.backend = "decision"
    cfg.model.gguf = "jevos-q8_0"
    cfg.serve.decision.registry = str(REGISTRY)
    cfg.serve.decision.models_dir = str(tmp_path / "models")
    cfg.serve.decision.llama_cpp.cache_dir = str(tmp_path / "runtimes")
    calibration = tmp_path / "cal.json"
    calibration.write_text("{}")
    cfg.serve.decision.calibration = str(calibration)
    calls = {}

    def fake_resolve(spec, models, models_dir, download, progress):
        calls["resolve"] = (spec, "jevos" in models, str(models_dir), download)
        return tmp_path / "w.gguf"

    def fake_install(accelerator, progress):
        calls["install"] = accelerator
        return tmp_path / "runtimes" / "lib"

    monkeypatch.setattr(loader.registry, "resolve", fake_resolve)
    monkeypatch.setattr(loader.llama_release, "install", fake_install)
    loader.provision(cfg)
    assert calls["resolve"] == ("jevos-q8_0", True, str(tmp_path / "models"), True)
    assert calls["install"] == "cuda"
    assert cfg.model.gguf == str(tmp_path / "w.gguf")
    assert cfg.serve.decision.llama_cpp.runtime_dir == str(tmp_path / "runtimes" / "lib")
    assert cfg.serve.decision.calibration == str(calibration.resolve())
    assert os.environ[llama_release.CACHE_ENV] == str(tmp_path / "runtimes")


def test_provision_requires_model_gguf():
    cfg = Config()
    cfg.serve.backend = "decision"
    with pytest.raises(ValueError, match="model.gguf"):
        loader.provision(cfg)


def _relative_cfg(tmp_path, monkeypatch):
    monkeypatch.delenv(llama_release.CACHE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.serve.backend = "decision"
    cfg.model.gguf = "jevos-q8_0"
    cfg.serve.decision.registry = str(REGISTRY)
    cfg.serve.decision.models_dir = str(tmp_path / "models")
    monkeypatch.setattr(loader.registry, "resolve", lambda *a, **k: tmp_path / "w.gguf")
    return cfg


def test_provision_resolves_relative_cache_dir(monkeypatch, tmp_path):
    cfg = _relative_cfg(tmp_path, monkeypatch)
    cfg.serve.decision.llama_cpp.cache_dir = "runtimes"
    monkeypatch.setattr(loader.llama_release, "install", lambda accelerator, progress: Path("runtimes/lib"))
    loader.provision(cfg)
    runtime_dir = cfg.serve.decision.llama_cpp.runtime_dir
    assert os.path.isabs(runtime_dir)
    assert runtime_dir == str((tmp_path / "runtimes" / "lib").resolve())
    assert os.path.isabs(os.environ[llama_release.CACHE_ENV])


def test_provision_explicit_runtime_dir_becomes_absolute(monkeypatch, tmp_path):
    cfg = _relative_cfg(tmp_path, monkeypatch)
    (tmp_path / "rt").mkdir()
    (tmp_path / "rt" / llama_release.library_name()).write_text("")
    cfg.serve.decision.llama_cpp.runtime_dir = "rt"
    loader.provision(cfg)
    assert cfg.serve.decision.llama_cpp.runtime_dir == str((tmp_path / "rt").resolve())


def test_provision_explicit_runtime_dir_without_library_raises(monkeypatch, tmp_path):
    cfg = _relative_cfg(tmp_path, monkeypatch)
    (tmp_path / "empty").mkdir()
    cfg.serve.decision.llama_cpp.runtime_dir = "empty"
    with pytest.raises(ValueError, match="not found"):
        loader.provision(cfg)


def test_decide_many_isolates_a_bad_request():
    backend = FakeBackend({"a": [0.0, math.log(4.0)], "b": [math.log(4.0), 0.0]})
    backend.max_requests = 4
    engine = Engine(backend, binary=True)
    bad = {
        "state": "s2",
        "questions": {
            "c": {
                "type": "choice",
                "instructions": "C?",
                "options": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}],
            }
        },
    }
    out = engine.decide_many([_boolean("a", state="s1"), bad, _boolean("b", state="s3")])
    assert out[0]["answers"]["a"]["noul"] == pytest.approx(0.8)
    assert isinstance(out[1], ValueError)
    assert out[2]["answers"]["b"]["noul"] == pytest.approx(0.2)
    assert backend.many_calls == [2]
    assert out[0]["timing"]["batch_requests"] == 2


def test_decide_many_model_matches_decide():
    backend = FakeBackend({"a": [0.0, math.log(4.0)]})
    engine = Engine(backend, binary=True)
    request = _boolean("a")
    assert engine.decide_many([request])[0]["model"] == engine.decide(request)["model"]


def test_decide_many_chunks_to_the_backend_capacity():
    backend = FakeBackend({q: [0.0, math.log(4.0)] for q in "abc"})
    backend.max_requests = 2
    engine = Engine(backend, binary=True)
    out = engine.decide_many([_boolean(q, state=f"s{q}") for q in "abc"])
    assert all(isinstance(o, dict) for o in out)
    assert backend.many_calls == [2, 1]


def test_decide_many_runtime_fault_fails_only_its_chunk():
    from bhaskera.serve.decision.runtime.llama_cpp import LlamaRuntimeError

    backend = FakeBackend({q: [0.0, math.log(4.0)] for q in "abc"})
    backend.max_requests = 2
    real = backend.score_many

    def flaky(requests):
        if not backend.many_calls:
            backend.many_calls.append(len(requests))
            raise LlamaRuntimeError("llama_decode returned -1: compute error")
        return real(requests)

    backend.score_many = flaky
    engine = Engine(backend, binary=True)
    out = engine.decide_many([_boolean(q, state=f"s{q}") for q in "abc"])
    assert isinstance(out[0], LlamaRuntimeError) and isinstance(out[1], LlamaRuntimeError)
    assert not isinstance(out[0], ValueError)
    assert isinstance(out[2], dict)


def test_load_engine_uses_the_configured_branch_even_when_batching(monkeypatch, tmp_path):
    from bhaskera.config import Config
    from bhaskera.serve.decision import backend as backend_module
    from bhaskera.serve.decision import loader

    seen = {}

    def fake_load(path, **kwargs):
        seen.update(kwargs)
        backend = FakeBackend({})
        backend.metadata["branch_strategy"] = "state-restore"
        return backend

    monkeypatch.setattr(backend_module.LlamaBackend, "load", staticmethod(fake_load))
    cfg = Config()
    cfg.model.gguf = "x.gguf"
    cfg.serve.decision.registry = "configs/models/gguf.yaml"
    cfg.serve.decision.batching.enabled = True
    cfg.serve.decision.batching.max_batch_size = 6
    loader.load_engine(cfg)
    assert seen["branch"] == cfg.serve.decision.branch == "auto"
    assert seen["max_requests"] == 6
