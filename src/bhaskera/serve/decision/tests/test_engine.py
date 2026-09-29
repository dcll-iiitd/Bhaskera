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
