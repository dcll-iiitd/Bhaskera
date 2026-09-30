import hashlib
from pathlib import Path

import pytest

from bhaskera.serve.decision import registry

ROOT = Path(__file__).resolve().parents[5]
REGISTRY = ROOT / "configs" / "models" / "gguf.yaml"


def test_repo_registry_pins_jevos_and_gemma():
    models = registry.load(REGISTRY)
    jevos = models["jevos"]
    assert jevos.files["q8_0"].sha256 == (
        "c9f412118808004f6c15fab38a4fc94e940c8d5e74f5675974fddf822b598683"
    )
    assert jevos.url("q4_k_m") == (
        "https://github.com/feder-cr/jev/releases/download/jevos-v2/jevos-v2-q4_k_m.gguf"
    )
    assert set(models["gemma-4-e4b"].files) == {"q8_0", "q4_k_m"}


def test_split_name():
    assert registry.split_name("jevos-q8_0") == ("jevos", "q8_0")
    assert registry.split_name("gemma-4-e4b-q4_k_m") == ("gemma-4-e4b", "q4_k_m")
    with pytest.raises(ValueError):
        registry.split_name("jevos")


def test_resolve_a_path(tmp_path):
    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"x")
    assert registry.resolve(str(weights), {}, tmp_path) == weights.resolve()
    with pytest.raises(ValueError, match="not found"):
        registry.resolve(str(tmp_path / "nope.gguf"), {}, tmp_path)


def test_resolve_a_name_downloads_into_models_dir(tmp_path):
    served = tmp_path / "server"
    served.mkdir()
    (served / "tiny-q8_0.gguf").write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    spec = registry.ModelSpec(
        "tiny", "src", "repo", "rev", "mit", None, served.as_uri(),
        {"q8_0": registry.GgufFile("q8_0", "tiny-q8_0.gguf", digest, 7)},
    )
    models = {"tiny": spec}
    path = registry.resolve("tiny-q8_0", models, tmp_path / "models")
    assert path == (tmp_path / "models" / "tiny" / "tiny-q8_0.gguf").resolve()
    assert path.read_bytes() == b"weights"
    assert registry.resolve("tiny-q8_0", models, tmp_path / "models", download=False) == path
    assert registry.identify(digest, models) == (spec, spec.files["q8_0"])
    assert registry.identify("0" * 64, models) is None


def test_resolve_rejects_unknown_names(tmp_path):
    models = registry.load(REGISTRY)
    with pytest.raises(ValueError, match="Unknown model"):
        registry.resolve("nope-q8_0", models, tmp_path, download=False)
    with pytest.raises(ValueError, match="no bf16 file"):
        registry.resolve("jevos-bf16", models, tmp_path, download=False)
    with pytest.raises(ValueError, match="bhaskera-decision-fetch"):
        registry.resolve("jevos-q8_0", models, tmp_path, download=False)
