"""Pinned GGUF files named in configs/models/gguf.yaml.

`resolve` turns a config's `model.gguf` (a registry name such as `jevos-q8_0`, or a path to a
.gguf file) into an absolute path to verified weights, downloading when asked. `identify`
maps a file hash back to its registry entry, for the engine's metadata and fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .runtime.llama_release import fetch

QUANTS = ("q8_0", "q4_k_m", "bf16")


@dataclass(frozen=True)
class GgufFile:
    quant: str
    file: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str  # the checkpoint the GGUF files were converted from
    repo: str
    revision: str
    license: str
    architecture: str | None  # `general.architecture` of the files, when known
    base_url: str  # directory URL the files are downloaded from
    files: dict[str, GgufFile]  # by quant

    def url(self, quant: str) -> str:
        return f"{self.base_url.rstrip('/')}/{self.files[quant].file}"

    def path(self, quant: str, models_dir) -> Path:
        return Path(models_dir).expanduser() / self.name / self.files[quant].file


def load(path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    models = {}
    for name, entry in (raw.get("models") or {}).items():
        files = {
            quant: GgufFile(quant, str(item["file"]), str(item["sha256"]), int(item["bytes"]))
            for quant, item in entry["files"].items()
        }
        unknown = sorted(set(files) - set(QUANTS))
        if unknown:
            raise ValueError(f"{path}: {name} has unknown quants {unknown}; expected {QUANTS}")
        models[name] = ModelSpec(
            name=name,
            source=str(entry["source"]),
            repo=str(entry["repo"]),
            revision=str(entry["revision"]),
            license=str(entry["license"]),
            architecture=entry.get("architecture"),
            base_url=str(entry["base_url"]),
            files=files,
        )
    return models


def split_name(spec: str) -> tuple[str, str]:
    model, _, quant = spec.rpartition("-")
    if not model or quant not in QUANTS:
        raise ValueError(f"{spec!r} is not '<model>-<quant>' with a quant in {QUANTS}")
    return model, quant


def resolve(spec: str, models: dict[str, ModelSpec], models_dir, download=True, progress=None) -> Path:
    if spec.lower().endswith(".gguf"):
        path = Path(spec).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"GGUF file not found at {path}")
        return path
    model, quant = split_name(spec)
    if model not in models:
        raise ValueError(f"Unknown model {model!r}; the registry has: {', '.join(sorted(models))}")
    entry = models[model]
    if quant not in entry.files:
        raise ValueError(f"{model} has no {quant} file; available: {', '.join(entry.files)}")
    target = entry.path(quant, models_dir)
    if download:
        return fetch(entry.url(quant), target, entry.files[quant].sha256, progress).resolve()
    if not target.is_file():
        raise ValueError(f"{target} is missing; run bhaskera-decision-fetch first")
    return target.resolve()


def identify(sha256: str, models: dict[str, ModelSpec]) -> tuple[ModelSpec, GgufFile] | None:
    for spec in models.values():
        for file in spec.files.values():
            if file.sha256 == sha256:
                return spec, file
    return None
