"""Weights and runtime for the decision backend.

`provision` runs once on the driver, before any replica starts: it downloads and verifies the
GGUF and the pinned llama.cpp runtime and rewrites the config with absolute paths, so replicas
(which may start in another working directory) only load. `load_engine` runs in each replica.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from . import registry
from .runtime import llama_release

if TYPE_CHECKING:
    from bhaskera.config import Config

    from .engine import Engine


def provision(cfg: "Config", progress=None) -> None:
    decision = cfg.serve.decision
    if not cfg.model.gguf:
        raise ValueError("serve.backend=decision needs model.gguf (a registry name or a .gguf path)")
    decision.registry = str(Path(decision.registry).expanduser().resolve())
    models = registry.load(decision.registry)
    cfg.model.gguf = str(
        registry.resolve(cfg.model.gguf, models, decision.models_dir, download=True, progress=progress)
    )
    if decision.calibration:
        decision.calibration = str(Path(decision.calibration).expanduser().resolve())
    runtime = decision.llama_cpp
    if runtime.runtime_dir is None:
        os.environ[llama_release.CACHE_ENV] = str(Path(runtime.cache_dir).expanduser().resolve())
        runtime.runtime_dir = str(Path(llama_release.install(runtime.accelerator, progress)).resolve())
    else:
        library = llama_release.find_library(Path(runtime.runtime_dir).expanduser())
        if library is None:
            raise ValueError(
                f"serve.decision.llama_cpp.runtime_dir={runtime.runtime_dir}: "
                f"{llama_release.library_name()} not found there"
            )
        runtime.runtime_dir = str(library.parent.resolve())


def load_engine(cfg: "Config") -> "Engine":
    from .backend import LlamaBackend
    from .calibration import Calibration
    from .engine import Engine

    decision = cfg.serve.decision
    models = registry.load(decision.registry)
    backend = LlamaBackend.load(
        cfg.model.gguf,
        device=decision.device,
        ctx=decision.ctx,
        batch_size=decision.batch_size,
        prefill_chunk=decision.prefill_chunk,
        runtime_dir=decision.llama_cpp.runtime_dir,
        branch=decision.branch,
        identify=lambda sha256: registry.identify(sha256, models),
    )
    calibration = Calibration.from_file(Path(decision.calibration)) if decision.calibration else None
    # A GGUF that names a binary prompt version (jevos) answers yes/no questions only.
    binary = bool(backend.metadata.get("binary_prompt_version"))
    return Engine(backend, ctx=decision.ctx, calibration=calibration, binary=binary)
