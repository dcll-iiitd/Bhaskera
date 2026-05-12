"""
Tests for bhaskera.inference — no GPU or real model needed.

Run with:
    pytest src/bhaskera/inference/tests/test_inference.py -v
"""
from __future__ import annotations

import math
import time
import sys
import types
import pytest
import torch
import torch.nn.functional as F


# ===========================================================================
# Lloyd-Max codebook
# ===========================================================================

class TestLloydMaxCodebook:
    def test_codebook_shape(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb = LloydMaxCodebook.get(d=128, bits=4)
        assert cb.centroids.shape == (16,)
        assert cb.boundaries.shape == (15,)

    def test_centroids_ordered(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb = LloydMaxCodebook.get(d=128, bits=4)
        assert (cb.centroids[1:] - cb.centroids[:-1] > 0).all()

    def test_round_trip_accuracy(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        d, bits = 128, 4
        cb = LloydMaxCodebook.get(d=d, bits=bits)
        torch.manual_seed(0)
        sigma = 1.0 / math.sqrt(d)
        x = torch.randn(10_000) * sigma
        idx   = cb.quantize(x)
        x_hat = cb.dequantize(idx)
        mse   = ((x - x_hat) ** 2).mean().item()
        upper = sigma**2 * (2 ** (-2 * bits)) * (math.pi**2 / 3) * 4
        assert mse < upper

    def test_cache_reuse(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        assert LloydMaxCodebook.get(128, 4) is LloydMaxCodebook.get(128, 4)

    def test_different_bitwidths(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        assert LloydMaxCodebook.get(128, 2).n_levels == 4
        assert LloydMaxCodebook.get(128, 8).n_levels == 256


# ===========================================================================
# FastLloydMaxCodebook (bucketize path)
# ===========================================================================

class TestFastLloydMaxCodebook:
    def test_matches_slow_codebook(self):
        from bhaskera.inference.kv_cache import FastLloydMaxCodebook
        from bhaskera.inference.lloyd_max import LloydMaxCodebook

        device = torch.device("cpu")
        cb_fast = FastLloydMaxCodebook.get(64, 4, device)
        cb_slow = LloydMaxCodebook.get(64, 4)

        torch.manual_seed(1)
        x = torch.randn(500) * (1.0 / math.sqrt(64))
        x = x.clamp(cb_slow.centroids[0].item() + 1e-4,
                    cb_slow.centroids[-1].item() - 1e-4)

        idx_fast = cb_fast.quantize(x.unsqueeze(-1)).squeeze(-1)
        idx_slow = cb_slow.quantize(x)
        assert (idx_fast == idx_slow).all()

    def test_round_trip_quality(self):
        from bhaskera.inference.kv_cache import FastLloydMaxCodebook
        device = torch.device("cpu")
        cb = FastLloydMaxCodebook.get(64, 4, device)
        x = torch.randn(1000, 64) * (1.0 / math.sqrt(64))
        idx = cb.quantize(x)
        x_hat = cb.dequantize(idx)
        assert x_hat.shape == x.shape


# ===========================================================================
# Sampling utilities
# ===========================================================================

class TestSampling:
    def test_temperature_scale_identity(self):
        from bhaskera.inference.sampling import temperature_scale
        logits = torch.randn(4, 100)
        assert torch.allclose(temperature_scale(logits, 1.0), logits)

    def test_top_k_zeroes_others(self):
        from bhaskera.inference.sampling import top_k_filter
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        probs = F.softmax(top_k_filter(logits, top_k=2), dim=-1)
        assert (probs[0, :3] < 1e-6).all()

    def test_greedy_no_sample(self):
        from bhaskera.inference.sampling import sample_from_logits
        logits = torch.tensor([[0.1, 9.0, 0.3]])
        assert sample_from_logits(logits, do_sample=False).item() == 1


# ===========================================================================
# StaticKVCache
# ===========================================================================

class TestStaticKVCache:
    def _make(self, **kw):
        from bhaskera.inference.kv_cache import StaticKVCache
        d = dict(num_layers=4, batch_size=2, num_heads=8, head_dim=64,
                 max_seq_len=512, dtype=torch.float32, device=torch.device("cpu"))
        d.update(kw)
        return StaticKVCache(**d)

    def test_update_shape(self):
        c = self._make()
        k = torch.randn(2, 8, 10, 64)
        fk, fv = c.update(k, torch.randn_like(k), 0)
        assert fk.shape == (2, 8, 10, 64)

    def test_reset(self):
        c = self._make()
        c.update(torch.randn(2,8,5,64), torch.randn(2,8,5,64), 0)
        c.advance(5)
        c.reset()
        assert c.seq_len == 0

    def test_overflow_raises(self):
        c = self._make(max_seq_len=8)
        with pytest.raises(ValueError, match="overflow"):
            c.update(torch.randn(2,8,10,64), torch.randn(2,8,10,64), 0)


# ===========================================================================
# TurboQuantKVCache
# ===========================================================================

class TestTurboQuantKVCache:
    def _make(self, **kw):
        from bhaskera.inference.kv_cache import TurboQuantKVCache
        d = dict(num_layers=4, batch_size=1, num_heads=4, head_dim=64,
                 key_bits=4, value_bits=2, residual_window=8, protected_layers=1,
                 dtype=torch.float32, device=torch.device("cpu"), max_seq_len=256)
        d.update(kw)
        return TurboQuantKVCache(**d)

    def test_update_returns_correct_shape(self):
        c = self._make()
        k = torch.randn(1, 4, 5, 64)
        fk, fv = c.update(k, torch.randn_like(k), 0)
        assert fk.shape == (1, 4, 5, 64)

    def test_reset(self):
        c = self._make()
        c.update(torch.randn(1,4,5,64), torch.randn(1,4,5,64), 0)
        c.reset()
        assert c.seq_len == 0

    def test_reconstruction_quality_k4v2(self):
        c = self._make(key_bits=4, value_bits=2, residual_window=0)
        torch.manual_seed(7)
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn_like(k)
        fk, fv = c.update(k, v, 0)
        cos_sim = F.cosine_similarity(
            k.reshape(-1, 64), fk.reshape(-1, 64), dim=-1
        ).mean().item()
        assert cos_sim > 0.92, f"Cosine similarity {cos_sim:.4f} too low"

    def test_incremental_matches_full(self):
        from bhaskera.inference.kv_cache import TurboQuantKVCache
        torch.manual_seed(42)

        c_batch = TurboQuantKVCache(
            num_layers=1, batch_size=1, num_heads=2, head_dim=32,
            key_bits=4, value_bits=2, residual_window=4, protected_layers=0,
            max_seq_len=64, dtype=torch.float32, device=torch.device("cpu"),
        )
        k_all = torch.randn(1, 2, 20, 32)
        v_all = torch.randn_like(k_all)
        fk_batch, _ = c_batch.update(k_all, v_all, layer_idx=0)

        c_incr = TurboQuantKVCache(
            num_layers=1, batch_size=1, num_heads=2, head_dim=32,
            key_bits=4, value_bits=2, residual_window=4, protected_layers=0,
            max_seq_len=64, dtype=torch.float32, device=torch.device("cpu"),
        )
        for t in range(20):
            fk_incr, _ = c_incr.update(
                k_all[:, :, t:t+1, :], v_all[:, :, t:t+1, :], layer_idx=0
            )

        assert fk_incr.shape[2] == fk_batch.shape[2]
        cos_batch = F.cosine_similarity(
            fk_batch.reshape(-1, 32), k_all.reshape(-1, 32), dim=-1
        ).mean().item()
        cos_incr = F.cosine_similarity(
            fk_incr.reshape(-1, 32), k_all.reshape(-1, 32), dim=-1
        ).mean().item()
        assert abs(cos_batch - cos_incr) < 0.05

    def test_amortised_O1_per_step(self):
        from bhaskera.inference.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            num_layers=1, batch_size=1, num_heads=4, head_dim=64,
            key_bits=4, value_bits=2, residual_window=16, protected_layers=0,
            max_seq_len=512, dtype=torch.float32, device=torch.device("cpu"),
        )

        for _ in range(10):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        cache.advance(10)

        t0 = time.perf_counter()
        for _ in range(10):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        t_short = (time.perf_counter() - t0) / 10

        for _ in range(180):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        cache.advance(200)

        t0 = time.perf_counter()
        for _ in range(10):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        t_long = (time.perf_counter() - t0) / 10

        ratio = t_long / t_short
        assert ratio < 5.0, (
            f"Step time grew {ratio:.1f}× — O(n²) regression detected "
            f"({t_short*1000:.1f}ms → {t_long*1000:.1f}ms)"
        )


# ===========================================================================
# Rotation matrix
# ===========================================================================

class TestRotationMatrix:
    def test_orthogonal(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R = _generate_rotation_matrix(64, seed=0)
        assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-5)

    def test_deterministic(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R1 = _generate_rotation_matrix(64, seed=42)
        R2 = _generate_rotation_matrix(64, seed=42)
        assert torch.allclose(R1, R2)


# ===========================================================================
# build_kv_cache factory
# ===========================================================================

class TestBuildKVCache:
    _base = dict(num_layers=2, batch_size=1, num_heads=4, head_dim=32,
                 max_seq_len=128, dtype=torch.float32, device=torch.device("cpu"))

    def test_static(self):
        from bhaskera.inference.kv_cache import build_kv_cache, StaticKVCache
        assert isinstance(build_kv_cache(strategy="static", tq_cfg=None, **self._base), StaticKVCache)

    def test_none(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        assert build_kv_cache(strategy="none", tq_cfg=None, **self._base) is None

    def test_turboquant_requires_cfg(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        with pytest.raises(ValueError):
            build_kv_cache(strategy="turboquant", tq_cfg=None, **self._base)

    def test_turboquant_with_cfg(self):
        from bhaskera.inference.kv_cache import build_kv_cache, TurboQuantKVCache
        from bhaskera.config import TurboQuantConfig
        tq = TurboQuantConfig(key_bits=4, value_bits=2, residual_window=16)
        assert isinstance(
            build_kv_cache(strategy="turboquant", tq_cfg=tq, **self._base),
            TurboQuantKVCache,
        )

    def test_unknown_strategy(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        with pytest.raises(ValueError):
            build_kv_cache(strategy="invalid", tq_cfg=None, **self._base)


# ===========================================================================
# Config
# ===========================================================================

class TestInferenceConfig:
    def test_defaults(self):
        from bhaskera.config import Config
        cfg = Config()
        assert cfg.inference.kv_cache == "static"
        assert cfg.inference.turboquant.key_bits == 4
        assert cfg.inference.turboquant.value_bits == 2

    def test_yaml_override(self, tmp_path):
        import yaml
        from bhaskera.config import load_config
        d = {"inference": {"max_new_tokens": 256, "kv_cache": "turboquant",
                           "turboquant": {"key_bits": 3, "residual_window": 64}}}
        p = tmp_path / "t.yaml"
        p.write_text(yaml.dump(d))
        cfg = load_config(str(p))
        assert cfg.inference.max_new_tokens == 256
        assert cfg.inference.turboquant.key_bits == 3


# ===========================================================================
# Ray+vLLM backend — unit tests (no real Ray/vLLM needed)
# ===========================================================================

class TestRayInit:
    """Test _ensure_ray_initialized behaviour without touching a real cluster."""

    def test_skips_when_already_initialised(self, monkeypatch):
        fake_ray = types.ModuleType("ray")
        fake_ray.is_initialized = lambda: True
        fake_ray.init = lambda **kw: (_ for _ in ()).throw(
            AssertionError("ray.init should not be called if already initialised")
        )
        fake_ray.available_resources = lambda: {}
        monkeypatch.setitem(sys.modules, "ray", fake_ray)

        from bhaskera.inference.backends import ray_vllm
        # Should not raise
        ray_vllm._ensure_ray_initialized(1)

    def test_uses_ray_address_env_var(self, monkeypatch):
        calls = []

        fake_ray = types.ModuleType("ray")
        fake_ray.is_initialized = lambda: False
        fake_ray.available_resources = lambda: {}

        def _fake_init(**kw):
            calls.append(kw)

        fake_ray.init = _fake_init
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")

        from bhaskera.inference.backends import ray_vllm
        ray_vllm._ensure_ray_initialized(2)

        assert any(kw.get("address") == "10.0.0.1:6379" for kw in calls)


class TestEngineBackendSelection:
    """Test that InferenceEngine picks the right backend."""

    def test_forces_hf_via_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BHASKERA_BACKEND", "hf")

        loaded = []

        class _FakeHF:
            def __init__(self, *a, **kw): loaded.append("hf")
            def generate(self, **kw): return ["ok"]

        monkeypatch.setattr(
            "bhaskera.inference.engine._vllm_available", lambda: True
        )

        import importlib
        import bhaskera.inference.engine as engine_mod
        original_load_hf = engine_mod.InferenceEngine._load_hf

        def _patched_load_hf(self):
            loaded.append("hf")
            self._backend = _FakeHF()
            self._backend_name = "hf"
            self._loaded = True

        monkeypatch.setattr(engine_mod.InferenceEngine, "_load_hf", _patched_load_hf)

        from bhaskera.config import Config
        engine = engine_mod.InferenceEngine(Config())
        engine.load()

        assert "hf" in loaded
        assert engine._backend_name == "hf"

    def test_warns_on_turboquant_with_vllm(self, monkeypatch, caplog):
        import logging
        import bhaskera.inference.engine as engine_mod

        def _noop_load_vllm(self):
            self._backend = object()
            self._backend_name = "ray_vllm"
            self._loaded = True

        monkeypatch.setattr(engine_mod.InferenceEngine, "_load_vllm", _noop_load_vllm)
        monkeypatch.setattr(engine_mod, "_vllm_available", lambda: True)
        monkeypatch.setenv("BHASKERA_BACKEND", "vllm")

        from bhaskera.config import Config
        cfg = Config()
        cfg.inference.kv_cache = "turboquant"

        import torch
        monkeypatch.setattr(
            "bhaskera.inference.engine._resolve_device",
            lambda _: torch.device("cuda"),
        )

        with caplog.at_level(logging.WARNING, logger="bhaskera.inference.engine"):
            e = engine_mod.InferenceEngine(cfg)
            e.load()

        assert any("turboquant" in r.message.lower() for r in caplog.records)