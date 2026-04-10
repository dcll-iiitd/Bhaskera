"""
Tests for bhaskera.inference — no GPU or real model needed.

Run with:
    pytest src/bhaskera/inference/tests/test_inference.py -v
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F


# ===========================================================================
# Lloyd-Max quantizer
# ===========================================================================

class TestLloydMaxCodebook:
    """Validate the Lloyd-Max codebook against theoretical bounds."""

    def test_codebook_shape(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb = LloydMaxCodebook.get(d=128, bits=4)
        assert cb.centroids.shape == (16,), "16 centroids for 4-bit codebook"
        assert cb.boundaries.shape == (15,), "15 boundaries for 4-bit codebook"

    def test_centroids_ordered(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb = LloydMaxCodebook.get(d=128, bits=4)
        diffs = cb.centroids[1:] - cb.centroids[:-1]
        assert (diffs > 0).all(), "Centroids must be strictly increasing"

    def test_round_trip_accuracy(self):
        """Quantize→dequantize should have low MSE on N(0,1/d) samples."""
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        d, bits = 128, 4
        cb = LloydMaxCodebook.get(d=d, bits=bits)

        torch.manual_seed(0)
        sigma = 1.0 / math.sqrt(d)
        x = torch.randn(10_000) * sigma

        idx     = cb.quantize(x)
        x_hat   = cb.dequantize(idx)
        mse     = ((x - x_hat) ** 2).mean().item()
        # Theoretical bound for Gaussian: σ² * 2^(-2b) * π²/3
        upper   = sigma**2 * (2 ** (-2 * bits)) * (math.pi**2 / 3) * 4
        assert mse < upper, f"MSE {mse:.2e} exceeds theoretical bound {upper:.2e}"

    def test_cache_reuse(self):
        """Same (d, bits) should return the same object."""
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        a = LloydMaxCodebook.get(128, 4)
        b = LloydMaxCodebook.get(128, 4)
        assert a is b, "Codebooks should be cached and reused"

    def test_different_bitwidths(self):
        """2-bit and 8-bit codebooks have different number of levels."""
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb2 = LloydMaxCodebook.get(128, 2)
        cb8 = LloydMaxCodebook.get(128, 8)
        assert cb2.n_levels == 4
        assert cb8.n_levels == 256


# ===========================================================================
# Sampling utilities
# ===========================================================================

class TestSampling:
    def test_temperature_scale_identity(self):
        from bhaskera.inference.sampling import temperature_scale
        logits = torch.randn(4, 100)
        out = temperature_scale(logits, 1.0)
        assert torch.allclose(out, logits), "temperature=1.0 should be identity"

    def test_temperature_scale_effect(self):
        from bhaskera.inference.sampling import temperature_scale
        logits = torch.tensor([[1.0, 2.0, 4.0]])
        hot = temperature_scale(logits, 2.0)
        cold = temperature_scale(logits, 0.5)
        # Hotter → flatter probabilities; colder → more peaked
        hot_probs  = F.softmax(hot, dim=-1)
        cold_probs = F.softmax(cold, dim=-1)
        assert hot_probs.std() < cold_probs.std()

    def test_top_k_zeroes_others(self):
        from bhaskera.inference.sampling import top_k_filter
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = top_k_filter(logits, top_k=2)
        probs = F.softmax(out, dim=-1)
        # Only top-2 should have non-negligible probability
        assert (probs[0, :3] < 1e-6).all(), "Bottom 3 should be ~0"
        assert probs[0, 3:].sum() > 0.99

    def test_top_k_no_filter_zero(self):
        from bhaskera.inference.sampling import top_k_filter
        logits = torch.randn(2, 50)
        assert torch.allclose(top_k_filter(logits, 0), logits)

    def test_top_p_reduces_mass(self):
        from bhaskera.inference.sampling import top_p_filter
        torch.manual_seed(1)
        logits = torch.randn(1, 1000)
        filtered = top_p_filter(logits, top_p=0.9)
        probs = F.softmax(filtered, dim=-1)
        # Number of tokens with non-negligible prob should be much less than 1000
        active = (probs > 1e-8).sum().item()
        assert active < 1000, "top_p should have zeroed some logits"

    def test_top_p_identity_at_one(self):
        from bhaskera.inference.sampling import top_p_filter
        logits = torch.randn(2, 100)
        assert torch.allclose(top_p_filter(logits, 1.0), logits)

    def test_greedy_sample_argmax(self):
        from bhaskera.inference.sampling import greedy_sample
        logits = torch.tensor([[0.1, 5.0, 0.3], [1.0, 0.5, 0.2]])
        out = greedy_sample(logits)
        assert out[0].item() == 1
        assert out[1].item() == 0

    def test_sample_from_logits_shape(self):
        from bhaskera.inference.sampling import sample_from_logits
        logits = torch.randn(4, 50257)
        out = sample_from_logits(logits, temperature=0.7, top_p=0.9, top_k=50)
        assert out.shape == (4,), "Should return (batch,) token ids"
        assert out.dtype in (torch.int64, torch.long)

    def test_greedy_no_sample(self):
        from bhaskera.inference.sampling import sample_from_logits
        logits = torch.tensor([[0.1, 9.0, 0.3]])
        out = sample_from_logits(logits, do_sample=False)
        assert out.item() == 1, "Greedy should always pick argmax"


# ===========================================================================
# KV cache — StaticKVCache
# ===========================================================================

class TestStaticKVCache:
    def _make_cache(self, **kwargs):
        from bhaskera.inference.kv_cache import StaticKVCache
        defaults = dict(
            num_layers=4, batch_size=2, num_heads=8,
            head_dim=64, max_seq_len=512,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        defaults.update(kwargs)
        return StaticKVCache(**defaults)

    def test_update_shape(self):
        cache = self._make_cache()
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        full_k, full_v = cache.update(key_states=k, value_states=v, layer_idx=0)
        assert full_k.shape == (2, 8, 10, 64)
        assert full_v.shape == (2, 8, 10, 64)

    def test_update_accumulates(self):
        cache = self._make_cache()
        k1 = torch.randn(2, 8, 5, 64)
        v1 = torch.randn(2, 8, 5, 64)
        k2 = torch.randn(2, 8, 3, 64)
        v2 = torch.randn(2, 8, 3, 64)
        cache.update(key_states=k1, value_states=v1, layer_idx=0)
        cache.advance(5)
        full_k, full_v = cache.update(key_states=k2, value_states=v2, layer_idx=0)
        assert full_k.shape[2] == 8, "Should accumulate 5+3=8 tokens"

    def test_reset_clears(self):
        cache = self._make_cache()
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        cache.update(key_states=k, value_states=v, layer_idx=0)
        cache.advance(10)
        cache.reset()
        assert cache.seq_len == 0

    def test_memory_bytes_positive(self):
        cache = self._make_cache()
        assert cache.memory_bytes() > 0

    def test_overflow_raises(self):
        cache = self._make_cache(max_seq_len=8)
        k = torch.randn(2, 8, 10, 64)   # 10 > max 8
        v = torch.randn(2, 8, 10, 64)
        with pytest.raises(ValueError, match="overflow"):
            cache.update(key_states=k, value_states=v, layer_idx=0)


# ===========================================================================
# KV cache — TurboQuantKVCache
# ===========================================================================

class TestTurboQuantKVCache:
    def _make_cache(self, **kwargs):
        from bhaskera.inference.kv_cache import TurboQuantKVCache
        defaults = dict(
            num_layers=4, batch_size=1, num_heads=4,
            head_dim=64, key_bits=4, value_bits=2,
            residual_window=8, protected_layers=1,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        defaults.update(kwargs)
        return TurboQuantKVCache(**defaults)

    def test_update_returns_correct_shape(self):
        cache = self._make_cache()
        k = torch.randn(1, 4, 5, 64)
        v = torch.randn(1, 4, 5, 64)
        full_k, full_v = cache.update(key_states=k, value_states=v, layer_idx=0)
        assert full_k.shape == (1, 4, 5, 64)
        assert full_v.shape == (1, 4, 5, 64)

    def test_reconstruction_quality_k4v2(self):
        """Round-trip cosine similarity should be > 0.95 with K4/V2."""
        cache = self._make_cache(key_bits=4, value_bits=2, residual_window=0)
        torch.manual_seed(7)
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn(1, 4, 32, 64)

        # Store and immediately evict window → forces compression
        cache.update(key_states=k, value_states=v, layer_idx=0)
        # Force eviction by making residual window smaller than stored tokens
        store = cache._stores[0]
        # Manually compress what's in the window
        if store._fp16_keys is not None:
            evict_k = store._fp16_keys
            evict_v = store._fp16_values
            store.compress_and_store(evict_k, evict_v)
            store._fp16_keys = None
            store._fp16_values = None

        recon_k, recon_v = store.reconstruct_all(1, 4, torch.float32)

        # Flatten for cosine similarity
        k_flat     = k.reshape(-1, 64)
        recon_flat = recon_k.reshape(-1, 64)

        cos_sim = F.cosine_similarity(k_flat, recon_flat, dim=-1).mean().item()
        assert cos_sim > 0.92, f"Cosine similarity {cos_sim:.4f} too low for K4 quantization"

    def test_reset_clears(self):
        cache = self._make_cache()
        k = torch.randn(1, 4, 5, 64)
        v = torch.randn(1, 4, 5, 64)
        cache.update(key_states=k, value_states=v, layer_idx=0)
        cache.reset()
        assert cache.seq_len == 0

    def test_memory_grows_with_tokens(self):
        cache = self._make_cache(residual_window=2)
        k = torch.randn(1, 4, 10, 64)
        v = torch.randn(1, 4, 10, 64)
        cache.update(key_states=k, value_states=v, layer_idx=0)
        assert cache.memory_bytes() > 0


# ===========================================================================
# Rotation matrix
# ===========================================================================

class TestRotationMatrix:
    def test_orthogonal(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R = _generate_rotation_matrix(64, seed=0)
        I = R @ R.T
        assert torch.allclose(I, torch.eye(64), atol=1e-5), "R @ R^T should be identity"

    def test_deterministic(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R1 = _generate_rotation_matrix(64, seed=42)
        R2 = _generate_rotation_matrix(64, seed=42)
        assert torch.allclose(R1, R2), "Same seed should yield same rotation"

    def test_different_seeds_differ(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R1 = _generate_rotation_matrix(64, seed=1)
        R2 = _generate_rotation_matrix(64, seed=2)
        assert not torch.allclose(R1, R2)


# ===========================================================================
# build_kv_cache factory
# ===========================================================================

class TestBuildKVCache:
    _base_kwargs = dict(
        num_layers=2, batch_size=1, num_heads=4,
        head_dim=32, max_seq_len=128,
        dtype=torch.float32, device=torch.device("cpu"),
    )

    def test_static(self):
        from bhaskera.inference.kv_cache import build_kv_cache, StaticKVCache
        cache = build_kv_cache(strategy="static", tq_cfg=None, **self._base_kwargs)
        assert isinstance(cache, StaticKVCache)

    def test_none(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        cache = build_kv_cache(strategy="none", tq_cfg=None, **self._base_kwargs)
        assert cache is None

    def test_turboquant_requires_cfg(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        with pytest.raises(ValueError):
            build_kv_cache(strategy="turboquant", tq_cfg=None, **self._base_kwargs)

    def test_turboquant_with_cfg(self):
        from bhaskera.inference.kv_cache import build_kv_cache, TurboQuantKVCache
        from bhaskera.config import TurboQuantConfig
        tq = TurboQuantConfig(key_bits=4, value_bits=2, residual_window=16)
        cache = build_kv_cache(strategy="turboquant", tq_cfg=tq, **self._base_kwargs)
        assert isinstance(cache, TurboQuantKVCache)

    def test_unknown_strategy(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        with pytest.raises(ValueError):
            build_kv_cache(strategy="invalid", tq_cfg=None, **self._base_kwargs)


# ===========================================================================
# Config loading
# ===========================================================================

class TestInferenceConfig:
    def test_defaults(self):
        from bhaskera.config import Config
        cfg = Config()
        assert cfg.inference.kv_cache == "static"
        assert cfg.inference.max_new_tokens == 512
        assert cfg.inference.temperature == 1.0
        assert cfg.inference.turboquant.key_bits == 4
        assert cfg.inference.turboquant.value_bits == 2
        assert cfg.inference.speculative.enabled is False

    def test_yaml_override(self, tmp_path):
        import yaml
        from bhaskera.config import load_config
        cfg_dict = {
            "inference": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "kv_cache": "turboquant",
                "turboquant": {"key_bits": 3, "value_bits": 2, "residual_window": 64},
            }
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(cfg_dict))
        cfg = load_config(str(p))
        assert cfg.inference.max_new_tokens == 256
        assert cfg.inference.temperature == 0.7
        assert cfg.inference.kv_cache == "turboquant"
        assert cfg.inference.turboquant.key_bits == 3
        assert cfg.inference.turboquant.residual_window == 64
