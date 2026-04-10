"""
bhaskera.inference.lloyd_max
============================
Lloyd-Max optimal scalar quantizer for the coordinate distribution arising
from random rotation of unit-norm vectors.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate follows a Beta-like distribution well-approximated by
N(0, 1/d) for d >= 64 (the practical range for LLM head dimensions).

We solve the Lloyd-Max conditions (continuous 1-D k-means) to find the
quantizer that minimises MSE for this specific distribution. This is the
theoretically optimal scalar quantizer used in TurboQuant (ICLR 2026).

Reference:
    TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    https://arxiv.org/abs/2504.19874
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: float, sigma2: float) -> float:
    """N(0, sigma2) probability density at x."""
    return math.exp(-x * x / (2.0 * sigma2)) / math.sqrt(2.0 * math.pi * sigma2)


def _beta_pdf(x: float, d: int) -> float:
    """Exact coordinate PDF after random rotation of a d-dim unit vector.

    f(x) ∝ (1 - x²)^((d-3)/2)  supported on [-1, 1].
    For d >= 64, the Gaussian approximation is accurate to < 0.1%.
    """
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1.0 - x * x) ** ((d - 3) / 2)


# ---------------------------------------------------------------------------
# Core Lloyd-Max solver
# ---------------------------------------------------------------------------

def solve_lloyd_max(
    d: int,
    bits: int,
    use_exact_pdf: bool = False,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the Lloyd-Max optimal quantizer for random-rotated coordinates.

    Args:
        d:              Vector dimension (head_dim in LLM context).
        bits:           Quantization bit-width.
        use_exact_pdf:  Use exact Beta PDF (slower); default uses Gaussian
                        approximation which is accurate for d >= 64.
        max_iter:       Maximum Lloyd-Max iterations.
        tol:            Convergence tolerance on centroid shift.

    Returns:
        centroids:   (2^bits,) tensor of optimal reconstruction levels.
        boundaries:  (2^bits - 1,) tensor of decision boundaries.
    """
    # Numerical integration — scipy is optional; fall back to Riemann sum.
    try:
        from scipy import integrate as _sci_integrate
        _has_scipy = True
    except ImportError:
        _has_scipy = False
        logger.warning(
            "scipy not found — using Riemann-sum integration for Lloyd-Max. "
            "Install scipy>=1.10 for higher accuracy: pip install scipy"
        )

    n_levels = 2 ** bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    if use_exact_pdf:
        pdf = lambda x: _beta_pdf(x, d)
    else:
        pdf = lambda x: _gaussian_pdf(x, sigma2)

    # Truncation range: covers 99.99%+ of the distribution
    lo = -4.0 * sigma
    hi = 4.0 * sigma

    def _integrate(fn, a, b):
        """Integrate fn over [a, b] using scipy or a Riemann fallback."""
        if _has_scipy:
            val, _ = _sci_integrate.quad(fn, a, b, limit=200)
            return val
        # Fallback: 1000-point Riemann sum
        n = 1000
        xs = [a + (b - a) * (i + 0.5) / n for i in range(n)]
        dx = (b - a) / n
        return sum(fn(x) * dx for x in xs)

    # Initialise centroids uniformly within [lo, hi]
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for iteration in range(max_iter):
        # Decision boundaries = midpoints between adjacent centroids
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 2] + boundaries + [hi * 2]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator   = _integrate(lambda x: x * pdf(x), a, b)
            denominator = _integrate(pdf, a, b)
            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])  # keep old if region is empty

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            logger.debug(f"Lloyd-Max converged in {iteration + 1} iterations (d={d}, bits={bits})")
            break

    # Final boundaries
    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Codebook (cached per (d, bits) pair)
# ---------------------------------------------------------------------------

class LloydMaxCodebook:
    """Pre-computed Lloyd-Max codebook for a given (dimension, bit-width) pair.

    Codebooks are computed once on first use and cached for the lifetime of
    the process. For a typical LLM with head_dim=128, this takes ~0.5 s on
    CPU and is never repeated.
    """

    # Module-level cache: (d, bits) -> LloydMaxCodebook
    _cache: dict[tuple[int, int], "LloydMaxCodebook"] = {}

    def __init__(self, d: int, bits: int, use_exact_pdf: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits

        logger.debug(f"Computing Lloyd-Max codebook for d={d}, bits={bits}…")
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact_pdf)

    @classmethod
    def get(cls, d: int, bits: int) -> "LloydMaxCodebook":
        """Return a cached codebook, computing it on first use."""
        key = (d, bits)
        if key not in cls._cache:
            cls._cache[key] = cls(d, bits)
        return cls._cache[key]

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map values to nearest centroid indices.

        Args:
            x: (...,) float tensor of values to quantize.

        Returns:
            (...,) int16 tensor of centroid indices.
        """
        centroids = self.centroids.to(x.device)
        diffs = x.unsqueeze(-1) - centroids          # (..., n_levels)
        return diffs.abs().argmin(dim=-1).to(torch.int16)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map centroid indices back to float values.

        Args:
            indices: (...,) int tensor of indices.

        Returns:
            (...,) float32 tensor of reconstructed values.
        """
        centroids = self.centroids.to(indices.device)
        return centroids[indices.long()]

    def expected_distortion(self) -> float:
        """Expected MSE distortion per coordinate (smaller = better)."""
        # Computed as sum over partitions of E[(x - c_i)^2 | x in partition_i]
        sigma2 = 1.0 / self.d
        pdf = lambda x: _gaussian_pdf(x, sigma2)
        try:
            from scipy import integrate as _sci_integrate
            edges = (
                [self.boundaries[0].item() - 10]
                + self.boundaries.tolist()
                + [self.boundaries[-1].item() + 10]
            )
            distortion = 0.0
            for i in range(self.n_levels):
                c = self.centroids[i].item()
                a, b = edges[i], edges[i + 1]
                d_i, _ = _sci_integrate.quad(lambda x: (x - c) ** 2 * pdf(x), a, b)
                distortion += d_i
            return distortion
        except ImportError:
            return float("nan")

    def __repr__(self) -> str:
        return (
            f"LloydMaxCodebook(d={self.d}, bits={self.bits}, "
            f"levels={self.n_levels})"
        )
