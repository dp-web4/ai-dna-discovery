"""Null controls and reproducible random transforms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def permutation_test(
    observed: float,
    statistic_for_permutation: Callable[[np.ndarray], float],
    n_samples: int,
    *,
    n_permutations: int = 999,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, float | np.ndarray]:
    """Permutation test over sample identities."""

    if n_samples < 3:
        raise ValueError("at least three samples are required")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        null[i] = statistic_for_permutation(rng.permutation(n_samples))

    if alternative == "greater":
        exceed = int(np.sum(null >= observed))
    elif alternative == "less":
        exceed = int(np.sum(null <= observed))
    else:
        center = float(np.mean(null))
        exceed = int(np.sum(np.abs(null - center) >= abs(observed - center)))

    p_value = (exceed + 1) / (n_permutations + 1)
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null": null,
    }
