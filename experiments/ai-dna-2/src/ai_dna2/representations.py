"""Representation containers with explicit coordinate-system provenance.

The central guardrail is simple: coordinate-wise operations are only legal when
both batches explicitly declare the same coordinate system and dimensionality.
Cross-architecture comparisons should use relational metrics or an explicit
learned alignment instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class CoordinateSystemMismatch(ValueError):
    """Raised when a coordinate-wise comparison has no declared shared basis."""


@dataclass(frozen=True)
class RepresentationBatch:
    """A matrix of representations plus provenance needed for safe comparison."""

    values: np.ndarray
    sample_ids: Sequence[str]
    coordinate_system: str
    model_id: str = "unknown"
    layer_id: str = "unknown"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"values must be a 2D matrix, got shape {values.shape}")
        if values.shape[0] != len(self.sample_ids):
            raise ValueError("sample_ids length must match representation rows")
        if values.shape[0] < 2:
            raise ValueError("at least two samples are required")
        if values.shape[1] < 1:
            raise ValueError("representation dimension must be positive")
        if not np.all(np.isfinite(values)):
            raise ValueError("representation matrix contains non-finite values")
        if not self.coordinate_system:
            raise ValueError("coordinate_system must be a non-empty identifier")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_ids", tuple(str(x) for x in self.sample_ids))

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def dimension(self) -> int:
        return self.values.shape[1]

    def subset(self, indices: Sequence[int]) -> "RepresentationBatch":
        idx = np.asarray(indices, dtype=int)
        return RepresentationBatch(
            values=self.values[idx],
            sample_ids=[self.sample_ids[i] for i in idx],
            coordinate_system=self.coordinate_system,
            model_id=self.model_id,
            layer_id=self.layer_id,
        )


def _require_same_samples(a: RepresentationBatch, b: RepresentationBatch) -> None:
    if a.sample_ids != b.sample_ids:
        raise ValueError("sample_ids must match in the same order for paired comparison")


def _require_shared_coordinates(a: RepresentationBatch, b: RepresentationBatch) -> None:
    if a.coordinate_system != b.coordinate_system:
        raise CoordinateSystemMismatch(
            "coordinate-wise comparison refused: batches declare different coordinate "
            f"systems ({a.coordinate_system!r} vs {b.coordinate_system!r}). Use a "
            "coordinate-free relational metric or fit an explicit alignment."
        )
    if a.dimension != b.dimension:
        raise CoordinateSystemMismatch(
            "coordinate-wise comparison refused: shared-coordinate batches have different "
            f"dimensions ({a.dimension} vs {b.dimension}); dimension truncation is forbidden."
        )


def direct_row_cosine(a: RepresentationBatch, b: RepresentationBatch) -> np.ndarray:
    """Cosine for paired rows only when a shared coordinate basis is declared."""

    _require_same_samples(a, b)
    _require_shared_coordinates(a, b)
    an = np.linalg.norm(a.values, axis=1)
    bn = np.linalg.norm(b.values, axis=1)
    denom = an * bn
    if np.any(denom == 0):
        raise ValueError("cosine is undefined for zero-norm rows")
    return np.sum(a.values * b.values, axis=1) / denom
