"""Explicit cross-space alignment and held-out retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .representations import RepresentationBatch


@dataclass(frozen=True)
class ProcrustesMap:
    matrix: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_coordinate_system: str
    target_coordinate_system: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        return (x - self.source_mean) @ self.matrix + self.target_mean


def _validate_indices(n: int, indices: Sequence[int], name: str) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1 or len(idx) == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"{name} contains an out-of-range index")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} contains duplicate indices")
    return idx


def fit_procrustes(
    source: RepresentationBatch,
    target: RepresentationBatch,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int] | None = None,
    *,
    center: bool = True,
) -> ProcrustesMap:
    """Fit a semi-orthogonal map on explicit training anchors only."""

    if source.sample_ids != target.sample_ids:
        raise ValueError("source and target sample_ids must match before alignment")
    train = _validate_indices(source.n_samples, train_indices, "train_indices")
    if holdout_indices is not None:
        holdout = _validate_indices(source.n_samples, holdout_indices, "holdout_indices")
        overlap = np.intersect1d(train, holdout)
        if len(overlap):
            raise ValueError(f"train/holdout leakage detected at indices {overlap.tolist()}")

    x = source.values[train]
    y = target.values[train]
    source_mean = x.mean(axis=0) if center else np.zeros(source.dimension)
    target_mean = y.mean(axis=0) if center else np.zeros(target.dimension)
    xc = x - source_mean
    yc = y - target_mean
    u, _, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    matrix = u @ vt
    return ProcrustesMap(
        matrix=matrix,
        source_mean=source_mean,
        target_mean=target_mean,
        source_coordinate_system=source.coordinate_system,
        target_coordinate_system=target.coordinate_system,
    )


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(an == 0) or np.any(bn == 0):
        raise ValueError("retrieval cosine undefined for zero-norm rows")
    return (a / an) @ (b / bn).T


def retrieval_metrics(
    mapping: ProcrustesMap,
    source: RepresentationBatch,
    target: RepresentationBatch,
    test_indices: Sequence[int],
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """Retrieve each held-out target concept among held-out candidates."""

    if source.sample_ids != target.sample_ids:
        raise ValueError("source and target sample_ids must match for retrieval")
    test = _validate_indices(source.n_samples, test_indices, "test_indices")
    mapped = mapping.transform(source.values[test])
    candidates = target.values[test]
    similarities = _cosine_matrix(mapped, candidates)

    ranks = []
    for row in range(len(test)):
        ordering = np.argsort(-similarities[row], kind="mergesort")
        rank = int(np.where(ordering == row)[0][0]) + 1
        ranks.append(rank)

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    k = max(1, min(int(top_k), len(test)))
    return {
        "top1": float(np.mean(ranks_arr == 1)),
        f"top{k}": float(np.mean(ranks_arr <= k)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
    }
