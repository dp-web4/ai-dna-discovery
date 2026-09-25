"""Coordinate-free relational similarity metrics."""

from __future__ import annotations

import numpy as np

from .representations import RepresentationBatch


def _normalized_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm rows are not supported")
    return x / norms


def pairwise_cosine_rdm(batch: RepresentationBatch) -> np.ndarray:
    """Representational dissimilarity matrix using within-space cosine distance."""

    x = _normalized_rows(batch.values)
    similarity = np.clip(x @ x.T, -1.0, 1.0)
    rdm = 1.0 - similarity
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("RDM must be a square matrix")
    if matrix.shape[0] < 3:
        raise ValueError("at least three samples are required for RDM correlation")
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.linalg.norm(ac) * np.linalg.norm(bc)
    if denom == 0:
        raise ValueError("correlation undefined for constant input")
    return float(np.dot(ac, bc) / denom)


def rdm_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman correlation of the upper triangles of two RDMs."""

    a = _upper_triangle_values(rdm_a)
    b = _upper_triangle_values(rdm_b)
    if len(a) != len(b):
        raise ValueError("RDMs must contain the same number of samples")
    return _pearson(_rankdata(a), _rankdata(b))


def _center_gram(gram: np.ndarray) -> np.ndarray:
    row_mean = gram.mean(axis=1, keepdims=True)
    col_mean = gram.mean(axis=0, keepdims=True)
    grand_mean = gram.mean()
    return gram - row_mean - col_mean + grand_mean


def linear_cka(a: RepresentationBatch, b: RepresentationBatch) -> float:
    """Linear CKA using sample Gram matrices; feature dimensions may differ."""

    if a.sample_ids != b.sample_ids:
        raise ValueError("sample_ids must match in the same order for CKA")
    k = _center_gram(a.values @ a.values.T)
    l = _center_gram(b.values @ b.values.T)
    numerator = float(np.sum(k * l))
    denominator = float(np.linalg.norm(k, "fro") * np.linalg.norm(l, "fro"))
    if denominator == 0:
        raise ValueError("CKA undefined for degenerate representations")
    return numerator / denominator
