"""AI-DNA 2.0 coordinate-safe representation analysis."""

from .representations import CoordinateSystemMismatch, RepresentationBatch, direct_row_cosine
from .relational import linear_cka, pairwise_cosine_rdm, rdm_spearman
from .align import ProcrustesMap, fit_procrustes, retrieval_metrics
from .controls import permutation_test, random_orthogonal

__all__ = [
    "CoordinateSystemMismatch",
    "RepresentationBatch",
    "direct_row_cosine",
    "linear_cka",
    "pairwise_cosine_rdm",
    "rdm_spearman",
    "ProcrustesMap",
    "fit_procrustes",
    "retrieval_metrics",
    "permutation_test",
    "random_orthogonal",
]
