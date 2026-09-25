#!/usr/bin/env python3
"""Run the AI-DNA 2.0 Phase A synthetic rotation sanity check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_dna2.align import fit_procrustes, retrieval_metrics
from ai_dna2.controls import permutation_test, random_orthogonal
from ai_dna2.relational import linear_cka, pairwise_cosine_rdm, rdm_spearman
from ai_dna2.representations import CoordinateSystemMismatch, RepresentationBatch, direct_row_cosine


def main() -> int:
    rng = np.random.default_rng(1337)
    n_samples = 256
    dimension = 24
    x = rng.normal(size=(n_samples, dimension))
    q = random_orthogonal(dimension, rng)
    y = x @ q
    ids = [f"anchor-{i:04d}" for i in range(n_samples)]

    source = RepresentationBatch(x, ids, "synthetic/source", "synthetic-A", "layer-0")
    rotated = RepresentationBatch(y, ids, "synthetic/rotated", "synthetic-B", "layer-0")

    naive = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    guard_raised = False
    try:
        direct_row_cosine(source, rotated)
    except CoordinateSystemMismatch:
        guard_raised = True

    rdm_source = pairwise_cosine_rdm(source)
    rdm_rotated = pairwise_cosine_rdm(rotated)
    rdm_corr = rdm_spearman(rdm_source, rdm_rotated)
    cka = linear_cka(source, rotated)

    order = rng.permutation(n_samples)
    split = int(n_samples * 0.75)
    train = order[:split]
    test = order[split:]
    mapping = fit_procrustes(source, rotated, train, test)
    retrieval = retrieval_metrics(mapping, source, rotated, test)

    perm = permutation_test(
        rdm_corr,
        lambda p: rdm_spearman(rdm_source, rdm_rotated[np.ix_(p, p)]),
        n_samples,
        n_permutations=199,
        seed=2026,
    )

    result = {
        "seed": 1337,
        "n_samples": n_samples,
        "dimension": dimension,
        "naive_coordinate_cosine_mean": float(np.mean(naive)),
        "naive_coordinate_cosine_abs_mean": float(np.mean(np.abs(naive))),
        "coordinate_guard_raised": guard_raised,
        "rdm_spearman": rdm_corr,
        "linear_cka": cka,
        "heldout_retrieval": retrieval,
        "permutation_test": {
            "p_value": perm["p_value"],
            "null_mean": perm["null_mean"],
            "null_std": perm["null_std"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    passed = (
        guard_raised
        and abs(rdm_corr - 1.0) < 1e-10
        and abs(cka - 1.0) < 1e-10
        and retrieval["top1"] == 1.0
        and perm["p_value"] <= 0.01
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
