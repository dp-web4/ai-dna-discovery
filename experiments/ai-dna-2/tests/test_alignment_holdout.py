import numpy as np

from ai_dna2.align import fit_procrustes, retrieval_metrics
from ai_dna2.controls import random_orthogonal
from ai_dna2.representations import RepresentationBatch


def test_procrustes_generalizes_to_unseen_samples():
    rng = np.random.default_rng(101)
    n, d = 240, 10
    x = rng.normal(size=(n, d))
    y = x @ random_orthogonal(d, rng)
    ids = [f"id-{i}" for i in range(n)]
    a = RepresentationBatch(x, ids, "a")
    b = RepresentationBatch(y, ids, "b")
    train = np.arange(180)
    test = np.arange(180, n)
    mapping = fit_procrustes(a, b, train, test)
    metrics = retrieval_metrics(mapping, a, b, test, top_k=5)
    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["mrr"] == 1.0
