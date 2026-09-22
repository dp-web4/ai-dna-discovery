import numpy as np

from ai_dna2.align import fit_procrustes, retrieval_metrics
from ai_dna2.controls import random_orthogonal
from ai_dna2.relational import linear_cka, pairwise_cosine_rdm, rdm_spearman
from ai_dna2.representations import RepresentationBatch


def test_rotation_destroys_coordinates_but_preserves_relations_and_retrieval():
    rng = np.random.default_rng(7)
    n, d = 320, 16
    x = rng.normal(size=(n, d))
    q = random_orthogonal(d, rng)
    y = x @ q
    ids = [f"c{i}" for i in range(n)]
    a = RepresentationBatch(x, ids, "space-a")
    b = RepresentationBatch(y, ids, "space-b")

    naive = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    assert np.mean(np.abs(naive)) < 0.35
    assert rdm_spearman(pairwise_cosine_rdm(a), pairwise_cosine_rdm(b)) > 0.999999999
    assert linear_cka(a, b) > 0.999999999

    train = np.arange(0, 256)
    test = np.arange(256, n)
    mapping = fit_procrustes(a, b, train, test)
    metrics = retrieval_metrics(mapping, a, b, test)
    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0
