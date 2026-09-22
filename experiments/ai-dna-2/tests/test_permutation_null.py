import numpy as np

from ai_dna2.controls import permutation_test, random_orthogonal
from ai_dna2.relational import pairwise_cosine_rdm, rdm_spearman
from ai_dna2.representations import RepresentationBatch


def test_identity_pairing_beats_permutation_null():
    rng = np.random.default_rng(11)
    n, d = 80, 12
    x = rng.normal(size=(n, d))
    y = x @ random_orthogonal(d, rng)
    ids = [str(i) for i in range(n)]
    a = RepresentationBatch(x, ids, "a")
    b = RepresentationBatch(y, ids, "b")
    ra = pairwise_cosine_rdm(a)
    rb = pairwise_cosine_rdm(b)
    observed = rdm_spearman(ra, rb)

    result = permutation_test(
        observed,
        lambda p: rdm_spearman(ra, rb[np.ix_(p, p)]),
        n,
        n_permutations=99,
        seed=19,
    )
    assert observed > 0.999999999
    assert result["p_value"] <= 0.02
    assert result["null_mean"] < 0.1
