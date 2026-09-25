import numpy as np
import pytest

from ai_dna2.align import fit_procrustes
from ai_dna2.representations import RepresentationBatch


def test_alignment_refuses_train_holdout_overlap():
    rng = np.random.default_rng(2)
    ids = [str(i) for i in range(20)]
    a = RepresentationBatch(rng.normal(size=(20, 4)), ids, "a")
    b = RepresentationBatch(rng.normal(size=(20, 4)), ids, "b")
    with pytest.raises(ValueError, match="leakage"):
        fit_procrustes(a, b, train_indices=range(12), holdout_indices=range(10, 20))
