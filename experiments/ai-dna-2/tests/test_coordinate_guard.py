import numpy as np
import pytest

from ai_dna2.representations import CoordinateSystemMismatch, RepresentationBatch, direct_row_cosine


def test_cross_space_cosine_is_refused_even_when_dimensions_match():
    ids = ["a", "b", "c"]
    x = RepresentationBatch(np.eye(3), ids, "model-a/layer-x")
    y = RepresentationBatch(np.eye(3), ids, "model-b/layer-y")
    with pytest.raises(CoordinateSystemMismatch, match="coordinate-wise comparison refused"):
        direct_row_cosine(x, y)


def test_dimension_truncation_is_refused_in_shared_space():
    ids = ["a", "b", "c"]
    a = RepresentationBatch(np.ones((3, 4)), ids, "shared")
    b = RepresentationBatch(np.ones((3, 5)), ids, "shared")
    with pytest.raises(CoordinateSystemMismatch, match="dimension truncation is forbidden"):
        direct_row_cosine(a, b)


def test_shared_coordinate_cosine_is_allowed():
    ids = ["a", "b", "c"]
    values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    a = RepresentationBatch(values, ids, "shared")
    b = RepresentationBatch(values.copy(), ids, "shared")
    np.testing.assert_allclose(direct_row_cosine(a, b), np.ones(3))
