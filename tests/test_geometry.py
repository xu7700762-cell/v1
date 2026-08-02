import numpy as np
import torch

from vestibular_fusion.evaluation.geometry import (
    lorentz_distance,
    minkowski_inner,
    project_lorentz,
)


def test_lorentz_projection_satisfies_hyperboloid_constraint():
    spatial = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float64)
    points = project_lorentz(spatial)
    torch.testing.assert_close(minkowski_inner(points, points), torch.full((2,), -1.0, dtype=torch.float64))


def test_lorentz_distance_matches_acosh_of_negative_inner_product():
    points = project_lorentz(torch.tensor([[0.2, 0.1], [-0.4, 0.3]], dtype=torch.float64))
    actual = lorentz_distance(points[0], points[1])
    expected = torch.acosh(-minkowski_inner(points[0], points[1]))
    torch.testing.assert_close(actual, expected)
    assert np.isfinite(float(actual))
