"""Tests for topological feature extraction."""

import numpy as np
import pytest

from src.tda.features import betti_curve, persistence_entropy, persistence_landscape


@pytest.fixture
def sample_diagram():
    """Create a sample persistence diagram."""
    return np.array([
        [0.0, 0.5],
        [0.1, 0.8],
        [0.2, 0.3],
        [0.4, 0.9],
    ])


def test_betti_curve_shape(sample_diagram):
    filt_vals, betti = betti_curve(sample_diagram, num_points=50)
    assert len(filt_vals) == 50
    assert len(betti) == 50


def test_betti_curve_nonnegative(sample_diagram):
    _, betti = betti_curve(sample_diagram)
    assert np.all(betti >= 0)


def test_betti_curve_empty():
    empty_dgm = np.empty((0, 2))
    _, betti = betti_curve(empty_dgm)
    assert np.all(betti == 0)


def test_persistence_entropy_positive(sample_diagram):
    entropy = persistence_entropy(sample_diagram)
    assert entropy > 0


def test_persistence_entropy_empty():
    empty_dgm = np.empty((0, 2))
    entropy = persistence_entropy(empty_dgm)
    assert entropy == 0.0


def test_persistence_landscape_shape(sample_diagram):
    landscapes = persistence_landscape(sample_diagram, num_landscapes=3, num_points=50)
    assert landscapes.shape == (3, 50)


def test_persistence_landscape_ordered(sample_diagram):
    """First landscape should dominate subsequent ones."""
    landscapes = persistence_landscape(sample_diagram, num_landscapes=3)
    # At each point, landscape k should be >= landscape k+1
    for k in range(landscapes.shape[0] - 1):
        assert np.all(landscapes[k] >= landscapes[k + 1] - 1e-10)
