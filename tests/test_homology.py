"""Tests for persistent homology computation."""

import numpy as np
import pytest

from src.tda.homology import compute_persistence, filter_infinite, persistence_summary


@pytest.fixture
def sample_distance_matrix():
    """Create a small symmetric distance matrix for testing."""
    np.random.seed(42)
    n = 10
    pts = np.random.rand(n, 3)
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(pts))


def test_compute_persistence_returns_diagrams(sample_distance_matrix):
    result = compute_persistence(sample_distance_matrix, maxdim=1)
    assert "dgms" in result
    assert len(result["dgms"]) >= 2  # H0 and H1


def test_compute_persistence_h0_count(sample_distance_matrix):
    result = compute_persistence(sample_distance_matrix, maxdim=0)
    # H0 should have n features (one per point)
    assert len(result["dgms"][0]) == sample_distance_matrix.shape[0]


def test_filter_infinite():
    dgm = np.array([[0.0, 0.5], [0.1, np.inf], [0.2, 0.8]])
    filtered = filter_infinite([dgm])
    assert len(filtered[0]) == 2
    assert np.all(np.isfinite(filtered[0]))


def test_persistence_summary():
    dgms = [
        np.array([[0.0, 0.5], [0.1, 0.3]]),
        np.array([[0.2, 0.8], [0.3, np.inf]]),
    ]
    summary = persistence_summary(dgms)
    assert "H0" in summary
    assert "H1" in summary
    assert summary["H0"]["count"] == 2
    assert summary["H0"]["mean_lifetime"] == pytest.approx(0.35)
