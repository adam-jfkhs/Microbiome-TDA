"""Tests for src.tda.sample_features — shared per-sample TDA utilities."""

import numpy as np
import pytest

from src.tda.sample_features import h1_features


def test_h1_features_empty_diagram():
    """Empty diagram should return all zeros."""
    dgm = np.empty((0, 2))
    result = h1_features(dgm)
    assert result == [0, 0.0, 0.0, 0.0, 0.0, 0]


def test_h1_features_single_feature():
    """Single finite feature should have count=1 and zero entropy."""
    dgm = np.array([[0.1, 0.5]])
    result = h1_features(dgm)
    assert result[0] == 1  # count
    assert result[1] == pytest.approx(0.0, abs=1e-6)  # entropy (single feature)
    assert result[2] == pytest.approx(0.4, abs=1e-6)  # total persistence
    assert result[3] == pytest.approx(0.4, abs=1e-6)  # mean lifetime
    assert result[4] == pytest.approx(0.4, abs=1e-6)  # max lifetime
    assert result[5] == 1  # max betti


def test_h1_features_multiple_features():
    """Multiple features should have count > 1 and positive entropy."""
    dgm = np.array([
        [0.1, 0.5],
        [0.2, 0.6],
        [0.3, 0.4],
    ])
    result = h1_features(dgm)
    assert result[0] == 3
    assert result[1] > 0  # entropy should be positive
    assert result[2] > 0  # total persistence


def test_h1_features_infinite_filtered():
    """Infinite-death features should be excluded."""
    dgm = np.array([
        [0.1, 0.5],
        [0.2, np.inf],
    ])
    result = h1_features(dgm)
    assert result[0] == 1  # only the finite one


def test_h1_features_all_infinite():
    """All infinite features should return zeros."""
    dgm = np.array([[0.1, np.inf], [0.2, np.inf]])
    result = h1_features(dgm)
    assert result == [0, 0.0, 0.0, 0.0, 0.0, 0]


def test_h1_features_returns_six_values():
    """Should always return exactly 6 values."""
    dgm = np.array([[0.1, 0.3], [0.2, 0.7]])
    assert len(h1_features(dgm)) == 6
    assert len(h1_features(np.empty((0, 2)))) == 6
