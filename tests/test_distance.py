"""Tests for src.networks.distance — distance matrix computation."""

import numpy as np
import pandas as pd
import pytest

from src.networks.distance import (
    correlation_distance,
    aitchison_distance,
    sample_distance,
)


@pytest.fixture
def corr_df():
    """Symmetric correlation matrix."""
    data = np.array([
        [1.0,  0.8, -0.5],
        [0.8,  1.0,  0.3],
        [-0.5, 0.3,  1.0],
    ])
    labels = ["A", "B", "C"]
    return pd.DataFrame(data, index=labels, columns=labels)


@pytest.fixture
def clr_df():
    """Small CLR-transformed DataFrame."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 3))
    return pd.DataFrame(data, columns=["t1", "t2", "t3"])


def test_correlation_distance_range(corr_df):
    """Distances should be in [0, 1]."""
    dist = correlation_distance(corr_df)
    assert (dist.values >= 0).all()
    assert (dist.values <= 1.0 + 1e-10).all()


def test_correlation_distance_zero_diagonal(corr_df):
    """Diagonal should be exactly zero."""
    dist = correlation_distance(corr_df)
    np.testing.assert_array_equal(np.diag(dist.values), 0.0)


def test_correlation_distance_symmetry(corr_df):
    """Distance matrix should be symmetric."""
    dist = correlation_distance(corr_df)
    np.testing.assert_array_equal(dist.values, dist.values.T)


def test_correlation_distance_strong_corr_is_small(corr_df):
    """Strongly correlated items should have small distance."""
    dist = correlation_distance(corr_df)
    # A-B correlation is 0.8, so distance = 1 - 0.8 = 0.2
    assert dist.loc["A", "B"] == pytest.approx(0.2, abs=1e-10)


def test_correlation_distance_negative_corr(corr_df):
    """Negative correlations should use absolute value."""
    dist = correlation_distance(corr_df)
    # A-C correlation is -0.5, so distance = 1 - 0.5 = 0.5
    assert dist.loc["A", "C"] == pytest.approx(0.5, abs=1e-10)


def test_aitchison_distance_shape(clr_df):
    """Aitchison distance should produce n x n matrix."""
    dist = aitchison_distance(clr_df)
    assert dist.shape == (5, 5)


def test_aitchison_distance_zero_diagonal(clr_df):
    """Diagonal should be zero (distance to self)."""
    dist = aitchison_distance(clr_df)
    np.testing.assert_allclose(np.diag(dist.values), 0.0, atol=1e-10)


def test_aitchison_distance_symmetry(clr_df):
    """Distance matrix should be symmetric."""
    dist = aitchison_distance(clr_df)
    np.testing.assert_allclose(dist.values, dist.values.T, atol=1e-10)


def test_sample_distance_braycurtis():
    """Bray-Curtis distance should be in [0, 1] for non-negative data."""
    df = pd.DataFrame({"A": [10, 0, 5], "B": [0, 20, 15], "C": [5, 5, 5]})
    dist = sample_distance(df, metric="braycurtis")
    assert (dist.values >= -1e-10).all()
    assert (dist.values <= 1.0 + 1e-10).all()
