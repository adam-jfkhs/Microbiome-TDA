"""Tests for src.tda.filtration — distance matrix preparation."""

import numpy as np
import pandas as pd
import pytest

from src.tda.filtration import prepare_distance_matrix, select_filtration_range


@pytest.fixture
def dist_df():
    """Small distance matrix with a deliberate asymmetry and NaN."""
    data = np.array([
        [0.0, 0.3, np.nan],
        [0.31, 0.0, 0.5],
        [0.4, 0.5, 0.0],
    ])
    labels = ["A", "B", "C"]
    return pd.DataFrame(data, index=labels, columns=labels)


def test_prepare_symmetry(dist_df):
    """Output should be symmetric."""
    result = prepare_distance_matrix(dist_df)
    np.testing.assert_allclose(result, result.T, atol=1e-10)


def test_prepare_zero_diagonal(dist_df):
    """Diagonal should be exactly zero."""
    result = prepare_distance_matrix(dist_df)
    np.testing.assert_array_equal(np.diag(result), 0.0)


def test_prepare_no_nan(dist_df):
    """Output should contain no NaN values."""
    result = prepare_distance_matrix(dist_df)
    assert not np.any(np.isnan(result))


def test_prepare_nan_replaced_with_max(dist_df):
    """NaN entries should be replaced with the maximum distance."""
    result = prepare_distance_matrix(dist_df)
    # The NaN was at (0,2), should be replaced and then symmetrised
    assert result[0, 2] > 0


def test_prepare_returns_numpy(dist_df):
    """Output should be a numpy array, not a DataFrame."""
    result = prepare_distance_matrix(dist_df)
    assert isinstance(result, np.ndarray)


def test_prepare_already_clean():
    """A clean symmetric matrix should pass through unchanged."""
    data = np.array([[0.0, 0.5], [0.5, 0.0]])
    df = pd.DataFrame(data)
    result = prepare_distance_matrix(df)
    np.testing.assert_allclose(result, data)


def test_select_filtration_range_order():
    """Low threshold should be less than high threshold."""
    dist = np.array([
        [0.0, 0.1, 0.9],
        [0.1, 0.0, 0.5],
        [0.9, 0.5, 0.0],
    ])
    low, high = select_filtration_range(dist, percentiles=(10, 90))
    assert low < high


def test_select_filtration_range_within_data():
    """Thresholds should be within the range of actual distances."""
    dist = np.array([
        [0.0, 0.2, 0.8],
        [0.2, 0.0, 0.6],
        [0.8, 0.6, 0.0],
    ])
    low, high = select_filtration_range(dist)
    assert low >= 0.2 - 0.01
    assert high <= 0.8 + 0.01
