"""Tests for enhanced statistics module."""

import numpy as np
import pytest

from src.analysis.statistics import (
    permutation_test, cohens_d, fdr_correction,
    compare_topological_features,
)


def test_permutation_test_detects_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(5, 1, 100)
    b = rng.normal(6, 1, 100)
    obs, pval = permutation_test(a, b, n_permutations=5000, seed=42)
    assert pval < 0.05
    assert obs < 0  # a mean < b mean


def test_permutation_test_null():
    rng = np.random.default_rng(42)
    a = rng.normal(5, 1, 50)
    b = rng.normal(5, 1, 50)
    obs, pval = permutation_test(a, b, n_permutations=5000, seed=42)
    assert pval > 0.05


def test_cohens_d_direction():
    a = np.array([10, 11, 12, 13])
    b = np.array([5, 6, 7, 8])
    d = cohens_d(a, b)
    assert d > 0  # a > b
    d2 = cohens_d(b, a)
    assert d2 < 0


def test_cohens_d_zero():
    a = np.array([5, 5, 5])
    d = cohens_d(a, a)
    assert d == 0.0


def test_fdr_correction():
    # Mix of significant and non-significant p-values
    p_values = np.array([0.001, 0.01, 0.04, 0.5, 0.8])
    rejected, adjusted = fdr_correction(p_values, alpha=0.05)
    # The first two should survive FDR
    assert rejected[0]
    assert rejected[1]
    # The last should not
    assert not rejected[-1]
    # Adjusted p-values should be >= original
    assert all(adjusted >= p_values - 1e-10)


def test_compare_topological_features_has_effect_sizes():
    rng = np.random.default_rng(42)
    features = {
        "entropy": rng.normal(2, 0.5, 100),
        "total_persistence": rng.normal(1, 0.3, 100),
    }
    labels = np.array(["A"] * 50 + ["B"] * 50)
    result = compare_topological_features(features, labels)
    assert "cohens_d" in result.columns
    assert "p_adjusted" in result.columns
    assert "significant" in result.columns
    assert len(result) == 2


def test_compare_topological_features_fdr():
    rng = np.random.default_rng(42)
    # Create many features, most with no real difference
    features = {}
    for i in range(20):
        features[f"feat_{i}"] = rng.normal(0, 1, 100)
    labels = np.array(["A"] * 50 + ["B"] * 50)
    result = compare_topological_features(features, labels)
    # FDR should reduce the number of significant results
    n_raw_sig = (result["p_value"] < 0.05).sum()
    n_fdr_sig = result["significant"].sum()
    assert n_fdr_sig <= n_raw_sig
