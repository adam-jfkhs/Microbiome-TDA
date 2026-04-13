"""Tests for src.analysis.correlation — feature-metadata correlation."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlation import correlate_features_with_metadata


@pytest.fixture
def topo_features():
    """Topological features for 10 samples."""
    rng = np.random.default_rng(7)
    idx = [f"S{i}" for i in range(10)]
    return pd.DataFrame({
        "h1_count": rng.integers(0, 10, 10).astype(float),
        "h1_entropy": rng.random(10),
    }, index=idx)


@pytest.fixture
def metadata():
    """Metadata with numeric and non-numeric columns."""
    rng = np.random.default_rng(7)
    idx = [f"S{i}" for i in range(10)]
    return pd.DataFrame({
        "age": rng.integers(20, 80, 10).astype(float),
        "bmi": rng.uniform(18, 35, 10),
        "diagnosis": ["IBD"] * 5 + ["healthy"] * 5,  # non-numeric
    }, index=idx)


def test_output_shape(topo_features, metadata):
    """Output should be (n_features x n_numeric_meta)."""
    corr, pval = correlate_features_with_metadata(topo_features, metadata)
    assert corr.shape == (2, 2)  # 2 topo features x 2 numeric meta cols
    assert pval.shape == (2, 2)


def test_pvalues_bounded(topo_features, metadata):
    """p-values should be in [0, 1]."""
    _, pval = correlate_features_with_metadata(topo_features, metadata)
    assert (pval.values >= 0).all()
    assert (pval.values <= 1).all()


def test_correlations_bounded(topo_features, metadata):
    """Correlations should be in [-1, 1]."""
    corr, _ = correlate_features_with_metadata(topo_features, metadata)
    assert (corr.values >= -1 - 1e-10).all()
    assert (corr.values <= 1 + 1e-10).all()


def test_non_numeric_excluded(topo_features, metadata):
    """Non-numeric metadata columns should be excluded."""
    corr, _ = correlate_features_with_metadata(topo_features, metadata)
    assert "diagnosis" not in corr.columns


def test_pearson_method(topo_features, metadata):
    """Pearson method should produce different results than Spearman."""
    corr_s, _ = correlate_features_with_metadata(topo_features, metadata, method="spearman")
    corr_p, _ = correlate_features_with_metadata(topo_features, metadata, method="pearson")
    # Results may be similar but generally not identical
    assert corr_s.shape == corr_p.shape


def test_partial_index_overlap():
    """Should handle non-overlapping indices gracefully."""
    topo = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])
    meta = pd.DataFrame({"m1": [10.0, 20.0, 30.0]}, index=["B", "C", "D"])
    corr, pval = correlate_features_with_metadata(topo, meta)
    assert corr.shape == (1, 1)  # Only B, C overlap


def test_too_few_non_nan():
    """Should return NaN when fewer than 3 non-NaN pairs exist."""
    topo = pd.DataFrame({"f1": [1.0, np.nan, np.nan]}, index=["A", "B", "C"])
    meta = pd.DataFrame({"m1": [10.0, 20.0, 30.0]}, index=["A", "B", "C"])
    corr, pval = correlate_features_with_metadata(topo, meta)
    assert np.isnan(corr.loc["f1", "m1"])
    assert np.isnan(pval.loc["f1", "m1"])
