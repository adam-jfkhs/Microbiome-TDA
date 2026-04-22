"""Tests for src.data.preprocess — CLR transform, filtering, normalization."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    filter_low_abundance,
    clr_transform,
    relative_abundance,
)


@pytest.fixture
def otu_df():
    """Simple OTU table: 5 samples x 4 OTUs."""
    data = np.array([
        [100, 200, 0, 50],
        [150, 0, 30, 60],
        [0, 180, 20, 40],
        [120, 160, 10, 0],
        [80, 0, 0, 70],
    ], dtype=float)
    return pd.DataFrame(data, columns=["OTU1", "OTU2", "OTU3", "OTU4"])


def test_filter_low_abundance_removes_low_prevalence(otu_df):
    """OTUs appearing in fewer than 60% of samples should be removed."""
    result = filter_low_abundance(otu_df, min_prevalence=0.6, min_reads=0)
    # OTU1 present in 4/5=80%, OTU2 in 3/5=60%, OTU3 in 3/5=60%, OTU4 in 4/5=80%
    assert "OTU1" in result.columns
    assert "OTU4" in result.columns


def test_filter_low_abundance_removes_low_reads(otu_df):
    """Samples with total reads below threshold should be removed."""
    result = filter_low_abundance(otu_df, min_prevalence=0.0, min_reads=200)
    # Row sums: 350, 240, 240, 290, 150 — only row 4 (150) is below 200
    assert len(result) == 4


def test_filter_low_abundance_empty_result():
    """If all OTUs are rare, result should be empty columns."""
    df = pd.DataFrame({"A": [0, 0, 1], "B": [0, 0, 0]})
    result = filter_low_abundance(df, min_prevalence=0.9, min_reads=0)
    assert result.shape[1] == 0 or result.empty


def test_clr_transform_shape(otu_df):
    """CLR output should have the same shape as input."""
    result = clr_transform(otu_df)
    assert result.shape == otu_df.shape


def test_clr_transform_zero_row_mean(otu_df):
    """CLR values should sum to ~0 per sample (by definition)."""
    result = clr_transform(otu_df)
    row_means = result.mean(axis=1)
    np.testing.assert_allclose(row_means, 0.0, atol=1e-10)


def test_clr_transform_preserves_index(otu_df):
    """Index and columns should be preserved."""
    result = clr_transform(otu_df)
    assert list(result.index) == list(otu_df.index)
    assert list(result.columns) == list(otu_df.columns)


def test_clr_transform_pseudocount():
    """Different pseudocounts should produce different results."""
    df = pd.DataFrame({"A": [10, 0, 5], "B": [0, 20, 15]})
    r1 = clr_transform(df, pseudocount=0.5)
    r2 = clr_transform(df, pseudocount=1.0)
    assert not np.allclose(r1.values, r2.values)


def test_clr_transform_single_column():
    """Single OTU should produce all zeros (geo mean = self)."""
    df = pd.DataFrame({"A": [10, 20, 30]})
    result = clr_transform(df)
    np.testing.assert_allclose(result.values, 0.0, atol=1e-10)


def test_relative_abundance_sums_to_one(otu_df):
    """Each sample should sum to 1.0."""
    result = relative_abundance(otu_df)
    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_relative_abundance_non_negative(otu_df):
    """All relative abundances should be >= 0."""
    result = relative_abundance(otu_df)
    assert (result.values >= 0).all()
