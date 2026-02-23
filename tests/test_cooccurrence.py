"""Tests for co-occurrence network construction."""

import numpy as np
import pandas as pd
import pytest

from src.networks.cooccurrence import spearman_correlation_matrix, build_network


@pytest.fixture
def sample_otu_df():
    """Create a small synthetic OTU table for testing."""
    np.random.seed(42)
    data = np.random.rand(20, 5)
    return pd.DataFrame(data, columns=[f"OTU_{i}" for i in range(5)])


def test_spearman_correlation_shape(sample_otu_df):
    corr_df, pval_df = spearman_correlation_matrix(sample_otu_df)
    n_otus = sample_otu_df.shape[1]
    assert corr_df.shape == (n_otus, n_otus)
    assert pval_df.shape == (n_otus, n_otus)


def test_spearman_correlation_diagonal(sample_otu_df):
    corr_df, _ = spearman_correlation_matrix(sample_otu_df)
    np.testing.assert_array_almost_equal(np.diag(corr_df.values), 1.0)


def test_build_network_nodes(sample_otu_df):
    corr_df, _ = spearman_correlation_matrix(sample_otu_df)
    G = build_network(corr_df, threshold=0.0)
    assert set(G.nodes()) == set(sample_otu_df.columns)


def test_build_network_threshold(sample_otu_df):
    corr_df, _ = spearman_correlation_matrix(sample_otu_df)
    G_low = build_network(corr_df, threshold=0.1)
    G_high = build_network(corr_df, threshold=0.9)
    assert G_high.number_of_edges() <= G_low.number_of_edges()
