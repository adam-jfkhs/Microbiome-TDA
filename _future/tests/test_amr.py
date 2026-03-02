"""Tests for AMR resistance module."""

import numpy as np
import pandas as pd
import pytest

from src.amr.resistance import (
    compute_amr_burden,
    compute_amr_diversity,
    hgt_edge_potential,
    identify_amr_carriers,
    network_resilience_to_amr,
)


@pytest.fixture
def amr_taxonomy():
    """Taxonomy with AMR and non-AMR genera."""
    return pd.DataFrame(
        {
            "Genus": [
                "Escherichia",      # AMR (high HGT)
                "Klebsiella",       # AMR (high HGT)
                "Enterococcus",     # AMR (high HGT)
                "Faecalibacterium", # non-AMR
                "Roseburia",        # non-AMR
                "Lactobacillus",    # non-AMR
                "Bifidobacterium",  # non-AMR
                "Ruminococcus",     # non-AMR
            ],
        },
        index=[f"OTU_{i}" for i in range(8)],
    )


@pytest.fixture
def amr_otu_table():
    """OTU table with AMR and non-AMR taxa."""
    np.random.seed(42)
    data = np.random.dirichlet(np.ones(8), size=10)
    return pd.DataFrame(
        data,
        index=[f"S_{i}" for i in range(10)],
        columns=[f"OTU_{i}" for i in range(8)],
    )


def test_identify_amr_carriers(amr_taxonomy):
    amr_otus, amr_info = identify_amr_carriers(amr_taxonomy)
    assert len(amr_otus) == 3  # Escherichia, Klebsiella, Enterococcus
    assert "OTU_0" in amr_otus
    assert "OTU_3" not in amr_otus  # Faecalibacterium is not AMR


def test_compute_amr_burden(amr_otu_table, amr_taxonomy):
    amr_otus, _ = identify_amr_carriers(amr_taxonomy)
    burden = compute_amr_burden(amr_otu_table, amr_otus)
    assert len(burden) == 10
    assert (burden >= 0).all()
    assert (burden <= 1).all()


def test_compute_amr_diversity(amr_otu_table, amr_taxonomy):
    diversity = compute_amr_diversity(amr_otu_table, amr_taxonomy)
    assert "amr_genera_count" in diversity.columns
    assert "resistance_class_count" in diversity.columns
    assert len(diversity) == 10


def test_network_resilience(amr_taxonomy):
    n = len(amr_taxonomy)
    rng = np.random.default_rng(42)
    corr = rng.normal(0, 0.3, (n, n))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(
        corr, index=amr_taxonomy.index, columns=amr_taxonomy.index
    )

    matrices, fractions = network_resilience_to_amr(
        corr_df, amr_taxonomy, removal_fraction=0.5
    )
    assert len(matrices) == len(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] > 0.0
    # Matrices should shrink as taxa are removed
    assert matrices[-1].shape[0] <= matrices[0].shape[0]


def test_hgt_edge_potential(amr_taxonomy):
    n = len(amr_taxonomy)
    # Create a correlation matrix with strong correlations between
    # high-HGT AMR carriers (OTU_0=Escherichia, OTU_1=Klebsiella, OTU_2=Enterococcus)
    corr = np.eye(n)
    corr[0, 1] = corr[1, 0] = 0.6  # Escherichia-Klebsiella
    corr[0, 2] = corr[2, 0] = 0.5  # Escherichia-Enterococcus
    corr_df = pd.DataFrame(
        corr, index=amr_taxonomy.index, columns=amr_taxonomy.index
    )

    edges = hgt_edge_potential(corr_df, amr_taxonomy)
    assert len(edges) > 0
    assert "source" in edges.columns
    assert "target" in edges.columns
    assert "correlation" in edges.columns
    # All edges should be above the threshold
    assert (edges["correlation"].abs() > 0.3).all()
