"""Tests for the two-axis mycobiome decomposition module."""

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, '.')

from src.cirs.mycobiome import (
    compute_exposure_proxy,
    compute_gut_mycobiome_burden,
    partition_by_exposure,
    partition_by_gut_mycobiome,
    interaction_groups,
    ENVIRONMENTAL_MOLD_TAXA,
    GUT_MYCOBIOME_TAXA,
)


@pytest.fixture
def sample_data():
    """Create a small OTU table with known fungal taxa."""
    taxonomy = pd.DataFrame({
        "Kingdom": ["Bacteria", "Bacteria", "Fungi", "Fungi", "Fungi"],
        "Phylum": ["Firmicutes", "Bacteroidetes", "Ascomycota", "Ascomycota", "Ascomycota"],
        "Genus": ["Faecalibacterium", "Bacteroides", "Aspergillus", "Candida", "Penicillium"],
    }, index=["Faecalibacterium", "Bacteroides", "Aspergillus", "Candida", "Penicillium"])

    otu_table = pd.DataFrame({
        "Faecalibacterium": [0.3, 0.1, 0.25],
        "Bacteroides": [0.4, 0.2, 0.3],
        "Aspergillus": [0.001, 0.05, 0.002],  # environmental mold
        "Candida": [0.01, 0.02, 0.15],         # gut mycobiome
        "Penicillium": [0.0, 0.03, 0.001],      # environmental mold
    }, index=["S1", "S2", "S3"])

    return otu_table, taxonomy


def test_environmental_and_gut_taxa_are_disjoint():
    """Environmental mold and gut mycobiome taxa must not overlap."""
    env_taxa = set(ENVIRONMENTAL_MOLD_TAXA.keys())
    gut_taxa = set(GUT_MYCOBIOME_TAXA.keys())
    assert env_taxa.isdisjoint(gut_taxa), (
        f"Overlap found: {env_taxa & gut_taxa}"
    )


def test_exposure_proxy_detects_environmental_molds(sample_data):
    otu_table, taxonomy = sample_data
    proxy = compute_exposure_proxy(otu_table, taxonomy)
    # S2 has highest Aspergillus + Penicillium
    assert proxy["S2"] > proxy["S1"]
    assert proxy["S2"] > proxy["S3"]


def test_gut_mycobiome_detects_candida(sample_data):
    otu_table, taxonomy = sample_data
    burden = compute_gut_mycobiome_burden(otu_table, taxonomy)
    assert "burden_Candida" in burden.columns
    # S3 has highest Candida
    assert burden.loc["S3", "burden_Candida"] > burden.loc["S1", "burden_Candida"]


def test_axes_are_independent(sample_data):
    """High exposure proxy should NOT imply high gut mycobiome burden."""
    otu_table, taxonomy = sample_data
    proxy = compute_exposure_proxy(otu_table, taxonomy)
    burden = compute_gut_mycobiome_burden(otu_table, taxonomy)

    # S2 has high exposure, S3 has high gut burden — they're different
    assert proxy["S2"] > proxy["S3"]  # S2 more exposed
    assert burden.loc["S3", "gut_mycobiome_total"] > burden.loc["S2", "gut_mycobiome_total"]  # S3 more burdened


def test_partition_by_exposure(sample_data):
    otu_table, taxonomy = sample_data
    high, low = partition_by_exposure(otu_table, taxonomy, quantile=0.5)
    assert len(high) + len(low) == len(otu_table)
    assert len(high) > 0
    assert len(low) > 0


def test_interaction_groups(sample_data):
    otu_table, taxonomy = sample_data
    groups = interaction_groups(otu_table, taxonomy)
    assert len(groups) == len(otu_table)
    assert all(g != "" for g in groups)
    valid_groups = {
        "low_exposure_low_burden", "low_exposure_high_burden",
        "high_exposure_low_burden", "high_exposure_high_burden",
    }
    assert set(groups.unique()).issubset(valid_groups)
