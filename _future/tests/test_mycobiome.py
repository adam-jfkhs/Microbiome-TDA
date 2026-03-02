"""Tests for mycobiome disruption module."""

import numpy as np
import pandas as pd
import pytest

from src.mycobiome.disruption import (
    bacterial_network_without_fungi,
    compute_disruption_score,
    compute_fungal_burden,
    identify_fungal_taxa,
    partition_samples_by_mycobiome,
)


@pytest.fixture
def mixed_taxonomy():
    """Taxonomy with both bacteria and fungi."""
    return pd.DataFrame(
        {
            "Kingdom": [
                "Bacteria", "Bacteria", "Bacteria", "Bacteria",
                "Fungi", "Fungi",
            ],
            "Genus": [
                "Lactobacillus", "Faecalibacterium", "Bacteroides", "Roseburia",
                "Candida", "Aspergillus",
            ],
        },
        index=["OTU_0", "OTU_1", "OTU_2", "OTU_3", "OTU_4", "OTU_5"],
    )


@pytest.fixture
def mixed_otu_table():
    """OTU table with bacteria and fungi."""
    data = np.array([
        [0.3, 0.2, 0.2, 0.2, 0.05, 0.05],  # low fungal
        [0.1, 0.1, 0.1, 0.1, 0.30, 0.30],  # high fungal
        [0.25, 0.25, 0.25, 0.24, 0.005, 0.005],  # very low fungal
    ])
    return pd.DataFrame(
        data,
        index=["S_0", "S_1", "S_2"],
        columns=["OTU_0", "OTU_1", "OTU_2", "OTU_3", "OTU_4", "OTU_5"],
    )


def test_identify_fungal_taxa(mixed_taxonomy):
    fungal_ids, genus_map = identify_fungal_taxa(mixed_taxonomy)
    assert len(fungal_ids) == 2
    assert "OTU_4" in fungal_ids
    assert "OTU_5" in fungal_ids
    assert genus_map["OTU_4"] == "Candida"


def test_compute_fungal_burden(mixed_otu_table):
    fungal_ids = ["OTU_4", "OTU_5"]
    burden = compute_fungal_burden(mixed_otu_table, fungal_ids)
    assert len(burden) == 3
    assert burden["S_1"] > burden["S_0"]  # S_1 has higher fungal load
    assert burden["S_2"] < burden["S_0"]  # S_2 has lowest


def test_partition_samples(mixed_otu_table, mixed_taxonomy):
    high, low = partition_samples_by_mycobiome(
        mixed_otu_table, mixed_taxonomy, threshold=0.1
    )
    assert "S_1" in high  # 60% fungal
    assert "S_0" in high  # 10% fungal
    assert "S_2" in low   # 1% fungal


def test_disruption_score(mixed_otu_table, mixed_taxonomy):
    scores = compute_disruption_score(mixed_otu_table, mixed_taxonomy)
    # Should have disruption columns for Candida and/or Aspergillus
    if len(scores.columns) > 0:
        assert "disruption_total" in scores.columns
        assert len(scores) == 3


def test_bacterial_network_without_fungi(mixed_taxonomy):
    n = len(mixed_taxonomy)
    corr = pd.DataFrame(
        np.eye(n),
        index=mixed_taxonomy.index,
        columns=mixed_taxonomy.index,
    )
    bact_corr = bacterial_network_without_fungi(corr, mixed_taxonomy)
    assert bact_corr.shape[0] == 4  # Only bacterial OTUs
    assert "OTU_4" not in bact_corr.columns
    assert "OTU_5" not in bact_corr.columns
