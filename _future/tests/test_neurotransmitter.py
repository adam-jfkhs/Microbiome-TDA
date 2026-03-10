"""Tests for neurotransmitter pathway module."""

import numpy as np
import pandas as pd
import pytest

from src.neurotransmitter.pathways import (
    compute_crossfeeding_potential,
    get_pathway_database,
    identify_nt_subnetwork,
    score_neurotransmitter_potential,
)


@pytest.fixture
def sample_otu_table():
    """Create a small OTU table with known NT-associated genera."""
    np.random.seed(42)
    data = np.random.dirichlet(np.ones(8), size=20)
    otus = [f"OTU_{i}" for i in range(8)]
    samples = [f"S_{i}" for i in range(20)]
    return pd.DataFrame(data, index=samples, columns=otus)


@pytest.fixture
def sample_taxonomy():
    """Create taxonomy mapping OTUs to known NT genera."""
    return pd.DataFrame(
        {
            "Kingdom": ["Bacteria"] * 8,
            "Genus": [
                "Enterococcus",   # serotonin + dopamine
                "Lactobacillus",  # GABA + serotonin
                "Bifidobacterium",  # GABA + SCFA
                "Faecalibacterium",  # SCFA (butyrate)
                "Roseburia",      # SCFA (butyrate)
                "Escherichia",    # serotonin + dopamine
                "Bacteroides",    # GABA + SCFA
                "Ruminococcus",   # serotonin (precursor)
            ],
        },
        index=[f"OTU_{i}" for i in range(8)],
    )


def test_pathway_database_nonempty():
    db = get_pathway_database()
    assert len(db) > 0
    assert "genus" in db.columns
    assert "neurotransmitter" in db.columns


def test_pathway_database_all_targets():
    db = get_pathway_database()
    targets = set(db["neurotransmitter"])
    assert "serotonin" in targets
    assert "GABA" in targets
    assert "dopamine" in targets
    assert "SCFA" in targets


def test_score_neurotransmitter_potential_shape(sample_otu_table, sample_taxonomy):
    scores = score_neurotransmitter_potential(
        sample_otu_table, sample_taxonomy, target="all"
    )
    assert scores.shape[0] == len(sample_otu_table)
    assert "serotonin_score" in scores.columns
    assert "GABA_score" in scores.columns
    assert "dopamine_score" in scores.columns
    assert "SCFA_score" in scores.columns


def test_score_single_target(sample_otu_table, sample_taxonomy):
    scores = score_neurotransmitter_potential(
        sample_otu_table, sample_taxonomy, target="serotonin"
    )
    assert scores.shape[1] == 1
    assert "serotonin_score" in scores.columns


def test_scores_nonnegative(sample_otu_table, sample_taxonomy):
    scores = score_neurotransmitter_potential(
        sample_otu_table, sample_taxonomy, target="all"
    )
    assert (scores >= 0).all().all()


def test_identify_nt_subnetwork(sample_taxonomy):
    n = len(sample_taxonomy)
    corr = pd.DataFrame(
        np.eye(n) + np.random.default_rng(42).normal(0, 0.1, (n, n)),
        index=sample_taxonomy.index,
        columns=sample_taxonomy.index,
    )
    sub, otu_ids = identify_nt_subnetwork(corr, sample_taxonomy, target="serotonin")
    # Should include Enterococcus, Lactobacillus, Escherichia, Ruminococcus
    assert len(otu_ids) >= 3
    assert sub.shape[0] == sub.shape[1]


def test_crossfeeding_potential(sample_otu_table, sample_taxonomy):
    cf = compute_crossfeeding_potential(sample_otu_table, sample_taxonomy)
    assert cf.shape[0] == len(sample_otu_table)
    assert cf.shape[1] > 0
    assert (cf >= 0).all().all()
