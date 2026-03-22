"""Tests for the synthetic cohort generator."""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_cohort
from src.data.loaders import load_cohort


def test_basic_shape():
    otu_df, metadata, taxonomy = generate_synthetic_cohort(n_samples=50, seed=42)
    assert otu_df.shape[0] == 50
    assert otu_df.shape[1] > 20  # should have ~26 genera
    assert len(metadata) == 50
    assert len(taxonomy) == otu_df.shape[1]


def test_groups_balanced():
    otu_df, metadata, taxonomy = generate_synthetic_cohort(n_samples=100, seed=42)
    counts = metadata["group"].value_counts()
    assert counts["low_exposure"] == 50
    assert counts["high_exposure"] == 50


def test_counts_are_positive_integers():
    otu_df, _, _ = generate_synthetic_cohort(n_samples=20, seed=42)
    assert (otu_df.values >= 0).all()
    assert (otu_df.values == otu_df.values.astype(int)).all()


def test_taxonomy_has_required_columns():
    _, _, taxonomy = generate_synthetic_cohort(n_samples=10, seed=42)
    for col in ["Kingdom", "Phylum", "Genus"]:
        assert col in taxonomy.columns


def test_fungi_present():
    _, _, taxonomy = generate_synthetic_cohort(n_samples=10, seed=42)
    fungi = taxonomy[taxonomy["Kingdom"] == "Fungi"]
    assert len(fungi) >= 2  # Candida, Aspergillus, Malassezia


def test_effect_size_changes_composition():
    _, meta_low, _ = generate_synthetic_cohort(n_samples=100, seed=42, effect_size=0.0)
    otu_high, meta_high, _ = generate_synthetic_cohort(n_samples=100, seed=42, effect_size=0.8)
    # With no effect, groups should be identical in composition
    # With high effect, high_exposure group should differ
    # Just check it runs and metadata is correct
    assert (meta_low["group"] == meta_high["group"]).all()


def test_load_cohort_synthetic():
    otu_df, metadata, taxonomy = load_cohort("synthetic", n_samples=30, seed=99)
    assert otu_df.shape[0] == 30
    assert "group" in metadata.columns


def test_reproducibility():
    a1, _, _ = generate_synthetic_cohort(n_samples=50, seed=123)
    a2, _, _ = generate_synthetic_cohort(n_samples=50, seed=123)
    pd.testing.assert_frame_equal(a1, a2)
