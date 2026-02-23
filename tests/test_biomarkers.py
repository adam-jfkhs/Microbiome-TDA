"""Tests for the evidence-weighted biomarker priors module."""

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, '.')

from src.cirs.biomarkers import (
    get_priors, get_biomarkers, score_biomarker_pressure,
    score_signature, BIOMARKERS, TAXON_BIOMARKER_PRIORS,
    EvidenceGrade, EVIDENCE_WEIGHTS,
)


def test_get_priors_returns_dataframe():
    priors = get_priors()
    assert isinstance(priors, pd.DataFrame)
    assert len(priors) > 0
    required_cols = {"taxon", "biomarker", "direction", "evidence_grade",
                     "evidence_weight", "mechanism", "citations"}
    assert required_cols.issubset(set(priors.columns))


def test_evidence_grades_are_valid():
    for edge in TAXON_BIOMARKER_PRIORS:
        assert edge.evidence_grade in EvidenceGrade
        assert edge.direction in ("+", "-")
        assert len(edge.citations) > 0, f"Missing citation for {edge.taxon}-{edge.biomarker}"


def test_evidence_weights_monotonic():
    assert EVIDENCE_WEIGHTS[EvidenceGrade.A] > EVIDENCE_WEIGHTS[EvidenceGrade.B]
    assert EVIDENCE_WEIGHTS[EvidenceGrade.B] > EVIDENCE_WEIGHTS[EvidenceGrade.C]


def test_biomarker_tiers():
    bms = get_biomarkers()
    tiers = bms["tier"].unique()
    assert "primary" in tiers
    assert "secondary" in tiers
    assert bms.loc[bms["name"] == "TGF-beta1", "tier"].iloc[0] == "primary"
    assert bms.loc[bms["name"] == "VIP", "tier"].iloc[0] == "secondary"


def test_score_biomarker_pressure_direction():
    """SCFA producers should decrease TGF-beta1 pressure (negative score contribution)."""
    taxonomy = pd.DataFrame({
        "Kingdom": ["Bacteria", "Bacteria"],
        "Genus": ["Faecalibacterium", "Escherichia"],
    }, index=["Faecalibacterium", "Escherichia"])

    # High Faecalibacterium → should push TGF-beta1 pressure DOWN
    abundances_protective = pd.Series(
        [0.9, 0.1], index=["Faecalibacterium", "Escherichia"]
    )
    score_protective = score_biomarker_pressure(abundances_protective, taxonomy, "TGF-beta1")

    # High Escherichia → should push TGF-beta1 pressure UP
    abundances_inflammatory = pd.Series(
        [0.1, 0.9], index=["Faecalibacterium", "Escherichia"]
    )
    score_inflammatory = score_biomarker_pressure(abundances_inflammatory, taxonomy, "TGF-beta1")

    assert score_inflammatory > score_protective


def test_score_signature_returns_series():
    """score_signature should return one score per sample."""
    from src.data.loaders import load_cohort
    from src.data.preprocess import relative_abundance

    otu_df, metadata, taxonomy = load_cohort("synthetic", n_samples=20, seed=99)
    rel_df = relative_abundance(otu_df)
    scores = score_signature(rel_df, taxonomy, "TGF-beta1")

    assert isinstance(scores, pd.Series)
    assert len(scores) == len(otu_df)
    assert scores.name == "TGF-beta1_pressure"


def test_unknown_biomarker_returns_zero():
    taxonomy = pd.DataFrame({"Genus": ["Foo"]}, index=["Foo"])
    abundances = pd.Series([1.0], index=["Foo"])
    score = score_biomarker_pressure(abundances, taxonomy, "NonexistentMarker")
    assert score == 0.0
