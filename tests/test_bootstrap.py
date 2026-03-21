"""Tests for src/analysis/bootstrap.py.

Covers the functions most critical to the paper's confounding-control claim
(matched_ids, make_strata) and the global-taxa selection logic, plus a smoke
test of the full paired_resample_test pipeline on synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.bootstrap import (
    FEATURES,
    select_global_taxa,
    make_strata,
    matched_ids,
    paired_resample_test,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_clr_df():
    """A small synthetic CLR-transformed DataFrame (30 samples × 15 taxa)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, 15))
    taxa = [f"taxon_{i}" for i in range(15)]
    samples = [f"s{i}" for i in range(30)]
    return pd.DataFrame(data, index=samples, columns=taxa)


@pytest.fixture
def small_meta():
    """Metadata for 30 samples with age, sex, and BMI."""
    rng = np.random.default_rng(1)
    n = 30
    meta = pd.DataFrame({
        "AGE": rng.uniform(20, 70, n).astype(int),
        "SEX": rng.choice(["female", "male"], n),
        "BMI": rng.uniform(18, 40, n),
    }, index=[f"s{i}" for i in range(n)])
    return meta


# ── select_global_taxa ────────────────────────────────────────────────────────

def test_select_global_taxa_count(small_clr_df):
    taxa = select_global_taxa(small_clr_df, n=8)
    assert len(taxa) == 8


def test_select_global_taxa_subset(small_clr_df):
    taxa = select_global_taxa(small_clr_df, n=10)
    assert all(t in small_clr_df.columns for t in taxa)


def test_select_global_taxa_reproducible(small_clr_df):
    taxa1 = select_global_taxa(small_clr_df, n=5)
    taxa2 = select_global_taxa(small_clr_df, n=5)
    assert taxa1 == taxa2


# ── make_strata ───────────────────────────────────────────────────────────────

def test_make_strata_length(small_meta):
    strata = make_strata(small_meta)
    assert len(strata) == len(small_meta)


def test_make_strata_nan_for_missing(small_meta):
    meta_missing = small_meta.copy()
    meta_missing.loc["s0", "AGE"] = np.nan
    strata = make_strata(meta_missing)
    assert pd.isna(strata.loc["s0"])
    assert strata.loc["s1"] is not np.nan


def test_make_strata_known_bins():
    meta = pd.DataFrame({
        "AGE": [22, 35, 55, 70],
        "SEX": ["female", "male", "female", "male"],
        "BMI": [22, 27, 32, 22],
    }, index=["a", "b", "c", "d"])
    strata = make_strata(meta)
    assert strata["a"] == "<25|female|<25"
    assert strata["b"] == "25-40|male|25-30"
    assert strata["c"] == "40-60|female|>30"
    assert strata["d"] == ">60|male|<25"


# ── matched_ids ───────────────────────────────────────────────────────────────

@pytest.fixture
def strata_series():
    """Strata for 30 samples: first 15 in stratum A, next 15 in stratum B."""
    labels = ["stratum_A"] * 15 + ["stratum_B"] * 15
    return pd.Series(labels, index=[f"s{i}" for i in range(30)])


def test_matched_ids_returns_subsets(strata_series):
    ids_a = [f"s{i}" for i in range(5)]    # first 5 in stratum_A
    ids_b = [f"s{i}" for i in range(15, 25)]  # first 10 of stratum_B group
    ma, mb = matched_ids(ids_a, ids_b, strata_series)
    # All returned ids must come from the original input sets
    assert all(i in ids_a for i in ma)
    assert all(i in ids_b for i in mb)


def test_matched_ids_excludes_nan():
    strata = pd.Series(
        ["A", np.nan, "A", "A", "A", "A"],
        index=["s0", "s1", "s2", "s3", "s4", "s5"],
    )
    ids_a = ["s0", "s1"]   # s1 has NaN stratum
    ids_b = ["s2", "s3", "s4", "s5"]
    ma, mb = matched_ids(ids_a, ids_b, strata)
    # s1 has NaN strata, must not appear in results
    assert "s1" not in ma
    assert "s1" not in mb


def test_matched_ids_no_cross_stratum():
    """Samples from stratum A must only be matched to stratum A counterparts."""
    strata = pd.Series(
        ["A"] * 5 + ["B"] * 5,
        index=[f"s{i}" for i in range(10)],
    )
    ids_a = ["s0", "s1"]        # stratum A
    ids_b = ["s5", "s6", "s7"]  # stratum B — should NOT match with ids_a
    ma, mb = matched_ids(ids_a, ids_b, strata)
    # No shared stratum → nothing matched
    assert len(ma) == 0
    assert len(mb) == 0


def test_matched_ids_b_capped_at_3x_a():
    """Group B draw is capped at 3× the size of group A in each stratum."""
    # 2 A-samples and 20 B-samples all in same stratum
    strata = pd.Series(["X"] * 22, index=[f"s{i}" for i in range(22)])
    ids_a = ["s0", "s1"]
    ids_b = [f"s{i}" for i in range(2, 22)]
    ma, mb = matched_ids(ids_a, ids_b, strata)
    assert len(ma) == 2
    assert len(mb) <= len(ma) * 3


def test_matched_ids_seed_reproducibility():
    strata = pd.Series(["X"] * 10, index=[f"s{i}" for i in range(10)])
    ids_a = ["s0", "s1"]
    ids_b = [f"s{i}" for i in range(2, 10)]
    ma1, mb1 = matched_ids(ids_a, ids_b, strata, seed=42)
    ma2, mb2 = matched_ids(ids_a, ids_b, strata, seed=42)
    assert mb1 == mb2


# ── paired_resample_test (smoke test) ─────────────────────────────────────────

def test_paired_resample_test_returns_six_rows(small_clr_df):
    taxa = select_global_taxa(small_clr_df, n=8)
    ids_a = [f"s{i}" for i in range(15)]
    ids_b = [f"s{i}" for i in range(15, 30)]
    rng = np.random.default_rng(99)
    df, raw = paired_resample_test(
        small_clr_df, ids_a, ids_b, taxa,
        n_iter=5, subsample_size=10, n_perm=20, rng=rng,
    )
    assert df is not None
    assert len(df) == len(FEATURES)
    assert set(df["feature"].tolist()) == set(FEATURES)


def test_paired_resample_test_columns(small_clr_df):
    taxa = select_global_taxa(small_clr_df, n=8)
    ids_a = [f"s{i}" for i in range(15)]
    ids_b = [f"s{i}" for i in range(15, 30)]
    rng = np.random.default_rng(7)
    df, raw = paired_resample_test(
        small_clr_df, ids_a, ids_b, taxa,
        n_iter=5, subsample_size=10, n_perm=10, rng=rng,
    )
    for col in ["permutation_p", "wilcoxon_p", "cohens_d", "mean_a", "mean_b"]:
        assert col in df.columns, f"Missing column: {col}"


def test_paired_resample_test_pvalues_in_range(small_clr_df):
    taxa = select_global_taxa(small_clr_df, n=8)
    ids_a = [f"s{i}" for i in range(15)]
    ids_b = [f"s{i}" for i in range(15, 30)]
    rng = np.random.default_rng(3)
    df, _ = paired_resample_test(
        small_clr_df, ids_a, ids_b, taxa,
        n_iter=5, subsample_size=10, n_perm=20, rng=rng,
    )
    assert df["permutation_p"].between(0, 1).all()
    assert df["wilcoxon_p"].between(0, 1).all()


def test_paired_resample_test_skip_when_too_few(small_clr_df):
    taxa = select_global_taxa(small_clr_df, n=8)
    ids_a = [f"s{i}" for i in range(5)]
    ids_b = [f"s{i}" for i in range(15, 30)]
    rng = np.random.default_rng(0)
    df, raw = paired_resample_test(
        small_clr_df, ids_a, ids_b, taxa,
        n_iter=5, subsample_size=10, n_perm=10, rng=rng,
        min_samples=20,   # require ≥20; group A has only 5
    )
    assert df is None
    assert raw is None
