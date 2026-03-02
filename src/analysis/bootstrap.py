"""Shared paired-bootstrap TDA utilities.

Used by run_agp_bootstrap_v2.py, run_ibdmdb_bootstrap.py, and
run_taxa_sensitivity.py.  Centralising here ensures that any bug fix
propagates to all three analyses automatically.

The canonical statistical workflow is:
  1. select_global_taxa() — once, before any group partitioning
  2. paired_resample_test() — sign-flip permutation primary p-value
  3. Wilcoxon signed-rank — secondary / confirmatory only
  4. BH-FDR — applied per-subset over the pre-specified test family

Effect sizes (Cohen's d) reported here are computed on per-iteration
bootstrap-distribution values (200 values per group), not on the original
sample-level observations.  This measures the effect size of the group-level
network topology difference, which is the quantity of biological interest.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.homology import compute_persistence, filter_infinite, persistence_summary
from src.tda.features import betti_curve, persistence_entropy

# The six H₁ scalar features used across all analyses.
FEATURES = [
    "h1_count",
    "h1_entropy",
    "h1_total_persistence",
    "h1_mean_lifetime",
    "h1_max_lifetime",
    "max_betti1",
]


def select_global_taxa(clr_df: pd.DataFrame, n: int) -> list:
    """Select the top-N taxa by above-median CLR prevalence across all samples.

    Prevalence is defined as the fraction of samples in which a taxon's CLR
    value exceeds that taxon's column-wise median.  Standard detection-based
    prevalence (proportion of samples where a taxon is non-zero) is not
    applicable after CLR transformation because CLR values are always real-
    valued — the transformation maps structural zeros to large negative values
    rather than to zero.  The above-median definition is a natural analogue:
    it selects taxa that are consistently in the upper half of their abundance
    distribution across the cohort, and is equivalent to selecting by the
    50th-percentile exceedance rate.  Sensitivity to this choice is evaluated
    at N ∈ {50, 80, 120} (see taxa-sensitivity analysis).

    Parameters
    ----------
    clr_df : DataFrame (samples × taxa), CLR-transformed.
    n : Number of taxa to select.

    Returns
    -------
    List of taxon names (length n), sorted by descending prevalence.
    """
    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top = prevalence.nlargest(n).index.tolist()
    print(f"Global taxa selected: {len(top)} (from {clr_df.shape[1]} total)")
    return top


def tda_features(clr_subset: pd.DataFrame, taxa: list) -> dict:
    """Run the H₁ TDA pipeline on clr_subset restricted to taxa.

    Parameters
    ----------
    clr_subset : DataFrame (samples × taxa), CLR-transformed.  Rows should
        already be the desired subsample; this function does not resample.
    taxa : List of taxon column names (the global fixed taxon set).

    Returns
    -------
    Dict with six scalar features:
        h1_count, h1_entropy, h1_total_persistence,
        h1_mean_lifetime, h1_max_lifetime, max_betti1
    """
    subset = clr_subset[taxa]
    corr_matrix, _ = spearman_correlation_matrix(subset)
    dist_df = correlation_distance(corr_matrix)
    dist_matrix = prepare_distance_matrix(dist_df)
    result = compute_persistence(dist_matrix, maxdim=1)

    dgms = result["dgms"]
    finite_dgms = filter_infinite(dgms)
    summary = persistence_summary(dgms)

    h1_entropy_val = persistence_entropy(dgms[1])
    _, betti1 = betti_curve(dgms[1], num_points=200)

    finite_h1 = finite_dgms[1]
    total_pers = (
        float(np.sum(finite_h1[:, 1] - finite_h1[:, 0]))
        if len(finite_h1) > 0 else 0.0
    )

    return {
        "h1_count":             summary["H1"]["count"],
        "h1_entropy":           h1_entropy_val,
        "h1_total_persistence": total_pers,
        "h1_mean_lifetime":     summary["H1"]["mean_lifetime"],
        "h1_max_lifetime":      summary["H1"]["max_lifetime"],
        "max_betti1":           int(betti1.max()),
    }


def paired_resample_test(
    clr_df, ids_a, ids_b, taxa,
    n_iter, subsample_size, n_perm, rng,
    label="", min_samples=0,
):
    """Paired resampling TDA comparison with sign-flip label-permutation null.

    Each of the n_iter iterations draws subsample_size samples from each group
    (without replacement, capped at group size), computes six H₁ scalar
    features, and records delta = feat_A − feat_B.

    Permutation null: each delta's sign is flipped independently (equivalent
    to relabelling group membership within each iteration).  The primary p-value
    is the fraction of n_perm shuffles whose |mean(delta)| meets or exceeds the
    observed |mean(delta)|, with a +1 continuity correction.

    Wilcoxon signed-rank on the delta vector is returned as a secondary
    confirmatory statistic and MUST NOT be used for FDR.

    Cohen's d is computed on the 200-value per-iteration feature distributions,
    measuring the effect size of the group-level network topology difference.

    Parameters
    ----------
    min_samples : int
        Skip and return (None, None) if either group falls below this size.

    Returns
    -------
    (results_df, raw_dict) or (None, None) if skipped.
    raw_dict keys: 'deltas', 'feat_a', 'feat_b' (each a dict feature→list).
    """
    ids_a = list(ids_a)
    ids_b = list(ids_b)
    n_a = min(subsample_size, len(ids_a))
    n_b = min(subsample_size, len(ids_b))

    if min_samples and (n_a < min_samples or n_b < min_samples):
        print(f"  SKIP {label}: too few samples (n_a={n_a}, n_b={n_b})")
        return None, None

    print(
        f"  {label}: {len(ids_a)} vs {len(ids_b)} | "
        f"drawing {n_a} vs {n_b} per iter × {n_iter}"
    )

    deltas    = {f: [] for f in FEATURES}
    feat_a_all = {f: [] for f in FEATURES}
    feat_b_all = {f: [] for f in FEATURES}

    t0 = time.time()
    for i in range(n_iter):
        boot_a = rng.choice(ids_a, size=n_a, replace=False)
        boot_b = rng.choice(ids_b, size=n_b, replace=False)

        fa = tda_features(clr_df.loc[boot_a].reset_index(drop=True), taxa)
        fb = tda_features(clr_df.loc[boot_b].reset_index(drop=True), taxa)

        for feat in FEATURES:
            deltas[feat].append(fa[feat] - fb[feat])
            feat_a_all[feat].append(fa[feat])
            feat_b_all[feat].append(fb[feat])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    iteration {i + 1}/{n_iter}  ({elapsed:.0f}s elapsed)")

    rows = []
    for feat in FEATURES:
        d = np.array(deltas[feat])
        observed_stat = np.mean(d)

        count_extreme = 0
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=len(d))
            if abs(np.mean(d * signs)) >= abs(observed_stat):
                count_extreme += 1
        perm_p = (count_extreme + 1) / (n_perm + 1)

        try:
            _, wilcox_p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            wilcox_p = 1.0

        vals_a = np.array(feat_a_all[feat])
        vals_b = np.array(feat_b_all[feat])
        pooled_std = np.sqrt(
            ((len(vals_a) - 1) * vals_a.var(ddof=1) + (len(vals_b) - 1) * vals_b.var(ddof=1))
            / (len(vals_a) + len(vals_b) - 2)
        )
        cohens_d = float((vals_a.mean() - vals_b.mean()) / pooled_std) if pooled_std > 0 else 0.0

        rows.append({
            "feature":            feat,
            "mean_a":             round(vals_a.mean(), 4),
            "mean_b":             round(vals_b.mean(), 4),
            "mean_delta":         round(observed_stat, 4),
            "cohens_d":           round(cohens_d, 4),
            "wilcoxon_p":         round(wilcox_p, 6),
            "permutation_p":      round(perm_p, 6),
            "n_iter":             n_iter,
            "n_a":                len(ids_a),
            "n_b":                len(ids_b),
            "subsample_a":        n_a,
            "subsample_b":        n_b,
        })

    raw = {"deltas": deltas, "feat_a": feat_a_all, "feat_b": feat_b_all}
    return pd.DataFrame(rows), raw


def make_strata(
    meta: pd.DataFrame,
    age_col: str = "AGE",
    sex_col: str = "SEX",
    bmi_col: str = "BMI",
) -> pd.Series:
    """Assign each sample a coarse stratum label for confounding control.

    Age bins: <25, 25–40, 40–60, >60.
    BMI bins: <25, 25–30, >30.
    Sex: female / male (all other values → NaN).

    Returns
    -------
    Series of stratum strings (NaN for samples missing any covariate).
    """
    age = pd.to_numeric(meta[age_col], errors="coerce")
    bmi = pd.to_numeric(meta[bmi_col], errors="coerce")
    sex = meta[sex_col].where(meta[sex_col].isin(["female", "male"]), other=np.nan)

    age_bin = pd.cut(age, bins=[0, 25, 40, 60, 200],
                     labels=["<25", "25-40", "40-60", ">60"])
    bmi_bin = pd.cut(bmi, bins=[0, 25, 30, 200],
                     labels=["<25", "25-30", ">30"])

    strata = (age_bin.astype(str) + "|" + sex.astype(str) + "|" + bmi_bin.astype(str))
    strata[age.isna() | bmi.isna() | sex.isna()] = np.nan
    return strata


def matched_ids(ids_a, ids_b, strata: pd.Series, seed: int = 141):
    """Match group B to group A by demographic stratum.

    For each stratum that contains at least one group-A sample, keeps all
    group-A samples and draws up to 3× that many group-B samples without
    replacement.  Samples whose stratum is NaN are excluded from both groups.

    Parameters
    ----------
    seed : int
        RNG seed for within-stratum sampling.  Default 141 preserves the
        original SEED+99 = 42+99 behaviour of the pre-refactor scripts.

    Returns
    -------
    (matched_a_ids, matched_b_ids) as plain lists.
    """
    s = strata.dropna()
    a_valid = [i for i in ids_a if i in s.index]
    b_valid = [i for i in ids_b if i in s.index]

    rng_local = np.random.default_rng(seed)
    matched_a, matched_b = [], []

    for stratum in s.loc[a_valid].unique():
        a_s = [i for i in a_valid if s.get(i) == stratum]
        b_s = [i for i in b_valid if s.get(i) == stratum]
        if not a_s or not b_s:
            continue
        b_sample = rng_local.choice(
            b_s, size=min(len(a_s) * 3, len(b_s)), replace=False
        ).tolist()
        matched_a.extend(a_s)
        matched_b.extend(b_sample)

    return matched_a, matched_b
