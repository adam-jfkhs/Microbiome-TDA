#!/usr/bin/env python3
"""Bootstrap TDA analysis on real AGP data — v2 (methodologically sound).

Three key upgrades over v1:

A. GLOBAL TAXA: Top-80 taxa selected once across all stool samples, not per
   resample. Every bootstrap iteration operates on the same feature space.

B. PAIRED RESAMPLING with LABEL PERMUTATION: Each iteration draws n=100 from
   group A and n=100 from group B independently (without replacement). The
   test statistic is the mean within-iteration delta. Null distribution is
   built by flipping group labels, giving a clean label-based permutation test.

C. CONFOUNDING CONTROL: Each comparison is run twice — on the full group and
   on a subset matched on age-bin / sex / BMI-bin. Results that survive
   matching are reported as robust.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.homology import compute_persistence, filter_infinite, persistence_summary
from src.tda.features import betti_curve, persistence_entropy
from src.analysis.statistics import fdr_correction

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200        # paired iterations (A vs B per iteration)
SUBSAMPLE_SIZE = 100      # samples drawn per group per iteration
N_PERMUTATIONS = 500      # label permutations for null distribution
N_GLOBAL_TAXA = 80        # fixed taxa set size
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES_TO_TEST = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]


# ── A. Global taxa selection ───────────────────────────────────────────────────

def select_global_taxa(clr_df, n=N_GLOBAL_TAXA):
    """Select top-N taxa by prevalence across ALL stool samples.

    Returns a list of taxon names that will be used identically in every
    bootstrap iteration and every comparison.
    """
    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top = prevalence.nlargest(n).index.tolist()
    print(f"Global taxa selected: {len(top)} (from {clr_df.shape[1]} total)")
    return top


# ── TDA pipeline on a subsample ───────────────────────────────────────────────

def tda_features_fixed_taxa(clr_subset, global_taxa):
    """Run TDA on a subsample using the pre-fixed global taxon set.

    clr_subset: DataFrame (samples × taxa), rows already selected.
    global_taxa: list of taxon names — same for every call.
    """
    subset = clr_subset[global_taxa]

    corr_matrix, _ = spearman_correlation_matrix(subset)
    dist_df = correlation_distance(corr_matrix)
    dist_matrix = prepare_distance_matrix(dist_df)
    result = compute_persistence(dist_matrix, maxdim=1)

    dgms = result["dgms"]
    finite_dgms = filter_infinite(dgms)
    summary = persistence_summary(dgms)

    h1_entropy = persistence_entropy(dgms[1])
    _, betti1 = betti_curve(dgms[1], num_points=200)

    finite_h1 = finite_dgms[1]
    total_pers = (
        float(np.sum(finite_h1[:, 1] - finite_h1[:, 0]))
        if len(finite_h1) > 0 else 0.0
    )

    return {
        "h1_count":            summary["H1"]["count"],
        "h1_entropy":          h1_entropy,
        "h1_total_persistence": total_pers,
        "h1_mean_lifetime":    summary["H1"]["mean_lifetime"],
        "h1_max_lifetime":     summary["H1"]["max_lifetime"],
        "max_betti1":          int(betti1.max()),
    }


# ── B. Paired resampling with label-permutation test ──────────────────────────

def paired_resample_test(clr_df, ids_a, ids_b, global_taxa, n_iter, subsample_size,
                         n_perm, rng, label=""):
    """Paired resampling TDA comparison with label-permutation null.

    Each iteration:
      1. Sample `subsample_size` from group A (without replacement, capped at group size).
      2. Sample `subsample_size` from group B (without replacement, capped at group size).
      3. Compute TDA features for each subsample.
      4. Record delta = metric_A - metric_B for each feature.

    Observed statistic: mean(delta) across iterations.
    Null: shuffle group labels (swap delta signs) n_perm times.

    Returns a DataFrame of per-feature statistics.
    """
    ids_a = list(ids_a)
    ids_b = list(ids_b)
    n_a = min(subsample_size, len(ids_a))
    n_b = min(subsample_size, len(ids_b))

    print(f"  {label}: {len(ids_a)} vs {len(ids_b)} | drawing {n_a} vs {n_b} per iter × {n_iter}")

    deltas = {feat: [] for feat in FEATURES_TO_TEST}
    feat_a_all = {feat: [] for feat in FEATURES_TO_TEST}
    feat_b_all = {feat: [] for feat in FEATURES_TO_TEST}

    t0 = time.time()
    for i in range(n_iter):
        boot_a_ids = rng.choice(ids_a, size=n_a, replace=False)
        boot_b_ids = rng.choice(ids_b, size=n_b, replace=False)

        fa = tda_features_fixed_taxa(clr_df.loc[boot_a_ids].reset_index(drop=True), global_taxa)
        fb = tda_features_fixed_taxa(clr_df.loc[boot_b_ids].reset_index(drop=True), global_taxa)

        for feat in FEATURES_TO_TEST:
            deltas[feat].append(fa[feat] - fb[feat])
            feat_a_all[feat].append(fa[feat])
            feat_b_all[feat].append(fb[feat])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    iteration {i + 1}/{n_iter}  ({elapsed:.0f}s elapsed)")

    # Label-permutation null: flip signs of deltas at random
    rows = []
    for feat in FEATURES_TO_TEST:
        d = np.array(deltas[feat])
        observed_stat = np.mean(d)

        # Null: each delta independently flips sign (equivalent to permuting labels)
        count_extreme = 0
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=len(d))
            perm_stat = np.mean(d * signs)
            if abs(perm_stat) >= abs(observed_stat):
                count_extreme += 1
        perm_p = (count_extreme + 1) / (n_perm + 1)

        # Wilcoxon signed-rank on deltas (paired nonparametric)
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
            "feature":           feat,
            "mean_a":            round(vals_a.mean(), 4),
            "mean_b":            round(vals_b.mean(), 4),
            "mean_delta":        round(observed_stat, 4),
            "cohens_d":          round(cohens_d, 4),
            "wilcoxon_p":        round(wilcox_p, 6),
            "permutation_p":     round(perm_p, 6),
            "n_iter":            n_iter,
            "n_a":               len(ids_a),
            "n_b":               len(ids_b),
            "subsample_a":       n_a,
            "subsample_b":       n_b,
        })

    return pd.DataFrame(rows), {
        "deltas": deltas,
        "feat_a": feat_a_all,
        "feat_b": feat_b_all,
    }


# ── C. Matching helpers ───────────────────────────────────────────────────────

def make_strata(meta, age_col="AGE", sex_col="SEX", bmi_col="BMI"):
    """Assign each sample a coarse stratum label for matching.

    Returns a Series of stratum strings, NaN for samples missing any covariate.
    """
    age = pd.to_numeric(meta[age_col], errors="coerce")
    bmi = pd.to_numeric(meta[bmi_col], errors="coerce")
    sex = meta[sex_col].where(meta[sex_col].isin(["female", "male"]), other=np.nan)

    age_bin = pd.cut(age, bins=[0, 25, 40, 60, 200],
                     labels=["<25", "25-40", "40-60", ">60"])
    bmi_bin = pd.cut(bmi, bins=[0, 25, 30, 200],
                     labels=["<25", "25-30", ">30"])

    strata = (age_bin.astype(str) + "|" + sex.astype(str) + "|" + bmi_bin.astype(str))
    # Mark as NaN if any component was NaN
    missing = age.isna() | bmi.isna() | sex.isna()
    strata[missing] = np.nan
    return strata


def matched_ids(ids_a, ids_b, strata):
    """Match group B to group A by stratum (1:1 or many:1 within strata).

    For each stratum, keeps all group-A members and up to len(A_stratum)
    group-B members. Drops samples without stratum info.

    Returns (matched_a_ids, matched_b_ids).
    """
    s = strata.dropna()
    a_valid = [i for i in ids_a if i in s.index]
    b_valid = [i for i in ids_b if i in s.index]

    rng_local = np.random.default_rng(SEED + 99)
    matched_a, matched_b = [], []

    for stratum in s.loc[a_valid].unique():
        a_s = [i for i in a_valid if s.get(i) == stratum]
        b_s = [i for i in b_valid if s.get(i) == stratum]
        if not a_s or not b_s:
            continue
        # Take all of A in this stratum; sample equal-or-more from B
        n = len(a_s)
        b_sample = rng_local.choice(b_s, size=min(n * 3, len(b_s)), replace=False).tolist()
        matched_a.extend(a_s)
        matched_b.extend(b_sample)

    return matched_a, matched_b


# ── Figure helper ──────────────────────────────────────────────────────────────

def plot_delta_distributions(raw_data, comp_name, label_a, label_b,
                             results_full, results_matched=None):
    """Plot delta distributions for 6 features, full vs matched overlay."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Paired Bootstrap TDA — {label_a} vs {label_b}\n"
        f"({N_ITERATIONS} iterations × {SUBSAMPLE_SIZE} samples, "
        f"global top-{N_GLOBAL_TAXA} taxa)",
        fontsize=12, fontweight="bold",
    )

    for idx, feat in enumerate(FEATURES_TO_TEST):
        ax = axes.flat[idx]
        d_full = raw_data["full"]["deltas"][feat]
        ax.hist(d_full, bins=25, alpha=0.7, color="#d32f2f",
                edgecolor="white", linewidth=0.5, label="Full groups", density=True)

        if results_matched is not None:
            d_match = raw_data["matched"]["deltas"][feat]
            ax.hist(d_match, bins=25, alpha=0.5, color="#388e3c",
                    edgecolor="white", linewidth=0.5, label="Matched", density=True)

        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

        row = results_full[results_full["feature"] == feat].iloc[0]
        title = (f"{feat}\nd={row['cohens_d']:.2f}  "
                 f"perm-p={row['permutation_p']:.3f}  "
                 f"wilcox-p={row['wilcoxon_p']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("delta (A − B)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f"agp_{comp_name}_v2.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("UPGRADED BOOTSTRAP TDA — v2")
    print("=" * 70)

    # Load & preprocess
    print("\nLoading AGP data...")
    otu_df, metadata = load_agp()
    stool_mask = metadata["BODY_SITE"] == "UBERON:feces"
    stool_ids = metadata.loc[stool_mask].index.intersection(otu_df.index)
    filtered = filter_low_abundance(otu_df.loc[stool_ids],
                                    min_prevalence=0.05, min_reads=1000)
    clr_df = clr_transform(filtered)
    meta = metadata.loc[clr_df.index]
    print(f"Preprocessed: {clr_df.shape[0]} samples × {clr_df.shape[1]} taxa")

    # A. Select global taxa ONCE
    print("\n--- UPGRADE A: Global taxa selection ---")
    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # C. Build strata for matching
    strata = make_strata(meta)
    print(f"Samples with complete age/sex/BMI: {strata.notna().sum()}")

    # Define comparisons
    comparisons = {
        "antibiotics": {
            "label_a": "Recent ABX",
            "label_b": "No ABX (>1yr)",
            "ids_a": meta.loc[
                meta["ANTIBIOTIC_SELECT"].isin(
                    ["In the past week", "In the past month", "In the past 6 months"]
                )
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["ANTIBIOTIC_SELECT"] == "Not in the last year"
            ].index.intersection(clr_df.index),
        },
        "ibd": {
            "label_a": "IBD (UC+Crohn)",
            "label_b": "Healthy",
            "ids_a": meta.loc[
                meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["IBD"] == "I do not have IBD"
            ].index.intersection(clr_df.index),
        },
        "diet": {
            "label_a": "Plant-based",
            "label_b": "Omnivore",
            "ids_a": meta.loc[
                meta["DIET_TYPE"].isin(["Vegan", "Vegetarian", "Vegetarian but eat seafood"])
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["DIET_TYPE"] == "Omnivore"
            ].index.intersection(clr_df.index),
        },
    }

    print("\nGroup sizes:")
    for name, comp in comparisons.items():
        print(f"  {name}: {comp['label_a']} n={len(comp['ids_a'])}  "
              f"{comp['label_b']} n={len(comp['ids_b'])}")

    # Run each comparison
    all_results = []
    for comp_name, comp in comparisons.items():
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {comp_name}")
        print(f"{'=' * 70}")

        raw_data = {}

        # --- Full groups ---
        print("\n[Full groups]")
        results_full, raw_full = paired_resample_test(
            clr_df, comp["ids_a"], comp["ids_b"], global_taxa,
            n_iter=N_ITERATIONS, subsample_size=SUBSAMPLE_SIZE,
            n_perm=N_PERMUTATIONS, rng=rng,
            label=f"{comp['label_a']} vs {comp['label_b']}",
        )
        results_full["comparison"] = comp_name
        results_full["subset"] = "full"
        raw_data["full"] = raw_full

        # --- Matched subset (C) ---
        print("\n[Matched on age/sex/BMI]")
        m_a, m_b = matched_ids(comp["ids_a"], comp["ids_b"], strata)
        print(f"  After matching: {len(m_a)} vs {len(m_b)}")

        results_matched = None
        raw_data["matched"] = None

        if len(m_a) >= SUBSAMPLE_SIZE and len(m_b) >= SUBSAMPLE_SIZE:
            results_matched, raw_matched = paired_resample_test(
                clr_df, m_a, m_b, global_taxa,
                n_iter=N_ITERATIONS, subsample_size=SUBSAMPLE_SIZE,
                n_perm=N_PERMUTATIONS, rng=rng,
                label=f"{comp['label_a']} vs {comp['label_b']} [matched]",
            )
            results_matched["comparison"] = comp_name
            results_matched["subset"] = "matched"
            raw_data["matched"] = raw_matched
            all_results.append(results_matched)
        else:
            print(f"  Skipping matched analysis (too few matched samples)")

        all_results.append(results_full)

        # Print summary
        print(f"\n  Results — {comp_name} (full):")
        print(f"  {'Feature':<28} {'Mean A':>7} {'Mean B':>7} "
              f"{'Cohen d':>8} {'perm-p':>8} {'wilcox-p':>9}")
        print(f"  {'─' * 68}")
        for _, row in results_full.iterrows():
            sig = "*" if row["permutation_p"] < 0.05 else ""
            print(f"  {row['feature']:<28} {row['mean_a']:>7.3f} {row['mean_b']:>7.3f} "
                  f"{row['cohens_d']:>8.3f} {row['permutation_p']:>8.4f} "
                  f"{row['wilcoxon_p']:>9.4f} {sig}")

        # Figure
        plot_delta_distributions(raw_data, comp_name,
                                 comp["label_a"], comp["label_b"],
                                 results_full, results_matched)

    # Collate results and apply FDR across all tests
    all_df = pd.concat(all_results, ignore_index=True)

    # FDR on permutation p-values
    rejected_perm, adj_perm = fdr_correction(all_df["permutation_p"].values)
    all_df["perm_p_fdr"] = adj_perm
    all_df["sig_perm_fdr"] = rejected_perm

    # FDR on Wilcoxon p-values
    rejected_wilcox, adj_wilcox = fdr_correction(all_df["wilcoxon_p"].values)
    all_df["wilcox_p_fdr"] = adj_wilcox
    all_df["sig_wilcox_fdr"] = rejected_wilcox

    out_path = os.path.join(RESULTS_DIR, "agp_bootstrap_v2.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — FDR-significant results (permutation p, BH-corrected)")
    print(f"{'=' * 70}")
    sig = all_df[all_df["sig_perm_fdr"]]
    if len(sig) > 0:
        print(sig[["comparison", "subset", "feature", "mean_a", "mean_b",
                    "cohens_d", "perm_p_fdr"]].to_string(index=False))
    else:
        print("No results significant after FDR correction.")

    print(f"\nNominally significant (perm p<0.05): "
          f"{(all_df['permutation_p'] < 0.05).sum()}/{len(all_df)}")
    print(f"FDR-significant (perm):  {all_df['sig_perm_fdr'].sum()}/{len(all_df)}")
    print(f"FDR-significant (wilcox): {all_df['sig_wilcox_fdr'].sum()}/{len(all_df)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
