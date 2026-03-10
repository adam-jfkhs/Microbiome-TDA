#!/usr/bin/env python3
"""Bootstrap TDA analysis on real AGP data.

Instead of computing ONE co-occurrence network per group and comparing
two single persistence diagrams, this script:

1. Repeatedly resamples (bootstrap) samples within each group
2. Computes a co-occurrence network and persistent homology for each resample
3. Extracts scalar topological features per resample
4. Compares the DISTRIBUTIONS of features between groups

This gives proper statistical power for testing whether group assignment
(e.g. antibiotic use, IBD status) produces systematically different
network topology.
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
from scipy.stats import mannwhitneyu

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.homology import compute_persistence, filter_infinite, persistence_summary
from src.tda.features import betti_curve, persistence_entropy
from src.analysis.statistics import cohens_d, fdr_correction

SEED = 42
N_BOOTSTRAP = 50  # number of bootstrap resamples per group
SUBSAMPLE_SIZE = 100  # samples per bootstrap resample
MAX_TAXA = 80  # top taxa for network construction
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_tda_features(clr_subset, max_taxa=MAX_TAXA):
    """Run TDA pipeline on a sample subset, return scalar features."""
    # Select top taxa by prevalence
    prevalence = (clr_subset > clr_subset.median()).mean(axis=0)
    top_taxa = prevalence.nlargest(max_taxa).index
    subset = clr_subset[top_taxa]

    # Spearman correlation → distance → persistent homology
    corr_matrix, _ = spearman_correlation_matrix(subset)
    dist_df = correlation_distance(corr_matrix)
    dist_matrix = prepare_distance_matrix(dist_df)
    result = compute_persistence(dist_matrix, maxdim=1)

    dgms = result["dgms"]
    finite_dgms = filter_infinite(dgms)
    summary = persistence_summary(dgms)

    # Scalar features
    h1_entropy = persistence_entropy(dgms[1])
    _, betti1 = betti_curve(dgms[1], num_points=200)

    finite_h1 = finite_dgms[1]
    total_pers = (
        float(np.sum(finite_h1[:, 1] - finite_h1[:, 0]))
        if len(finite_h1) > 0
        else 0.0
    )
    mean_lifetime = summary["H1"]["mean_lifetime"]
    max_lifetime = summary["H1"]["max_lifetime"]

    return {
        "h1_count": summary["H1"]["count"],
        "h1_entropy": h1_entropy,
        "h1_total_persistence": total_pers,
        "h1_mean_lifetime": mean_lifetime,
        "h1_max_lifetime": max_lifetime,
        "max_betti1": int(betti1.max()),
        "h0_count": summary["H0"]["count"],
    }


def bootstrap_tda(clr_df, sample_ids, n_bootstrap, subsample_size, rng, label=""):
    """Run bootstrap TDA: resample → compute features → return distribution."""
    all_features = []
    ids = list(sample_ids)

    for i in range(n_bootstrap):
        # Bootstrap resample
        boot_ids = rng.choice(ids, size=min(subsample_size, len(ids)), replace=True)
        boot_clr = clr_df.loc[boot_ids]

        # De-duplicate index for Spearman (bootstrap can repeat samples)
        boot_clr = boot_clr.reset_index(drop=True)

        features = extract_tda_features(boot_clr)
        features["bootstrap_iter"] = i
        all_features.append(features)

        if (i + 1) % 10 == 0:
            print(f"    {label}: {i + 1}/{n_bootstrap} bootstrap iterations")

    return pd.DataFrame(all_features)


# ── 1. Load and preprocess ───────────────────────────────────────────
print("=" * 70)
print("BOOTSTRAP TDA ANALYSIS ON REAL AGP DATA")
print("=" * 70)

otu_df, metadata = load_agp()
stool_mask = metadata["BODY_SITE"] == "UBERON:feces"
stool_ids = metadata.loc[stool_mask].index.intersection(otu_df.index)
otu_stool = otu_df.loc[stool_ids]
meta_stool = metadata.loc[stool_ids]

filtered = filter_low_abundance(otu_stool, min_prevalence=0.05, min_reads=1000)
clr_df = clr_transform(filtered)
meta_filtered = meta_stool.loc[clr_df.index]
print(f"Preprocessed: {clr_df.shape[0]} samples, {clr_df.shape[1]} taxa")

# ── 2. Define comparisons ───────────────────────────────────────────
comparisons = {
    "antibiotics": {
        "group_a_name": "Recent ABX",
        "group_b_name": "No ABX",
        "group_a_ids": meta_filtered.loc[
            meta_filtered["ANTIBIOTIC_SELECT"].isin(
                ["In the past week", "In the past month", "In the past 6 months"]
            )
        ].index.intersection(clr_df.index),
        "group_b_ids": meta_filtered.loc[
            meta_filtered["ANTIBIOTIC_SELECT"] == "Not in the last year"
        ].index.intersection(clr_df.index),
    },
    "ibd": {
        "group_a_name": "IBD",
        "group_b_name": "Healthy",
        "group_a_ids": meta_filtered.loc[
            meta_filtered["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
        ].index.intersection(clr_df.index),
        "group_b_ids": meta_filtered.loc[
            meta_filtered["IBD"] == "I do not have IBD"
        ].index.intersection(clr_df.index),
    },
    "diet": {
        "group_a_name": "Plant-based",
        "group_b_name": "Omnivore",
        "group_a_ids": meta_filtered.loc[
            meta_filtered["DIET_TYPE"].isin(
                ["Vegan", "Vegetarian", "Vegetarian but eat seafood"]
            )
        ].index.intersection(clr_df.index),
        "group_b_ids": meta_filtered.loc[
            meta_filtered["DIET_TYPE"] == "Omnivore"
        ].index.intersection(clr_df.index),
    },
}

for name, comp in comparisons.items():
    print(
        f"  {name}: {comp['group_a_name']} (n={len(comp['group_a_ids'])}) vs "
        f"{comp['group_b_name']} (n={len(comp['group_b_ids'])})"
    )

# ── 3. Run bootstrap TDA ────────────────────────────────────────────
rng = np.random.default_rng(SEED)
all_bootstrap_results = {}

for comp_name, comp in comparisons.items():
    print(f"\n{'─' * 50}")
    print(f"BOOTSTRAPPING: {comp_name}")
    print(f"{'─' * 50}")

    t0 = time.time()
    feat_a = bootstrap_tda(
        clr_df, comp["group_a_ids"], N_BOOTSTRAP, SUBSAMPLE_SIZE, rng,
        label=comp["group_a_name"],
    )
    feat_b = bootstrap_tda(
        clr_df, comp["group_b_ids"], N_BOOTSTRAP, SUBSAMPLE_SIZE, rng,
        label=comp["group_b_name"],
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")

    feat_a["group"] = comp["group_a_name"]
    feat_b["group"] = comp["group_b_name"]

    all_bootstrap_results[comp_name] = {
        "feat_a": feat_a,
        "feat_b": feat_b,
        "combined": pd.concat([feat_a, feat_b], ignore_index=True),
    }

# ── 4. Statistical tests ────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("STATISTICAL RESULTS (Bootstrap distributions)")
print(f"{'=' * 70}")

feature_names = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]

stat_rows = []

for comp_name, comp in comparisons.items():
    br = all_bootstrap_results[comp_name]
    fa, fb = br["feat_a"], br["feat_b"]

    print(f"\n{comp_name}: {comp['group_a_name']} vs {comp['group_b_name']}")
    print(f"  {'Feature':<25} {'Mean A':>8} {'Mean B':>8} {'Cohen d':>8} {'MW-U p':>10} {'Sig':>5}")
    print(f"  {'─' * 65}")

    for feat in feature_names:
        vals_a = fa[feat].values
        vals_b = fb[feat].values
        d = cohens_d(vals_a, vals_b)
        stat, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")

        print(
            f"  {feat:<25} {np.mean(vals_a):>8.3f} {np.mean(vals_b):>8.3f} "
            f"{d:>8.3f} {p:>10.4f} {'*' if p < 0.05 else '':>5}"
        )

        stat_rows.append({
            "comparison": comp_name,
            "feature": feat,
            "mean_a": round(np.mean(vals_a), 4),
            "mean_b": round(np.mean(vals_b), 4),
            "std_a": round(np.std(vals_a), 4),
            "std_b": round(np.std(vals_b), 4),
            "cohens_d": round(d, 4),
            "mannwhitney_p": round(p, 6),
        })

stat_df = pd.DataFrame(stat_rows)

# FDR correction across all tests
rejected, adj_p = fdr_correction(stat_df["mannwhitney_p"].values)
stat_df["p_adjusted"] = adj_p
stat_df["significant_fdr"] = rejected

results_path = os.path.join(RESULTS_DIR, "agp_bootstrap_results.csv")
stat_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")

# Print FDR results
print(f"\nFDR-corrected significant results:")
sig = stat_df[stat_df["significant_fdr"]]
if len(sig) > 0:
    print(sig[["comparison", "feature", "mean_a", "mean_b", "cohens_d", "p_adjusted"]].to_string(index=False))
else:
    print("  No results significant after FDR correction")

# ── 5. Figures ───────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("GENERATING BOOTSTRAP DISTRIBUTION FIGURES")
print(f"{'=' * 70}")

for comp_name, comp in comparisons.items():
    br = all_bootstrap_results[comp_name]
    fa, fb = br["feat_a"], br["feat_b"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Bootstrap TDA: {comp['group_a_name']} vs {comp['group_b_name']}\n"
        f"({N_BOOTSTRAP} resamples of {SUBSAMPLE_SIZE} samples each)",
        fontsize=13, fontweight="bold",
    )

    for idx, feat in enumerate(feature_names):
        ax = axes.flat[idx]
        vals_a = fa[feat].values
        vals_b = fb[feat].values

        ax.hist(vals_a, bins=15, alpha=0.6, color="#d32f2f", label=comp["group_a_name"],
                edgecolor="white", linewidth=0.5)
        ax.hist(vals_b, bins=15, alpha=0.6, color="#1976d2", label=comp["group_b_name"],
                edgecolor="white", linewidth=0.5)

        d = cohens_d(vals_a, vals_b)
        _, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")

        ax.set_title(f"{feat}\nd={d:.2f}, p={p:.4f}{'*' if p < 0.05 else ''}", fontsize=10)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f"agp_{comp_name}_bootstrap.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ── 6. Summary ───────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("FINAL SUMMARY")
print(f"{'=' * 70}")
print(f"Dataset: American Gut Project (real data)")
print(f"Samples: {len(clr_df)} stool, {clr_df.shape[1]} taxa")
print(f"Bootstrap: {N_BOOTSTRAP} resamples of {SUBSAMPLE_SIZE} samples")
print(f"Comparisons: {', '.join(comparisons.keys())}")
n_sig = stat_df["significant_fdr"].sum()
n_nom = (stat_df["mannwhitney_p"] < 0.05).sum()
print(f"Nominally significant (p<0.05): {n_nom}/{len(stat_df)}")
print(f"FDR-significant: {n_sig}/{len(stat_df)}")
print("=" * 70)
