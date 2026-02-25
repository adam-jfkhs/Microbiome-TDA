#!/usr/bin/env python3
"""Bootstrap TDA analysis on IBDMDB / HMP2 metagenomics data.

Replicates the AGP v2 methodology in an independent clinical cohort, and adds
two novel biomarker-activity comparisons not possible with AGP:

  COMPARISONS
  ───────────
  1. ibd_vs_nonibd       — CD+UC vs. healthy controls  (replication)
  2. cd_vs_nonibd        — Crohn's disease vs. healthy controls
  3. uc_vs_nonibd        — Ulcerative colitis vs. healthy controls
  4. calprotectin        — high (≥250 μg/g) vs. low (<50 μg/g) inflammation
  5. hbi                 — active CD (HBI≥5) vs. remission CD (HBI<5)
  6. sccai               — active UC (SCCAI≥3) vs. remission UC (SCCAI<3)

Methodological notes
  - Global taxa selected once from all IBDMDB metagenomics samples.
  - MetaPhlAn2 species-level relative abundances → CLR-transformed.
  - Paired bootstrap identical to AGP v2 (sign-flip permutation null,
    Wilcoxon confirmatory, BH FDR across all tests per dataset).
  - IBDMDB is longitudinal; we treat each sample as independent (standard
    for cross-sectional TDA comparison; longitudinal analysis is future work).
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

from src.data.ibdmdb_loader import load_ibdmdb, ibdmdb_group_ids
from src.data.preprocess import filter_low_abundance, clr_transform
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.homology import compute_persistence, filter_infinite, persistence_summary
from src.tda.features import betti_curve, persistence_entropy
from src.analysis.statistics import fdr_correction

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200
SUBSAMPLE_SIZE = 60       # smaller than AGP because nonIBD group is n=364
N_PERMUTATIONS = 500
N_GLOBAL_TAXA = 80
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES_TO_TEST = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]

COMPARISONS = [
    "ibd_vs_nonibd",
    "cd_vs_nonibd",
    "uc_vs_nonibd",
    "high_vs_low_calprotectin",
    "high_vs_low_hbi",
    "high_vs_low_sccai",
]


# ── Helpers (identical to AGP v2) ─────────────────────────────────────────────

def select_global_taxa(clr_df, n=N_GLOBAL_TAXA):
    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top = prevalence.nlargest(n).index.tolist()
    print(f"Global taxa selected: {len(top)} (from {clr_df.shape[1]} total)")
    return top


def tda_features_fixed_taxa(clr_subset, global_taxa):
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
        "h1_count":             summary["H1"]["count"],
        "h1_entropy":           h1_entropy,
        "h1_total_persistence": total_pers,
        "h1_mean_lifetime":     summary["H1"]["mean_lifetime"],
        "h1_max_lifetime":      summary["H1"]["max_lifetime"],
        "max_betti1":           int(betti1.max()),
    }


def paired_resample_test(clr_df, ids_a, ids_b, global_taxa,
                         n_iter, subsample_size, n_perm, rng, label=""):
    ids_a = list(ids_a)
    ids_b = list(ids_b)
    n_a = min(subsample_size, len(ids_a))
    n_b = min(subsample_size, len(ids_b))

    if n_a < 20 or n_b < 20:
        print(f"  SKIP {label}: too few samples (n_a={n_a}, n_b={n_b})")
        return None, None

    print(f"  {label}: {len(ids_a)} vs {len(ids_b)} | drawing {n_a} vs {n_b} per iter × {n_iter}")

    deltas = {feat: [] for feat in FEATURES_TO_TEST}
    feat_a_all = {feat: [] for feat in FEATURES_TO_TEST}
    feat_b_all = {feat: [] for feat in FEATURES_TO_TEST}

    t0 = time.time()
    for i in range(n_iter):
        boot_a = rng.choice(ids_a, size=n_a, replace=False)
        boot_b = rng.choice(ids_b, size=n_b, replace=False)

        fa = tda_features_fixed_taxa(clr_df.loc[boot_a].reset_index(drop=True), global_taxa)
        fb = tda_features_fixed_taxa(clr_df.loc[boot_b].reset_index(drop=True), global_taxa)

        for feat in FEATURES_TO_TEST:
            deltas[feat].append(fa[feat] - fb[feat])
            feat_a_all[feat].append(fa[feat])
            feat_b_all[feat].append(fb[feat])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    iteration {i + 1}/{n_iter}  ({elapsed:.0f}s elapsed)")

    rows = []
    for feat in FEATURES_TO_TEST:
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


def plot_delta_distributions(deltas, comp_name, label_a, label_b, results_df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"IBDMDB Bootstrap TDA — {label_a} vs {label_b}\n"
        f"({N_ITERATIONS} iterations × {SUBSAMPLE_SIZE} samples, "
        f"global top-{N_GLOBAL_TAXA} taxa)",
        fontsize=12, fontweight="bold",
    )

    for idx, feat in enumerate(FEATURES_TO_TEST):
        ax = axes.flat[idx]
        d = deltas[feat]
        ax.hist(d, bins=25, alpha=0.75, color="#1565c0",
                edgecolor="white", linewidth=0.5, density=True)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

        row = results_df[results_df["feature"] == feat].iloc[0]
        title = (f"{feat}\nd={row['cohens_d']:.2f}  "
                 f"perm-p={row['permutation_p']:.3f}  "
                 f"wilcox-p={row['wilcoxon_p']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("delta (A − B)")
        ax.set_ylabel("density")

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f"ibdmdb_{comp_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("IBDMDB BOOTSTRAP TDA — Independent Validation")
    print("=" * 70)

    # Load & preprocess
    print("\nLoading IBDMDB data...")
    abundance_df, metadata_df = load_ibdmdb()
    print(f"Raw: {abundance_df.shape[0]} samples × {abundance_df.shape[1]} species")
    print(f"Diagnosis breakdown: {metadata_df['diagnosis'].value_counts().to_dict()}")

    # Filter and CLR-transform
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    meta = metadata_df.loc[clr_df.index]
    print(f"After filtering: {clr_df.shape[0]} samples × {clr_df.shape[1]} taxa")

    # Select global taxa once
    print("\nSelecting global taxa...")
    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # Report biomarker availability
    calp = meta["Tube B:Fecal Calprotectin"].notna().sum()
    hbi_n = meta["hbi"].notna().sum()
    sccai_n = meta["sccai"].notna().sum()
    print(f"\nBiomarker availability: calprotectin={calp}, HBI={hbi_n}, SCCAI={sccai_n}")

    # Run comparisons
    all_results = []

    for comp_name in COMPARISONS:
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {comp_name}")
        print(f"{'=' * 70}")

        ids_a, ids_b, label_a, label_b = ibdmdb_group_ids(meta, comp_name)
        ids_a = [i for i in ids_a if i in clr_df.index]
        ids_b = [i for i in ids_b if i in clr_df.index]
        print(f"  Group sizes: {label_a} n={len(ids_a)}, {label_b} n={len(ids_b)}")

        results_df, raw = paired_resample_test(
            clr_df, ids_a, ids_b, global_taxa,
            n_iter=N_ITERATIONS, subsample_size=SUBSAMPLE_SIZE,
            n_perm=N_PERMUTATIONS, rng=rng,
            label=f"{label_a} vs {label_b}",
        )

        if results_df is None:
            continue

        results_df["comparison"] = comp_name
        results_df["label_a"] = label_a
        results_df["label_b"] = label_b
        all_results.append(results_df)

        print(f"\n  Results:")
        print(f"  {'Feature':<28} {'Mean A':>7} {'Mean B':>7} "
              f"{'Cohen d':>8} {'perm-p':>8} {'wilcox-p':>9}")
        print(f"  {'─' * 68}")
        for _, row in results_df.iterrows():
            sig = "*" if row["permutation_p"] < 0.05 else ""
            print(f"  {row['feature']:<28} {row['mean_a']:>7.3f} {row['mean_b']:>7.3f} "
                  f"{row['cohens_d']:>8.3f} {row['permutation_p']:>8.4f} "
                  f"{row['wilcoxon_p']:>9.4f} {sig}")

        plot_delta_distributions(raw["deltas"], comp_name, label_a, label_b, results_df)

    if not all_results:
        print("\nNo comparisons completed.")
        return

    # Collate and apply FDR
    all_df = pd.concat(all_results, ignore_index=True)

    rejected_perm, adj_perm = fdr_correction(all_df["permutation_p"].values)
    all_df["perm_p_fdr"] = adj_perm
    all_df["sig_perm_fdr"] = rejected_perm

    rejected_wilcox, adj_wilcox = fdr_correction(all_df["wilcoxon_p"].values)
    all_df["wilcox_p_fdr"] = adj_wilcox
    all_df["sig_wilcox_fdr"] = rejected_wilcox

    out_path = os.path.join(RESULTS_DIR, "ibdmdb_bootstrap.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — FDR-significant results (permutation p, BH-corrected)")
    print(f"{'=' * 70}")
    sig = all_df[all_df["sig_perm_fdr"]]
    if len(sig) > 0:
        print(sig[["comparison", "feature", "mean_a", "mean_b",
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
