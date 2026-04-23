#!/usr/bin/env python3
"""
Third-Dataset Validation & Shannon Comparison
==============================================

Runs the TDA pipeline on the IBDMDB dataset and directly compares
topological features vs classical Shannon diversity for separating
IBD from non-IBD samples.

Outputs:
  - Cohen's d for each topological feature (IBD vs nonIBD)
  - Cohen's d for Shannon diversity (IBD vs nonIBD)
  - AUC comparison: TDA features vs Shannon-only vs combined
  - results/validation_cohens_d.csv
  - results/validation_auc_comparison.csv
  - figures/validation_tda_vs_shannon.png
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
from src.tda.sample_features import h1_features, compute_per_sample_topology

SEED = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS = 40
DATA_DIR = os.path.join(ROOT, "data", "raw")
FIGURE_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]


def shannon_diversity(abundance_df):
    """Compute Shannon diversity index for each sample."""
    rel = abundance_df.div(abundance_df.sum(axis=1), axis=0)
    rel = rel.replace(0, np.nan)
    H = -(rel * np.log(rel)).sum(axis=1)
    return H.fillna(0).values


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def main():
    print("=" * 60)
    print("VALIDATION: TDA vs Shannon Diversity on IBDMDB")
    print("=" * 60)

    # ── Load and preprocess ──────────────────────────────────────
    print("\n1. Loading IBDMDB data...")
    abundance, metadata = load_ibdmdb(os.path.join(DATA_DIR, "ibdmdb"))

    metadata["ibd_label"] = (metadata["diagnosis"].isin(["CD", "UC"])).astype(int)
    common = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[common]
    metadata = metadata.loc[common]
    print(f"   {len(common)} samples: {metadata['diagnosis'].value_counts().to_dict()}")

    # ── Shannon diversity (on raw abundances) ────────────────────
    print("\n2. Computing Shannon diversity...")
    shannon = shannon_diversity(abundance)
    print(f"   Shannon range: {shannon.min():.2f} – {shannon.max():.2f}")

    # ── Preprocess for TDA ───────────────────────────────────────
    print("\n3. Preprocessing for TDA...")
    filtered = filter_low_abundance(abundance, min_prevalence=0.05, min_reads=0)
    clr = clr_transform(filtered)
    top_taxa = select_global_taxa(clr, n=N_GLOBAL_TAXA)
    clr_matrix = clr[top_taxa].values.astype(np.float64)
    print(f"   CLR matrix: {clr_matrix.shape}")

    # ── Per-sample topology ──────────────────────────────────────
    print(f"\n4. Computing per-sample TDA features (k={K_NEIGHBOURS})...")
    print(f"   This will process {clr_matrix.shape[0]} samples...")
    t0 = time.time()
    topo_features = compute_per_sample_topology(clr_matrix, k=K_NEIGHBOURS, verbose=True)
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s")

    # ── Labels ───────────────────────────────────────────────────
    labels = metadata["ibd_label"].values
    ibd_mask = labels == 1
    non_mask = labels == 0

    # ── Cohen's d: TDA features ──────────────────────────────────
    print("\n5. Cohen's d — TDA features (IBD vs nonIBD):")
    print(f"   {'Feature':<25} {'Cohen d':>10} {'p-value':>12} {'Significant':>12}")
    print("   " + "-" * 60)

    d_rows = []
    for j, feat_name in enumerate(FEATURE_NAMES):
        vals_ibd = topo_features[ibd_mask, j]
        vals_non = topo_features[non_mask, j]
        d = cohens_d(vals_ibd, vals_non)
        _, pval = mannwhitneyu(vals_ibd, vals_non, alternative="two-sided")
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"   {feat_name:<25} {d:>10.4f} {pval:>12.6f} {sig:>12}")
        d_rows.append({
            "feature": feat_name, "cohens_d": round(d, 4),
            "p_value": round(pval, 8), "type": "TDA"
        })

    # ── Cohen's d: Shannon ───────────────────────────────────────
    print(f"\n   {'Shannon diversity':<25}", end="")
    d_shannon = cohens_d(shannon[ibd_mask], shannon[non_mask])
    _, p_shannon = mannwhitneyu(shannon[ibd_mask], shannon[non_mask], alternative="two-sided")
    sig_sh = "***" if p_shannon < 0.001 else ("**" if p_shannon < 0.01 else ("*" if p_shannon < 0.05 else ""))
    print(f"{d_shannon:>10.4f} {p_shannon:>12.6f} {sig_sh:>12}")
    d_rows.append({
        "feature": "shannon_diversity", "cohens_d": round(d_shannon, 4),
        "p_value": round(p_shannon, 8), "type": "Shannon"
    })

    pd.DataFrame(d_rows).to_csv(
        os.path.join(RESULTS_DIR, "validation_cohens_d.csv"), index=False
    )
    print(f"\n   Saved: results/validation_cohens_d.csv")

    # ── AUC comparison: TDA vs Shannon vs Combined ───────────────
    print("\n6. AUC comparison (5-fold cross-validation):")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=SEED, max_iter=1000)),
    ])

    # Shannon only
    auc_shannon = cross_val_score(
        pipe, shannon.reshape(-1, 1), labels, cv=cv, scoring="roc_auc"
    )

    # TDA only
    auc_tda = cross_val_score(
        pipe, topo_features, labels, cv=cv, scoring="roc_auc"
    )

    # Combined
    X_combined = np.column_stack([topo_features, shannon.reshape(-1, 1)])
    auc_combined = cross_val_score(
        pipe, X_combined, labels, cv=cv, scoring="roc_auc"
    )

    print(f"   Shannon only:   AUC = {auc_shannon.mean():.4f} ± {auc_shannon.std():.4f}")
    print(f"   TDA only:       AUC = {auc_tda.mean():.4f} ± {auc_tda.std():.4f}")
    print(f"   TDA + Shannon:  AUC = {auc_combined.mean():.4f} ± {auc_combined.std():.4f}")

    auc_df = pd.DataFrame({
        "method": ["Shannon only", "TDA only", "TDA + Shannon"],
        "mean_auc": [auc_shannon.mean(), auc_tda.mean(), auc_combined.mean()],
        "std_auc": [auc_shannon.std(), auc_tda.std(), auc_combined.std()],
    })
    auc_df.to_csv(os.path.join(RESULTS_DIR, "validation_auc_comparison.csv"), index=False)
    print(f"   Saved: results/validation_auc_comparison.csv")

    # ── Figure ───────────────────────────────────────────────────
    print("\n7. Generating comparison figure...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Cohen's d comparison
    ax = axes[0]
    features_for_plot = FEATURE_NAMES + ["Shannon"]
    d_values = [d_rows[i]["cohens_d"] for i in range(6)] + [d_shannon]
    colors = ["#2196F3" if abs(d) >= 0.8 else "#90CAF9" if abs(d) >= 0.5 else "#E0E0E0"
              for d in d_values]
    bars = ax.barh(features_for_plot, d_values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=-0.8, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Large effect (|d|=0.8)")
    ax.axvline(x=0.8, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Cohen's d (IBD vs nonIBD)")
    ax.set_title("A. Effect Sizes: TDA Features vs Shannon")
    ax.legend(fontsize=8)

    # Panel B: AUC comparison
    ax = axes[1]
    methods = ["Shannon\nonly", "TDA\nonly", "TDA +\nShannon"]
    means = [auc_shannon.mean(), auc_tda.mean(), auc_combined.mean()]
    stds = [auc_shannon.std(), auc_tda.std(), auc_combined.std()]
    bar_colors = ["#FF9800", "#2196F3", "#4CAF50"]
    ax.bar(methods, means, yerr=stds, color=bar_colors, edgecolor="black",
           linewidth=0.5, capsize=5)
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("B. Classification: IBD vs nonIBD")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax.legend(fontsize=8)

    # Panel C: Distribution comparison (best TDA feature vs Shannon)
    ax = axes[2]
    best_idx = int(np.argmax([abs(d_rows[i]["cohens_d"]) for i in range(6)]))
    best_feat = FEATURE_NAMES[best_idx]
    ax.hist(topo_features[non_mask, best_idx], bins=30, alpha=0.6,
            label=f"nonIBD ({best_feat})", color="#4CAF50", density=True)
    ax.hist(topo_features[ibd_mask, best_idx], bins=30, alpha=0.6,
            label=f"IBD ({best_feat})", color="#F44336", density=True)
    ax.set_xlabel(best_feat)
    ax.set_ylabel("Density")
    ax.set_title(f"C. Distribution: Best TDA Feature")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "validation_tda_vs_shannon.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {fig_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    tda_advantage = auc_tda.mean() - auc_shannon.mean()
    combined_advantage = auc_combined.mean() - auc_shannon.mean()
    print(f"  TDA captures {tda_advantage:+.4f} AUC over Shannon alone")
    print(f"  Combined captures {combined_advantage:+.4f} AUC over Shannon alone")
    max_d_tda = max(abs(d_rows[i]["cohens_d"]) for i in range(6))
    print(f"  Largest TDA Cohen's d:  {max_d_tda:.4f}")
    print(f"  Shannon Cohen's d:      {abs(d_shannon):.4f}")
    print(f"  TDA effect is {max_d_tda/abs(d_shannon):.1f}x larger than Shannon")
    print("=" * 60)


if __name__ == "__main__":
    main()
