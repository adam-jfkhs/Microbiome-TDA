#!/usr/bin/env python3
"""
Robustness test: How stable is the classifier AUC across random seeds,
split ratios, and cross-validation configurations?

Tests:
  1. Seed stability: 20 different random seeds, same 5-fold CV
  2. Split ratio stability: 60/40, 70/30, 80/20, 90/10 holdout splits
  3. CV fold stability: 3, 5, 7, 10-fold CV
  4. Subsampling stability: what happens with fewer samples?

All on IBDMDB data (n=1,338, IBD=974, nonIBD=364).

Outputs
-------
  Console:  summary statistics
  figures/auc_robustness.png
  results/auc_robustness.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
from src.tda.homology import compute_persistence, filter_infinite
from src.tda.sample_features import h1_features, compute_per_sample_topology

N_GLOBAL_TAXA = 80
K_NEIGHBOURS = 40  # Smaller than scripts/ (60) — appropriate for IBDMDB size
DATA_DIR = os.path.join(ROOT, "data", "raw")
FIGURE_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# _h1_features and compute_per_sample_topology imported from src.tda.sample_features


def compute_shannon(abundance_df):
    raw = abundance_df.values.astype(np.float64)
    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    p = raw / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -(p * logp).sum(axis=1).reshape(-1, 1)


def cv_auc(X, y, seed, n_splits=5):
    """Run stratified CV and return per-fold AUCs."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=seed)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        probas = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], probas))
    return fold_aucs


def holdout_auc(X, y, seed, test_size=0.20):
    """Single train/test split AUC."""
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             random_state=seed)
    clf.fit(scaler.transform(X[train_idx]), y[train_idx])
    probas = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    return roc_auc_score(y[test_idx], probas)


def main():
    t_start = time.time()

    # ── Load and preprocess ───────────────────────────────────────────────────
    print("Loading IBDMDB data ...")
    abundance_df, metadata = load_ibdmdb(os.path.join(DATA_DIR, "ibdmdb"))
    diag = metadata["diagnosis"]
    keep = diag.isin(["CD", "UC", "nonIBD"])
    abundance_df = abundance_df.loc[keep]
    metadata = metadata.loc[keep]
    y_series = diag.loc[keep].isin(["CD", "UC"]).astype(int)

    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    taxa = select_global_taxa(clr_df, N_GLOBAL_TAXA)
    clr_taxa = clr_df[taxa]
    common = clr_taxa.index
    y = y_series.loc[common].values.astype(int)

    print(f"  n={len(y)} | IBD={y.sum()} | nonIBD={(y==0).sum()}")

    # ── Compute features (once) ───────────────────────────────────────────────
    print("Computing features ...")
    X_shannon = compute_shannon(abundance_df.loc[common])
    print("  Computing per-sample topology ...")
    X_topo = compute_per_sample_topology(clr_taxa.values.astype(np.float64))

    # Feature sets to test
    feature_sets = {
        "Shannon only": X_shannon,
        "Topology only": X_topo,
        "Topo + Shannon": np.hstack([X_topo, X_shannon]),
    }

    all_rows = []

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: Seed stability — 20 seeds, 5-fold CV
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: SEED STABILITY (20 random seeds, 5-fold CV)")
    print("=" * 70)

    seeds = list(range(20))
    for name, X in feature_sets.items():
        aucs = []
        for seed in seeds:
            fold_aucs = cv_auc(X, y, seed=seed, n_splits=5)
            mean_auc = np.mean(fold_aucs)
            aucs.append(mean_auc)
            all_rows.append({
                "test": "seed_stability",
                "feature_set": name,
                "seed": seed,
                "n_folds": 5,
                "test_size": None,
                "subsample_frac": 1.0,
                "mean_auc": round(mean_auc, 4),
                "std_auc": round(np.std(fold_aucs), 4),
            })

        print(f"  {name:<20s}  AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  "
              f"[range: {np.min(aucs):.3f} - {np.max(aucs):.3f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: Split ratio stability — holdout at different ratios
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: SPLIT RATIO STABILITY (5 seeds x 4 ratios)")
    print("=" * 70)

    test_sizes = [0.40, 0.30, 0.20, 0.10]
    for name, X in feature_sets.items():
        print(f"\n  {name}:")
        for ts in test_sizes:
            aucs = [holdout_auc(X, y, seed=s, test_size=ts) for s in range(5)]
            print(f"    {1-ts:.0%}/{ts:.0%} split:  "
                  f"AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
            for s, a in zip(range(5), aucs):
                all_rows.append({
                    "test": "split_ratio",
                    "feature_set": name,
                    "seed": s,
                    "n_folds": None,
                    "test_size": ts,
                    "subsample_frac": 1.0,
                    "mean_auc": round(a, 4),
                    "std_auc": None,
                })

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: CV fold count stability
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: CV FOLD STABILITY (3, 5, 7, 10-fold, seed=42)")
    print("=" * 70)

    fold_counts = [3, 5, 7, 10]
    for name, X in feature_sets.items():
        print(f"\n  {name}:")
        for nf in fold_counts:
            fold_aucs = cv_auc(X, y, seed=42, n_splits=nf)
            mean_a = np.mean(fold_aucs)
            std_a = np.std(fold_aucs)
            print(f"    {nf:>2d}-fold:  AUC = {mean_a:.3f} +/- {std_a:.3f}  "
                  f"[folds: {', '.join(f'{a:.3f}' for a in fold_aucs)}]")
            all_rows.append({
                "test": "cv_folds",
                "feature_set": name,
                "seed": 42,
                "n_folds": nf,
                "test_size": None,
                "subsample_frac": 1.0,
                "mean_auc": round(mean_a, 4),
                "std_auc": round(std_a, 4),
            })

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4: Subsampling stability — what if we had fewer samples?
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: SUBSAMPLING STABILITY (25%, 50%, 75%, 100% of data)")
    print("=" * 70)

    fracs = [0.25, 0.50, 0.75, 1.0]
    rng = np.random.default_rng(42)
    for name, X in feature_sets.items():
        print(f"\n  {name}:")
        for frac in fracs:
            aucs = []
            for s in range(5):
                rng_sub = np.random.default_rng(s)
                n_sub = max(int(len(y) * frac), 20)
                idx = rng_sub.choice(len(y), size=n_sub, replace=False)
                # Ensure both classes present
                if len(np.unique(y[idx])) < 2:
                    continue
                fold_aucs = cv_auc(X[idx], y[idx], seed=s, n_splits=min(5, n_sub // 10))
                aucs.append(np.mean(fold_aucs))
                all_rows.append({
                    "test": "subsampling",
                    "feature_set": name,
                    "seed": s,
                    "n_folds": min(5, n_sub // 10),
                    "test_size": None,
                    "subsample_frac": frac,
                    "mean_auc": round(np.mean(fold_aucs), 4),
                    "std_auc": round(np.std(fold_aucs), 4),
                })
            if aucs:
                n_sub = int(len(y) * frac)
                print(f"    {frac:>4.0%} (n={n_sub:>5d}):  "
                      f"AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 70)
    print("#" + " " * 22 + "ROBUSTNESS SUMMARY" + " " * 28 + "#")
    print("#" * 70)

    results_df = pd.DataFrame(all_rows)

    # Seed stability summary
    seed_results = results_df[results_df["test"] == "seed_stability"]
    print("\n  Seed stability (20 seeds, 5-fold CV):")
    for name in feature_sets:
        subset = seed_results[seed_results["feature_set"] == name]["mean_auc"]
        print(f"    {name:<20s}  {subset.mean():.3f} +/- {subset.std():.3f}  "
              f"  95% CI: [{subset.mean() - 1.96*subset.std():.3f}, "
              f"{subset.mean() + 1.96*subset.std():.3f}]")

    # Is the signal real?
    topo_shannon = seed_results[seed_results["feature_set"] == "Topo + Shannon"]["mean_auc"]
    print(f"\n  VERDICT:")
    print(f"    Topo + Shannon AUC across 20 seeds: "
          f"{topo_shannon.mean():.3f} +/- {topo_shannon.std():.3f}")
    print(f"    Minimum AUC observed: {topo_shannon.min():.3f}")
    print(f"    Maximum AUC observed: {topo_shannon.max():.3f}")
    if topo_shannon.min() > 0.55:
        print(f"    All 20 seeds above 0.55 — THE SIGNAL IS REAL.")
    else:
        print(f"    Some seeds below 0.55 — signal may be marginal.")

    # Does topology add value over Shannon?
    shannon_aucs = seed_results[seed_results["feature_set"] == "Shannon only"]["mean_auc"].values
    combined_aucs = seed_results[seed_results["feature_set"] == "Topo + Shannon"]["mean_auc"].values
    improvement = combined_aucs - shannon_aucs
    print(f"\n    Topology improvement over Shannon (per seed):")
    print(f"      Mean improvement: {improvement.mean():+.3f}")
    print(f"      Positive in {(improvement > 0).sum()}/20 seeds "
          f"({(improvement > 0).mean():.0%})")

    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(improvement, alternative="greater")
    print(f"      Wilcoxon one-sided p = {pval:.4f}")
    if pval < 0.05:
        print(f"      TOPOLOGY SIGNIFICANTLY IMPROVES OVER SHANNON (p < 0.05)")
    else:
        print(f"      Improvement not statistically significant (p = {pval:.3f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "auc_robustness.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AUC Robustness Analysis — IBDMDB (n=1,338)\n"
                 "Is the topological signal real and stable?",
                 fontsize=12, fontweight="bold")

    colors = {"Shannon only": "#a6cee3", "Topology only": "#ff7f00",
              "Topo + Shannon": "#1f78b4"}

    # Panel 1: Seed stability
    ax = axes[0, 0]
    for name in feature_sets:
        subset = seed_results[seed_results["feature_set"] == name]
        ax.plot(subset["seed"].values, subset["mean_auc"].values,
                "o-", color=colors[name], label=name, markersize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Mean 5-fold AUC")
    ax.set_title("Seed Stability (20 seeds)", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(0.45, 0.80)

    # Panel 2: Split ratio
    ax = axes[0, 1]
    split_results = results_df[results_df["test"] == "split_ratio"]
    for name in feature_sets:
        subset = split_results[split_results["feature_set"] == name]
        means = subset.groupby("test_size")["mean_auc"].mean()
        stds = subset.groupby("test_size")["mean_auc"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt="o-", color=colors[name], label=name,
                    markersize=5, capsize=3)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Test Set Fraction")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("Split Ratio Stability", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(0.45, 0.80)

    # Panel 3: CV folds
    ax = axes[1, 0]
    cv_results = results_df[results_df["test"] == "cv_folds"]
    for name in feature_sets:
        subset = cv_results[cv_results["feature_set"] == name]
        ax.errorbar(subset["n_folds"].values, subset["mean_auc"].values,
                    yerr=subset["std_auc"].values,
                    fmt="o-", color=colors[name], label=name,
                    markersize=5, capsize=3)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Number of CV Folds")
    ax.set_ylabel("Mean AUC")
    ax.set_title("CV Fold Stability", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xticks(fold_counts)
    ax.set_ylim(0.45, 0.80)

    # Panel 4: Subsampling
    ax = axes[1, 1]
    sub_results = results_df[results_df["test"] == "subsampling"]
    for name in feature_sets:
        subset = sub_results[sub_results["feature_set"] == name]
        means = subset.groupby("subsample_frac")["mean_auc"].mean()
        stds = subset.groupby("subsample_frac")["mean_auc"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt="o-", color=colors[name], label=name,
                    markersize=5, capsize=3)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Fraction of Data Used")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Subsampling Stability", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(0.45, 0.80)

    fig.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "auc_robustness.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
