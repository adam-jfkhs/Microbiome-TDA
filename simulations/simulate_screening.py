#!/usr/bin/env python3
"""
Clinical Simulation 2: Population Screening Cost-Benefit Analysis
=================================================================

Simulates "What if a hospital screened 10,000 patients with this tool?"
Uses real classifier performance from IBDMDB data to project screening
outcomes at different IBD prevalence rates.

Compares three screening strategies:
  1. Shannon-only pre-screening
  2. Topology-only pre-screening
  3. Combined Topology + Shannon pre-screening

At two operating points:
  - Balanced threshold (t=0.50): maximise detection
  - High-specificity threshold (95% spec): minimise false referrals

Outputs
-------
  Console:  summary tables
  figures/screening_cost_benefit.png
  results/screening_simulation.csv

Usage
-----
  python simulations/simulate_screening.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
from src.tda.homology import compute_persistence, filter_infinite

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS = 40
N_CV_FOLDS = 5
POPULATION = 10_000
PREVALENCES = [0.01, 0.03, 0.05, 0.10]
COLONOSCOPY_COST = 3_000  # USD approximate

DATA_DIR = os.path.join(ROOT, "data", "raw")
FIGURE_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _h1_features(dgm_h1):
    """Extract 6 H1 scalars from a persistence diagram."""
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else dgm_h1
    if len(finite) == 0:
        return [0, 0.0, 0.0, 0.0, 0.0, 0]
    lifetimes = finite[:, 1] - finite[:, 0]
    total_pers = float(lifetimes.sum())
    norm_lt = lifetimes / total_pers if total_pers > 0 else lifetimes
    entropy = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))
    births, deaths = finite[:, 0], finite[:, 1]
    thresholds = np.unique(np.concatenate([births, deaths]))
    max_betti = int(max(
        np.sum((births <= t) & (deaths > t)) for t in thresholds
    )) if len(thresholds) > 0 else 0
    return [len(finite), entropy, total_pers, float(lifetimes.mean()),
            float(lifetimes.max()), max_betti]


def compute_per_sample_topology(clr_matrix, k=K_NEIGHBOURS):
    """Per-sample H1 features via k-NN neighbourhood TDA."""
    n = clr_matrix.shape[0]
    k_actual = min(k, n - 1)
    out = np.zeros((n, 6), dtype=np.float32)
    dists = cdist(clr_matrix, clr_matrix, metric="euclidean")

    t0 = time.time()
    for i in range(n):
        if i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"    {i}/{n}  {elapsed:.0f}s elapsed")

        nn_idx = np.argsort(dists[i])[1:k_actual + 1]
        neighbourhood = clr_matrix[nn_idx]
        if neighbourhood.shape[0] < 3:
            continue
        corr_mat, _ = spearmanr(neighbourhood, axis=0)
        if corr_mat.ndim == 0:
            corr_mat = np.array([[1.0]])
        dist_mat = np.clip(1.0 - np.abs(corr_mat), 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)
        result = compute_persistence(dist_mat, maxdim=1, thresh=1.0)
        dgms = filter_infinite(result["dgms"])
        dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        out[i] = _h1_features(dgm_h1)
    return out


def compute_shannon(abundance_df):
    """Shannon H from abundance table."""
    raw = abundance_df.values.astype(np.float64)
    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    p = raw / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -(p * logp).sum(axis=1).reshape(-1, 1)


def get_classifier_metrics(X_topo, X_shannon, y):
    """5-fold CV to get sensitivity/specificity at two operating points."""
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    feature_sets = {}

    for name, X in [
        ("Shannon only", X_shannon),
        ("Topology only", X_topo),
        ("Topology + Shannon", np.hstack([X_topo, X_shannon])),
    ]:
        all_probas = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            scaler = StandardScaler().fit(X[train_idx])
            clf = LogisticRegression(C=1.0, max_iter=1000,
                                     class_weight="balanced",
                                     random_state=SEED)
            clf.fit(scaler.transform(X[train_idx]), y[train_idx])
            all_probas[test_idx] = clf.predict_proba(
                scaler.transform(X[test_idx]))[:, 1]

        auc = roc_auc_score(y, all_probas)
        fpr, tpr, thresholds = roc_curve(y, all_probas)
        specificity = 1 - fpr

        # Balanced (t=0.50)
        idx_b = np.argmin(np.abs(thresholds - 0.50))
        # 95% specificity
        idx_h = np.argmin(np.abs(specificity - 0.95))

        feature_sets[name] = {
            "auc": auc,
            "balanced": (float(tpr[idx_b]), float(specificity[idx_b])),
            "high_spec": (float(tpr[idx_h]), float(specificity[idx_h])),
        }
        print(f"  {name}: AUC={auc:.3f}  "
              f"balanced=({tpr[idx_b]:.2f}/{specificity[idx_b]:.2f})  "
              f"high_spec=({tpr[idx_h]:.2f}/{specificity[idx_h]:.2f})")

    return feature_sets


def simulate_screening(sensitivity, specificity, prevalence, population):
    """Simulate screening outcomes."""
    n_ibd = int(population * prevalence)
    n_healthy = population - n_ibd
    tp = int(n_ibd * sensitivity)
    fn = n_ibd - tp
    fp = int(n_healthy * (1 - specificity))
    tn = n_healthy - fp
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv,
        "referrals": tp + fp, "missed": fn, "unnecessary": fp,
    }


def print_screening_table(results_df, operating_point):
    """Print formatted screening table."""
    subset = results_df[results_df["operating_point"] == operating_point]
    op_label = "BALANCED (t=0.50)" if operating_point == "balanced" else "HIGH SPECIFICITY (95%)"

    print(f"\n{'=' * 90}")
    print(f"  SCREENING SIMULATION: {op_label}  ({POPULATION:,} patients)")
    print(f"{'=' * 90}")
    print(f"  {'Strategy':<28s} {'Prev':>5s} {'Detected':>10s} "
          f"{'Missed':>7s} {'FalsePos':>9s} {'PPV':>6s} {'Referrals':>10s}")
    print(f"  {'-' * 82}")

    for _, row in subset.iterrows():
        n_ibd = int(POPULATION * row["prevalence"])
        print(f"  {row['strategy']:<28s} {row['prevalence']:>5.0%} "
              f"{row['tp']:>5d}/{n_ibd:<4d} "
              f"{row['missed']:>5d}   {row['unnecessary']:>7d}   "
              f"{row['ppv']:>5.1%}  {row['referrals']:>8d}")
    print()


def plot_screening(results_df, out_path):
    """Generate screening cost-benefit figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Population Screening Simulation: {POPULATION:,} Patients\n"
        "Microbiome Topology-Based IBD Screening",
        fontsize=12, fontweight="bold",
    )

    strategies = results_df["strategy"].unique()
    colors = {
        "Shannon only": "#a6cee3",
        "Topology only": "#ff7f00",
        "Topology + Shannon": "#1f78b4",
    }

    for col, op in enumerate(["balanced", "high_spec"]):
        subset = results_df[results_df["operating_point"] == op]
        op_label = "Balanced (t=0.50)" if op == "balanced" else "High Specificity (95%)"

        # Top: detected cases
        ax = axes[0, col]
        x = np.arange(len(PREVALENCES))
        w = 0.25
        for i, strat in enumerate(strategies):
            sd = subset[subset["strategy"] == strat]
            ax.bar(x + i * w, sd["tp"].values, w * 0.9,
                   color=colors.get(strat, "#999"), label=strat, edgecolor="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"{p:.0%}" for p in PREVALENCES])
        ax.set_xlabel("IBD Prevalence")
        ax.set_ylabel("IBD Cases Detected")
        ax.set_title(f"Cases Detected — {op_label}", fontsize=10)
        ax.legend(fontsize=7)

        # Bottom: false positives
        ax = axes[1, col]
        for i, strat in enumerate(strategies):
            sd = subset[subset["strategy"] == strat]
            ax.bar(x + i * w, sd["unnecessary"].values, w * 0.9,
                   color=colors.get(strat, "#999"), label=strat, edgecolor="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"{p:.0%}" for p in PREVALENCES])
        ax.set_xlabel("IBD Prevalence")
        ax.set_ylabel("Unnecessary Colonoscopies")
        ax.set_title(f"False Positives — {op_label}", fontsize=10)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def main():
    t_start = time.time()

    # ── Load IBDMDB ───────────────────────────────────────────────────────────
    print("Loading IBDMDB data ...")
    abundance_df, metadata = load_ibdmdb(os.path.join(DATA_DIR, "ibdmdb"))

    diag = metadata["diagnosis"]
    keep = diag.isin(["CD", "UC", "nonIBD"])
    abundance_df = abundance_df.loc[keep]
    metadata = metadata.loc[keep]
    y_series = diag.loc[keep].isin(["CD", "UC"]).astype(int)

    print(f"  n = {len(abundance_df)} | IBD = {y_series.sum()} | "
          f"nonIBD = {(y_series == 0).sum()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    taxa = select_global_taxa(clr_df, N_GLOBAL_TAXA)
    clr_taxa = clr_df[taxa]
    common = clr_taxa.index
    y = y_series.loc[common].values.astype(int)

    # ── Features ──────────────────────────────────────────────────────────────
    X_shannon = compute_shannon(abundance_df.loc[common])
    print("Computing per-sample topology ...")
    X_topo = compute_per_sample_topology(clr_taxa.values.astype(np.float64))

    # ── Classifier metrics ────────────────────────────────────────────────────
    print("\nComputing classifier metrics (5-fold CV) ...")
    feature_sets = get_classifier_metrics(X_topo, X_shannon, y)

    # ── Simulate screening ────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("#" + " " * 16 + "POPULATION SCREENING SIMULATION" + " " * 21 + "#")
    print("#" * 70)

    rows = []
    for op_name in ["balanced", "high_spec"]:
        for strat_name, metrics in feature_sets.items():
            sens, spec = metrics[op_name]
            for prev in PREVALENCES:
                result = simulate_screening(sens, spec, prev, POPULATION)
                rows.append({
                    "strategy": strat_name,
                    "operating_point": op_name,
                    "prevalence": prev,
                    **result,
                })

    results_df = pd.DataFrame(rows)
    print_screening_table(results_df, "balanced")
    print_screening_table(results_df, "high_spec")

    # ── Cost analysis ─────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  COST ANALYSIS (@ ${COLONOSCOPY_COST:,}/colonoscopy)")
    print("=" * 70)
    print(f"  {'Strategy':<28s} {'Prev':>5s} {'Screening Cost':>15s} "
          f"{'vs All-Colon':>15s}")
    print(f"  {'-' * 68}")

    best = results_df[results_df["operating_point"] == "high_spec"]
    cost_all = POPULATION * COLONOSCOPY_COST
    for _, row in best.iterrows():
        cost = row["referrals"] * COLONOSCOPY_COST
        saved = cost_all - cost
        print(f"  {row['strategy']:<28s} {row['prevalence']:>5.0%} "
              f"${cost:>13,.0f}  ${saved:>13,.0f} saved")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "screening_simulation.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    fig_path = os.path.join(FIGURE_DIR, "screening_cost_benefit.png")
    plot_screening(results_df, fig_path)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
