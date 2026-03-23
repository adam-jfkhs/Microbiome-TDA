#!/usr/bin/env python3
"""
Clinical Simulation 4: Batch Triage Dashboard
==============================================

Scores all holdout patients and triages them into risk tiers (Low / Moderate /
High / Critical).  Compares topology-enhanced triage vs Shannon-only.

Demonstrates:
  - Risk stratification from microbiome topology
  - Cases where topology catches IBD that Shannon misses
  - Feature importance: which H1 features drive high-risk scores

Uses IBDMDB clinical data (n=1,338, with calprotectin and disease activity).

Outputs
-------
  Console:  triage summary tables
  figures/triage_dashboard.png
  results/triage_simulation.csv

Usage
-----
  python simulations/simulate_triage.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
DATA_DIR = os.path.join(ROOT, "data", "raw")
FIGURE_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TIERS = [
    ("LOW", 0, 25, "#2ca02c"),
    ("MODERATE", 25, 50, "#ff7f0e"),
    ("HIGH", 50, 75, "#d62728"),
    ("CRITICAL", 75, 100.01, "#7b2d8e"),
]

FEATURE_NAMES = [
    "H1 Count", "H1 Entropy", "H1 Total Persistence",
    "H1 Mean Lifetime", "H1 Max Lifetime", "Max Betti-1",
]


def _h1_features(dgm_h1):
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
    n = clr_matrix.shape[0]
    k_actual = min(k, n - 1)
    out = np.zeros((n, 6), dtype=np.float32)
    dists = cdist(clr_matrix, clr_matrix, metric="euclidean")
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{n}  {time.time() - t0:.0f}s elapsed")
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
    raw = abundance_df.values.astype(np.float64)
    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    p = raw / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -(p * logp).sum(axis=1).reshape(-1, 1)


def assign_tier(tds):
    for name, lo, hi, _ in TIERS:
        if lo <= tds < hi:
            return name
    return "CRITICAL"


def print_triage_table(triage_df, strategy_name, y_test):
    total_ibd = (y_test == 1).sum()
    cumulative_ibd = 0
    print(f"\n  {'=' * 60}")
    print(f"  TRIAGE: {strategy_name}")
    print(f"  {'=' * 60}")
    print(f"  {'Tier':<12s} {'Count':>6s} {'IBD':>5s} {'Healthy':>8s} "
          f"{'IBD Rate':>9s} {'Cumul Sens':>11s}")
    print(f"  {'-' * 55}")

    for tier_name, _, _, _ in TIERS:
        mask = triage_df["tier"] == tier_name
        n = mask.sum()
        n_ibd = (triage_df.loc[mask, "true_label"] == 1).sum()
        n_healthy = n - n_ibd
        ibd_rate = n_ibd / n if n > 0 else 0
        cumulative_ibd += n_ibd
        cumul_sens = cumulative_ibd / total_ibd if total_ibd > 0 else 0
        marker = " <--" if ibd_rate > 0.15 else ""
        print(f"  {tier_name:<12s} {n:>6d} {n_ibd:>5d} {n_healthy:>8d} "
              f"{ibd_rate:>8.1%} {cumul_sens:>10.1%}{marker}")

    print(f"  {'-' * 55}")
    print(f"  {'TOTAL':<12s} {len(triage_df):>6d} {total_ibd:>5d} "
          f"{(y_test == 0).sum():>8d}")


def plot_triage_dashboard(all_triage, topo_coefs, out_path):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Batch Triage Dashboard — Microbiome Topological Dysbiosis Screening\n"
        "Risk tier assignment on 20% holdout (IBDMDB clinical data)",
        fontsize=12, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    strategies = list(all_triage.keys())

    # Panel 1: IBD rate by tier
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(strategies))
    w = 0.18
    for i, (tier_name, _, _, color) in enumerate(TIERS):
        rates = []
        for strat in strategies:
            df = all_triage[strat]
            mask = df["tier"] == tier_name
            n = mask.sum()
            n_ibd = (df.loc[mask, "true_label"] == 1).sum()
            rates.append(n_ibd / n if n > 0 else 0)
        ax1.bar(x + i * w, rates, w * 0.9, color=color, label=tier_name,
                edgecolor="white")
    ax1.set_xticks(x + 1.5 * w)
    ax1.set_xticklabels([s[:18] for s in strategies], fontsize=7, rotation=15)
    ax1.set_ylabel("IBD Rate in Tier")
    ax1.set_title("IBD Prevalence by Risk Tier", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Panel 2: Patient distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (tier_name, _, _, color) in enumerate(TIERS):
        counts = [(all_triage[s]["tier"] == tier_name).sum() for s in strategies]
        ax2.bar(x + i * w, counts, w * 0.9, color=color, label=tier_name,
                edgecolor="white")
    ax2.set_xticks(x + 1.5 * w)
    ax2.set_xticklabels([s[:18] for s in strategies], fontsize=7, rotation=15)
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Patient Distribution by Tier", fontsize=10)
    ax2.legend(fontsize=7)

    # Panel 3: Feature importance
    ax3 = fig.add_subplot(gs[0, 2])
    importance = np.abs(topo_coefs)
    sorted_idx = np.argsort(importance)[::-1]
    colors_imp = ["#1f78b4" if topo_coefs[i] > 0 else "#d62728"
                  for i in sorted_idx]
    ax3.barh(np.arange(6), importance[sorted_idx], color=colors_imp,
             edgecolor="white")
    ax3.set_yticks(np.arange(6))
    ax3.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=8)
    ax3.set_xlabel("|Coefficient|")
    ax3.set_title("Topology Feature Importance\n(Combined LR)", fontsize=10)
    ax3.invert_yaxis()

    # Panel 4: Shannon vs Combined scatter
    ax4 = fig.add_subplot(gs[1, :2])
    if "Shannon only" in all_triage and "Topo + Shannon" in all_triage:
        s_df = all_triage["Shannon only"]
        c_df = all_triage["Topo + Shannon"]
        merged = pd.DataFrame({
            "true_label": s_df["true_label"].values,
            "shannon_tds": s_df["tds"].values,
            "combined_tds": c_df["tds"].values,
        })
        ibd = merged[merged["true_label"] == 1]
        healthy = merged[merged["true_label"] == 0]

        ax4.scatter(healthy["shannon_tds"], healthy["combined_tds"],
                    alpha=0.3, s=8, c="#2ca02c", label="nonIBD", zorder=3)
        ax4.scatter(ibd["shannon_tds"], ibd["combined_tds"],
                    alpha=0.7, s=25, c="#d62728", marker="^", label="IBD",
                    zorder=4)
        ax4.axhline(50, color="gray", lw=0.8, ls="--")
        ax4.axvline(50, color="gray", lw=0.8, ls="--")

        topo_catches = ((ibd["shannon_tds"] < 50) & (ibd["combined_tds"] >= 50)).sum()
        shannon_catches = ((ibd["shannon_tds"] >= 50) & (ibd["combined_tds"] < 50)).sum()
        both_catch = ((ibd["shannon_tds"] >= 50) & (ibd["combined_tds"] >= 50)).sum()
        both_miss = ((ibd["shannon_tds"] < 50) & (ibd["combined_tds"] < 50)).sum()

        ax4.text(75, 25, f"Shannon only\n(n={shannon_catches})", ha="center",
                 va="center", fontsize=7, color="#666")
        ax4.text(25, 75, f"Topology adds\n(n={topo_catches})", ha="center",
                 va="center", fontsize=7, color="#1f78b4", fontweight="bold")
        ax4.text(75, 75, f"Both catch\n(n={both_catch})", ha="center",
                 va="center", fontsize=7, color="#2ca02c")
        ax4.text(25, 25, f"Both miss\n(n={both_miss})", ha="center",
                 va="center", fontsize=7, color="#999")

        ax4.set_xlabel("Shannon-Only TDS", fontsize=9)
        ax4.set_ylabel("Topology + Shannon TDS", fontsize=9)
        ax4.set_title("Where Topology Adds Value", fontsize=10)
        ax4.legend(fontsize=8, loc="lower right")
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)

    # Panel 5: Score distributions
    ax5 = fig.add_subplot(gs[1, 2])
    if "Topo + Shannon" in all_triage:
        df = all_triage["Topo + Shannon"]
        bins = np.linspace(0, 100, 25)
        ax5.hist(df.loc[df["true_label"] == 0, "tds"], bins=bins, alpha=0.6,
                 color="#2ca02c", label="nonIBD", density=True)
        ax5.hist(df.loc[df["true_label"] == 1, "tds"], bins=bins, alpha=0.7,
                 color="#d62728", label="IBD", density=True)
        ax5.axvline(50, color="gray", lw=1, ls="--")
        ax5.set_xlabel("Topological Dysbiosis Score")
        ax5.set_ylabel("Density")
        ax5.set_title("Score Distribution", fontsize=10)
        ax5.legend(fontsize=7)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def main():
    t_start = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading IBDMDB data ...")
    abundance_df, metadata = load_ibdmdb(os.path.join(DATA_DIR, "ibdmdb"))
    diag = metadata["diagnosis"]
    keep = diag.isin(["CD", "UC", "nonIBD"])
    abundance_df = abundance_df.loc[keep]
    metadata = metadata.loc[keep]
    y_series = diag.loc[keep].isin(["CD", "UC"]).astype(int)
    print(f"  n={len(abundance_df)} | IBD={y_series.sum()} | nonIBD={(y_series==0).sum()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    taxa = select_global_taxa(clr_df, N_GLOBAL_TAXA)
    clr_taxa = clr_df[taxa]
    common = clr_taxa.index
    y = y_series.loc[common].values.astype(int)
    metadata = metadata.loc[common]

    # ── Features ──────────────────────────────────────────────────────────────
    X_shannon = compute_shannon(abundance_df.loc[common])
    print("Computing per-sample topology ...")
    X_topo = compute_per_sample_topology(clr_taxa.values.astype(np.float64))

    # ── Train/test ────────────────────────────────────────────────────────────
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )
    y_test = y[test_idx]

    # ── Train classifiers ─────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "BATCH TRIAGE DASHBOARD" + " " * 26 + "#")
    print("#" * 70)

    probas_dict = {}
    topo_coefs = None

    for name, X in [
        ("Shannon only", X_shannon),
        ("Topology only", X_topo),
        ("Topo + Shannon", np.hstack([X_topo, X_shannon])),
    ]:
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        probas = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        probas_dict[name] = probas

        auc = roc_auc_score(y_test, probas)
        print(f"\n  {name}: AUC = {auc:.3f}")

        if name == "Topo + Shannon":
            topo_coefs = clf.coef_[0][:6]  # first 6 are topology

    # ── Triage ────────────────────────────────────────────────────────────────
    all_triage = {}
    all_rows = []

    for strat_name, probas in probas_dict.items():
        tds_scores = probas * 100
        tiers = [assign_tier(s) for s in tds_scores]

        triage_df = pd.DataFrame({
            "true_label": y_test,
            "tds": tds_scores,
            "tier": tiers,
        })
        all_triage[strat_name] = triage_df
        print_triage_table(triage_df, strat_name, y_test)

        for i, (_, row) in enumerate(triage_df.iterrows()):
            all_rows.append({
                "strategy": strat_name,
                "sample_idx": test_idx[i],
                "true_label": "IBD" if row["true_label"] == 1 else "nonIBD",
                "tds": round(row["tds"], 1),
                "tier": row["tier"],
            })

    # ── Value-add analysis ────────────────────────────────────────────────────
    if "Shannon only" in probas_dict and "Topo + Shannon" in probas_dict:
        s_high = probas_dict["Shannon only"] >= 0.5
        c_high = probas_dict["Topo + Shannon"] >= 0.5
        ibd_mask = y_test == 1

        topo_catches = (ibd_mask & ~s_high & c_high).sum()
        shannon_catches = (ibd_mask & s_high & ~c_high).sum()
        both_catch = (ibd_mask & s_high & c_high).sum()
        both_miss = (ibd_mask & ~s_high & ~c_high).sum()

        print(f"\n  {'=' * 60}")
        print(f"  TOPOLOGY VALUE-ADD (IBD cases in holdout, n={ibd_mask.sum()})")
        print(f"  {'=' * 60}")
        print(f"  Both catch:              {both_catch:>5d}")
        print(f"  Topology adds:           {topo_catches:>5d}  <-- extra IBD caught")
        print(f"  Shannon adds:            {shannon_catches:>5d}")
        print(f"  Both miss:               {both_miss:>5d}")

    # ── Feature importance ────────────────────────────────────────────────────
    if topo_coefs is not None:
        print(f"\n  {'=' * 60}")
        print(f"  TOPOLOGY FEATURE IMPORTANCE (Topo+Shannon LR)")
        print(f"  {'=' * 60}")
        sorted_idx = np.argsort(np.abs(topo_coefs))[::-1]
        for i in sorted_idx:
            d = "+" if topo_coefs[i] > 0 else "-"
            print(f"    {FEATURE_NAMES[i]:<24s}  coef={topo_coefs[i]:+.4f}  "
                  f"[{d} IBD risk]")

    # ── Save ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "triage_simulation.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    fig_path = os.path.join(FIGURE_DIR, "triage_dashboard.png")
    plot_triage_dashboard(all_triage, topo_coefs, fig_path)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
