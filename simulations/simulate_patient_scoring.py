#!/usr/bin/env python3
"""
Clinical Simulation 1: Individual Patient Risk Scoring
=======================================================

Demonstrates how per-sample topological features could generate individual
patient risk reports.  Uses real IBDMDB clinical samples, scores them through
a trained classifier, and produces mock "clinical reports" with a Topological
Dysbiosis Score (TDS) and feature-level breakdown.

This is a simulation for demonstration purposes — not a diagnostic tool.

Outputs
-------
  Console:  5 mock patient reports
  figures/patient_scoring_profiles.png
  results/simulated_patient_scores.csv

Usage
-----
  python simulations/simulate_patient_scoring.py
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

FEATURE_NAMES = [
    "H1 Count", "H1 Entropy", "H1 Total Persistence",
    "H1 Mean Lifetime", "H1 Max Lifetime", "Max Betti-1",
]
SHORT_NAMES = [
    "h1_count", "h1_entropy", "h1_total_pers",
    "h1_mean_life", "h1_max_life", "max_betti1",
]


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

    print(f"  Computing per-sample topology (n={n}, k={k_actual}) ...")
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (n - i)
            print(f"    {i}/{n}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

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

    print(f"  Done in {time.time() - t0:.1f}s")
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


def risk_tier(tds):
    """Map Topological Dysbiosis Score to risk tier."""
    if tds < 25:
        return "LOW", "No topological dysbiosis detected"
    elif tds < 50:
        return "MODERATE", "Mild topological disruption — consider monitoring"
    elif tds < 75:
        return "HIGH", "Significant topological simplification — recommend follow-up"
    else:
        return "CRITICAL", "Severe topological dysbiosis — prioritise clinical evaluation"


def print_patient_report(patient_id, idx, tds, prob, topo_row, topo_ref_mean,
                         topo_ref_std, shannon_val, true_label, calprotectin=None):
    """Print a mock clinical report for one patient."""
    tier, recommendation = risk_tier(tds)

    print("=" * 68)
    print(f"  TOPOLOGICAL DYSBIOSIS REPORT  —  Patient #{idx+1}")
    print("=" * 68)
    print(f"  Sample ID:                          {patient_id}")
    print(f"  Topological Dysbiosis Score (TDS):  {tds:.0f} / 100")
    print(f"  Risk Tier:                          {tier}")
    print(f"  Shannon Diversity:                  {shannon_val:.3f}")
    if calprotectin is not None and not np.isnan(calprotectin):
        print(f"  Fecal Calprotectin:                 {calprotectin:.0f} ug/g")
    print(f"  True Diagnosis:                     {true_label}")
    print()
    print("  Feature Breakdown (Z-score vs healthy reference):")
    print("  " + "-" * 56)

    for i, (name, short) in enumerate(zip(FEATURE_NAMES, SHORT_NAMES)):
        val = topo_row[i]
        z = (val - topo_ref_mean[i]) / topo_ref_std[i] if topo_ref_std[i] > 0 else 0
        flag = " ***" if abs(z) > 2 else " *" if abs(z) > 1 else ""
        bar_len = int(min(abs(z), 4) * 5)
        bar_dir = "-" if z < 0 else "+"
        bar = bar_dir * bar_len
        print(f"    {name:<24s}  Z = {z:+6.2f}  [{bar:<20s}]{flag}")

    print()
    print(f"  Recommendation: {recommendation}")
    print(f"  Classifier confidence: {prob:.1%}")
    print("=" * 68)
    print()


def select_example_patients(y, probas, metadata_aligned, n_each=2):
    """Select 5 patients: 2 confident healthy, 2 confident IBD, 1 borderline."""
    rng = np.random.default_rng(SEED)
    healthy_idx = np.where(y == 0)[0]
    ibd_idx = np.where(y == 1)[0]

    confident_healthy = healthy_idx[np.argsort(probas[healthy_idx])[:20]]
    selected_healthy = rng.choice(confident_healthy,
                                  size=min(n_each, len(confident_healthy)),
                                  replace=False)

    confident_ibd = ibd_idx[np.argsort(probas[ibd_idx])[-20:]]
    selected_ibd = rng.choice(confident_ibd,
                              size=min(n_each, len(confident_ibd)),
                              replace=False)

    border_dist = np.abs(probas - 0.5)
    borderline_candidates = np.argsort(border_dist)[:20]
    selected_border = rng.choice(borderline_candidates, size=1, replace=False)

    return np.concatenate([selected_healthy, selected_ibd, selected_border])


def plot_patient_profiles(patients_df, X_topo, topo_ref_mean, topo_ref_std,
                          out_path):
    """Generate bar plots for example patients."""
    n_patients = len(patients_df)
    fig, axes = plt.subplots(1, n_patients, figsize=(3.2 * n_patients, 4.5),
                             sharey=True)
    if n_patients == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, patients_df.iterrows()):
        idx = int(row["array_idx"])
        topo = X_topo[idx]
        z_scores = np.where(topo_ref_std > 0,
                            (topo - topo_ref_mean) / topo_ref_std, 0)

        colors = []
        for z in z_scores:
            if abs(z) > 2:
                colors.append("#d62728")
            elif abs(z) > 1:
                colors.append("#ff7f0e")
            else:
                colors.append("#2ca02c")

        y_pos = np.arange(len(FEATURE_NAMES))
        ax.barh(y_pos, z_scores, color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(-2, color="red", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(2, color="red", lw=0.5, ls="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(FEATURE_NAMES if ax == axes[0] else [], fontsize=7)
        ax.set_xlim(-5, 5)
        ax.set_xlabel("Z-score", fontsize=8)

        tds = row["tds"]
        tier, _ = risk_tier(tds)
        ax.set_title(f"Patient #{row['patient_num']}\nTDS={tds:.0f} ({tier})\n"
                     f"Dx: {row['true_label']}",
                     fontsize=8, fontweight="bold")

    fig.suptitle("Individual Patient Topological Feature Profiles\n"
                 "(Z-scores vs healthy reference  |  red dashed = +/-2 SD)",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def main():
    t_start = time.time()

    # ── Load IBDMDB data ──────────────────────────────────────────────────────
    print("Loading IBDMDB data ...")
    abundance_df, metadata = load_ibdmdb(os.path.join(DATA_DIR, "ibdmdb"))

    # Binary label: IBD (CD+UC) vs nonIBD
    diag = metadata["diagnosis"]
    ibd_mask = diag.isin(["CD", "UC"])
    nonibd_mask = diag == "nonIBD"
    keep = ibd_mask | nonibd_mask
    abundance_df = abundance_df.loc[keep]
    metadata = metadata.loc[keep]
    y_series = ibd_mask.loc[keep].astype(int)

    print(f"  n = {len(abundance_df)} | IBD = {y_series.sum()} | "
          f"nonIBD = {(y_series == 0).sum()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing ...")
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    taxa = select_global_taxa(clr_df, N_GLOBAL_TAXA)
    clr_taxa = clr_df[taxa]

    # Align everything
    common = clr_taxa.index
    y = y_series.loc[common].values.astype(int)
    metadata = metadata.loc[common]

    # ── Features ──────────────────────────────────────────────────────────────
    X_shannon = compute_shannon(abundance_df.loc[common])
    print("Computing per-sample topology ...")
    X_topo = compute_per_sample_topology(clr_taxa.values.astype(np.float64))

    # Aitchison PCoA
    clr_matrix = clr_taxa.values.astype(np.float64)

    # ── Train/test split ──────────────────────────────────────────────────────
    print("\nSplitting 80/20 ...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )

    # Simple feature matrix (topology + Shannon)
    X = np.hstack([X_topo, X_shannon])
    scaler = StandardScaler().fit(X[train_idx])
    X_train_s = scaler.transform(X[train_idx])
    X_test_s = scaler.transform(X[test_idx])

    # ── Train classifier ──────────────────────────────────────────────────────
    print("Training Logistic Regression ...")
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(X_train_s, y[train_idx])
    probas_test = clf.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y[test_idx], probas_test)
    print(f"  Holdout AUC: {auc:.3f}")

    # ── Healthy reference ─────────────────────────────────────────────────────
    healthy_train = y[train_idx] == 0
    topo_ref_mean = X_topo[train_idx][healthy_train].mean(axis=0)
    topo_ref_std = X_topo[train_idx][healthy_train].std(axis=0)

    # ── Select example patients ───────────────────────────────────────────────
    example_local = select_example_patients(y[test_idx], probas_test, metadata)

    print("\n" + "#" * 68)
    print("#" + " " * 20 + "PATIENT RISK REPORTS" + " " * 26 + "#")
    print("#" * 68 + "\n")

    sample_ids = common.values
    rows = []
    for i, local_idx in enumerate(example_local):
        global_idx = test_idx[local_idx]
        patient_id = sample_ids[global_idx]
        tds = probas_test[local_idx] * 100
        topo_row = X_topo[global_idx]
        shannon_val = X_shannon[global_idx, 0]
        true_label = metadata.loc[patient_id, "diagnosis"]

        # Get calprotectin if available
        calp = pd.to_numeric(metadata.loc[patient_id].get("fecalcal", np.nan),
                             errors="coerce")

        print_patient_report(
            patient_id=patient_id,
            idx=i, tds=tds, prob=probas_test[local_idx],
            topo_row=topo_row, topo_ref_mean=topo_ref_mean,
            topo_ref_std=topo_ref_std, shannon_val=shannon_val,
            true_label=true_label, calprotectin=calp,
        )

        rows.append({
            "patient_num": i + 1,
            "patient_id": patient_id,
            "true_label": true_label,
            "tds": round(tds, 1),
            "risk_tier": risk_tier(tds)[0],
            "classifier_prob": round(probas_test[local_idx], 4),
            "shannon": round(shannon_val, 4),
            "calprotectin": round(calp, 1) if pd.notna(calp) else "",
            "array_idx": global_idx,
            **{short: round(float(topo_row[j]), 4)
               for j, short in enumerate(SHORT_NAMES)},
        })

    patients_df = pd.DataFrame(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "simulated_patient_scores.csv")
    patients_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # ── Generate figure ───────────────────────────────────────────────────────
    fig_path = os.path.join(FIGURE_DIR, "patient_scoring_profiles.png")
    plot_patient_profiles(patients_df, X_topo, topo_ref_mean, topo_ref_std,
                          fig_path)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
