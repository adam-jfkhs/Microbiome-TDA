#!/usr/bin/env python3
"""
Clinical Simulation 3: Treatment Monitoring via Longitudinal Topology Tracking
===============================================================================

Uses real IBDMDB longitudinal data (multiple timepoints per patient) to
simulate how topology scores could track disease activity over time.

For each patient with >= 4 timepoints, we compute per-sample topological
features and plot them as a time series.  Alert zones are defined from the
healthy reference distribution.

This demonstrates:
  - Real-time monitoring of microbial network topology
  - Alert thresholds for clinical deterioration
  - Correlation between topology scores and fecal calprotectin

Outputs
-------
  Console:  patient trajectory summaries
  figures/treatment_monitoring.png
  results/longitudinal_topology_scores.csv

Usage
-----
  python simulations/simulate_treatment_monitoring.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
from src.tda.homology import compute_persistence, filter_infinite
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.features import betti_curve, persistence_entropy
from src.analysis.bootstrap import FEATURES as FEATURE_COLS

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS = 40       # neighbourhood size for per-sample TDA
MIN_TIMEPOINTS = 4      # minimum timepoints per patient to include
N_DISPLAY_PATIENTS = 5  # number of patients to show in the figure

DATA_DIR = os.path.join(ROOT, "data", "raw")
FIGURE_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def per_sample_topology(clr_matrix, k=K_NEIGHBOURS):
    """Compute per-sample H1 features via k-NN neighbourhood TDA.

    Same method as in run_classification_benchmark_v2.py but adapted for
    smaller IBDMDB matrices.
    """
    from scipy.stats import spearmanr

    n_samples, n_taxa = clr_matrix.shape
    k_actual = min(k, n_samples - 1)

    if k_actual < 5:
        # Too few samples for meaningful neighbourhood TDA
        return np.zeros((n_samples, 6), dtype=np.float32)

    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(clr_matrix, clr_matrix, metric="euclidean")

    out = np.zeros((n_samples, 6), dtype=np.float32)
    for i in range(n_samples):
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

        finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else dgm_h1
        if len(finite) == 0:
            continue

        lifetimes = finite[:, 1] - finite[:, 0]
        total_pers = float(lifetimes.sum())
        norm_lt = lifetimes / total_pers if total_pers > 0 else lifetimes
        entropy = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))

        births, deaths = finite[:, 0], finite[:, 1]
        thresholds = np.unique(np.concatenate([births, deaths]))
        max_betti = int(max(
            np.sum((births <= t) & (deaths > t)) for t in thresholds
        )) if len(thresholds) > 0 else 0

        out[i] = [
            len(finite), entropy, total_pers,
            float(lifetimes.mean()), float(lifetimes.max()), max_betti,
        ]

    return out


def select_display_patients(patient_data, metadata):
    """Select patients with interesting trajectories for display.

    Criteria: patients with most timepoints, mix of diagnoses.
    """
    rng = np.random.default_rng(SEED)

    # Group by patient, count timepoints
    counts = patient_data.groupby("participant_id").size()
    eligible = counts[counts >= MIN_TIMEPOINTS].index.tolist()

    if len(eligible) == 0:
        print("WARNING: No patients with enough timepoints")
        return []

    # Get diagnosis for each patient
    diag_map = {}
    for pid in eligible:
        mask = patient_data["participant_id"] == pid
        diags = metadata.loc[patient_data.loc[mask, "sample_id"].values, "diagnosis"]
        diag_map[pid] = diags.iloc[0] if len(diags) > 0 else "unknown"

    # Try to get a mix: some CD, some UC, some nonIBD
    selected = []
    for dx in ["CD", "UC", "nonIBD"]:
        candidates = [p for p in eligible if diag_map.get(p) == dx]
        if candidates:
            # Pick the ones with the most timepoints
            candidates.sort(key=lambda p: counts[p], reverse=True)
            n_pick = min(2, len(candidates))
            selected.extend(candidates[:n_pick])

    # If we don't have enough, fill with highest-timepoint patients
    remaining = [p for p in eligible if p not in selected]
    remaining.sort(key=lambda p: counts[p], reverse=True)
    while len(selected) < N_DISPLAY_PATIENTS and remaining:
        selected.append(remaining.pop(0))

    return selected[:N_DISPLAY_PATIENTS]


def plot_monitoring(patient_data, display_patients, metadata,
                    healthy_ref_mean, healthy_ref_std, out_path):
    """Generate longitudinal monitoring figure."""
    n_patients = len(display_patients)
    if n_patients == 0:
        print("No patients to plot")
        return

    # Use h1_total_persistence as the primary monitoring metric
    metric = "h1_total_persistence"
    metric_label = "H1 Total Persistence (Topological Complexity)"

    fig, axes = plt.subplots(n_patients, 1, figsize=(12, 3 * n_patients),
                             sharex=False)
    if n_patients == 1:
        axes = [axes]

    fig.suptitle(
        "Longitudinal Topology Monitoring — IBDMDB Patients\n"
        "Per-sample H1 Total Persistence over time | "
        "Green = healthy range, Yellow = caution, Red = alert",
        fontsize=11, fontweight="bold", y=1.01,
    )

    ref_mean = healthy_ref_mean
    ref_std = healthy_ref_std

    for ax, pid in zip(axes, display_patients):
        pdata = patient_data[patient_data["participant_id"] == pid].sort_values("week_num")

        # Get diagnosis
        sample_ids = pdata["sample_id"].values
        diag = metadata.loc[sample_ids[0], "diagnosis"] if sample_ids[0] in metadata.index else "?"

        weeks = pdata["week_num"].values
        values = pdata[metric].values

        # Get calprotectin if available
        calp_values = []
        for sid in sample_ids:
            if sid in metadata.index and "fecalcal" in metadata.columns:
                c = metadata.loc[sid, "fecalcal"]
                calp_values.append(c if pd.notna(c) else np.nan)
            else:
                calp_values.append(np.nan)
        calp_values = np.array(calp_values)

        # Draw alert zones
        ax.axhspan(ref_mean - ref_std, ref_mean + ref_std,
                    color="#2ca02c", alpha=0.15, label="Healthy ±1 SD")
        ax.axhspan(ref_mean - 2 * ref_std, ref_mean - ref_std,
                    color="#ff7f0e", alpha=0.1)
        ax.axhspan(ref_mean + ref_std, ref_mean + 2 * ref_std,
                    color="#ff7f0e", alpha=0.1, label="Caution zone")
        ax.axhline(ref_mean, color="#2ca02c", lw=1, ls="--", alpha=0.5)

        # Alert threshold
        alert_low = ref_mean - 2 * ref_std
        ax.axhline(alert_low, color="red", lw=1, ls=":", alpha=0.7,
                    label="Alert threshold")

        # Plot topology score
        ax.plot(weeks, values, "o-", color="#1f78b4", lw=2, markersize=6,
                label=f"Patient topology", zorder=5)

        # Color points by alert status
        for w, v in zip(weeks, values):
            if v < alert_low:
                ax.plot(w, v, "o", color="red", markersize=10, zorder=6,
                        markeredgecolor="darkred", markeredgewidth=1.5)

        # Overlay calprotectin as secondary axis if available
        valid_calp = ~np.isnan(calp_values)
        if valid_calp.sum() > 1:
            ax2 = ax.twinx()
            ax2.plot(weeks[valid_calp], calp_values[valid_calp],
                     "s--", color="#d62728", alpha=0.6, markersize=4,
                     label="Calprotectin (μg/g)")
            ax2.set_ylabel("Fecal Calprotectin (μg/g)", fontsize=8,
                           color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=7)

        n_alerts = int((values < alert_low).sum())
        ax.set_ylabel(metric_label, fontsize=7)
        ax.set_title(
            f"Patient {pid[:15]}  |  Diagnosis: {diag}  |  "
            f"{len(weeks)} timepoints  |  {n_alerts} alerts",
            fontsize=9, fontweight="bold", loc="left",
        )
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Week", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def main():
    t_start = time.time()

    # ── Load IBDMDB data ──────────────────────────────────────────────────────
    print("Loading IBDMDB data ...")
    abundance_df, metadata = load_ibdmdb(os.path.join(DATA_DIR, "ibdmdb"))
    print(f"  {len(abundance_df)} samples, {len(abundance_df.columns)} species")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing ...")
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)

    # Select global taxa
    taxa = select_global_taxa(clr_df, N_GLOBAL_TAXA)
    clr_taxa = clr_df[taxa]

    # ── Compute per-sample topology ───────────────────────────────────────────
    print(f"Computing per-sample topology for {len(clr_taxa)} samples ...")
    topo_features = per_sample_topology(clr_taxa.values.astype(np.float64))

    # ── Build patient-level longitudinal table ────────────────────────────────
    print("Building longitudinal table ...")
    rows = []
    for i, sample_id in enumerate(clr_taxa.index):
        if sample_id not in metadata.index:
            continue
        meta_row = metadata.loc[sample_id]
        pid = meta_row.get("Participant ID", "unknown")
        week = pd.to_numeric(meta_row.get("week_num", np.nan), errors="coerce")
        diag = meta_row.get("diagnosis", "unknown")

        row = {
            "sample_id": sample_id,
            "participant_id": pid,
            "diagnosis": diag,
            "week_num": week,
        }
        for j, col in enumerate(FEATURE_COLS):
            row[col] = float(topo_features[i, j])

        rows.append(row)

    patient_data = pd.DataFrame(rows)
    patient_data = patient_data.dropna(subset=["week_num"])

    n_patients = patient_data["participant_id"].nunique()
    print(f"  {len(patient_data)} samples across {n_patients} patients "
          f"with valid week data")

    # ── Healthy reference distribution ────────────────────────────────────────
    healthy_mask = patient_data["diagnosis"] == "nonIBD"
    healthy_ref_mean = patient_data.loc[healthy_mask, "h1_total_persistence"].mean()
    healthy_ref_std = patient_data.loc[healthy_mask, "h1_total_persistence"].std()
    print(f"  Healthy reference: mean={healthy_ref_mean:.3f}, std={healthy_ref_std:.3f}")

    # ── Select display patients ───────────────────────────────────────────────
    display_patients = select_display_patients(patient_data, metadata)
    print(f"  Selected {len(display_patients)} patients for display")

    # ── Print patient summaries ───────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("#" + " " * 15 + "LONGITUDINAL MONITORING SUMMARIES" + " " * 21 + "#")
    print("#" * 70)

    alert_threshold = healthy_ref_mean - 2 * healthy_ref_std

    for pid in display_patients:
        pdata = patient_data[patient_data["participant_id"] == pid].sort_values("week_num")
        diag = pdata["diagnosis"].iloc[0]
        metric_vals = pdata["h1_total_persistence"].values
        weeks = pdata["week_num"].values

        n_alerts = int((metric_vals < alert_threshold).sum())
        trend = "IMPROVING" if len(metric_vals) >= 3 and metric_vals[-1] > metric_vals[0] else \
                "WORSENING" if len(metric_vals) >= 3 and metric_vals[-1] < metric_vals[0] else "STABLE"

        print(f"\n  Patient: {pid}")
        print(f"  Diagnosis: {diag}")
        print(f"  Timepoints: {len(weeks)} (weeks {weeks.min():.0f}–{weeks.max():.0f})")
        print(f"  H1 Total Persistence: {metric_vals.mean():.3f} ± {metric_vals.std():.3f}")
        print(f"  Alert events: {n_alerts}/{len(metric_vals)} below threshold")
        print(f"  Trajectory: {trend}")

        # Show calprotectin correlation if available
        sample_ids = pdata["sample_id"].values
        calp = []
        for sid in sample_ids:
            if sid in metadata.index:
                c = pd.to_numeric(metadata.loc[sid].get("fecalcal", np.nan), errors="coerce")
                calp.append(c)
            else:
                calp.append(np.nan)
        calp = np.array(calp)
        valid = ~np.isnan(calp) & ~np.isnan(metric_vals)
        if valid.sum() >= 3:
            from scipy.stats import spearmanr
            r, p = spearmanr(metric_vals[valid], calp[valid])
            print(f"  Calprotectin correlation: r={r:.3f}, p={p:.3f}")

    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "longitudinal_topology_scores.csv")
    patient_data.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # ── Generate figure ───────────────────────────────────────────────────────
    fig_path = os.path.join(FIGURE_DIR, "treatment_monitoring.png")
    plot_monitoring(patient_data, display_patients, metadata,
                    healthy_ref_mean, healthy_ref_std, fig_path)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
