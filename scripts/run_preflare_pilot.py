#!/usr/bin/env python3
"""Pre-flare topology pilot — IBDMDB longitudinal analysis.

Identifies participants with a documented low-calprotectin (remission)
→ high-calprotectin (flare) transition and computes H₁ TDA features at
each timepoint to test whether topological simplification precedes the
calprotectin spike.

Clinical hypothesis
-------------------
If topological simplification causes (or is an early marker of) mucosal
inflammation, we should see H₁ features declining at the pre-flare
timepoint BEFORE calprotectin rises above the clinical threshold.

Design
------
  Remission timepoints: fecalcal ≤ 50 µg/g
  Flare timepoints:     fecalcal ≥ 250 µg/g
  Pre-flare:            most recent remission sample before the first flare

For each participant with a complete low→high transition:
  Compare H₁ features at remission vs pre-flare (same participant,
  different timepoints) — within-participant effect size.

Outputs
-------
  results/preflare_pilot.csv
  figures/preflare_pilot.png
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, spearmanr

from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.homology import compute_persistence, filter_infinite, persistence_summary
from src.tda.features import betti_curve, persistence_entropy

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ibdmdb")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURE_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIGH_CAL = 250   # flare threshold (µg/g)
LOW_CAL  = 50    # remission threshold (µg/g)
N_TAXA   = 50    # taxa for IBDMDB (fewer than AGP due to smaller cohort)
SEED     = 42


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ibdmdb():
    meta = pd.read_csv(os.path.join(DATA_DIR, "hmp2_metadata.csv"), low_memory=False)
    tax  = pd.read_csv(os.path.join(DATA_DIR, "taxonomic_profiles.tsv"),
                       sep="\t", index_col=0)
    tax.columns = [c.replace("_taxonomic_profile", "") for c in tax.columns]

    # Filter to metagenomics only
    meta_mg = meta[meta["data_type"] == "metagenomics"].copy()
    meta_mg["fecalcal"] = pd.to_numeric(meta_mg["fecalcal"], errors="coerce")
    meta_mg["week_num"] = pd.to_numeric(meta_mg["week_num"],  errors="coerce")

    # Align with taxonomic profiles
    meta_mg = meta_mg[meta_mg["External ID"].isin(tax.columns)].copy()
    meta_mg = meta_mg.dropna(subset=["Participant ID"])
    return meta_mg, tax


def clr_transform(df):
    """Row-wise CLR on relative abundances with +0.5 pseudocount.

    Uses the same pseudocount approach as src/data/preprocess.py to ensure
    consistency across all analyses in this repo.
    """
    data = df.values + 0.5
    log_data = np.log(data)
    geo_mean = log_data.mean(axis=1, keepdims=True)
    clr_data = log_data - geo_mean
    return pd.DataFrame(clr_data, index=df.index, columns=df.columns)


def select_taxa(tax_df, sample_ids, n=N_TAXA):
    """Select top-N taxa by prevalence across given samples."""
    sub = tax_df[sample_ids].T
    prevalence = (sub > 0).mean()
    return prevalence.nlargest(n).index.tolist()


def tda_features_from_samples(tax_df, sample_ids, taxa):
    """Compute H₁ features from a CLR-transformed sub-cohort."""
    sub = tax_df.loc[taxa, sample_ids].T
    sub = clr_transform(sub)

    corr, _ = spearmanr(sub)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    dist = np.clip(1.0 - np.abs(corr), 0, 1)
    np.fill_diagonal(dist, 0)

    result    = compute_persistence(dist, maxdim=1)
    dgms      = result["dgms"]
    finite    = filter_infinite(dgms)
    summary   = persistence_summary(dgms)
    h1_ent    = persistence_entropy(dgms[1])
    _, betti1 = betti_curve(dgms[1], num_points=200)
    fh1       = finite[1]
    total_p   = float(np.sum(fh1[:, 1] - fh1[:, 0])) if len(fh1) > 0 else 0.0

    return {
        "h1_count":             summary["H1"]["count"],
        "h1_entropy":           h1_ent,
        "h1_total_persistence": total_p,
        "h1_mean_lifetime":     summary["H1"]["mean_lifetime"],
        "h1_max_lifetime":      summary["H1"]["max_lifetime"],
        "max_betti1":           int(betti1.max()),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading IBDMDB data…")
    meta, tax = load_ibdmdb()

    # Global taxa for IBDMDB
    all_sample_ids = meta["External ID"].tolist()
    taxa = select_taxa(tax, all_sample_ids, n=N_TAXA)
    print(f"Using {len(taxa)} global taxa")

    # ── Find participants with pre-flare transitions ───────────────
    participants = meta["Participant ID"].unique()
    transitions = []

    for pid in participants:
        p = meta[meta["Participant ID"] == pid].sort_values("week_num")
        p = p.dropna(subset=["week_num", "fecalcal"])

        high_rows = p[p["fecalcal"] >= HIGH_CAL]
        low_rows  = p[p["fecalcal"] <= LOW_CAL]

        if len(high_rows) == 0 or len(low_rows) == 0:
            continue

        # First flare week
        first_flare_week = high_rows["week_num"].min()

        # Most recent remission before flare
        prior_low = low_rows[low_rows["week_num"] < first_flare_week]
        if len(prior_low) == 0:
            continue

        remission_row = prior_low.sort_values("week_num").iloc[-1]
        flare_row     = high_rows.sort_values("week_num").iloc[0]

        transitions.append({
            "pid":              pid,
            "diagnosis":        p["diagnosis"].iloc[0],
            "remission_week":   remission_row["week_num"],
            "flare_week":       flare_row["week_num"],
            "remission_cal":    remission_row["fecalcal"],
            "flare_cal":        flare_row["fecalcal"],
            "remission_sample": remission_row["External ID"],
            "flare_sample":     flare_row["External ID"],
            "weeks_to_flare":   flare_row["week_num"] - remission_row["week_num"],
        })

    print(f"\nParticipants with low→high calprotectin transition: {len(transitions)}")
    for t in transitions:
        print(f"  {t['pid']}  {t['diagnosis']}  "
              f"week {t['remission_week']:.0f}→{t['flare_week']:.0f}  "
              f"cal {t['remission_cal']:.0f}→{t['flare_cal']:.0f}  "
              f"({t['weeks_to_flare']:.0f} weeks apart)")

    if len(transitions) == 0:
        print("No pre-flare transitions found. Exiting.")
        return

    # ── Compute TDA features for each timepoint ────────────────────
    # Use a group-level approach: pool remission samples vs pool flare samples
    # from these participants (more taxa per group = more stable topology)

    remission_samples = [t["remission_sample"] for t in transitions]
    flare_samples     = [t["flare_sample"]     for t in transitions]

    # Also collect ALL remission and flare timepoints for these participants
    # (not just the boundary ones) for supplemental comparison
    all_pid_meta = meta[meta["Participant ID"].isin([t["pid"] for t in transitions])]
    all_remission = all_pid_meta[all_pid_meta["fecalcal"] <= LOW_CAL]["External ID"].tolist()
    all_flare     = all_pid_meta[all_pid_meta["fecalcal"] >= HIGH_CAL]["External ID"].tolist()

    print(f"\nComputing TDA features…")
    print(f"  Boundary remission samples (n={len(remission_samples)})")
    print(f"  Boundary flare samples     (n={len(flare_samples)})")
    print(f"  All remission samples      (n={len(all_remission)})")
    print(f"  All flare samples          (n={len(all_flare)})")

    results = {}
    for label, samples in [
        ("boundary_remission", remission_samples),
        ("boundary_flare",     flare_samples),
        ("all_remission",      all_remission),
        ("all_flare",          all_flare),
    ]:
        valid = [s for s in samples if s in tax.columns]
        if len(valid) < 3:
            print(f"  {label}: too few samples ({len(valid)}), skipping")
            results[label] = None
            continue
        print(f"  Computing {label} (n={len(valid)})…")
        results[label] = tda_features_from_samples(tax, valid, taxa)

    # ── Per-participant within-subject comparison ──────────────────
    per_participant = []
    for t in transitions:
        rem_id = t["remission_sample"]
        fla_id = t["flare_sample"]
        if rem_id not in tax.columns or fla_id not in tax.columns:
            continue
        # Use a 2-sample neighbourhood: the transition pair + nearest timepoints
        pid_meta = meta[meta["Participant ID"] == t["pid"]].sort_values("week_num")
        pid_samples = pid_meta["External ID"].tolist()
        pid_valid   = [s for s in pid_samples if s in tax.columns]

        # Compute per-timepoint topology (use all samples for taxa selection)
        sub_tax = tax.loc[taxa, pid_valid].T
        sub_clr = clr_transform(sub_tax)

        # We need at least 5 samples to get a meaningful correlation matrix
        if len(pid_valid) < 5:
            print(f"  {t['pid']}: only {len(pid_valid)} samples, skipping per-participant")
            continue

        corr, _ = spearmanr(sub_clr)
        dist     = np.clip(1.0 - np.abs(corr), 0, 1)
        np.fill_diagonal(dist, 0)
        res      = compute_persistence(dist, maxdim=1)
        summary  = persistence_summary(res["dgms"])
        fin_h1   = filter_infinite(res["dgms"])[1]
        total_p  = float(np.sum(fin_h1[:, 1] - fin_h1[:, 0])) if len(fin_h1) > 0 else 0.0

        per_participant.append({
            "pid":               t["pid"],
            "diagnosis":         t["diagnosis"],
            "weeks_to_flare":    t["weeks_to_flare"],
            "remission_cal":     t["remission_cal"],
            "flare_cal":         t["flare_cal"],
            "n_samples":         len(pid_valid),
            "h1_count":          summary["H1"]["count"],
            "h1_total_pers":     total_p,
            "h1_mean_lifetime":  summary["H1"]["mean_lifetime"],
            "h1_max_lifetime":   summary["H1"]["max_lifetime"],
        })

    # ── Save results ───────────────────────────────────────────────
    rows = []
    FEATS = ["h1_count","h1_entropy","h1_total_persistence",
             "h1_mean_lifetime","h1_max_lifetime","max_betti1"]

    for label in ["boundary_remission","boundary_flare","all_remission","all_flare"]:
        if results.get(label) is None:
            continue
        row = {"condition": label}
        row.update(results[label])
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_part = pd.DataFrame(per_participant)

    out_csv = os.path.join(RESULTS_DIR, "preflare_pilot.csv")
    df_out.to_csv(out_csv, index=False)
    df_part.to_csv(os.path.join(RESULTS_DIR, "preflare_pilot_perparticipant.csv"), index=False)
    print(f"\nResults saved to {out_csv}")

    # ── Print comparison ───────────────────────────────────────────
    print("\n── Group-level H₁ comparison (boundary timepoints) ──")
    if results["boundary_remission"] and results["boundary_flare"]:
        for f in FEATS:
            rem_val = results["boundary_remission"][f]
            fla_val = results["boundary_flare"][f]
            delta   = fla_val - rem_val
            direction = "↓" if delta < 0 else "↑"
            print(f"  {f:28s}  remission={rem_val:.4f}  flare={fla_val:.4f}  "
                  f"Δ={delta:+.4f} {direction}")

    print("\n── Per-participant summary ──")
    if not df_part.empty:
        print(df_part[["pid","diagnosis","weeks_to_flare",
                        "remission_cal","flare_cal","n_samples",
                        "h1_count","h1_total_pers"]].to_string(index=False))

    # ── Figure ─────────────────────────────────────────────────────
    if results["boundary_remission"] and results["boundary_flare"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"Pre-flare Topology Pilot: n={len(transitions)} participant transitions\n"
            f"Remission (fecalcal ≤{LOW_CAL}) vs Flare (fecalcal ≥{HIGH_CAL}) µg/g",
            fontsize=11, fontweight="bold"
        )

        plot_feats = [
            ("h1_total_persistence", "H₁ Total Persistence"),
            ("h1_mean_lifetime",     "H₁ Mean Lifetime"),
            ("h1_count",             "H₁ Count (loops)"),
        ]
        colors = {"boundary_remission": "#43A047", "boundary_flare": "#E53935",
                  "all_remission":      "#81C784", "all_flare":       "#EF9A9A"}

        for ax, (feat, title) in zip(axes, plot_feats):
            conditions = ["boundary_remission","boundary_flare"]
            vals = [results[c][feat] for c in conditions if results.get(c)]
            labels = ["Remission", "Flare"]
            bars = ax.bar(labels, vals,
                          color=[colors["boundary_remission"], colors["boundary_flare"]],
                          width=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Feature value")
            ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        out_fig = os.path.join(FIGURE_DIR, "preflare_pilot.png")
        fig.savefig(out_fig, dpi=150)
        print(f"\nFigure saved to {out_fig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
