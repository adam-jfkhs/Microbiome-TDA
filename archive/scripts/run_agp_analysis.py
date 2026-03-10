#!/usr/bin/env python3
"""End-to-end TDA analysis on real American Gut Project data.

Runs persistent homology on microbial co-occurrence networks and
compares topological features across clinically relevant groups:
  1. Antibiotic recency (recent vs. never)
  2. IBD status (UC/Crohn's vs. healthy)
  3. Diet type (omnivore vs. plant-based)

Produces publication-quality figures and statistical results.
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
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.homology import compute_persistence, filter_infinite, persistence_summary
from src.tda.features import betti_curve, persistence_entropy, persistence_landscape
from src.analysis.statistics import (
    diagram_distance_permutation_test,
    cohens_d,
    permutation_test,
)

SEED = 42
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load real AGP data ────────────────────────────────────────────
print("=" * 70)
print("LOADING AMERICAN GUT PROJECT DATA")
print("=" * 70)

otu_df, metadata = load_agp()
print(f"Raw: {otu_df.shape[0]} samples, {otu_df.shape[1]} OTUs")

# Filter to stool samples only
stool_mask = metadata["BODY_SITE"] == "UBERON:feces"
stool_ids = metadata.loc[stool_mask].index.intersection(otu_df.index)
otu_stool = otu_df.loc[stool_ids]
meta_stool = metadata.loc[stool_ids]
print(f"Stool samples: {len(otu_stool)}")

# ── 2. Preprocess ────────────────────────────────────────────────────
print("\nPreprocessing...")
filtered = filter_low_abundance(otu_stool, min_prevalence=0.05, min_reads=1000)
clr_df = clr_transform(filtered)
print(f"After filtering: {clr_df.shape[0]} samples, {clr_df.shape[1]} taxa")

# Align metadata
meta_filtered = meta_stool.loc[clr_df.index]


# ── 3. Define comparison groups ──────────────────────────────────────

def define_groups(meta, clr):
    """Define biologically meaningful comparison groups."""
    comparisons = {}

    # --- Antibiotic recency ---
    abx_col = "ANTIBIOTIC_SELECT"
    recent_abx = meta[abx_col].isin(
        ["In the past week", "In the past month", "In the past 6 months"]
    )
    never_abx = meta[abx_col] == "Not in the last year"
    abx_recent_ids = meta.loc[recent_abx].index.intersection(clr.index)
    abx_never_ids = meta.loc[never_abx].index.intersection(clr.index)
    comparisons["antibiotics"] = {
        "group_a_name": "Recent antibiotics",
        "group_b_name": "No recent antibiotics",
        "group_a_ids": abx_recent_ids,
        "group_b_ids": abx_never_ids,
    }

    # --- IBD status ---
    ibd_col = "IBD"
    has_ibd = meta[ibd_col].isin(["Ulcerative colitis", "Crohn's disease"])
    no_ibd = meta[ibd_col] == "I do not have IBD"
    ibd_ids = meta.loc[has_ibd].index.intersection(clr.index)
    healthy_ids = meta.loc[no_ibd].index.intersection(clr.index)
    comparisons["ibd"] = {
        "group_a_name": "IBD (UC + Crohn's)",
        "group_b_name": "No IBD",
        "group_a_ids": ibd_ids,
        "group_b_ids": healthy_ids,
    }

    # --- Diet type ---
    diet_col = "DIET_TYPE"
    omnivore = meta[diet_col] == "Omnivore"
    plant = meta[diet_col].isin(
        [
            "Vegan",
            "Vegetarian",
            "Vegetarian but eat seafood",
        ]
    )
    omni_ids = meta.loc[omnivore].index.intersection(clr.index)
    plant_ids = meta.loc[plant].index.intersection(clr.index)
    comparisons["diet"] = {
        "group_a_name": "Plant-based diet",
        "group_b_name": "Omnivore",
        "group_a_ids": plant_ids,
        "group_b_ids": omni_ids,
    }

    return comparisons


comparisons = define_groups(meta_filtered, clr_df)

for name, comp in comparisons.items():
    print(
        f"  {name}: {comp['group_a_name']} (n={len(comp['group_a_ids'])}) vs "
        f"{comp['group_b_name']} (n={len(comp['group_b_ids'])})"
    )


# ── 4. Run TDA per group ────────────────────────────────────────────

def run_tda_on_group(clr_subset, group_name, max_taxa=80):
    """Run the full TDA pipeline on a sample subset.

    For computational tractability, we subsample to max_taxa most
    prevalent taxa across the subset, compute Spearman correlations,
    convert to distance, and run persistent homology.
    """
    # Select top taxa by prevalence in this group
    prevalence = (clr_subset > clr_subset.median()).mean(axis=0)
    top_taxa = prevalence.nlargest(max_taxa).index
    subset = clr_subset[top_taxa]

    print(f"    {group_name}: {subset.shape[0]} samples, {subset.shape[1]} taxa")

    # Spearman correlation matrix (taxa × taxa)
    corr_matrix, _ = spearman_correlation_matrix(subset)

    # Distance = 1 - |corr|
    dist_df = correlation_distance(corr_matrix)
    dist_matrix = prepare_distance_matrix(dist_df)

    # Persistent homology (max dim=1 for speed; H1 = loops)
    t0 = time.time()
    result = compute_persistence(dist_matrix, maxdim=1)
    elapsed = time.time() - t0
    print(f"    Ripser completed in {elapsed:.1f}s")

    dgms = result["dgms"]
    finite_dgms = filter_infinite(dgms)
    summary = persistence_summary(dgms)

    # Feature extraction
    h1_entropy = persistence_entropy(dgms[1])
    filt_vals, betti1 = betti_curve(dgms[1], num_points=200)
    landscapes = persistence_landscape(dgms[1], num_landscapes=3, num_points=200)

    # Total persistence
    finite_h1 = finite_dgms[1]
    total_pers = (
        float(np.sum(finite_h1[:, 1] - finite_h1[:, 0]))
        if len(finite_h1) > 0
        else 0.0
    )

    return {
        "dgms": dgms,
        "finite_dgms": finite_dgms,
        "summary": summary,
        "h1_entropy": h1_entropy,
        "betti1_curve": (filt_vals, betti1),
        "landscapes": landscapes,
        "total_persistence_h1": total_pers,
        "n_h1_features": summary["H1"]["count"],
        "max_betti1": int(betti1.max()),
        "corr_matrix": corr_matrix,
        "n_samples": subset.shape[0],
        "n_taxa": subset.shape[1],
    }


# Run for each comparison
MAX_SAMPLES_PER_GROUP = 500  # subsample large groups for tractability
all_results = {}

for comp_name, comp in comparisons.items():
    print(f"\n{'─' * 50}")
    print(f"COMPARISON: {comp_name}")
    print(f"{'─' * 50}")

    rng = np.random.default_rng(SEED)

    for group_key, id_key, label in [
        ("a", "group_a_ids", comp["group_a_name"]),
        ("b", "group_b_ids", comp["group_b_name"]),
    ]:
        ids = comp[id_key]
        if len(ids) > MAX_SAMPLES_PER_GROUP:
            ids = pd.Index(rng.choice(ids, MAX_SAMPLES_PER_GROUP, replace=False))
            print(f"  Subsampled {label} to {MAX_SAMPLES_PER_GROUP}")

        result = run_tda_on_group(clr_df.loc[ids], label)
        all_results[f"{comp_name}_{group_key}"] = result

        s = result["summary"]
        print(f"    H0: {s['H0']['count']} features")
        print(f"    H1: {s['H1']['count']} features, entropy={result['h1_entropy']:.3f}")
        print(f"    Total H1 persistence: {result['total_persistence_h1']:.3f}")
        print(f"    Max Betti-1: {result['max_betti1']}")


# ── 5. Statistical tests ────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("STATISTICAL TESTS")
print(f"{'=' * 70}")

stat_results = []

for comp_name, comp in comparisons.items():
    res_a = all_results[f"{comp_name}_a"]
    res_b = all_results[f"{comp_name}_b"]

    # Wasserstein distance permutation test on H1 diagrams
    w_dist, w_pval = diagram_distance_permutation_test(
        res_a["dgms"][1], res_b["dgms"][1], n_permutations=2000, seed=SEED
    )

    # Effect sizes on scalar summaries
    d_features = res_a["n_h1_features"] - res_b["n_h1_features"]
    d_entropy = res_a["h1_entropy"] - res_b["h1_entropy"]
    d_persistence = res_a["total_persistence_h1"] - res_b["total_persistence_h1"]

    row = {
        "comparison": comp_name,
        "group_a": comp["group_a_name"],
        "group_b": comp["group_b_name"],
        "n_a": res_a["n_samples"],
        "n_b": res_b["n_samples"],
        "h1_features_a": res_a["n_h1_features"],
        "h1_features_b": res_b["n_h1_features"],
        "h1_entropy_a": round(res_a["h1_entropy"], 3),
        "h1_entropy_b": round(res_b["h1_entropy"], 3),
        "total_pers_a": round(res_a["total_persistence_h1"], 3),
        "total_pers_b": round(res_b["total_persistence_h1"], 3),
        "wasserstein_dist": round(w_dist, 4),
        "wasserstein_pval": round(w_pval, 4),
        "significant": w_pval < 0.05,
    }
    stat_results.append(row)

    print(f"\n{comp_name}: {comp['group_a_name']} vs {comp['group_b_name']}")
    print(f"  H1 features: {res_a['n_h1_features']} vs {res_b['n_h1_features']}")
    print(f"  H1 entropy: {res_a['h1_entropy']:.3f} vs {res_b['h1_entropy']:.3f}")
    print(f"  Total H1 persistence: {res_a['total_persistence_h1']:.3f} vs {res_b['total_persistence_h1']:.3f}")
    print(f"  Wasserstein distance: {w_dist:.4f}")
    print(f"  Permutation p-value: {w_pval:.4f}")
    print(f"  Significant (alpha=0.05): {w_pval < 0.05}")

results_df = pd.DataFrame(stat_results)
results_path = os.path.join(RESULTS_DIR, "agp_tda_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")


# ── 6. Publication figures ───────────────────────────────────────────
print(f"\n{'=' * 70}")
print("GENERATING FIGURES")
print(f"{'=' * 70}")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
    }
)

for comp_name, comp in comparisons.items():
    res_a = all_results[f"{comp_name}_a"]
    res_b = all_results[f"{comp_name}_b"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Persistence diagrams side by side + Betti curves overlay
    # -- Persistence diagram A
    ax1 = fig.add_subplot(gs[0, 0])
    for dim, (color, marker) in enumerate(
        [("#1f77b4", "o"), ("#ff7f0e", "^")]
    ):
        dgm = res_a["finite_dgms"][dim]
        if len(dgm) > 0:
            ax1.scatter(
                dgm[:, 0], dgm[:, 1], c=color, marker=marker, s=15,
                alpha=0.6, label=f"H{dim} ({len(dgm)})",
            )
    max_val = max(
        np.max(res_a["finite_dgms"][0][:, 1]) if len(res_a["finite_dgms"][0]) > 0 else 1,
        np.max(res_a["finite_dgms"][1][:, 1]) if len(res_a["finite_dgms"][1]) > 0 else 1,
    )
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=0.8)
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Death")
    ax1.set_title(f"{comp['group_a_name']}\n(n={res_a['n_samples']})")
    ax1.legend(fontsize=8)

    # -- Persistence diagram B
    ax2 = fig.add_subplot(gs[0, 1])
    for dim, (color, marker) in enumerate(
        [("#1f77b4", "o"), ("#ff7f0e", "^")]
    ):
        dgm = res_b["finite_dgms"][dim]
        if len(dgm) > 0:
            ax2.scatter(
                dgm[:, 0], dgm[:, 1], c=color, marker=marker, s=15,
                alpha=0.6, label=f"H{dim} ({len(dgm)})",
            )
    max_val_b = max(
        np.max(res_b["finite_dgms"][0][:, 1]) if len(res_b["finite_dgms"][0]) > 0 else 1,
        np.max(res_b["finite_dgms"][1][:, 1]) if len(res_b["finite_dgms"][1]) > 0 else 1,
    )
    ax2.plot([0, max_val_b], [0, max_val_b], "k--", alpha=0.3, linewidth=0.8)
    ax2.set_xlabel("Birth")
    ax2.set_ylabel("Death")
    ax2.set_title(f"{comp['group_b_name']}\n(n={res_b['n_samples']})")
    ax2.legend(fontsize=8)

    # -- Betti-1 curves overlay
    ax3 = fig.add_subplot(gs[0, 2])
    fv_a, bc_a = res_a["betti1_curve"]
    fv_b, bc_b = res_b["betti1_curve"]
    ax3.plot(fv_a, bc_a, color="#d32f2f", linewidth=2, label=comp["group_a_name"])
    ax3.plot(fv_b, bc_b, color="#1976d2", linewidth=2, label=comp["group_b_name"])
    ax3.set_xlabel("Filtration value")
    ax3.set_ylabel("Betti-1 number")
    ax3.set_title("Betti-1 curves")
    ax3.legend(fontsize=8)
    ax3.fill_between(fv_a, bc_a, alpha=0.15, color="#d32f2f")
    ax3.fill_between(fv_b, bc_b, alpha=0.15, color="#1976d2")

    # Row 2: Landscapes + summary bar chart + stats text
    # -- Landscape 1 overlay
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(res_a["landscapes"][0], color="#d32f2f", linewidth=1.5, label=comp["group_a_name"])
    ax4.plot(res_b["landscapes"][0], color="#1976d2", linewidth=1.5, label=comp["group_b_name"])
    ax4.set_xlabel("Index")
    ax4.set_ylabel(r"$\lambda_1(t)$")
    ax4.set_title("Persistence landscape 1 (H1)")
    ax4.legend(fontsize=8)

    # -- Summary bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ["H1 features", "H1 entropy", "Total H1\npersistence", "Max Betti-1"]
    vals_a = [
        res_a["n_h1_features"],
        res_a["h1_entropy"],
        res_a["total_persistence_h1"],
        res_a["max_betti1"],
    ]
    vals_b = [
        res_b["n_h1_features"],
        res_b["h1_entropy"],
        res_b["total_persistence_h1"],
        res_b["max_betti1"],
    ]
    x = np.arange(len(metrics))
    w = 0.35
    ax5.bar(x - w / 2, vals_a, w, color="#d32f2f", alpha=0.8, label=comp["group_a_name"])
    ax5.bar(x + w / 2, vals_b, w, color="#1976d2", alpha=0.8, label=comp["group_b_name"])
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=8)
    ax5.set_ylabel("Value")
    ax5.set_title("Topological summary")
    ax5.legend(fontsize=7)

    # -- Stats text panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    stat_row = results_df.loc[results_df["comparison"] == comp_name].iloc[0]
    txt = (
        f"Wasserstein distance: {stat_row['wasserstein_dist']:.4f}\n"
        f"Permutation p-value: {stat_row['wasserstein_pval']:.4f}\n"
        f"Significant (p<0.05): {'YES' if stat_row['significant'] else 'NO'}\n\n"
        f"H1 features: {stat_row['h1_features_a']} vs {stat_row['h1_features_b']}\n"
        f"H1 entropy: {stat_row['h1_entropy_a']} vs {stat_row['h1_entropy_b']}\n"
        f"Total persistence: {stat_row['total_pers_a']} vs {stat_row['total_pers_b']}\n\n"
        f"Samples: {stat_row['n_a']} vs {stat_row['n_b']}\n"
        f"Taxa: {res_a['n_taxa']} (top by prevalence)"
    )
    ax6.text(
        0.05, 0.95, txt, transform=ax6.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )
    ax6.set_title("Statistical results")

    fig.suptitle(
        f"Persistent Homology: {comp['group_a_name']} vs {comp['group_b_name']}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    fig_path = os.path.join(FIGURE_DIR, f"agp_{comp_name}_tda.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ── 7. Print final summary ──────────────────────────────────────────
print(f"\n{'=' * 70}")
print("FINAL RESULTS SUMMARY")
print(f"{'=' * 70}")
print(f"Dataset: American Gut Project (real data)")
print(f"Stool samples: {len(clr_df)}, Taxa after filtering: {clr_df.shape[1]}")
print()
print(results_df.to_string(index=False))
print(f"\n{'=' * 70}")
