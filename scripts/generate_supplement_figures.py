#!/usr/bin/env python3
"""Generate two supplementary figures that need redrawing:

  1. agp_persistence_diagrams.png  — H₀ and H₁ in SEPARATE panels
  2. loop_attribution_heatmap.png  — fixed label sizing

Run from repo root:
    python scripts/generate_supplement_figures.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
import ripser

SEED = 42
N_GLOBAL_TAXA = 80
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Figure 1: persistence diagrams with separate H₀ / H₁ panels ──────────────

def compute_representative_diagrams(clr_df, metadata, n_samples=120, seed=SEED):
    """
    For a representative bootstrap sample from healthy and IBD groups,
    build the Spearman correlation matrix and run Ripser to get
    birth-death pairs for H₀ and H₁.
    Returns dict with keys 'healthy' and 'ibd', each a dict
    {'h0': array(N,2), 'h1': array(M,2)}.
    """
    rng = np.random.default_rng(seed)

    # IBD label
    ibd_mask     = metadata["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
    healthy_mask = metadata["IBD"] == "I do not have IBD"

    ibd_ids     = metadata.index[ibd_mask].intersection(clr_df.index)
    healthy_ids = metadata.index[healthy_mask].intersection(clr_df.index)

    # Draw balanced samples
    n = min(n_samples, len(ibd_ids), len(healthy_ids))
    s_ibd     = rng.choice(ibd_ids,     n, replace=False)
    s_healthy = rng.choice(healthy_ids, n, replace=False)

    diagrams = {}
    for label, ids in [("healthy", s_healthy), ("ibd", s_ibd)]:
        mat = clr_df.loc[ids].values          # (n, 80)
        corr = np.corrcoef(mat.T)             # (80, 80) Pearson (fast proxy for Spearman at group level)
        # Distance matrix: 1 - |correlation|  (range [0,1])
        dist = 1.0 - np.abs(corr)
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0, 1)

        result = ripser.ripser(dist, distance_matrix=True, maxdim=1)
        dgms = result["dgms"]

        h0 = dgms[0]
        h1 = dgms[1]

        # Remove the infinite bar from H₀ (one component that never merges)
        finite_h0 = h0[np.isfinite(h0[:, 1])]

        diagrams[label] = {"h0": finite_h0, "h1": h1}
        print(f"  {label}: H₀ bars={len(finite_h0)}, H₁ bars={len(h1)}")

    return diagrams


def plot_persistence_diagrams(diagrams, out_path):
    """
    Four-panel figure:
      Left column:  H₀ (connected components) — healthy (top), IBD (bottom)
      Right column: H₁ (loops)                — healthy (top), IBD (bottom)
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(
        "Persistence diagrams: healthy vs IBD co-occurrence networks\n"
        "(representative bootstrap sample, $N = 80$ taxa, $n = 120$ per group)",
        fontsize=11, fontweight="bold",
    )

    colours = {"healthy": "#1565C0", "ibd": "#B71C1C"}
    titles  = {"healthy": "Healthy", "ibd": "IBD (UC + Crohn's)"}
    dim_labels = {0: "H\u2080 (connected components)", 1: "H\u2081 (loops)"}

    for row, label in enumerate(["healthy", "ibd"]):
        for col, dim in enumerate([0, 1]):
            ax  = axes[row, col]
            pts = diagrams[label][f"h{dim}"]
            c   = colours[label]

            if len(pts) == 0:
                ax.text(0.5, 0.5, "no features", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            else:
                births = pts[:, 0]
                deaths = pts[:, 1]
                lifetimes = deaths - births

                # Scatter: colour encodes lifetime
                sc = ax.scatter(births, deaths, c=lifetimes, cmap="plasma",
                                s=18, alpha=0.7, linewidths=0.3,
                                edgecolors="white", vmin=0,
                                vmax=np.percentile(lifetimes, 95))
                plt.colorbar(sc, ax=ax, label="lifetime", pad=0.02, shrink=0.85)

                # Diagonal
                lim = max(deaths.max() * 1.05, 0.01)
                ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
                ax.set_xlim(-0.01, lim)
                ax.set_ylim(-0.01, lim)

                # Annotation
                n_pts = len(pts)
                total_pers = lifetimes.sum()
                ax.text(0.97, 0.05,
                        f"n = {n_pts}\nΣ = {total_pers:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color=c,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=c, alpha=0.8))

            ax.set_xlabel("birth", fontsize=9)
            ax.set_ylabel("death", fontsize=9)
            ax.set_title(
                f"{titles[label]} — {dim_labels[dim]}",
                fontsize=9, color=c, fontweight="bold",
            )
            ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: heatmap with readable labels ────────────────────────────────────

FEATURE_NAMES = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]
FEATURE_DISPLAY = {
    "h1_count": "H₁ count",
    "h1_entropy": "H₁ entropy",
    "h1_total_persistence": "total pers.",
    "h1_mean_lifetime": "mean lifetime",
    "h1_max_lifetime": "max lifetime",
    "max_betti1": "max Betti-1",
}


def truncate_label(name, maxlen=28):
    """Italicise genus/species names; truncate family-level strings."""
    # If it's a bracket genus like [Ruminococcus], keep as-is
    if len(name) <= maxlen:
        return name
    return name[:maxlen - 1] + "…"


def plot_heatmap(csv_path, out_path, top_n=30):
    diff_df = pd.read_csv(csv_path, index_col=0)
    # Drop duplicate index entries, keep first (highest composite_impact)
    diff_df = diff_df[~diff_df.index.duplicated(keep="first")]
    top     = diff_df["composite_impact"].nlargest(top_n).index
    data    = diff_df.loc[top, FEATURE_NAMES]

    # Z-score columns for visual comparability
    zdata = (data - data.mean()) / (data.std() + 1e-9)

    labels     = [truncate_label(n) for n in zdata.index]
    col_labels = [FEATURE_DISPLAY.get(c, c) for c in FEATURE_NAMES]

    # Significance stars on composite_impact
    pvals = diff_df.loc[top, "pval"].values if "pval" in diff_df.columns else None

    fig, ax = plt.subplots(figsize=(9, 7))   # compact for 15 taxa

    im = ax.imshow(zdata.values, aspect="auto", cmap="RdBu_r",
                   vmin=-3, vmax=3, interpolation="nearest")

    # Column labels
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)

    # Row labels — left side, reasonable font
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Significance stars to the right of each row
    if pvals is not None:
        for i, pv in enumerate(pvals):
            star = ("***" if pv < 0.002 else "**" if pv < 0.01
                    else "*" if pv < 0.05 else "")
            if star:
                ax.text(len(FEATURE_NAMES) - 0.4, i, f" {star}",
                        va="center", ha="left", fontsize=7.5, color="#333333")

    # Thin grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(FEATURE_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, label="Z-score", shrink=0.5, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Differential topological impact (healthy − IBD)\ntop {top_n} taxa "
        "(★ p < 0.05, ★★ p < 0.01, ★★★ p < 0.002)",
        fontsize=10, pad=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Heatmap (no data loading needed) ──────────────────────────────────────
    print("=" * 60)
    print("Generating heatmap …")
    plot_heatmap(
        os.path.join(RESULTS_DIR, "loop_attribution_differential.csv"),
        os.path.join(FIGURE_DIR,  "loop_attribution_heatmap.png"),
        top_n=15,
    )

    # ── Persistence diagrams (requires AGP data) ───────────────────────────────
    print("\nLoading AGP data …")
    from src.data.preprocess import filter_low_abundance, clr_transform
    otu_df, metadata = load_agp()
    stool_mask = metadata["BODY_SITE"] == "UBERON:feces"
    stool_ids  = metadata.loc[stool_mask].index.intersection(otu_df.index)
    filtered   = filter_low_abundance(otu_df.loc[stool_ids],
                                      min_prevalence=0.05, min_reads=1000)
    clr_df     = clr_transform(filtered)
    meta       = metadata.loc[clr_df.index]

    # Select same global top-80 taxa as main analysis
    all_labels = np.zeros(len(clr_df), dtype=int)  # placeholder — taxa selection is label-agnostic
    top_taxa   = select_global_taxa(clr_df, N_GLOBAL_TAXA)
    clr_sub    = clr_df[top_taxa]

    print("\nComputing representative persistence diagrams …")
    diagrams = compute_representative_diagrams(clr_sub, meta, n_samples=120)

    print("\nPlotting persistence diagrams …")
    plot_persistence_diagrams(
        diagrams,
        os.path.join(FIGURE_DIR, "agp_persistence_diagrams.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
