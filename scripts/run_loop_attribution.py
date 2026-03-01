#!/usr/bin/env python3
"""Loop attribution analysis: which taxa anchor co-occurrence loops in healthy vs IBD?

For each of the 80 CLR taxa we run a leave-one-out (LOO) experiment:
  1. Remove taxon j from the group-level CLR matrix (n_samples × 79).
  2. Compute the Spearman taxon–taxon correlation matrix (79×79).
  3. Build distance matrix  d = 1 − |r|.
  4. Run Ripser (H₁) and extract the same six scalar features used elsewhere.
  5. Compare to the full-80-taxon baseline → impact = baseline − LOO.

Steps 1–2 are repeated for healthy and IBD groups separately, giving:
  - healthy_impact[j]   (6-vector)
  - ibd_impact[j]       (6-vector)
  - diff_impact[j] = healthy_impact[j] − ibd_impact[j]

Taxa with high diff_impact are "healthy anchors" — structurally critical in
healthy networks but not in IBD networks.  These are the bacteria whose
partnership loops are specifically dismantled by disease.

Outputs
-------
  results/loop_attribution_healthy.csv
  results/loop_attribution_ibd.csv
  results/loop_attribution_differential.csv   ← primary result
  figures/loop_attribution_heatmap.png
  figures/loop_attribution_top20.png

Runtime: ~1–2 min (160 Ripser runs on 79×79 matrices).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from ripser import ripser

from biom import load_table

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform

# ── Configuration ───────────────────────────────────────────────────────────────
SEED = 42
N_GLOBAL_TAXA = 80
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIG_DIR   = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

FEATURE_NAMES = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]


# ── Taxonomy mapping ────────────────────────────────────────────────────────────

def load_taxonomy_map() -> dict:
    """Return {otu_id: 'Genus species'} from BIOM observation metadata."""
    biom_path = os.path.join(DATA_DIR, "agp", "agp_otu_table.biom")
    table = load_table(biom_path)
    tax_map = {}
    for otu_id in table.ids(axis="observation"):
        meta = table.metadata(otu_id, axis="observation")
        if meta and "taxonomy" in meta:
            parts = [p.strip() for p in meta["taxonomy"]]
            # Walk from species → genus → … to find lowest non-empty name
            label = None
            for p in reversed(parts):
                stripped = p.split("__", 1)[-1].strip()
                if stripped:
                    label = stripped
                    break
            tax_map[str(otu_id)] = label or str(otu_id)
        else:
            tax_map[str(otu_id)] = str(otu_id)
    return tax_map


# ── Data loading (mirrors run_classification_benchmark.py) ──────────────────────

def load_and_prepare():
    """Return (clr_df, labels) filtered to stool samples with binary IBD label."""
    otu_df, meta = load_agp(os.path.join(DATA_DIR, "agp"))

    stool_ids = meta.index[meta["BODY_SITE"] == "UBERON:feces"]
    otu_df = otu_df.loc[otu_df.index.intersection(stool_ids)]
    meta   = meta.loc[otu_df.index]

    ibd_mask     = meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
    healthy_mask = meta["IBD"] == "I do not have IBD"
    keep   = ibd_mask | healthy_mask
    otu_df = otu_df.loc[keep]
    meta   = meta.loc[keep]

    labels = pd.Series(
        (meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])).astype(int),
        index=meta.index,
    )

    otu_filtered = filter_low_abundance(otu_df, min_prevalence=0.05, min_reads=1000)
    clr_df = clr_transform(otu_filtered)

    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top_taxa   = prevalence.nlargest(N_GLOBAL_TAXA).index.tolist()
    clr_df     = clr_df[top_taxa]
    labels     = labels.loc[clr_df.index]

    return clr_df, labels


# ── H₁ feature extraction ───────────────────────────────────────────────────────

def h1_features(dgm_h1: np.ndarray) -> np.ndarray:
    """Return the six-vector of H₁ scalar features."""
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else np.empty((0, 2))
    if len(finite) == 0:
        return np.zeros(len(FEATURE_NAMES))

    lifetimes  = finite[:, 1] - finite[:, 0]
    total_pers = lifetimes.sum()
    norm_lt    = lifetimes / total_pers if total_pers > 0 else lifetimes
    entropy    = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))

    births, deaths = finite[:, 0], finite[:, 1]
    thresholds = np.unique(np.concatenate([births, deaths]))
    max_betti  = int(max(
        np.sum((births <= t) & (deaths > t)) for t in thresholds
    )) if len(thresholds) > 0 else 0

    return np.array([
        len(finite),
        entropy,
        float(total_pers),
        float(lifetimes.mean()),
        float(lifetimes.max()),
        float(max_betti),
    ])


# ── Group-level correlation → distance → Ripser ────────────────────────────────

def group_h1(clr_group: np.ndarray) -> np.ndarray:
    """Compute H₁ features from the group-level taxon correlation matrix."""
    n_taxa = clr_group.shape[1]
    if n_taxa < 3:
        return np.zeros(len(FEATURE_NAMES))

    corr, _ = spearmanr(clr_group, axis=0)   # shape (n_taxa, n_taxa)
    if n_taxa == 1:
        corr = np.array([[1.0]])
    dist = 1.0 - np.abs(np.asarray(corr))
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    result = ripser(dist, maxdim=1, distance_matrix=True)
    return h1_features(result["dgms"][1])


def loo_impact(clr_group: np.ndarray, taxa_names: list) -> pd.DataFrame:
    """Run LOO for every taxon.  Returns DataFrame (taxa × features) of impact scores."""
    n_taxa   = len(taxa_names)
    baseline = group_h1(clr_group)

    rows = []
    t0   = time.time()
    for j, taxon in enumerate(taxa_names):
        idx_keep = [k for k in range(n_taxa) if k != j]
        loo_feat = group_h1(clr_group[:, idx_keep])
        impact   = baseline - loo_feat          # positive → taxon inflates feature
        rows.append({"taxon": taxon, **dict(zip(FEATURE_NAMES, impact))})

        if (j + 1) % 20 == 0 or j == n_taxa - 1:
            elapsed = time.time() - t0
            eta     = elapsed / (j + 1) * (n_taxa - j - 1)
            print(f"    {j+1}/{n_taxa}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    df = pd.DataFrame(rows).set_index("taxon")
    # Add composite score: mean of z-scored individual impacts
    z = (df - df.mean()) / (df.std() + 1e-9)
    df["composite_impact"] = z.mean(axis=1)
    return df


# ── Figures ─────────────────────────────────────────────────────────────────────

def plot_top20(diff_df: pd.DataFrame, out_path: str):
    """Bar chart of top 20 differential topological anchors."""
    top = diff_df["composite_impact"].nlargest(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top.values[::-1], color="#e6550d", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_xlabel("Differential composite topological impact\n(healthy anchor score)", fontsize=9)
    ax.set_title(
        "Top 20 taxa: topological anchors in healthy but not IBD networks",
        fontsize=9, pad=8,
    )
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def plot_heatmap(diff_df: pd.DataFrame, out_path: str, top_n: int = 30):
    """Heatmap of per-feature impact for top N differential taxa."""
    top = diff_df["composite_impact"].nlargest(top_n).index
    data = diff_df.loc[top, FEATURE_NAMES]

    # Z-score columns for visual comparability
    zdata = (data - data.mean()) / (data.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(zdata.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=7)
    ax.set_title(
        f"Differential topological impact (healthy − IBD), top {top_n} taxa",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Loading AGP data …")
    clr_df, labels = load_and_prepare()
    taxa_names = clr_df.columns.tolist()

    print("Loading taxonomy map …")
    tax_map = load_taxonomy_map()
    # Build display names: prefer genus-level label, keep OTU id as fallback
    display = {otu: tax_map.get(str(otu), str(otu)) for otu in taxa_names}
    print(f"  n_samples={len(clr_df)}, n_taxa={len(taxa_names)}")
    print(f"  IBD={labels.sum()}, healthy={(labels == 0).sum()}")

    healthy_clr = clr_df.loc[labels == 0].values
    ibd_clr     = clr_df.loc[labels == 1].values

    # ── Healthy LOO ─────────────────────────────────────────────────────────────
    print(f"\nRunning LOO on healthy samples (n={len(healthy_clr)}) …")
    healthy_df = loo_impact(healthy_clr, taxa_names)
    healthy_df.index = [display[t] for t in healthy_df.index]
    healthy_df.to_csv(os.path.join(RES_DIR, "loop_attribution_healthy.csv"))
    print(f"  Saved: results/loop_attribution_healthy.csv")

    # ── IBD LOO ─────────────────────────────────────────────────────────────────
    print(f"\nRunning LOO on IBD samples (n={len(ibd_clr)}) …")
    ibd_df = loo_impact(ibd_clr, taxa_names)
    ibd_df.index = [display[t] for t in ibd_df.index]
    ibd_df.to_csv(os.path.join(RES_DIR, "loop_attribution_ibd.csv"))
    print(f"  Saved: results/loop_attribution_ibd.csv")

    # ── Differential impact ──────────────────────────────────────────────────────
    diff_df = healthy_df[FEATURE_NAMES].subtract(ibd_df[FEATURE_NAMES])
    z = (diff_df - diff_df.mean()) / (diff_df.std() + 1e-9)
    diff_df["composite_impact"] = z.mean(axis=1)
    diff_df = diff_df.sort_values("composite_impact", ascending=False)
    diff_df.to_csv(os.path.join(RES_DIR, "loop_attribution_differential.csv"))
    print(f"  Saved: results/loop_attribution_differential.csv")

    # ── Print top 20 ─────────────────────────────────────────────────────────────
    print("\nTop 20 differential topological anchors (healthy − IBD):")
    print(f"{'Rank':<5} {'Taxon':<35} {'Composite':>10}")
    print("-" * 55)
    for rank, (taxon, row) in enumerate(diff_df.head(20).iterrows(), 1):
        print(f"{rank:<5} {taxon:<35} {row['composite_impact']:>10.3f}")

    # ── Figures ──────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_top20(diff_df, os.path.join(FIG_DIR, "loop_attribution_top20.png"))
    plot_heatmap(diff_df, os.path.join(FIG_DIR, "loop_attribution_heatmap.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
