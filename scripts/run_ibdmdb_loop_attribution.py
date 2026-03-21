#!/usr/bin/env python3
"""IBDMDB (HMP2) replication of AGP loop attribution analysis.

Applies the same leave-one-out topological attribution pipeline to the
IBDMDB WGS metagenomics dataset (n=1,338 samples, clinically confirmed
IBD diagnoses) and compares top anchors to the AGP 16S results.

This serves as independent cross-dataset validation:
  - Different measurement technology (WGS vs 16S)
  - Clinically confirmed diagnoses vs self-reported (AGP)
  - ~7× more IBD cases (1,209 vs 165)

If the same genera appear as top topological anchors in both datasets,
the finding is replication-grade evidence.

Outputs
-------
  results/ibdmdb_loop_attribution_nonIBD.csv
  results/ibdmdb_loop_attribution_ibd.csv
  results/ibdmdb_loop_attribution_differential.csv
  results/ibdmdb_agp_overlap.csv            ← cross-dataset comparison
  figures/ibdmdb_loop_attribution_top20.png
  figures/ibdmdb_agp_overlap.png
"""

import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from ripser import ripser

from src.data.preprocess import clr_transform

# ── Configuration ───────────────────────────────────────────────────────────────
SEED         = 42
N_GLOBAL_TAXA = 80
N_PERMS      = 500
N_BOOT       = 200
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ibdmdb")
AGP_RES      = os.path.join(os.path.dirname(__file__), "..", "results",
                             "loop_attribution_differential.csv")
FIG_DIR      = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

FEATURE_NAMES = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]


# ── Data loading ────────────────────────────────────────────────────────────────

def load_ibdmdb() -> tuple:
    """Load IBDMDB MetaPhlAn2 profiles and return (clr_df, labels).

    Returns
    -------
    clr_df : DataFrame (samples × top-N species)
    labels : Series (0=nonIBD, 1=IBD)
    """
    tax_path  = os.path.join(DATA_DIR, "taxonomic_profiles.tsv")
    meta_path = os.path.join(DATA_DIR, "hmp2_metadata.csv")

    # Load profiles
    raw = pd.read_csv(tax_path, sep="\t", index_col=0)
    # Columns are sample IDs with _taxonomic_profile suffix
    raw.columns = [c.replace("_taxonomic_profile", "") for c in raw.columns]

    # Keep species-level rows only (7 pipe-delimited levels, ends with s__)
    species_mask = raw.index.str.count(r"\|") == 6
    raw = raw.loc[species_mask]

    # Transpose → samples × species
    abund = raw.T   # (n_samples, n_species)

    # Load metadata, filter to MGX samples
    meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta["sample_id"] = meta["External ID"].str.replace("_P$", "", regex=True)
    meta = meta.set_index("sample_id")
    meta = meta[~meta.index.duplicated(keep="first")]   # remove dup metadata rows

    # Align
    common = abund.index.intersection(meta.index)
    abund = abund.loc[common]
    meta  = meta.loc[common]

    # Binary label: IBD (CD + UC) vs nonIBD
    ibd_mask    = meta["diagnosis"].isin(["CD", "UC"])
    nonibd_mask = meta["diagnosis"] == "nonIBD"
    keep = ibd_mask | nonibd_mask
    abund = abund.loc[keep]
    meta  = meta.loc[keep]
    labels = pd.Series(
        meta["diagnosis"].isin(["CD", "UC"]).astype(int),
        index=meta.index,
    )

    print(f"  Samples: {len(abund)}  (IBD={labels.sum()}, nonIBD={(labels==0).sum()})")

    # Convert relative abundance → pseudo-counts (scale to reads-like values),
    # then CLR-transform.  MetaPhlAn2 outputs proportions (0–100), multiply by 1000.
    abund = abund * 1000.0

    # Filter: keep species present in ≥5% of samples
    prevalence = (abund > 0).mean(axis=0)
    abund = abund.loc[:, prevalence >= 0.05]
    print(f"  Species after prevalence filter: {abund.shape[1]}")

    clr_df = clr_transform(abund)

    # Top N species by above-median prevalence (same criterion as AGP)
    above_median_prev = (clr_df > clr_df.median()).mean(axis=0)
    top_taxa = above_median_prev.nlargest(N_GLOBAL_TAXA).index.tolist()
    clr_df   = clr_df[top_taxa]

    print(f"  CLR matrix shape: {clr_df.shape}")
    return clr_df, labels


# ── TDA core (same as run_loop_attribution.py) ──────────────────────────────────

def h1_features(dgm_h1: np.ndarray) -> np.ndarray:
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
    return np.array([len(finite), entropy, float(total_pers),
                     float(lifetimes.mean()), float(lifetimes.max()), float(max_betti)])


def group_h1(clr_group: np.ndarray) -> np.ndarray:
    n_taxa = clr_group.shape[1]
    if n_taxa < 3:
        return np.zeros(len(FEATURE_NAMES))
    corr, _ = spearmanr(clr_group, axis=0)
    dist = 1.0 - np.abs(np.asarray(corr))
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)
    result = ripser(dist, maxdim=1, distance_matrix=True)
    return h1_features(result["dgms"][1])


def loo_impact(clr_group: np.ndarray, taxa_names: list) -> pd.DataFrame:
    n_taxa   = len(taxa_names)
    baseline = group_h1(clr_group)
    rows, t0 = [], time.time()
    for j, taxon in enumerate(taxa_names):
        idx_keep = [k for k in range(n_taxa) if k != j]
        impact   = baseline - group_h1(clr_group[:, idx_keep])
        rows.append({"taxon": taxon, **dict(zip(FEATURE_NAMES, impact))})
        if (j + 1) % 20 == 0 or j == n_taxa - 1:
            elapsed = time.time() - t0
            eta     = elapsed / (j + 1) * (n_taxa - j - 1)
            print(f"    {j+1}/{n_taxa}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    df = pd.DataFrame(rows).set_index("taxon")
    z  = (df - df.mean()) / (df.std() + 1e-9)
    df["composite_impact"] = z.mean(axis=1)
    return df


# ── Parallel workers ─────────────────────────────────────────────────────────────

def _diff_composite(clr_a, clr_b, n_taxa):
    baseline_a = group_h1(clr_a)
    baseline_b = group_h1(clr_b)
    impacts_a  = np.stack([
        baseline_a - group_h1(clr_a[:, [k for k in range(n_taxa) if k != j]])
        for j in range(n_taxa)])
    impacts_b  = np.stack([
        baseline_b - group_h1(clr_b[:, [k for k in range(n_taxa) if k != j]])
        for j in range(n_taxa)])
    diff_raw   = impacts_a - impacts_b
    z          = (diff_raw - diff_raw.mean(axis=0)) / (diff_raw.std(axis=0) + 1e-9)
    return z.mean(axis=1)


def _perm_worker(args):
    clr_matrix, n_nonibd, n_taxa, seed = args
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clr_matrix))
    return _diff_composite(clr_matrix[idx[:n_nonibd]],
                           clr_matrix[idx[n_nonibd:]], n_taxa)


def _boot_worker(args):
    nonibd_clr, ibd_clr, n_taxa, seed = args
    rng = np.random.default_rng(seed)
    h_idx = rng.integers(0, len(nonibd_clr), size=len(nonibd_clr))
    i_idx = rng.integers(0, len(ibd_clr),    size=len(ibd_clr))
    return _diff_composite(nonibd_clr[h_idx], ibd_clr[i_idx], n_taxa)


def run_permutation_test(clr_matrix, n_nonibd, observed, n_perms=N_PERMS):
    n_taxa = clr_matrix.shape[1]
    n_jobs = max(1, cpu_count() - 1)
    args   = [(clr_matrix, n_nonibd, n_taxa, SEED + s) for s in range(n_perms)]
    print(f"  Running {n_perms} permutations on {n_jobs} CPUs …")
    t0 = time.time()
    with Pool(n_jobs) as pool:
        null = np.stack(pool.map(_perm_worker, args))
    print(f"  Done in {time.time()-t0:.0f}s")
    return (null >= observed[np.newaxis, :]).mean(axis=0)


def run_bootstrap(nonibd_clr, ibd_clr, observed, n_boot=N_BOOT):
    n_taxa = nonibd_clr.shape[1]
    n_jobs = max(1, cpu_count() - 1)
    args   = [(nonibd_clr, ibd_clr, n_taxa, SEED + 10000 + s) for s in range(n_boot)]
    print(f"  Running {n_boot} bootstrap iterations on {n_jobs} CPUs …")
    t0 = time.time()
    with Pool(n_jobs) as pool:
        boot = np.stack(pool.map(_boot_worker, args))
    print(f"  Done in {time.time()-t0:.0f}s")
    ci_lo = np.percentile(boot, 2.5,  axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    return ci_lo, ci_hi


# ── Cross-dataset comparison ─────────────────────────────────────────────────────

def genus_from_species(species_str: str) -> str:
    """Extract genus name from MetaPhlAn2 species string."""
    parts = species_str.split("|")
    for p in reversed(parts):
        if p.startswith("g__"):
            return p[3:]
        if p.startswith("s__"):
            # genus is part of species name: s__Genus_species
            sp = p[3:]
            return sp.split("_")[0] if "_" in sp else sp
    return species_str.split("|")[-1]


def compare_to_agp(ibdmdb_diff: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Compare top IBDMDB anchors to AGP results by genus name."""
    if not os.path.exists(AGP_RES):
        print("  AGP results not found — skipping comparison.")
        return pd.DataFrame()

    agp_diff = pd.read_csv(AGP_RES, index_col=0)
    agp_top  = set(agp_diff["composite_impact"].nlargest(top_n).index)

    ibdmdb_diff = ibdmdb_diff.copy()
    ibdmdb_diff["genus"] = [genus_from_species(s) for s in ibdmdb_diff.index]
    ibdmdb_top_genera = set(
        ibdmdb_diff.nlargest(top_n, "composite_impact")["genus"].tolist()
    )

    rows = []
    for rank, (taxon, row) in enumerate(
        ibdmdb_diff.nlargest(top_n, "composite_impact").iterrows(), 1
    ):
        genus    = row["genus"]
        in_agp   = any(genus.lower() in a.lower() for a in agp_top)
        rows.append({
            "ibdmdb_rank":       rank,
            "ibdmdb_taxon":      taxon,
            "genus":             genus,
            "ibdmdb_composite":  row["composite_impact"],
            "ibdmdb_pval":       row.get("pval", np.nan),
            "replicated_in_agp": in_agp,
        })

    overlap_df = pd.DataFrame(rows)
    n_overlap  = overlap_df["replicated_in_agp"].sum()
    print(f"\n  Cross-dataset overlap (top {top_n}): {n_overlap}/{top_n} genera "
          f"replicated in AGP")
    return overlap_df


# ── Figures ──────────────────────────────────────────────────────────────────────

def plot_top20(diff_df: pd.DataFrame, out_path: str):
    top    = diff_df["composite_impact"].nlargest(20)
    top_df = diff_df.loc[top.index]
    ci_lo  = top_df.get("ci_lo", pd.Series(np.nan, index=top.index))
    ci_hi  = top_df.get("ci_hi", pd.Series(np.nan, index=top.index))
    pvals  = top_df.get("pval",  pd.Series(np.nan, index=top.index))

    # Short display name: genus only
    labels_short = [genus_from_species(t) for t in top.index[::-1]]

    xerr_lo = (top - ci_lo).fillna(0).values[::-1]
    xerr_hi = (ci_hi - top).fillna(0).values[::-1]
    has_ci  = not ci_lo.isna().all()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(20), top.values[::-1], color="#31a354", alpha=0.85)
    if has_ci:
        ax.errorbar(top.values[::-1], range(20),
                    xerr=[xerr_lo, xerr_hi],
                    fmt="none", color="black", capsize=3, linewidth=0.8)
    if not pvals.isna().all():
        for i, (taxon, pv) in enumerate(zip(top.index[::-1], pvals.values[::-1])):
            star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            if star:
                ax.text(top.loc[taxon] + 0.02, i, star, va="center", fontsize=7)

    ax.set_yticks(range(20))
    ax.set_yticklabels(labels_short, fontsize=8)
    ax.set_xlabel("Differential composite topological impact\n(nonIBD anchor score)", fontsize=9)
    ax.set_title("IBDMDB: top 20 topological anchors lost in IBD (WGS, clinically confirmed)",
                 fontsize=9, pad=8)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def plot_overlap(overlap_df: pd.DataFrame, out_path: str):
    """Grouped bar: IBDMDB composite scores, coloured by AGP replication."""
    if overlap_df.empty:
        return
    top    = overlap_df.head(20)
    colors = ["#e6550d" if r else "#bdbdbd" for r in top["replicated_in_agp"]]
    labels = top["genus"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), top["ibdmdb_composite"].values[::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("IBDMDB composite topological impact", fontsize=9)
    ax.set_title("Cross-dataset replication: orange = also top anchor in AGP",
                 fontsize=9, pad=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#e6550d", label="Replicated in AGP (16S)"),
        Patch(facecolor="#bdbdbd", label="IBDMDB only"),
    ], fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Loading IBDMDB data …")
    clr_df, labels = load_ibdmdb()
    taxa_names = clr_df.columns.tolist()

    nonibd_clr = clr_df.loc[labels == 0].values
    ibd_clr    = clr_df.loc[labels == 1].values

    # ── nonIBD LOO ──────────────────────────────────────────────────────────────
    print(f"\nRunning LOO on nonIBD samples (n={len(nonibd_clr)}) …")
    nonibd_df = loo_impact(nonibd_clr, taxa_names)
    nonibd_df.to_csv(os.path.join(RES_DIR, "ibdmdb_loop_attribution_nonIBD.csv"))
    print(f"  Saved: results/ibdmdb_loop_attribution_nonIBD.csv")

    # ── IBD LOO ─────────────────────────────────────────────────────────────────
    print(f"\nRunning LOO on IBD samples (n={len(ibd_clr)}) …")
    ibd_df = loo_impact(ibd_clr, taxa_names)
    ibd_df.to_csv(os.path.join(RES_DIR, "ibdmdb_loop_attribution_ibd.csv"))
    print(f"  Saved: results/ibdmdb_loop_attribution_ibd.csv")

    # ── Differential impact ──────────────────────────────────────────────────────
    diff_df = nonibd_df[FEATURE_NAMES].subtract(ibd_df[FEATURE_NAMES])
    z = (diff_df - diff_df.mean()) / (diff_df.std() + 1e-9)
    diff_df["composite_impact"] = z.mean(axis=1)

    # ── Permutation test ─────────────────────────────────────────────────────────
    print("\nRunning permutation test …")
    observed = diff_df["composite_impact"].values
    pvals    = run_permutation_test(clr_df.values, len(nonibd_clr), observed)
    diff_df["pval"] = pvals

    # ── Bootstrap CI ─────────────────────────────────────────────────────────────
    print("\nRunning bootstrap stability …")
    ci_lo, ci_hi    = run_bootstrap(nonibd_clr, ibd_clr, observed)
    diff_df["ci_lo"] = ci_lo
    diff_df["ci_hi"] = ci_hi

    diff_df = diff_df.sort_values("composite_impact", ascending=False)
    diff_df.to_csv(os.path.join(RES_DIR, "ibdmdb_loop_attribution_differential.csv"))
    print(f"  Saved: results/ibdmdb_loop_attribution_differential.csv")

    # ── Print top 20 ─────────────────────────────────────────────────────────────
    print("\nTop 20 IBDMDB differential topological anchors (nonIBD − IBD):")
    print(f"{'Rank':<5} {'Genus':<30} {'Composite':>10} {'p-val':>8}")
    print("-" * 58)
    for rank, (taxon, row) in enumerate(diff_df.head(20).iterrows(), 1):
        genus = genus_from_species(taxon)
        star  = "***" if row["pval"] < 0.001 else "**" if row["pval"] < 0.01 else "*" if row["pval"] < 0.05 else ""
        print(f"{rank:<5} {genus:<30} {row['composite_impact']:>10.3f} {row['pval']:>7.3f} {star}")

    # ── Cross-dataset comparison ─────────────────────────────────────────────────
    print("\nComparing to AGP results …")
    overlap_df = compare_to_agp(diff_df)
    if not overlap_df.empty:
        overlap_df.to_csv(os.path.join(RES_DIR, "ibdmdb_agp_overlap.csv"), index=False)
        print(f"  Saved: results/ibdmdb_agp_overlap.csv")

    # ── Figures ──────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_top20(diff_df, os.path.join(FIG_DIR, "ibdmdb_loop_attribution_top20.png"))
    plot_overlap(overlap_df, os.path.join(FIG_DIR, "ibdmdb_agp_overlap.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
