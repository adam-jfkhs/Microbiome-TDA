#!/usr/bin/env python3
"""Bootstrap TDA analysis on IBDMDB / HMP2 metagenomics data.

Replicates the AGP v2 methodology in an independent clinical cohort, and adds
two novel biomarker-activity comparisons not possible with AGP:

  COMPARISONS
  ───────────
  1. ibd_vs_nonibd       — CD+UC vs. healthy controls  (replication)
  2. cd_vs_nonibd        — Crohn's disease vs. healthy controls
  3. uc_vs_nonibd        — Ulcerative colitis vs. healthy controls
  4. calprotectin        — high (≥250 μg/g) vs. low (<50 μg/g) inflammation
  5. hbi                 — active CD (HBI≥5) vs. remission CD (HBI<5)
  6. sccai               — active UC (SCCAI≥3) vs. remission UC (SCCAI<3)

Methodological notes
  - Global taxa selected once from all IBDMDB metagenomics samples.
  - MetaPhlAn2 species-level relative abundances → CLR-transformed.
  - Paired bootstrap identical to AGP v2 (sign-flip permutation null,
    Wilcoxon confirmatory, BH FDR across all tests per dataset).
  - IBDMDB is longitudinal; we treat each sample as independent (standard
    for cross-sectional TDA comparison; longitudinal analysis is future work).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.ibdmdb_loader import load_ibdmdb, ibdmdb_group_ids
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import FEATURES, select_global_taxa, tda_features, paired_resample_test
from src.analysis.statistics import fdr_correction

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200
SUBSAMPLE_SIZE = 60       # smaller than AGP because nonIBD group is n=364
N_PERMUTATIONS = 500
N_GLOBAL_TAXA = 80
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

COMPARISONS = [
    "ibd_vs_nonibd",
    "cd_vs_nonibd",
    "uc_vs_nonibd",
    "high_vs_low_calprotectin",
    "high_vs_low_hbi",
    "high_vs_low_sccai",
]


# ── Longitudinal de-duplication ────────────────────────────────────────────────

def one_per_subject(ids, metadata, subject_col="Participant ID", seed=SEED):
    """Return a subset of ids with at most one sample per subject.

    For subjects with multiple longitudinal samples, one is chosen uniformly at
    random (seeded for reproducibility).  This sensitivity check tests whether
    treating IBDMDB timepoints as independent inflates effective sample size.

    Parameters
    ----------
    ids : Iterable of sample identifiers (index values in metadata).
    metadata : DataFrame indexed by sample ID, must contain subject_col.
    subject_col : Column name for subject/participant identifier.
    seed : RNG seed.

    Returns
    -------
    List of sample IDs — one per unique subject, in the intersection of ids
    with metadata.index.
    """
    rng_local = np.random.default_rng(seed)
    meta_sub = metadata.loc[[i for i in ids if i in metadata.index]]
    if subject_col not in meta_sub.columns:
        return list(meta_sub.index)   # no subject info; return as-is

    selected = []
    for _, grp in meta_sub.groupby(subject_col, sort=False):
        chosen = rng_local.choice(grp.index.tolist())
        selected.append(chosen)
    return selected


def plot_delta_distributions(deltas, comp_name, label_a, label_b, results_df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"IBDMDB Bootstrap TDA — {label_a} vs {label_b}\n"
        f"({N_ITERATIONS} iterations × {SUBSAMPLE_SIZE} samples, "
        f"global top-{N_GLOBAL_TAXA} taxa)",
        fontsize=12, fontweight="bold",
    )

    for idx, feat in enumerate(FEATURES_TO_TEST):
        ax = axes.flat[idx]
        d = deltas[feat]
        ax.hist(d, bins=25, alpha=0.75, color="#1565c0",
                edgecolor="white", linewidth=0.5, density=True)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

        row = results_df[results_df["feature"] == feat].iloc[0]
        title = (f"{feat}\nd={row['cohens_d']:.2f}  "
                 f"perm-p={row['permutation_p']:.3f}  "
                 f"wilcox-p={row['wilcoxon_p']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("delta (A − B)")
        ax.set_ylabel("density")

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f"ibdmdb_{comp_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("IBDMDB BOOTSTRAP TDA — Independent Validation")
    print("=" * 70)

    # Load & preprocess
    print("\nLoading IBDMDB data...")
    abundance_df, metadata_df = load_ibdmdb()
    print(f"Raw: {abundance_df.shape[0]} samples × {abundance_df.shape[1]} species")
    print(f"Diagnosis breakdown: {metadata_df['diagnosis'].value_counts().to_dict()}")

    # Filter and CLR-transform
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    meta = metadata_df.loc[clr_df.index]
    print(f"After filtering: {clr_df.shape[0]} samples × {clr_df.shape[1]} taxa")

    # Select global taxa once
    print("\nSelecting global taxa...")
    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # Report biomarker and longitudinal availability
    calp = meta["fecalcal"].notna().sum() if "fecalcal" in meta.columns else 0
    hbi_n = meta["hbi"].notna().sum()
    sccai_n = meta["sccai"].notna().sum()
    n_subjects = meta["Participant ID"].nunique() if "Participant ID" in meta.columns else None
    print(f"\nBiomarker availability: calprotectin={calp}, HBI={hbi_n}, SCCAI={sccai_n}")
    if n_subjects is not None:
        print(f"Unique participants: {n_subjects} ({len(meta)} samples → "
              f"{len(meta) / n_subjects:.1f} timepoints/subject on average)")

    # Run comparisons
    all_results = []

    for comp_name in COMPARISONS:
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {comp_name}")
        print(f"{'=' * 70}")

        ids_a, ids_b, label_a, label_b = ibdmdb_group_ids(meta, comp_name)
        ids_a = [i for i in ids_a if i in clr_df.index]
        ids_b = [i for i in ids_b if i in clr_df.index]
        print(f"  Group sizes: {label_a} n={len(ids_a)}, {label_b} n={len(ids_b)}")

        results_df, raw = paired_resample_test(
            clr_df, ids_a, ids_b, global_taxa,
            n_iter=N_ITERATIONS, subsample_size=SUBSAMPLE_SIZE,
            n_perm=N_PERMUTATIONS, rng=rng,
            label=f"{label_a} vs {label_b}",
        )

        if results_df is None:
            continue

        results_df["comparison"] = comp_name
        results_df["label_a"] = label_a
        results_df["label_b"] = label_b
        all_results.append(results_df)

        print(f"\n  Results:")
        print(f"  {'Feature':<28} {'Mean A':>7} {'Mean B':>7} "
              f"{'Cohen d':>8} {'perm-p':>8} {'wilcox-p':>9}")
        print(f"  {'─' * 68}")
        for _, row in results_df.iterrows():
            sig = "*" if row["permutation_p"] < 0.05 else ""
            print(f"  {row['feature']:<28} {row['mean_a']:>7.3f} {row['mean_b']:>7.3f} "
                  f"{row['cohens_d']:>8.3f} {row['permutation_p']:>8.4f} "
                  f"{row['wilcoxon_p']:>9.4f} {sig}")

        plot_delta_distributions(raw["deltas"], comp_name, label_a, label_b, results_df)

    if not all_results:
        print("\nNo comparisons completed.")
        return

    # Collate and apply FDR
    all_df = pd.concat(all_results, ignore_index=True)

    rejected_perm, adj_perm = fdr_correction(all_df["permutation_p"].values)
    all_df["perm_p_fdr"] = adj_perm
    all_df["sig_perm_fdr"] = rejected_perm

    rejected_wilcox, adj_wilcox = fdr_correction(all_df["wilcoxon_p"].values)
    all_df["wilcox_p_fdr"] = adj_wilcox
    all_df["sig_wilcox_fdr"] = rejected_wilcox

    out_path = os.path.join(RESULTS_DIR, "ibdmdb_bootstrap.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — FDR-significant results (permutation p, BH-corrected)")
    print(f"{'=' * 70}")
    sig = all_df[all_df["sig_perm_fdr"]]
    if len(sig) > 0:
        print(sig[["comparison", "feature", "mean_a", "mean_b",
                   "cohens_d", "perm_p_fdr"]].to_string(index=False))
    else:
        print("No results significant after FDR correction.")

    print(f"\nNominally significant (perm p<0.05): "
          f"{(all_df['permutation_p'] < 0.05).sum()}/{len(all_df)}")
    print(f"FDR-significant (perm):  {all_df['sig_perm_fdr'].sum()}/{len(all_df)}")
    print(f"FDR-significant (wilcox): {all_df['sig_wilcox_fdr'].sum()}/{len(all_df)}")
    print("=" * 70)

    # ── Longitudinal sensitivity check ────────────────────────────────────────
    # IBDMDB is a longitudinal study; treating timepoints as independent
    # inflates effective sample size.  This check subsamples one random
    # timepoint per participant and re-runs the primary ibd_vs_nonibd
    # comparison to confirm that the finding is not an artefact of
    # within-subject temporal correlation.
    print(f"\n{'=' * 70}")
    print("LONGITUDINAL SENSITIVITY CHECK — one timepoint per subject")
    print(f"{'=' * 70}")

    if "Participant ID" not in meta.columns:
        print("  Skipping: 'Participant ID' column not found in metadata.")
    else:
        ids_a_all, ids_b_all, la, lb = ibdmdb_group_ids(meta, "ibd_vs_nonibd")
        ids_a_all = [i for i in ids_a_all if i in clr_df.index]
        ids_b_all = [i for i in ids_b_all if i in clr_df.index]

        ids_a_1ps = one_per_subject(ids_a_all, meta)
        ids_b_1ps = one_per_subject(ids_b_all, meta)
        print(f"  IBD: {len(ids_a_all)} timepoints → {len(ids_a_1ps)} subjects")
        print(f"  nonIBD: {len(ids_b_all)} timepoints → {len(ids_b_1ps)} subjects")

        subsample_1ps = min(SUBSAMPLE_SIZE, len(ids_a_1ps), len(ids_b_1ps))
        sens_df, _ = paired_resample_test(
            clr_df, ids_a_1ps, ids_b_1ps, global_taxa,
            n_iter=N_ITERATIONS, subsample_size=subsample_1ps,
            n_perm=N_PERMUTATIONS, rng=np.random.default_rng(SEED + 1),
            label="IBD vs nonIBD (1 timepoint/subject)",
            min_samples=10,
        )
        if sens_df is not None:
            sens_df["comparison"] = "ibd_vs_nonibd_1ps"
            out_1ps = os.path.join(RESULTS_DIR, "ibdmdb_bootstrap_1ps.csv")
            sens_df.to_csv(out_1ps, index=False)
            print(f"\n  1-per-subject results saved to {out_1ps}")
            print(f"  {'Feature':<28} {'Cohen d':>8} {'perm-p':>8}")
            print(f"  {'─' * 50}")
            for _, row in sens_df.iterrows():
                sig = "*" if row["permutation_p"] < 0.05 else ""
                print(f"  {row['feature']:<28} {row['cohens_d']:>8.3f} "
                      f"{row['permutation_p']:>8.4f} {sig}")
        print("=" * 70)


if __name__ == "__main__":
    main()
