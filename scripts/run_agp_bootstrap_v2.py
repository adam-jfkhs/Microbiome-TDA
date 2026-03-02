#!/usr/bin/env python3
"""Bootstrap TDA analysis on real AGP data — v2 (methodologically sound).

Three key upgrades over v1:

A. GLOBAL TAXA: Top-80 taxa selected once across all stool samples, not per
   resample. Every bootstrap iteration operates on the same feature space.

B. PAIRED RESAMPLING with LABEL PERMUTATION: Each iteration draws n=100 from
   group A and n=100 from group B independently (without replacement). The
   test statistic is the mean within-iteration delta. Null distribution is
   built by flipping group labels, giving a clean label-based permutation test.

C. CONFOUNDING CONTROL: Each comparison is run twice — on the full group and
   on a subset matched on age-bin / sex / BMI-bin. Results that survive
   matching are reported as robust.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import (
    FEATURES, select_global_taxa, tda_features,
    paired_resample_test, make_strata, matched_ids,
)
from src.analysis.statistics import fdr_correction

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200        # paired iterations (A vs B per iteration)
SUBSAMPLE_SIZE = 100      # samples drawn per group per iteration
N_PERMUTATIONS = 500      # label permutations for null distribution
N_GLOBAL_TAXA = 80        # fixed taxa set size
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Figure helper ──────────────────────────────────────────────────────────────

def plot_delta_distributions(raw_data, comp_name, label_a, label_b,
                             results_full, results_matched=None):
    """Plot delta distributions for 6 features, full vs matched overlay."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Paired Bootstrap TDA — {label_a} vs {label_b}\n"
        f"({N_ITERATIONS} iterations × {SUBSAMPLE_SIZE} samples, "
        f"global top-{N_GLOBAL_TAXA} taxa)",
        fontsize=12, fontweight="bold",
    )

    for idx, feat in enumerate(FEATURES):
        ax = axes.flat[idx]
        d_full = raw_data["full"]["deltas"][feat]
        ax.hist(d_full, bins=25, alpha=0.7, color="#d32f2f",
                edgecolor="white", linewidth=0.5, label="Full groups", density=True)

        if results_matched is not None:
            d_match = raw_data["matched"]["deltas"][feat]
            ax.hist(d_match, bins=25, alpha=0.5, color="#388e3c",
                    edgecolor="white", linewidth=0.5, label="Matched", density=True)

        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

        row = results_full[results_full["feature"] == feat].iloc[0]
        title = (f"{feat}\nd={row['cohens_d']:.2f}  "
                 f"perm-p={row['permutation_p']:.3f}  "
                 f"wilcox-p={row['wilcoxon_p']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("delta (A − B)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f"agp_{comp_name}_v2.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("UPGRADED BOOTSTRAP TDA — v2")
    print("=" * 70)

    # Load & preprocess
    print("\nLoading AGP data...")
    otu_df, metadata = load_agp()
    stool_mask = metadata["BODY_SITE"] == "UBERON:feces"
    stool_ids = metadata.loc[stool_mask].index.intersection(otu_df.index)
    filtered = filter_low_abundance(otu_df.loc[stool_ids],
                                    min_prevalence=0.05, min_reads=1000)
    clr_df = clr_transform(filtered)
    meta = metadata.loc[clr_df.index]
    print(f"Preprocessed: {clr_df.shape[0]} samples × {clr_df.shape[1]} taxa")

    # A. Select global taxa ONCE
    print("\n--- UPGRADE A: Global taxa selection ---")
    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # C. Build strata for matching
    strata = make_strata(meta)
    print(f"Samples with complete age/sex/BMI: {strata.notna().sum()}")

    # Define comparisons
    comparisons = {
        "antibiotics": {
            "label_a": "Recent ABX",
            "label_b": "No ABX (>1yr)",
            "ids_a": meta.loc[
                meta["ANTIBIOTIC_SELECT"].isin(
                    ["In the past week", "In the past month", "In the past 6 months"]
                )
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["ANTIBIOTIC_SELECT"] == "Not in the last year"
            ].index.intersection(clr_df.index),
        },
        "ibd": {
            "label_a": "IBD (UC+Crohn)",
            "label_b": "Healthy",
            "ids_a": meta.loc[
                meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["IBD"] == "I do not have IBD"
            ].index.intersection(clr_df.index),
        },
        "diet": {
            "label_a": "Plant-based",
            "label_b": "Omnivore",
            "ids_a": meta.loc[
                meta["DIET_TYPE"].isin(["Vegan", "Vegetarian", "Vegetarian but eat seafood"])
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["DIET_TYPE"] == "Omnivore"
            ].index.intersection(clr_df.index),
        },
    }

    print("\nGroup sizes:")
    for name, comp in comparisons.items():
        print(f"  {name}: {comp['label_a']} n={len(comp['ids_a'])}  "
              f"{comp['label_b']} n={len(comp['ids_b'])}")

    # Run each comparison
    all_results = []
    for comp_name, comp in comparisons.items():
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {comp_name}")
        print(f"{'=' * 70}")

        raw_data = {}

        # --- Full groups ---
        print("\n[Full groups]")
        results_full, raw_full = paired_resample_test(
            clr_df, comp["ids_a"], comp["ids_b"], global_taxa,
            n_iter=N_ITERATIONS, subsample_size=SUBSAMPLE_SIZE,
            n_perm=N_PERMUTATIONS, rng=rng,
            label=f"{comp['label_a']} vs {comp['label_b']}",
        )
        results_full["comparison"] = comp_name
        results_full["subset"] = "full"
        raw_data["full"] = raw_full

        # --- Matched subset (C) ---
        print("\n[Matched on age/sex/BMI]")
        m_a, m_b = matched_ids(comp["ids_a"], comp["ids_b"], strata)
        print(f"  After matching: {len(m_a)} vs {len(m_b)}")

        results_matched = None
        raw_data["matched"] = None

        if len(m_a) >= SUBSAMPLE_SIZE and len(m_b) >= SUBSAMPLE_SIZE:
            results_matched, raw_matched = paired_resample_test(
                clr_df, m_a, m_b, global_taxa,
                n_iter=N_ITERATIONS, subsample_size=SUBSAMPLE_SIZE,
                n_perm=N_PERMUTATIONS, rng=rng,
                label=f"{comp['label_a']} vs {comp['label_b']} [matched]",
            )
            results_matched["comparison"] = comp_name
            results_matched["subset"] = "matched"
            raw_data["matched"] = raw_matched
            all_results.append(results_matched)
        else:
            print(f"  Skipping matched analysis (too few matched samples)")

        all_results.append(results_full)

        # Print summary
        print(f"\n  Results — {comp_name} (full):")
        print(f"  {'Feature':<28} {'Mean A':>7} {'Mean B':>7} "
              f"{'Cohen d':>8} {'perm-p':>8} {'wilcox-p':>9}")
        print(f"  {'─' * 68}")
        for _, row in results_full.iterrows():
            sig = "*" if row["permutation_p"] < 0.05 else ""
            print(f"  {row['feature']:<28} {row['mean_a']:>7.3f} {row['mean_b']:>7.3f} "
                  f"{row['cohens_d']:>8.3f} {row['permutation_p']:>8.4f} "
                  f"{row['wilcoxon_p']:>9.4f} {sig}")

        # Figure
        plot_delta_distributions(raw_data, comp_name,
                                 comp["label_a"], comp["label_b"],
                                 results_full, results_matched)

    # Collate results and apply FDR across all tests
    all_df = pd.concat(all_results, ignore_index=True)

    # FDR on permutation p-values
    rejected_perm, adj_perm = fdr_correction(all_df["permutation_p"].values)
    all_df["perm_p_fdr"] = adj_perm
    all_df["sig_perm_fdr"] = rejected_perm

    # FDR on Wilcoxon p-values
    rejected_wilcox, adj_wilcox = fdr_correction(all_df["wilcoxon_p"].values)
    all_df["wilcox_p_fdr"] = adj_wilcox
    all_df["sig_wilcox_fdr"] = rejected_wilcox

    out_path = os.path.join(RESULTS_DIR, "agp_bootstrap_v2.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — FDR-significant results (permutation p, BH-corrected)")
    print(f"{'=' * 70}")
    sig = all_df[all_df["sig_perm_fdr"]]
    if len(sig) > 0:
        print(sig[["comparison", "subset", "feature", "mean_a", "mean_b",
                    "cohens_d", "perm_p_fdr"]].to_string(index=False))
    else:
        print("No results significant after FDR correction.")

    print(f"\nNominally significant (perm p<0.05): "
          f"{(all_df['permutation_p'] < 0.05).sum()}/{len(all_df)}")
    print(f"FDR-significant (perm):  {all_df['sig_perm_fdr'].sum()}/{len(all_df)}")
    print(f"FDR-significant (wilcox): {all_df['sig_wilcox_fdr'].sum()}/{len(all_df)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
