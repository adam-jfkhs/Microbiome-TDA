#!/usr/bin/env python3
"""Taxa-set sensitivity analysis for bootstrap TDA.

Tests whether topological findings are robust to the number of taxa
included in co-occurrence networks. Runs the full paired-resampling
pipeline at three taxon-set sizes: 50, 80, 120.

Statistical framework (explicit):
  - Unit of analysis: one paired iteration (subsample A + subsample B)
  - N_ITERATIONS paired iterations per comparison
  - Primary p-value: sign-flip label permutation on mean(delta)
  - FDR correction: Benjamini-Hochberg across 18 tests per subset
    (3 comparisons × 6 features), applied SEPARATELY for full and
    matched subsets. Wilcoxon signed-rank is reported as a secondary
    confirmatory column only, not used for FDR.
  - Taxa selected globally once per taxa-size N, before any resampling.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import FEATURES, select_global_taxa, tda_features, paired_resample_test, make_strata, matched_ids
from src.analysis.statistics import fdr_correction

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200
SUBSAMPLE_SIZE = 100
N_PERMUTATIONS = 500
TAXA_SIZES = [50, 80, 120]

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_LABELS = {
    "h1_count":            "H1 count",
    "h1_entropy":          "H1 entropy",
    "h1_total_persistence": "Total persistence",
    "h1_mean_lifetime":    "Mean lifetime",
    "h1_max_lifetime":     "Max lifetime",
    "max_betti1":          "Max Betti-1",
}

COMPARISONS_DEF = {
    "antibiotics": ("Recent ABX",   "No ABX (>1yr)"),
    "ibd":         ("IBD (UC+Crohn)", "Healthy"),
    "diet":        ("Plant-based",  "Omnivore"),
}


def apply_fdr_18(df):
    """Apply BH-FDR across exactly 18 tests (3 comparisons × 6 features).

    FDR POLICY: correction is over the 18 primary (sign-flip) p-values
    within each subset (full and matched treated separately).
    Wilcoxon p-values are NOT used for FDR.
    """
    assert len(df) == 18, f"Expected 18 rows for FDR, got {len(df)}"
    p = df["permutation_p"].values
    n = len(p)
    order = np.argsort(p)
    adj = np.zeros(n)
    adj[order[-1]] = p[order[-1]]
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i + 1]], p[order[i]] * n / (i + 1))
    adj = np.clip(adj, 0, 1)
    df = df.copy()
    df["perm_p_fdr18"] = adj
    df["sig_fdr18"] = adj <= 0.05
    return df


# ── Sensitivity figure ─────────────────────────────────────────────────────────

def plot_sensitivity(pivot_d, pivot_sig, subset_label):
    """Heatmap of Cohen's d values across taxa sizes, marked for significance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"Taxa-set sensitivity: Cohen's d by N taxa ({subset_label} groups)\n"
        f"* = FDR-significant (18-test BH correction, sign-flip permutation p)",
        fontsize=12, fontweight="bold",
    )

    comps = list(COMPARISONS_DEF.keys())
    cmap = plt.cm.RdBu_r
    vmax = max(abs(pivot_d.values.max()), abs(pivot_d.values.min()), 0.5)

    for ax, comp in zip(axes, comps):
        comp_d = pivot_d.loc[comp]   # rows=feature, cols=taxa_size
        comp_s = pivot_sig.loc[comp]

        im = ax.imshow(comp_d.values, cmap=cmap, vmin=-vmax, vmax=vmax,
                       aspect="auto")

        feat_labels = [FEATURE_LABELS[f] for f in comp_d.index]
        taxa_labels = [str(n) for n in comp_d.columns]

        ax.set_xticks(range(len(taxa_labels)))
        ax.set_xticklabels(taxa_labels, fontsize=10)
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_xlabel("N taxa", fontsize=10)
        ax.set_title(f"{COMPARISONS_DEF[comp][0]}\nvs {COMPARISONS_DEF[comp][1]}",
                     fontsize=10, fontweight="bold")

        # Annotate cells
        for i, feat in enumerate(comp_d.index):
            for j, n_taxa in enumerate(comp_d.columns):
                d_val = comp_d.loc[feat, n_taxa]
                sig = comp_s.loc[feat, n_taxa]
                star = "*" if sig else ""
                ax.text(j, i, f"{d_val:.2f}{star}", ha="center", va="center",
                        fontsize=8, color="white" if abs(d_val) > vmax * 0.6 else "black")

        plt.colorbar(im, ax=ax, label="Cohen's d")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, f"taxa_sensitivity_{subset_label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TAXA-SET SENSITIVITY ANALYSIS")
    print(f"N taxa tested: {TAXA_SIZES}")
    print()
    print("FDR POLICY:")
    print("  Primary p-value: sign-flip label permutation (500 shuffles)")
    print("  Correction: BH-FDR across 18 tests (3 comparisons × 6 features)")
    print("  Applied SEPARATELY for full and matched subsets")
    print("  Wilcoxon is secondary/confirmatory only — not used for FDR")
    print("=" * 70)

    # Load & preprocess once
    print("\nLoading AGP data...")
    otu_df, metadata = load_agp()
    stool_mask = metadata["BODY_SITE"] == "UBERON:feces"
    stool_ids = metadata.loc[stool_mask].index.intersection(otu_df.index)
    filtered = filter_low_abundance(otu_df.loc[stool_ids],
                                    min_prevalence=0.05, min_reads=1000)
    clr_df = clr_transform(filtered)
    meta = metadata.loc[clr_df.index]
    print(f"Preprocessed: {clr_df.shape[0]} samples × {clr_df.shape[1]} taxa")

    strata = make_strata(meta)
    print(f"Samples with complete age/sex/BMI covariates: {strata.notna().sum()}")

    # Define group IDs (constant across taxa sizes)
    groups = {
        "antibiotics": {
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
            "ids_a": meta.loc[
                meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["IBD"] == "I do not have IBD"
            ].index.intersection(clr_df.index),
        },
        "diet": {
            "ids_a": meta.loc[
                meta["DIET_TYPE"].isin(["Vegan", "Vegetarian", "Vegetarian but eat seafood"])
            ].index.intersection(clr_df.index),
            "ids_b": meta.loc[
                meta["DIET_TYPE"] == "Omnivore"
            ].index.intersection(clr_df.index),
        },
    }

    # Pre-compute matched subsets (constant across taxa sizes)
    matched = {}
    for comp_name, g in groups.items():
        ma, mb = matched_ids(g["ids_a"], g["ids_b"], strata)
        matched[comp_name] = {"ids_a": ma, "ids_b": mb}

    # Container for all results
    all_rows = []

    # ── Loop over taxa sizes ───────────────────────────────────────────────────
    for n_taxa in TAXA_SIZES:
        print(f"\n{'=' * 70}")
        print(f"N TAXA = {n_taxa}")
        print("=" * 70)

        # Global taxa selection for this N (uses canonical function)
        taxa_list = select_global_taxa(clr_df, n=n_taxa)

        t_total = time.time()

        for comp_idx, (comp_name, g) in enumerate(groups.items()):
            la, lb = COMPARISONS_DEF[comp_name]

            # Full groups
            print(f"\n  [{comp_name}] Full — {la} (n={len(g['ids_a'])}) "
                  f"vs {lb} (n={len(g['ids_b'])})")
            comp_rng = np.random.default_rng(SEED + n_taxa * 100 + comp_idx)
            df_full, _ = paired_resample_test(
                clr_df, g["ids_a"], g["ids_b"], taxa_list,
                N_ITERATIONS, SUBSAMPLE_SIZE, N_PERMUTATIONS, comp_rng,
                label=f"{comp_name}/full/N={n_taxa}",
            )
            df_full["comparison"] = comp_name
            df_full["subset"] = "full"
            df_full["n_taxa"] = n_taxa
            all_rows.append(df_full)

            # Matched subset
            ma, mb = matched[comp_name]["ids_a"], matched[comp_name]["ids_b"]
            if len(ma) >= SUBSAMPLE_SIZE and len(mb) >= SUBSAMPLE_SIZE:
                print(f"  [{comp_name}] Matched — {la} (n={len(ma)}) vs {lb} (n={len(mb)})")
                matched_rng = np.random.default_rng(SEED + n_taxa * 100 + comp_idx + 1000)
                df_match, _ = paired_resample_test(
                    clr_df, ma, mb, taxa_list,
                    N_ITERATIONS, SUBSAMPLE_SIZE, N_PERMUTATIONS, matched_rng,
                    label=f"{comp_name}/matched/N={n_taxa}",
                )
                df_match["comparison"] = comp_name
                df_match["subset"] = "matched"
                df_match["n_taxa"] = n_taxa
                all_rows.append(df_match)

        print(f"\n  N={n_taxa} total time: {time.time() - t_total:.0f}s")

    # ── Assemble and apply FDR ─────────────────────────────────────────────────
    full_df = pd.concat(all_rows, ignore_index=True)

    fdr_chunks = []
    for n_taxa in TAXA_SIZES:
        for subset in ["full", "matched"]:
            chunk = full_df[(full_df["n_taxa"] == n_taxa) & (full_df["subset"] == subset)].copy()
            if len(chunk) != 18:
                # matched may be absent for a taxa size; skip gracefully
                continue
            chunk = apply_fdr_18(chunk.reset_index(drop=True))
            fdr_chunks.append(chunk)

    results = pd.concat(fdr_chunks, ignore_index=True)

    out_path = os.path.join(RESULTS_DIR, "taxa_sensitivity.csv")
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SENSITIVITY SUMMARY — FDR-significant (18-test BH, sign-flip p)")
    print(f"{'=' * 70}")

    for subset in ["full", "matched"]:
        print(f"\nSubset: {subset}")
        sub = results[results["subset"] == subset]
        # Count sig per comparison × n_taxa
        pivot_nsig = sub.groupby(["comparison", "n_taxa"])["sig_fdr18"].sum().unstack("n_taxa")  # noqa: E501
        print(f"  Number of FDR-significant features (out of 6):")
        print(pivot_nsig.to_string())

    # ── Heatmap figures ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("GENERATING SENSITIVITY FIGURES")
    print(f"{'=' * 70}")

    for subset in ["full", "matched"]:
        sub = results[results["subset"] == subset]
        if sub.empty:
            continue

        # Build pivot tables: rows=comparison+feature, cols=n_taxa
        d_rows = {}
        s_rows = {}
        for comp in COMPARISONS_DEF:
            for feat in FEATURES:
                key = (comp, feat)
                d_rows[key] = {}
                s_rows[key] = {}
                for n_taxa in TAXA_SIZES:
                    cell = sub[(sub["comparison"] == comp) &
                               (sub["feature"] == feat) &
                               (sub["n_taxa"] == n_taxa)]
                    if not cell.empty:
                        d_rows[key][n_taxa] = cell.iloc[0]["cohens_d"]
                        s_rows[key][n_taxa] = bool(cell.iloc[0]["sig_fdr18"])
                    else:
                        d_rows[key][n_taxa] = np.nan
                        s_rows[key][n_taxa] = False

        pivot_d = pd.DataFrame(d_rows).T
        pivot_d.index = pd.MultiIndex.from_tuples(pivot_d.index, names=["comparison", "feature"])
        pivot_sig = pd.DataFrame(s_rows).T
        pivot_sig.index = pivot_d.index

        plot_sensitivity(pivot_d, pivot_sig, subset)

    # ── Clean text table ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DETAILED RESULTS: IBD (full) — showing stability across N taxa")
    print(f"{'=' * 70}")
    ibd_full = results[(results["comparison"] == "ibd") & (results["subset"] == "full")]
    print(
        ibd_full[["n_taxa", "feature", "mean_a", "mean_b", "cohens_d",
                  "permutation_p", "perm_p_fdr18", "sig_fdr18", "wilcoxon_p"]]
        .sort_values(["feature", "n_taxa"])
        .to_string(index=False)
    )

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
