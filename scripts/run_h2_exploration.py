#!/usr/bin/env python3
"""H₂ (void) exploration: do 2-dimensional cavities add signal beyond H₁?

Runs a single IBD vs Healthy bootstrap comparison at maxdim=2 and reports
effect sizes for all 12 features (6 × H₁ + 6 × H₂).  The goal is to
empirically test the paper's claim that "H₁ saturates the topological
signal for 80-node networks."

If H₂ features show |d| > 0.3 or p < 0.05, a full analysis is warranted.
If not, the null result confirms the saturation claim.
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import (
    FEATURES, H2_FEATURES, select_global_taxa, tda_features,
)

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200
SUBSAMPLE_SIZE = 100
N_PERMUTATIONS = 500
N_GLOBAL_TAXA = 80
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_FEATURES = FEATURES + H2_FEATURES


def main():
    print("=" * 70)
    print("H₂ EXPLORATION — Do voids add signal beyond loops?")
    print("=" * 70)

    # ── Load & preprocess ────────────────────────────────────────────────────
    print("\nLoading IBDMDB data...")
    otu_df, metadata = load_ibdmdb()
    filtered = filter_low_abundance(otu_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    meta = metadata.loc[clr_df.index.intersection(metadata.index)]
    clr_df = clr_df.loc[meta.index]
    print(f"Preprocessed: {clr_df.shape[0]} samples × {clr_df.shape[1]} taxa")

    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # ── IBD vs nonIBD ────────────────────────────────────────────────────────
    ids_ibd = meta.loc[
        meta["diagnosis"].isin(["CD", "UC"])
    ].index.intersection(clr_df.index)
    ids_healthy = meta.loc[
        meta["diagnosis"] == "nonIBD"
    ].index.intersection(clr_df.index)
    print(f"\nIBD: n={len(ids_ibd)}   Healthy: n={len(ids_healthy)}")

    n_a = min(SUBSAMPLE_SIZE, len(ids_ibd))
    n_b = min(SUBSAMPLE_SIZE, len(ids_healthy))
    rng = np.random.default_rng(SEED)

    # ── Bootstrap with maxdim=2 ──────────────────────────────────────────────
    deltas = {f: [] for f in ALL_FEATURES}
    feat_a_all = {f: [] for f in ALL_FEATURES}
    feat_b_all = {f: [] for f in ALL_FEATURES}

    print(f"\nRunning {N_ITERATIONS} bootstrap iterations with maxdim=2 ...")
    t0 = time.time()

    for i in range(N_ITERATIONS):
        boot_a = rng.choice(list(ids_ibd), size=n_a, replace=False)
        boot_b = rng.choice(list(ids_healthy), size=n_b, replace=False)

        fa = tda_features(clr_df.loc[boot_a].reset_index(drop=True),
                          global_taxa, maxdim=2)
        fb = tda_features(clr_df.loc[boot_b].reset_index(drop=True),
                          global_taxa, maxdim=2)

        for feat in ALL_FEATURES:
            deltas[feat].append(fa[feat] - fb[feat])
            feat_a_all[feat].append(fa[feat])
            feat_b_all[feat].append(fb[feat])

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_ITERATIONS - i - 1) / rate
            print(f"  iteration {i+1}/{N_ITERATIONS}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nCompleted in {total_time:.1f}s "
          f"({total_time / N_ITERATIONS:.2f}s per iteration)")

    # ── Statistics ────────────────────────────────────────────────────────────
    rows = []
    for feat in ALL_FEATURES:
        d = np.array(deltas[feat])
        observed_stat = np.mean(d)

        # Permutation test
        signs = rng.choice([-1, 1], size=(N_PERMUTATIONS, len(d)))
        perm_means = np.abs((signs * d).mean(axis=1))
        count_extreme = int(np.sum(perm_means >= abs(observed_stat)))
        perm_p = (count_extreme + 1) / (N_PERMUTATIONS + 1)

        # Wilcoxon
        try:
            _, wilcox_p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            wilcox_p = 1.0

        # Cohen's d
        vals_a = np.array(feat_a_all[feat])
        vals_b = np.array(feat_b_all[feat])
        pooled_std = np.sqrt(
            ((len(vals_a) - 1) * vals_a.var(ddof=1)
             + (len(vals_b) - 1) * vals_b.var(ddof=1))
            / (len(vals_a) + len(vals_b) - 2)
        )
        cohens_d = (float((vals_a.mean() - vals_b.mean()) / pooled_std)
                    if pooled_std > 0 else 0.0)

        dimension = "H1" if feat.startswith("h1") or feat == "max_betti1" else "H2"

        rows.append({
            "dimension": dimension,
            "feature": feat,
            "mean_ibd": round(vals_a.mean(), 4),
            "mean_healthy": round(vals_b.mean(), 4),
            "mean_delta": round(observed_stat, 4),
            "cohens_d": round(cohens_d, 4),
            "permutation_p": round(perm_p, 6),
            "wilcoxon_p": round(wilcox_p, 6),
        })

    results = pd.DataFrame(rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "h2_exploration.csv")
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Print comparison table ───────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("H₁ vs H₂ FEATURE COMPARISON — IBD vs Healthy")
    print(f"{'=' * 78}")
    print(f"{'Dim':<4} {'Feature':<28} {'IBD':>8} {'Healthy':>8} "
          f"{'d':>7} {'perm-p':>8} {'Signal?':>8}")
    print(f"{'─' * 78}")

    for _, row in results.iterrows():
        sig = "***" if row["permutation_p"] < 0.005 else \
              "**" if row["permutation_p"] < 0.01 else \
              "*" if row["permutation_p"] < 0.05 else ""
        print(f"{row['dimension']:<4} {row['feature']:<28} "
              f"{row['mean_ibd']:>8.3f} {row['mean_healthy']:>8.3f} "
              f"{row['cohens_d']:>7.3f} {row['permutation_p']:>8.4f} {sig:>8}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    h2_rows = results[results["dimension"] == "H2"]
    h1_rows = results[results["dimension"] == "H1"]
    h2_max_d = h2_rows["cohens_d"].abs().max()
    h1_max_d = h1_rows["cohens_d"].abs().max()
    h2_any_sig = (h2_rows["permutation_p"] < 0.05).any()

    print(f"\n{'=' * 78}")
    print("VERDICT")
    print(f"{'=' * 78}")
    print(f"  H₁ max |Cohen's d|: {h1_max_d:.3f}")
    print(f"  H₂ max |Cohen's d|: {h2_max_d:.3f}")
    print(f"  H₂ any p < 0.05:    {h2_any_sig}")
    print(f"  Runtime (maxdim=2):  {total_time:.1f}s for {N_ITERATIONS} iterations")

    if h2_any_sig and h2_max_d > 0.3:
        print("\n  → H₂ SHOWS SIGNAL. Full analysis warranted.")
    elif h2_any_sig:
        print("\n  → H₂ shows weak signal (significant p but small effect).")
        print("    Consider reporting as supplementary finding.")
    else:
        print("\n  → H₂ shows NO signal. H₁ saturation confirmed.")
        print("    Report as empirical validation in methods/supplementary.")


if __name__ == "__main__":
    main()
