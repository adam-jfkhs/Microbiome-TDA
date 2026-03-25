#!/usr/bin/env python3
"""Benchmark: standard network metrics vs TDA topological features.

Runs the same paired bootstrap comparison as the TDA analysis (v2) but
replaces the H1 persistence pipeline with standard graph-theoretic metrics
(density, clustering, transitivity, mean degree, modularity, components).

This answers: "Do standard graph metrics detect the IBD signal, or is
topology (persistent homology) necessary?"

Statistical workflow mirrors run_agp_bootstrap_v2.py exactly:
  1. Load & preprocess AGP data (same filtering, CLR, global taxa)
  2. 200 paired bootstrap iterations, n=100 per group
  3. Compute network_metrics() at thresholds [0.2, 0.3, 0.4]
  4. Sign-flip permutation test (500 shuffles) + Wilcoxon confirmatory
  5. BH-FDR correction per subset
  6. Compare effect sizes with TDA results side by side
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data.ibdmdb_loader import load_ibdmdb, ibdmdb_group_ids
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.metrics import network_metrics
from src.analysis.statistics import fdr_correction

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200
SUBSAMPLE_SIZE = 100
N_PERMUTATIONS = 500
N_GLOBAL_TAXA = 80
THRESHOLDS = [0.2, 0.3, 0.4]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Metric names (one per threshold)
BASE_METRICS = [
    "edge_density",
    "clustering_coeff",
    "transitivity",
    "mean_degree",
    "modularity",
    "n_components",
]


def metric_feature_names():
    """Return the full list of feature names: metric@threshold."""
    names = []
    for t in THRESHOLDS:
        for m in BASE_METRICS:
            names.append(f"{m}@{t}")
    return names


FEATURES = metric_feature_names()


def compute_network_features(clr_subset, taxa):
    """Compute network metrics at multiple thresholds for a subsample.

    Parameters
    ----------
    clr_subset : DataFrame (samples x taxa), CLR-transformed subsample.
    taxa : list of taxon column names (global fixed set).

    Returns
    -------
    dict mapping feature name (metric@threshold) to scalar value.
    """
    subset = clr_subset[taxa]
    corr_df, _ = spearman_correlation_matrix(subset)
    corr_matrix = corr_df.values

    result = {}
    for t in THRESHOLDS:
        metrics = network_metrics(corr_matrix, threshold=t)
        for m in BASE_METRICS:
            result[f"{m}@{t}"] = metrics[m]
    return result


def paired_network_test(
    clr_df, ids_a, ids_b, taxa,
    n_iter, subsample_size, n_perm, rng,
    label="",
):
    """Paired resampling network-metrics comparison with sign-flip null.

    Mirrors paired_resample_test() from src.analysis.bootstrap but uses
    network_metrics instead of TDA features.
    """
    ids_a = list(ids_a)
    ids_b = list(ids_b)
    n_a = min(subsample_size, len(ids_a))
    n_b = min(subsample_size, len(ids_b))

    print(f"  {label}: {len(ids_a)} vs {len(ids_b)} | "
          f"drawing {n_a} vs {n_b} per iter x {n_iter}")

    deltas = {f: [] for f in FEATURES}
    feat_a_all = {f: [] for f in FEATURES}
    feat_b_all = {f: [] for f in FEATURES}

    t0 = time.time()
    for i in range(n_iter):
        boot_a = rng.choice(ids_a, size=n_a, replace=False)
        boot_b = rng.choice(ids_b, size=n_b, replace=False)

        fa = compute_network_features(
            clr_df.loc[boot_a].reset_index(drop=True), taxa
        )
        fb = compute_network_features(
            clr_df.loc[boot_b].reset_index(drop=True), taxa
        )

        for feat in FEATURES:
            deltas[feat].append(fa[feat] - fb[feat])
            feat_a_all[feat].append(fa[feat])
            feat_b_all[feat].append(fb[feat])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    iteration {i + 1}/{n_iter}  ({elapsed:.0f}s elapsed)")

    # Statistical tests per feature
    rows = []
    for feat in FEATURES:
        d = np.array(deltas[feat])
        observed_stat = np.mean(d)

        # Sign-flip permutation test
        signs = rng.choice([-1, 1], size=(n_perm, len(d)))
        perm_means = np.abs((signs * d).mean(axis=1))
        count_extreme = int(np.sum(perm_means >= abs(observed_stat)))
        perm_p = (count_extreme + 1) / (n_perm + 1)

        # Wilcoxon signed-rank (confirmatory)
        try:
            _, wilcox_p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            wilcox_p = 1.0

        # Cohen's d on per-iteration distributions
        vals_a = np.array(feat_a_all[feat])
        vals_b = np.array(feat_b_all[feat])
        pooled_std = np.sqrt(
            ((len(vals_a) - 1) * vals_a.var(ddof=1)
             + (len(vals_b) - 1) * vals_b.var(ddof=1))
            / (len(vals_a) + len(vals_b) - 2)
        )
        cohens_d = (
            float((vals_a.mean() - vals_b.mean()) / pooled_std)
            if pooled_std > 0 else 0.0
        )

        rows.append({
            "feature": feat,
            "mean_a": round(float(vals_a.mean()), 6),
            "mean_b": round(float(vals_b.mean()), 6),
            "mean_delta": round(float(observed_stat), 6),
            "cohens_d": round(cohens_d, 4),
            "wilcoxon_p": round(wilcox_p, 6),
            "permutation_p": round(perm_p, 6),
            "n_iter": n_iter,
            "n_a": len(ids_a),
            "n_b": len(ids_b),
            "subsample_a": n_a,
            "subsample_b": n_b,
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NETWORK METRICS BENCHMARK — Standard graph metrics vs TDA")
    print("=" * 70)

    # Load IBDMDB (data available in this environment)
    print("\nLoading IBDMDB data...")
    abundance_df, metadata_df = load_ibdmdb()
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    meta = metadata_df.loc[clr_df.index]
    print(f"Preprocessed: {clr_df.shape[0]} samples x {clr_df.shape[1]} taxa")

    # Global taxa selection (same as TDA pipeline)
    print("\n--- Global taxa selection ---")
    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # Define comparisons using IBDMDB groups
    comparisons = {}
    for comp_name in ["ibd_vs_nonibd", "cd_vs_nonibd", "high_vs_low_calprotectin"]:
        ids_a, ids_b, label_a, label_b = ibdmdb_group_ids(meta, comp_name)
        ids_a = [i for i in ids_a if i in clr_df.index]
        ids_b = [i for i in ids_b if i in clr_df.index]
        comparisons[comp_name] = {
            "label_a": label_a, "label_b": label_b,
            "ids_a": ids_a, "ids_b": ids_b,
        }

    print("\nGroup sizes:")
    for name, comp in comparisons.items():
        print(f"  {name}: {comp['label_a']} n={len(comp['ids_a'])}  "
              f"{comp['label_b']} n={len(comp['ids_b'])}")

    # Run each comparison
    subsample = 60  # match IBDMDB bootstrap
    all_results = []
    for comp_idx, (comp_name, comp) in enumerate(comparisons.items()):
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {comp_name}")
        print(f"{'=' * 70}")

        comp_rng = np.random.default_rng(SEED + comp_idx)

        results_full = paired_network_test(
            clr_df, comp["ids_a"], comp["ids_b"], global_taxa,
            n_iter=N_ITERATIONS, subsample_size=subsample,
            n_perm=N_PERMUTATIONS, rng=comp_rng,
            label=f"{comp['label_a']} vs {comp['label_b']}",
        )
        results_full["comparison"] = comp_name
        results_full["subset"] = "full"
        all_results.append(results_full)

        # Print summary
        print(f"\n  Results - {comp_name}:")
        print(f"  {'Feature':<30} {'Mean A':>10} {'Mean B':>10} "
              f"{'Cohen d':>8} {'perm-p':>8}")
        print(f"  {'-' * 68}")
        for _, row in results_full.iterrows():
            sig = "*" if row["permutation_p"] < 0.05 else ""
            print(f"  {row['feature']:<30} {row['mean_a']:>10.4f} "
                  f"{row['mean_b']:>10.4f} {row['cohens_d']:>8.3f} "
                  f"{row['permutation_p']:>8.4f} {sig}")

    # Collate and apply FDR
    all_df = pd.concat(all_results, ignore_index=True)
    rejected_perm, adj_perm = fdr_correction(all_df["permutation_p"].values)
    all_df["perm_p_fdr"] = adj_perm
    all_df["sig_perm_fdr"] = rejected_perm

    out_path = os.path.join(RESULTS_DIR, "network_metrics_benchmark.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Comparison with IBDMDB TDA results ────────────────────────────────────
    tda_path = os.path.join(RESULTS_DIR, "ibdmdb_bootstrap.csv")
    if os.path.exists(tda_path):
        tda_df = pd.read_csv(tda_path)
        tda_ibd = tda_df[tda_df["comparison"] == "ibd_vs_nonibd"]
        net_ibd = all_df[all_df["comparison"] == "ibd_vs_nonibd"]

        print(f"\n{'=' * 70}")
        print("COMPARISON: TDA vs Network Metrics (IBDMDB IBD vs nonIBD)")
        print(f"{'=' * 70}")
        print(f"  {'Method':<10} {'Feature':<30} {'|d|':>6} {'perm-p':>8}")
        print(f"  {'-' * 58}")
        for _, row in tda_ibd.iterrows():
            print(f"  {'TDA':<10} {row['feature']:<30} "
                  f"{abs(row['cohens_d']):>6.3f} {row['permutation_p']:>8.4f}")
        for _, row in net_ibd.iterrows():
            print(f"  {'Network':<10} {row['feature']:<30} "
                  f"{abs(row['cohens_d']):>6.3f} {row['permutation_p']:>8.4f}")

        # Summary
        tda_max = tda_ibd["cohens_d"].abs().max()
        net_max = net_ibd["cohens_d"].abs().max()
        print(f"\n  TDA max |d| = {tda_max:.3f}, Network max |d| = {net_max:.3f}")
        if tda_max > net_max:
            print("  >> TDA captures LARGER effect sizes than standard network metrics")
        else:
            print("  >> Standard network metrics show comparable or larger effect sizes")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
