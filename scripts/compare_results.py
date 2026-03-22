#!/usr/bin/env python3
"""Compare old (pre-fix) and new (post-fix) results side by side.

Usage:
    python scripts/compare_results.py

Expects:
    results/old_v1/*.csv   — archived pre-fix results
    results/*.csv          — newly generated post-fix results

Prints a comparison table for each analysis, highlighting changed values.
"""

import os
import sys

import numpy as np
import pandas as pd

OLD_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "old_v1")
NEW_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Column mappings: old name → new name (where they differ)
AGP_COL_MAP = {
    "perm_p_fdr": "perm_p_fdr",
    "sig_perm_fdr": "sig_perm_fdr",
}

TAXA_COL_MAP = {
    "perm_p_fdr18": "perm_p_fdr18",
    "sig_fdr18": "sig_fdr18",
}

IBDMDB_COL_MAP = {
    "perm_p_fdr": "perm_p_fdr",
    "sig_perm_fdr": "sig_perm_fdr",
}


def load_pair(filename):
    """Load old and new versions of a results file."""
    old_path = os.path.join(OLD_DIR, filename)
    new_path = os.path.join(NEW_DIR, filename)

    if not os.path.exists(old_path):
        print(f"  OLD missing: {old_path}")
        return None, None
    if not os.path.exists(new_path):
        print(f"  NEW missing: {new_path} (re-run needed)")
        return None, None

    return pd.read_csv(old_path), pd.read_csv(new_path)


def compare_agp(old, new):
    """Compare AGP bootstrap results."""
    # Key columns to compare
    value_cols = ["mean_a", "mean_b", "cohens_d", "permutation_p"]
    fdr_col_old = "perm_p_fdr"
    sig_col_old = "sig_perm_fdr"
    fdr_col_new = "perm_p_fdr"
    sig_col_new = "sig_perm_fdr"

    # Merge on feature + comparison + subset
    merge_keys = ["feature", "comparison", "subset"]
    merged = old.merge(new, on=merge_keys, suffixes=("_old", "_new"), how="outer")

    print(f"\n  {'Feature':<25} {'Comp':<15} {'Subset':<8} "
          f"{'d_old':>7} {'d_new':>7} {'Δd':>7}  "
          f"{'p_old':>8} {'p_new':>8}  "
          f"{'sig_old':>7} {'sig_new':>7}")
    print("  " + "-" * 120)

    changes = 0
    for _, row in merged.sort_values(merge_keys).iterrows():
        d_old = row.get("cohens_d_old", np.nan)
        d_new = row.get("cohens_d_new", np.nan)
        delta_d = d_new - d_old if pd.notna(d_old) and pd.notna(d_new) else np.nan

        p_old = row.get(f"{fdr_col_old}_old", np.nan)
        p_new = row.get(f"{fdr_col_new}_new", np.nan)

        sig_old = row.get(f"{sig_col_old}_old", "")
        sig_new = row.get(f"{sig_col_new}_new", "")

        changed = ""
        if pd.notna(delta_d) and abs(delta_d) > 0.01:
            changed = " ←"
            changes += 1

        print(f"  {row['feature']:<25} {row['comparison']:<15} {row['subset']:<8} "
              f"{d_old:>7.3f} {d_new:>7.3f} {delta_d:>+7.3f}  "
              f"{p_old:>8.4f} {p_new:>8.4f}  "
              f"{str(sig_old):>7} {str(sig_new):>7}{changed}")

    print(f"\n  → {changes} rows with |Δd| > 0.01")
    return changes


def compare_taxa(old, new):
    """Compare taxa sensitivity results."""
    merge_keys = ["feature", "comparison", "subset", "n_taxa"]

    # Normalize old column names if needed
    if "perm_p" in old.columns and "permutation_p" not in old.columns:
        old = old.rename(columns={"perm_p": "permutation_p"})

    merged = old.merge(new, on=merge_keys, suffixes=("_old", "_new"), how="outer")

    print(f"\n  {'Feature':<25} {'Comp':<15} {'Sub':<8} {'N':>4} "
          f"{'d_old':>7} {'d_new':>7} {'Δd':>7}  "
          f"{'sig_old':>7} {'sig_new':>7}")
    print("  " + "-" * 110)

    changes = 0
    for _, row in merged.sort_values(merge_keys).iterrows():
        d_old = row.get("cohens_d_old", np.nan)
        d_new = row.get("cohens_d_new", np.nan)
        delta_d = d_new - d_old if pd.notna(d_old) and pd.notna(d_new) else np.nan

        sig_old = row.get("sig_fdr18_old", "")
        sig_new = row.get("sig_fdr18_new", "")

        changed = ""
        if pd.notna(delta_d) and abs(delta_d) > 0.01:
            changed = " ←"
            changes += 1

        n_taxa = row.get("n_taxa", "")
        print(f"  {row['feature']:<25} {row['comparison']:<15} {row['subset']:<8} {n_taxa:>4.0f} "
              f"{d_old:>7.3f} {d_new:>7.3f} {delta_d:>+7.3f}  "
              f"{str(sig_old):>7} {str(sig_new):>7}{changed}")

    print(f"\n  → {changes} rows with |Δd| > 0.01")
    return changes


def compare_ibdmdb(old, new):
    """Compare IBDMDB bootstrap results."""
    merge_keys = ["feature", "comparison"]
    if "subset" in old.columns:
        merge_keys.append("subset")

    merged = old.merge(new, on=merge_keys, suffixes=("_old", "_new"), how="outer")

    print(f"\n  {'Feature':<25} {'Comp':<20} "
          f"{'d_old':>7} {'d_new':>7} {'Δd':>7}  "
          f"{'sig_old':>7} {'sig_new':>7}")
    print("  " + "-" * 100)

    changes = 0
    for _, row in merged.sort_values(merge_keys).iterrows():
        d_old = row.get("cohens_d_old", np.nan)
        d_new = row.get("cohens_d_new", np.nan)
        delta_d = d_new - d_old if pd.notna(d_old) and pd.notna(d_new) else np.nan

        sig_old = row.get("sig_perm_fdr_old", "")
        sig_new = row.get("sig_perm_fdr_new", "")

        changed = ""
        if pd.notna(delta_d) and abs(delta_d) > 0.01:
            changed = " ←"
            changes += 1

        print(f"  {row['feature']:<25} {row['comparison']:<20} "
              f"{d_old:>7.3f} {d_new:>7.3f} {delta_d:>+7.3f}  "
              f"{str(sig_old):>7} {str(sig_new):>7}{changed}")

    print(f"\n  → {changes} rows with |Δd| > 0.01")
    return changes


def main():
    print("=" * 70)
    print("RESULTS COMPARISON: old (pre-fix) vs new (post-fix)")
    print("=" * 70)

    total_changes = 0

    # AGP bootstrap
    print("\n" + "=" * 70)
    print("1. AGP BOOTSTRAP (agp_bootstrap_v2.csv)")
    print("=" * 70)
    old, new = load_pair("agp_bootstrap_v2.csv")
    if old is not None and new is not None:
        total_changes += compare_agp(old, new)

    # Taxa sensitivity
    print("\n" + "=" * 70)
    print("2. TAXA SENSITIVITY (taxa_sensitivity.csv)")
    print("=" * 70)
    old, new = load_pair("taxa_sensitivity.csv")
    if old is not None and new is not None:
        total_changes += compare_taxa(old, new)

    # IBDMDB bootstrap
    print("\n" + "=" * 70)
    print("3. IBDMDB BOOTSTRAP (ibdmdb_bootstrap.csv)")
    print("=" * 70)
    old, new = load_pair("ibdmdb_bootstrap.csv")
    if old is not None and new is not None:
        total_changes += compare_ibdmdb(old, new)

    # IBDMDB 1-per-subject
    print("\n" + "=" * 70)
    print("4. IBDMDB 1-PER-SUBJECT (ibdmdb_bootstrap_1ps.csv)")
    print("=" * 70)
    old, new = load_pair("ibdmdb_bootstrap_1ps.csv")
    if old is not None and new is not None:
        total_changes += compare_ibdmdb(old, new)

    # Classification benchmark
    print("\n" + "=" * 70)
    print("5. CLASSIFICATION BENCHMARK (classification_benchmark_v2.csv)")
    print("=" * 70)
    old, new = load_pair("classification_benchmark_v2.csv")
    if old is not None and new is not None:
        print("  (No results-affecting code changes — should be identical)")
        diff = old.compare(new) if old.shape == new.shape else "SHAPE MISMATCH"
        if isinstance(diff, str) or len(diff) > 0:
            print(f"  WARNING: differences found!")
        else:
            print(f"  ✓ Identical")

    # Summary
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_changes} values changed by |Δd| > 0.01")
    print("=" * 70)

    if total_changes > 0:
        print("\nNOTE: Changed values are expected due to the RNG-per-comparison fix.")
        print("Check that significance directions are preserved and effect sizes")
        print("remain in a similar range. Then update the LaTeX paper accordingly.")


if __name__ == "__main__":
    main()
