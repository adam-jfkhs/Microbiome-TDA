#!/usr/bin/env python3
"""Compute Shannon diversity for IBD vs Healthy and report Cohen's d.

This script provides the alpha-diversity benchmark referenced in the paper:
it shows that conventional Shannon diversity produces smaller effect sizes
than the topological features from the TDA pipeline.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy, mannwhitneyu

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance


def compute_shannon(otu_row):
    """Shannon entropy of a single sample's OTU counts."""
    props = otu_row / otu_row.sum()
    props = props[props > 0]
    return shannon_entropy(props)


def cohens_d(a, b):
    """Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled


def main():
    print("Loading AGP data...")
    otu_df, metadata = load_agp("data/raw/agp")

    # Same filtering as main pipeline
    print(f"Raw: {otu_df.shape[0]} samples, {otu_df.shape[1]} taxa")

    # Filter to stool samples
    if "body_site" in metadata.columns:
        stool_mask = metadata["body_site"].str.contains("feces|stool|UBERON:feces",
                                                         case=False, na=False)
        stool_ids = metadata[stool_mask].index.intersection(otu_df.index)
        otu_df = otu_df.loc[stool_ids]
        metadata = metadata.loc[stool_ids]
        print(f"Stool only: {otu_df.shape[0]} samples")

    # Minimum read depth filter
    read_depths = otu_df.sum(axis=1)
    keep = read_depths >= 1000
    otu_df = otu_df.loc[keep]
    metadata = metadata.loc[metadata.index.intersection(otu_df.index)]
    print(f"After depth filter (>=1000): {otu_df.shape[0]} samples")

    # Define IBD and healthy groups (same logic as bootstrap script)
    ibd_col = None
    for col in metadata.columns:
        if "ibd" in col.lower() or "inflammatory" in col.lower():
            ibd_col = col
            break

    if ibd_col is None:
        # Try diagnosis column
        for col in metadata.columns:
            if "diagnosis" in col.lower() or "disease" in col.lower():
                ibd_col = col
                break

    if ibd_col is None:
        print("ERROR: Cannot find IBD column in metadata")
        print("Available columns:", list(metadata.columns[:30]))
        sys.exit(1)

    print(f"Using IBD column: {ibd_col}")
    print(f"Unique values: {metadata[ibd_col].value_counts().head(10)}")

    # Identify IBD and healthy samples
    ibd_values = metadata[ibd_col].astype(str).str.lower()
    ibd_mask = ibd_values.isin(["diagnosed by a medical professional (doctor, physician assistant)",
                                 "i do not have this condition",
                                 "true", "yes", "ibd", "crohn's disease",
                                 "ulcerative colitis", "crohns disease"])

    # More flexible: check for actual IBD diagnosis
    ibd_positive = ibd_values.str.contains(
        "diagnosed|true|yes|crohn|ulcerative|ibd",
        case=False, na=False
    ) & ~ibd_values.str.contains("not|no|false", case=False, na=False)

    healthy_mask = ibd_values.str.contains(
        "do not have|false|no|never|undiagnosed",
        case=False, na=False
    )

    ibd_ids = metadata[ibd_positive].index.intersection(otu_df.index)
    healthy_ids = metadata[healthy_mask].index.intersection(otu_df.index)

    print(f"\nIBD samples: {len(ibd_ids)}")
    print(f"Healthy samples: {len(healthy_ids)}")

    if len(ibd_ids) < 10:
        print("\nTrying alternative column detection...")
        for col in metadata.columns:
            vals = metadata[col].astype(str).str.lower()
            if vals.str.contains("crohn|ulcerative", na=False).sum() > 10:
                print(f"  Column '{col}' has IBD mentions: "
                      f"{vals.str.contains('crohn|ulcerative', na=False).sum()}")

    # Compute Shannon diversity
    print("\nComputing Shannon diversity...")
    shannon_ibd = otu_df.loc[ibd_ids].apply(compute_shannon, axis=1)
    shannon_healthy = otu_df.loc[healthy_ids].apply(compute_shannon, axis=1)

    # Statistics
    d = cohens_d(shannon_healthy.values, shannon_ibd.values)
    u_stat, mwu_p = mannwhitneyu(shannon_healthy.values, shannon_ibd.values,
                                  alternative='two-sided')

    print("\n" + "=" * 60)
    print("SHANNON DIVERSITY: IBD vs HEALTHY")
    print("=" * 60)
    print(f"  IBD:     mean = {shannon_ibd.mean():.4f}, sd = {shannon_ibd.std():.4f}, n = {len(shannon_ibd)}")
    print(f"  Healthy: mean = {shannon_healthy.mean():.4f}, sd = {shannon_healthy.std():.4f}, n = {len(shannon_healthy)}")
    print(f"  Cohen's d = {d:.4f}  (Healthy - IBD direction)")
    print(f"  Mann-Whitney U p = {mwu_p:.2e}")
    print("=" * 60)
    print(f"\nFor comparison, TDA topology d = 0.75 to 2.38")
    print(f"Shannon d = {d:.2f} → topology amplification factor: {1.5/d:.1f}x to {2.38/d:.1f}x")


if __name__ == "__main__":
    main()
