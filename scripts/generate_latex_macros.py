#!/usr/bin/env python3
"""Generate LaTeX macros from results CSVs for automatic paper updates.

Usage:
    python scripts/generate_latex_macros.py

Reads results/*.csv and writes paper/generated_macros.tex with
\\newcommand definitions for every number that appears in the paper.
Include this file in your main .tex with \\input{generated_macros}.

After re-running analyses, re-run this script and recompile LaTeX —
all numbers update automatically.
"""

import os
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")

FEATURE_ORDER = [
    "h1_count", "h1_entropy", "h1_total_persistence",
    "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
]

FEATURE_SHORT = {
    "h1_count": "hcount",
    "h1_entropy": "hentropy",
    "h1_total_persistence": "htotalpers",
    "h1_mean_lifetime": "hmeanlife",
    "h1_max_lifetime": "hmaxlife",
    "max_betti1": "maxbetti",
}


def fmt(val, decimals=None):
    """Format a number: auto-detect decimals if not specified."""
    if pd.isna(val):
        return "---"
    if decimals is not None:
        return f"{val:.{decimals}f}"
    # Auto: use 3 decimals for values < 1, 2 for values < 100, 1 for larger
    av = abs(val)
    if av < 0.01:
        return f"{val:.4f}"
    if av < 1:
        return f"{val:.3f}"
    if av < 100:
        return f"{val:.2f}"
    return f"{val:.1f}"


def sign_prefix(val):
    """Return +/- prefix string for Cohen's d in LaTeX."""
    if val > 0:
        return "+"
    elif val < 0:
        return "$-$"
    return ""


def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        return None
    return pd.read_csv(path)


def generate_agp_macros(macros):
    """Generate macros from agp_bootstrap_v2.csv."""
    df = load_csv("agp_bootstrap_v2.csv")
    if df is None:
        return

    for _, row in df.iterrows():
        feat = FEATURE_SHORT[row["feature"]]
        comp = row["comparison"]
        subset = row["subset"]
        prefix = f"agp{comp}{subset}"

        macros[f"{prefix}{feat}MeanA"] = fmt(row["mean_a"])
        macros[f"{prefix}{feat}MeanB"] = fmt(row["mean_b"])
        macros[f"{prefix}{feat}D"] = fmt(row["cohens_d"])
        macros[f"{prefix}{feat}PermP"] = fmt(row["permutation_p"])

        fdr_col = "perm_p_fdr" if "perm_p_fdr" in df.columns else "perm_p_fdr18"
        sig_col = "sig_perm_fdr" if "sig_perm_fdr" in df.columns else "sig_fdr18"
        macros[f"{prefix}{feat}FDR"] = fmt(row[fdr_col])
        macros[f"{prefix}{feat}Sig"] = "true" if row[sig_col] else "false"

    # Compute d-range for IBD full
    ibd_full = df[(df["comparison"] == "ibd") & (df["subset"] == "full")]
    if not ibd_full.empty:
        d_vals = ibd_full["cohens_d"].values
        macros["agpIbdFullDmin"] = fmt(min(abs(d_vals)), 2)
        macros["agpIbdFullDmax"] = fmt(max(abs(d_vals)), 2)

    # Group sizes (from n_a, n_b columns)
    for _, row in df.drop_duplicates(subset=["comparison", "subset"]).iterrows():
        prefix = f"agp{row['comparison']}{row['subset']}"
        macros[f"{prefix}Na"] = str(int(row["n_a"]))
        macros[f"{prefix}Nb"] = str(int(row["n_b"]))


def generate_taxa_macros(macros):
    """Generate macros from taxa_sensitivity.csv."""
    df = load_csv("taxa_sensitivity.csv")
    if df is None:
        return

    # Sensitivity summary: count FDR-sig per comparison × subset × n_taxa
    for subset in ["full", "matched"]:
        for comp in ["ibd", "diet", "antibiotics"]:
            for n_taxa in [50, 80, 120]:
                chunk = df[(df["subset"] == subset) &
                           (df["comparison"] == comp) &
                           (df["n_taxa"] == n_taxa)]
                sig_col = "sig_fdr18" if "sig_fdr18" in df.columns else "sig_perm_fdr"
                n_sig = int(chunk[sig_col].sum()) if not chunk.empty else 0
                macros[f"taxa{subset.title()}{comp.title()}N{n_taxa}Sig"] = str(n_sig)

    # IBD d-range across all taxa sizes (full)
    ibd_full = df[(df["comparison"] == "ibd") & (df["subset"] == "full")]
    if not ibd_full.empty:
        d_vals = ibd_full["cohens_d"].values
        macros["taxaIbdFullDmin"] = fmt(min(abs(d_vals)), 2)
        macros["taxaIbdFullDmax"] = fmt(max(abs(d_vals)), 2)

    # IBD d-range matched
    ibd_matched = df[(df["comparison"] == "ibd") & (df["subset"] == "matched")]
    if not ibd_matched.empty:
        d_vals = ibd_matched["cohens_d"].values
        macros["taxaIbdMatchedDmin"] = fmt(min(abs(d_vals)), 2)
        macros["taxaIbdMatchedDmax"] = fmt(max(abs(d_vals)), 2)


def generate_ibdmdb_macros(macros):
    """Generate macros from ibdmdb_bootstrap.csv."""
    df = load_csv("ibdmdb_bootstrap.csv")
    if df is None:
        return

    for _, row in df.iterrows():
        feat = FEATURE_SHORT[row["feature"]]
        comp = row["comparison"].replace("_", "").replace("vs", "V")
        prefix = f"ibdmdb{comp}"

        macros[f"{prefix}{feat}MeanA"] = fmt(row["mean_a"])
        macros[f"{prefix}{feat}MeanB"] = fmt(row["mean_b"])
        macros[f"{prefix}{feat}D"] = fmt(row["cohens_d"])

        fdr_col = "perm_p_fdr" if "perm_p_fdr" in df.columns else "perm_p_fdr18"
        sig_col = "sig_perm_fdr" if "sig_perm_fdr" in df.columns else "sig_fdr18"
        macros[f"{prefix}{feat}FDR"] = fmt(row[fdr_col])
        macros[f"{prefix}{feat}Sig"] = "true" if row[sig_col] else "false"

    # Overall d-range for ibd_vs_nonibd
    ibd = df[df["comparison"] == "ibd_vs_nonibd"]
    if not ibd.empty:
        d_vals = ibd["cohens_d"].values
        macros["ibdmdbIbdDmin"] = fmt(min(abs(d_vals)), 2)
        macros["ibdmdbIbdDmax"] = fmt(max(abs(d_vals)), 2)


def generate_ibdmdb_1ps_macros(macros):
    """Generate macros from ibdmdb_bootstrap_1ps.csv."""
    df = load_csv("ibdmdb_bootstrap_1ps.csv")
    if df is None:
        return

    for _, row in df.iterrows():
        feat = FEATURE_SHORT[row["feature"]]
        comp = row["comparison"].replace("_", "").replace("vs", "V")
        prefix = f"ibdmdb1ps{comp}"

        macros[f"{prefix}{feat}MeanA"] = fmt(row["mean_a"])
        macros[f"{prefix}{feat}MeanB"] = fmt(row["mean_b"])
        macros[f"{prefix}{feat}D"] = fmt(row["cohens_d"])
        macros[f"{prefix}{feat}PermP"] = fmt(row["permutation_p"])


def write_macros(macros, outpath):
    """Write all macros to a .tex file."""
    lines = [
        "% Auto-generated by scripts/generate_latex_macros.py",
        "% DO NOT EDIT MANUALLY — re-run the script after updating results.",
        "%",
        "",
    ]
    for name, val in sorted(macros.items()):
        # Sanitize macro name: only letters allowed
        safe_name = "".join(c for c in name if c.isalpha() or c.isdigit())
        lines.append(f"\\newcommand{{\\res{safe_name}}}{{{val}}}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(macros)} macros to {outpath}")


def main():
    print("Generating LaTeX macros from results CSVs...")
    macros = {}

    generate_agp_macros(macros)
    generate_taxa_macros(macros)
    generate_ibdmdb_macros(macros)
    generate_ibdmdb_1ps_macros(macros)

    outpath = os.path.join(PAPER_DIR, "generated_macros.tex")
    os.makedirs(PAPER_DIR, exist_ok=True)
    write_macros(macros, outpath)

    print(f"\nTo use in your paper, add this near the top of your main .tex file:")
    print(f"  \\input{{generated_macros}}")
    print(f"\nThen replace hardcoded numbers with macro calls, e.g.:")
    print(f"  \\resagpibdfullhcountD  → Cohen's d for IBD full h1_count")
    print(f"  \\resagpibdfullhcountMeanA → Mean IBD for h1_count")


if __name__ == "__main__":
    main()
