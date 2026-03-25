#!/usr/bin/env python3
"""Treatment-response TDA analysis on IBDMDB longitudinal data.

Investigates whether topological features of the gut microbiome track
treatment response using three complementary approaches:

  ANALYSES
  ────────
  1. Calprotectin quartile dose-response
     Split all IBDMDB samples into calprotectin quartiles (Q1–Q4) and run
     pairwise bootstrap TDA comparisons.  If topology tracks inflammation,
     features should degrade monotonically from Q1 (lowest) to Q4 (highest).

  2. Immunosuppressant on/off comparison
     Among IBD subjects who have samples both ON and OFF immunosuppressants,
     compare the two conditions.  This within-subject design controls for
     between-subject variation.

  3. Calprotectin trajectory classification
     Classify IBD subjects into trajectory groups based on their calprotectin
     over time (improving, worsening, stable-low, stable-high, intermediate)
     and compare topology between clinically distinct trajectories.

All analyses use the same paired bootstrap TDA framework as the primary
IBDMDB and AGP analyses (sign-flip permutation null, BH FDR).
"""

import os
import logging

import numpy as np
import pandas as pd

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import FEATURES, select_global_taxa, paired_resample_test
from src.analysis.statistics import fdr_correction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 200
SUBSAMPLE_SIZE = 60
N_PERMUTATIONS = 500
N_GLOBAL_TAXA = 80
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMMUNOSUPPRESSANT_COL = "Immunosuppressants (e.g. oral corticosteroids)"


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_results_table(results_df, label_a, label_b):
    """Print a formatted results table for a single comparison."""
    print(f"\n  {'Feature':<28} {'Mean A':>7} {'Mean B':>7} "
          f"{'Cohen d':>8} {'perm-p':>8} {'wilcox-p':>9}")
    print(f"  {'─' * 68}")
    for _, row in results_df.iterrows():
        sig = "*" if row["permutation_p"] < 0.05 else ""
        print(f"  {row['feature']:<28} {row['mean_a']:>7.3f} {row['mean_b']:>7.3f} "
              f"{row['cohens_d']:>8.3f} {row['permutation_p']:>8.4f} "
              f"{row['wilcoxon_p']:>9.4f} {sig}")


def run_comparison(clr_df, ids_a, ids_b, taxa, label_a, label_b, comp_name,
                   comp_idx, min_samples=20):
    """Run a single paired bootstrap comparison and return results_df or None."""
    ids_a = [i for i in ids_a if i in clr_df.index]
    ids_b = [i for i in ids_b if i in clr_df.index]
    print(f"  Group sizes: {label_a} n={len(ids_a)}, {label_b} n={len(ids_b)}")

    if len(ids_a) < min_samples or len(ids_b) < min_samples:
        print(f"  SKIP: too few samples (need {min_samples} per group)")
        return None

    subsample = min(SUBSAMPLE_SIZE, len(ids_a), len(ids_b))
    comp_rng = np.random.default_rng(SEED + comp_idx)

    results_df, raw = paired_resample_test(
        clr_df, ids_a, ids_b, taxa,
        n_iter=N_ITERATIONS, subsample_size=subsample,
        n_perm=N_PERMUTATIONS, rng=comp_rng,
        label=f"{label_a} vs {label_b}",
        min_samples=10,
    )

    if results_df is None:
        print("  SKIP: paired_resample_test returned None")
        return None

    results_df["comparison"] = comp_name
    results_df["label_a"] = label_a
    results_df["label_b"] = label_b

    print_results_table(results_df, label_a, label_b)
    return results_df


# ── Analysis 1: Calprotectin quartile dose-response ─────────────────────────

def calprotectin_quartile_analysis(clr_df, meta, taxa):
    """Split samples by calprotectin quartiles and compare pairwise."""
    print(f"\n{'=' * 70}")
    print("ANALYSIS 1: Calprotectin Quartile Dose-Response")
    print(f"{'=' * 70}")

    calp = pd.to_numeric(meta["fecalcal"], errors="coerce")
    valid = calp.dropna()
    valid = valid[valid.index.isin(clr_df.index)]
    print(f"  Samples with calprotectin: {len(valid)}")

    if len(valid) < 40:
        print("  SKIP: too few samples with calprotectin data")
        return []

    # Compute quartile boundaries
    q_labels = pd.qcut(valid, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    quartile_counts = q_labels.value_counts().sort_index()
    print(f"  Quartile sizes: {quartile_counts.to_dict()}")

    boundaries = pd.qcut(valid, q=4, retbins=True)[1]
    print(f"  Quartile boundaries (ug/g): "
          f"Q1<{boundaries[1]:.0f}, Q2<{boundaries[2]:.0f}, "
          f"Q3<{boundaries[3]:.0f}, Q4>={boundaries[3]:.0f}")

    results_list = []
    comp_idx_offset = 100  # offset to avoid seed collision

    # Pairwise comparisons: Q4 vs Q1, Q4 vs Q2, Q3 vs Q1
    pairs = [
        ("Q4_high", "Q1_low", "calp_Q4_vs_Q1"),
        ("Q4_high", "Q2",     "calp_Q4_vs_Q2"),
        ("Q3",      "Q1_low", "calp_Q3_vs_Q1"),
        ("Q3",      "Q2",     "calp_Q3_vs_Q2"),
    ]

    for q_a, q_b, comp_name in pairs:
        print(f"\n  --- {q_a} vs {q_b} ---")
        ids_a = q_labels[q_labels == q_a].index.tolist()
        ids_b = q_labels[q_labels == q_b].index.tolist()

        res = run_comparison(
            clr_df, ids_a, ids_b, taxa,
            label_a=q_a, label_b=q_b,
            comp_name=comp_name, comp_idx=comp_idx_offset,
            min_samples=15,
        )
        comp_idx_offset += 1
        if res is not None:
            results_list.append(res)

    return results_list


# ── Analysis 2: Immunosuppressant on/off ──────────────────────────────────────

def immunosuppressant_analysis(clr_df, meta, taxa):
    """Compare samples from subjects ON vs OFF immunosuppressants (within-subject)."""
    print(f"\n{'=' * 70}")
    print("ANALYSIS 2: Immunosuppressant ON vs OFF (Within-Subject)")
    print(f"{'=' * 70}")

    if IMMUNOSUPPRESSANT_COL not in meta.columns:
        print(f"  SKIP: column '{IMMUNOSUPPRESSANT_COL}' not found in metadata")
        print(f"  Available columns: {[c for c in meta.columns if 'mmuno' in c.lower() or 'steroid' in c.lower()]}")
        return []

    ibd_mask = meta["diagnosis"].isin(["CD", "UC"])
    immuno = meta[IMMUNOSUPPRESSANT_COL]

    # Find subjects with both ON and OFF samples
    ibd_meta = meta[ibd_mask & immuno.isin(["Yes", "No"])].copy()
    print(f"  IBD samples with immunosuppressant data: {len(ibd_meta)}")
    print(f"  ON={( ibd_meta[IMMUNOSUPPRESSANT_COL] == 'Yes').sum()}, "
          f"OFF={(ibd_meta[IMMUNOSUPPRESSANT_COL] == 'No').sum()}")

    if "Participant ID" not in ibd_meta.columns:
        print("  SKIP: 'Participant ID' column not found")
        return []

    # Identify subjects who switched (have both ON and OFF samples)
    subject_status = ibd_meta.groupby("Participant ID")[IMMUNOSUPPRESSANT_COL].apply(
        lambda x: set(x.values)
    )
    switchers = subject_status[subject_status.apply(lambda s: {"Yes", "No"}.issubset(s))].index
    print(f"  Subjects with both ON and OFF timepoints: {len(switchers)}")

    if len(switchers) < 5:
        print("  SKIP: too few subjects who switched immunosuppressant status")
        return []

    # Collect ON and OFF sample IDs from switching subjects only
    switcher_meta = ibd_meta[ibd_meta["Participant ID"].isin(switchers)]
    ids_on = switcher_meta.index[switcher_meta[IMMUNOSUPPRESSANT_COL] == "Yes"].tolist()
    ids_off = switcher_meta.index[switcher_meta[IMMUNOSUPPRESSANT_COL] == "No"].tolist()

    print(f"  Switcher samples: ON={len(ids_on)}, OFF={len(ids_off)}")

    results_list = []
    res = run_comparison(
        clr_df, ids_on, ids_off, taxa,
        label_a="Immunosuppressant ON",
        label_b="Immunosuppressant OFF",
        comp_name="immunosuppressant_on_vs_off",
        comp_idx=200,
        min_samples=15,
    )
    if res is not None:
        results_list.append(res)

    return results_list


# ── Analysis 3: Calprotectin trajectory classification ────────────────────────

def classify_trajectories(meta):
    """Classify IBD subjects into calprotectin trajectory groups.

    For each subject with >=3 calprotectin-matched timepoints, split timepoints
    into first-half and second-half (by week_num), then classify:
      - Improving:   first-half mean >=200, second-half <100
      - Worsening:   first-half <100, second-half >=200
      - Stable low:  both halves <100
      - Stable high: both halves >=200
      - Intermediate: everything else

    Returns
    -------
    DataFrame with columns: Participant ID, trajectory, sample_ids, half
    where each row is a subject, trajectory is the classification, sample_ids
    is a dict with 'first' and 'second' lists of sample IDs.
    """
    ibd_mask = meta["diagnosis"].isin(["CD", "UC"])
    calp = pd.to_numeric(meta["fecalcal"], errors="coerce")
    week = pd.to_numeric(meta["week_num"], errors="coerce")

    valid_mask = ibd_mask & calp.notna() & week.notna()
    sub = meta[valid_mask][["Participant ID", "week_num"]].copy()
    sub["fecalcal"] = calp[valid_mask]
    sub["week_num"] = week[valid_mask]

    trajectories = []

    for pid, grp in sub.groupby("Participant ID"):
        if len(grp) < 3:
            continue

        grp_sorted = grp.sort_values("week_num")
        mid = len(grp_sorted) // 2
        first_half = grp_sorted.iloc[:mid]
        second_half = grp_sorted.iloc[mid:]

        mean_first = first_half["fecalcal"].mean()
        mean_second = second_half["fecalcal"].mean()

        if mean_first >= 200 and mean_second < 100:
            traj = "improving"
        elif mean_first < 100 and mean_second >= 200:
            traj = "worsening"
        elif mean_first < 100 and mean_second < 100:
            traj = "stable_low"
        elif mean_first >= 200 and mean_second >= 200:
            traj = "stable_high"
        else:
            traj = "intermediate"

        trajectories.append({
            "Participant ID": pid,
            "trajectory": traj,
            "first_ids": first_half.index.tolist(),
            "second_ids": second_half.index.tolist(),
            "all_ids": grp_sorted.index.tolist(),
            "mean_calp_first": mean_first,
            "mean_calp_second": mean_second,
        })

    return pd.DataFrame(trajectories)


def trajectory_analysis(clr_df, meta, taxa):
    """Compare topology across calprotectin trajectory groups."""
    print(f"\n{'=' * 70}")
    print("ANALYSIS 3: Calprotectin Trajectory Classification")
    print(f"{'=' * 70}")

    if "Participant ID" not in meta.columns or "week_num" not in meta.columns:
        print("  SKIP: missing 'Participant ID' or 'week_num' columns")
        return []

    traj_df = classify_trajectories(meta)

    if len(traj_df) == 0:
        print("  SKIP: no subjects with >=3 calprotectin-matched timepoints")
        return []

    traj_counts = traj_df["trajectory"].value_counts()
    print(f"  Trajectory classification ({len(traj_df)} subjects):")
    for traj, count in traj_counts.items():
        print(f"    {traj}: {count} subjects")

    results_list = []
    comp_idx_offset = 300

    # Comparison 1: Stable high vs stable low
    print(f"\n  --- Stable High vs Stable Low ---")
    stable_high = traj_df[traj_df["trajectory"] == "stable_high"]
    stable_low = traj_df[traj_df["trajectory"] == "stable_low"]

    if len(stable_high) >= 2 and len(stable_low) >= 2:
        ids_high = [sid for row in stable_high["all_ids"] for sid in row]
        ids_low = [sid for row in stable_low["all_ids"] for sid in row]

        res = run_comparison(
            clr_df, ids_high, ids_low, taxa,
            label_a=f"Stable high ({len(stable_high)} subj)",
            label_b=f"Stable low ({len(stable_low)} subj)",
            comp_name="traj_stable_high_vs_low",
            comp_idx=comp_idx_offset,
            min_samples=10,
        )
        comp_idx_offset += 1
        if res is not None:
            results_list.append(res)
    else:
        print(f"  SKIP: stable_high={len(stable_high)}, stable_low={len(stable_low)} subjects")

    # Comparison 2: Improving subjects — first half (inflamed) vs second half (resolved)
    print(f"\n  --- Improving: First Half (Inflamed) vs Second Half (Resolved) ---")
    improving = traj_df[traj_df["trajectory"] == "improving"]

    if len(improving) >= 2:
        ids_first = [sid for row in improving["first_ids"] for sid in row]
        ids_second = [sid for row in improving["second_ids"] for sid in row]

        res = run_comparison(
            clr_df, ids_first, ids_second, taxa,
            label_a=f"Improving first-half ({len(improving)} subj)",
            label_b=f"Improving second-half ({len(improving)} subj)",
            comp_name="traj_improving_before_vs_after",
            comp_idx=comp_idx_offset,
            min_samples=5,
        )
        comp_idx_offset += 1
        if res is not None:
            results_list.append(res)
    else:
        print(f"  SKIP: only {len(improving)} improving subjects")

    # Comparison 3: Worsening subjects — first half vs second half
    print(f"\n  --- Worsening: First Half (Quiescent) vs Second Half (Inflamed) ---")
    worsening = traj_df[traj_df["trajectory"] == "worsening"]

    if len(worsening) >= 2:
        ids_first = [sid for row in worsening["first_ids"] for sid in row]
        ids_second = [sid for row in worsening["second_ids"] for sid in row]

        res = run_comparison(
            clr_df, ids_first, ids_second, taxa,
            label_a=f"Worsening first-half ({len(worsening)} subj)",
            label_b=f"Worsening second-half ({len(worsening)} subj)",
            comp_name="traj_worsening_before_vs_after",
            comp_idx=comp_idx_offset,
            min_samples=5,
        )
        comp_idx_offset += 1
        if res is not None:
            results_list.append(res)
    else:
        print(f"  SKIP: only {len(worsening)} worsening subjects")

    # Comparison 4: Improving second-half vs worsening second-half
    # (Both are "after" timepoints but divergent outcomes)
    if len(improving) >= 2 and len(worsening) >= 2:
        print(f"\n  --- Improving Late vs Worsening Late (Divergent Outcomes) ---")
        ids_imp_late = [sid for row in improving["second_ids"] for sid in row]
        ids_wor_late = [sid for row in worsening["second_ids"] for sid in row]

        res = run_comparison(
            clr_df, ids_imp_late, ids_wor_late, taxa,
            label_a=f"Improving late ({len(improving)} subj)",
            label_b=f"Worsening late ({len(worsening)} subj)",
            comp_name="traj_improving_vs_worsening_late",
            comp_idx=comp_idx_offset,
            min_samples=5,
        )
        comp_idx_offset += 1
        if res is not None:
            results_list.append(res)

    return results_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("IBDMDB TREATMENT RESPONSE — Longitudinal TDA Analysis")
    print("=" * 70)

    # Load & preprocess
    print("\nLoading IBDMDB data...")
    abundance_df, metadata_df = load_ibdmdb()
    print(f"Raw: {abundance_df.shape[0]} samples x {abundance_df.shape[1]} species")
    print(f"Diagnosis breakdown: {metadata_df['diagnosis'].value_counts().to_dict()}")

    # Filter and CLR-transform
    filtered = filter_low_abundance(abundance_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    meta = metadata_df.loc[clr_df.index]
    print(f"After filtering: {clr_df.shape[0]} samples x {clr_df.shape[1]} taxa")

    # Select global taxa once
    print("\nSelecting global taxa...")
    global_taxa = select_global_taxa(clr_df, n=N_GLOBAL_TAXA)

    # Report availability
    calp_n = meta["fecalcal"].notna().sum() if "fecalcal" in meta.columns else 0
    n_subjects = meta["Participant ID"].nunique() if "Participant ID" in meta.columns else 0
    immuno_n = meta[IMMUNOSUPPRESSANT_COL].notna().sum() if IMMUNOSUPPRESSANT_COL in meta.columns else 0
    print(f"\nData availability:")
    print(f"  Calprotectin values: {calp_n}")
    print(f"  Immunosuppressant status: {immuno_n}")
    print(f"  Unique participants: {n_subjects}")

    # ── Run all analyses ──────────────────────────────────────────────────────
    all_results = []

    # Analysis 1: Calprotectin quartiles
    quartile_results = calprotectin_quartile_analysis(clr_df, meta, global_taxa)
    all_results.extend(quartile_results)

    # Analysis 2: Immunosuppressant on/off
    immuno_results = immunosuppressant_analysis(clr_df, meta, global_taxa)
    all_results.extend(immuno_results)

    # Analysis 3: Trajectory classification
    traj_results = trajectory_analysis(clr_df, meta, global_taxa)
    all_results.extend(traj_results)

    # ── Collate and save ──────────────────────────────────────────────────────
    if not all_results:
        print("\nNo comparisons completed.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # Apply FDR correction across all treatment-response tests
    rejected_perm, adj_perm = fdr_correction(all_df["permutation_p"].values)
    all_df["perm_p_fdr"] = adj_perm
    all_df["sig_perm_fdr"] = rejected_perm

    rejected_wilcox, adj_wilcox = fdr_correction(all_df["wilcoxon_p"].values)
    all_df["wilcox_p_fdr"] = adj_wilcox
    all_df["sig_wilcox_fdr"] = rejected_wilcox

    out_path = os.path.join(RESULTS_DIR, "treatment_response.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    print(f"\nTotal comparisons run: {all_df['comparison'].nunique()}")
    print(f"Total feature tests: {len(all_df)}")

    print(f"\nNominally significant (perm p<0.05): "
          f"{(all_df['permutation_p'] < 0.05).sum()}/{len(all_df)}")
    print(f"FDR-significant (perm):  {all_df['sig_perm_fdr'].sum()}/{len(all_df)}")

    # Per-analysis summary
    for analysis_prefix, analysis_name in [
        ("calp_Q", "Calprotectin quartile"),
        ("immunosuppressant", "Immunosuppressant"),
        ("traj_", "Trajectory"),
    ]:
        subset = all_df[all_df["comparison"].str.startswith(analysis_prefix)]
        if len(subset) == 0:
            continue
        n_sig = (subset["permutation_p"] < 0.05).sum()
        n_fdr = subset["sig_perm_fdr"].sum()
        print(f"\n  {analysis_name} analysis: "
              f"{n_sig}/{len(subset)} nominal, {n_fdr}/{len(subset)} FDR")

        # Show FDR-significant results
        sig_rows = subset[subset["sig_perm_fdr"]]
        if len(sig_rows) > 0:
            print(f"    FDR-significant features:")
            for _, row in sig_rows.iterrows():
                print(f"      {row['comparison']}: {row['feature']} "
                      f"(d={row['cohens_d']:.2f}, p_fdr={row['perm_p_fdr']:.4f})")

    # Dose-response check: show mean features across quartiles
    quartile_comps = all_df[all_df["comparison"].str.startswith("calp_Q")]
    if len(quartile_comps) > 0:
        print(f"\n  Dose-response pattern (Q4=highest inflammation):")
        for feat in FEATURES:
            q4_q1 = quartile_comps[
                (quartile_comps["comparison"] == "calp_Q4_vs_Q1")
                & (quartile_comps["feature"] == feat)
            ]
            if len(q4_q1) > 0:
                row = q4_q1.iloc[0]
                direction = "higher" if row["mean_a"] > row["mean_b"] else "lower"
                print(f"    {feat}: Q4 {direction} than Q1 "
                      f"(d={row['cohens_d']:.2f}, p={row['permutation_p']:.4f})")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
