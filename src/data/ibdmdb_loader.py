"""Loader for the IBDMDB / HMP2 dataset.

Data source:
  Metadata:   hmp2_metadata_2018-08-20.csv   (Globus IBDMDB endpoint)
  Profiles:   taxonomic_profiles.tsv.gz       (MetaPhlAn2 species abundances)

The taxonomic profiles TSV has rows = full lineage strings and columns =
sample IDs with the suffix '_taxonomic_profile'.  We extract species-level
rows (contain 's__', do not contain 't__'), strip the suffix, transpose to
get a samples × species abundance matrix, and return it alongside the
metagenomics-only rows of the metadata.

The abundance values are MetaPhlAn2 relative abundances (0–100 scale).
The existing clr_transform() adds a 0.5 pseudocount and works correctly on
these values without any pre-scaling.

Key metadata columns used downstream:
    diagnosis          - 'CD', 'UC', 'nonIBD'
    External ID        - matches columns of the abundance matrix (after
                         stripping '_taxonomic_profile' suffix)
    Tube B:Fecal Calprotectin  - continuous inflammation marker (μg/g)
    hbi                - Harvey-Bradshaw Index (Crohn's disease activity)
    sccai              - Simple Clinical Colitis Activity Index (UC activity)
    Participant ID     - subject identifier (for longitudinal grouping)
    week_num           - collection week
"""

import os

import numpy as np
import pandas as pd


def load_ibdmdb(data_dir="data/raw/ibdmdb"):
    """Load IBDMDB metagenomics abundances and paired clinical metadata.

    Args:
        data_dir: Directory containing hmp2_metadata.csv and
                  taxonomic_profiles.tsv (or .tsv.gz).

    Returns:
        Tuple of (abundance_df, metadata_df) where:
        - abundance_df: DataFrame (samples × species), relative abundances.
          Index = External ID strings matching metadata_df index.
        - metadata_df: DataFrame indexed by External ID, columns include
          'diagnosis', 'Tube B:Fecal Calprotectin', 'hbi', 'sccai',
          'Participant ID', 'week_num'.
    """
    meta_path = os.path.join(data_dir, "hmp2_metadata.csv")
    tsv_gz = os.path.join(data_dir, "taxonomic_profiles.tsv.gz")
    tsv_plain = os.path.join(data_dir, "taxonomic_profiles.tsv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"IBDMDB metadata not found at {meta_path}. "
            "Run: bash scripts/download_ibdmdb.sh"
        )

    profile_path = tsv_gz if os.path.exists(tsv_gz) else tsv_plain
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"IBDMDB taxonomic profiles not found. Expected: {tsv_gz} or {tsv_plain}. "
            "Run: bash scripts/download_ibdmdb.sh"
        )

    # ── Load metadata ─────────────────────────────────────────────────────────
    meta_raw = pd.read_csv(meta_path, low_memory=False)

    # Keep only metagenomics rows (other data types share metadata rows too)
    meta_mgx = meta_raw[meta_raw["data_type"] == "metagenomics"].copy()
    meta_mgx = meta_mgx.set_index("External ID")

    # ── Load taxonomic profiles ───────────────────────────────────────────────
    profiles = pd.read_csv(profile_path, sep="\t", index_col=0)

    # Strip '_taxonomic_profile' suffix from column names
    profiles.columns = [c.replace("_taxonomic_profile", "") for c in profiles.columns]

    # Keep only species-level rows (exactly one 's__' token, no 't__' strain level)
    is_species = profiles.index.str.contains("s__") & ~profiles.index.str.contains("t__")
    species_profiles = profiles.loc[is_species].copy()

    # Use only the terminal species name as column label (cleaner index)
    species_profiles.index = species_profiles.index.str.split("|").str[-1]

    # Transpose → samples × species
    abundance_df = species_profiles.T

    # ── Align abundance and metadata ──────────────────────────────────────────
    shared = abundance_df.index.intersection(meta_mgx.index)
    abundance_df = abundance_df.loc[shared]
    metadata_df = meta_mgx.loc[shared]

    # Coerce biomarker columns to numeric.
    # Note: 'fecalcal' holds the numeric calprotectin values (μg/g).
    # 'Tube B:Fecal Calprotectin' contains tube/sample IDs, not measurements.
    for col in ["fecalcal", "hbi", "sccai"]:
        if col in metadata_df.columns:
            metadata_df[col] = pd.to_numeric(metadata_df[col], errors="coerce")

    return abundance_df, metadata_df


def ibdmdb_group_ids(metadata_df, comparison):
    """Return (ids_a, ids_b, label_a, label_b) for a named comparison.

    Supported comparisons:
        'cd_vs_nonibd'    - Crohn's disease vs. healthy controls
        'uc_vs_nonibd'    - Ulcerative colitis vs. healthy controls
        'ibd_vs_nonibd'   - CD+UC vs. healthy controls
        'high_vs_low_calprotectin'  - ≥250 μg/g vs. <50 μg/g (active vs. quiescent)
        'high_vs_low_hbi'           - HBI ≥5 vs. HBI <5 (Crohn's only)
        'high_vs_low_sccai'         - SCCAI ≥3 vs. SCCAI <3 (UC only)
    """
    diag = metadata_df["diagnosis"]

    if comparison == "cd_vs_nonibd":
        ids_a = metadata_df.index[diag == "CD"].tolist()
        ids_b = metadata_df.index[diag == "nonIBD"].tolist()
        return ids_a, ids_b, "Crohn's disease", "nonIBD"

    if comparison == "uc_vs_nonibd":
        ids_a = metadata_df.index[diag == "UC"].tolist()
        ids_b = metadata_df.index[diag == "nonIBD"].tolist()
        return ids_a, ids_b, "Ulcerative colitis", "nonIBD"

    if comparison == "ibd_vs_nonibd":
        ids_a = metadata_df.index[diag.isin(["CD", "UC"])].tolist()
        ids_b = metadata_df.index[diag == "nonIBD"].tolist()
        return ids_a, ids_b, "IBD (CD+UC)", "nonIBD"

    if comparison == "high_vs_low_calprotectin":
        calp = metadata_df["fecalcal"]
        ids_a = metadata_df.index[calp >= 250].tolist()   # active mucosal inflammation
        ids_b = metadata_df.index[calp < 50].tolist()     # quiescent / healthy
        return ids_a, ids_b, "High calprotectin (≥250 μg/g)", "Low calprotectin (<50 μg/g)"

    if comparison == "high_vs_low_hbi":
        hbi = metadata_df["hbi"]
        cd_mask = diag == "CD"
        ids_a = metadata_df.index[cd_mask & (hbi >= 5)].tolist()   # active CD
        ids_b = metadata_df.index[cd_mask & (hbi < 5)].tolist()    # remission CD
        return ids_a, ids_b, "Active CD (HBI≥5)", "Remission CD (HBI<5)"

    if comparison == "high_vs_low_sccai":
        sccai = metadata_df["sccai"]
        uc_mask = diag == "UC"
        ids_a = metadata_df.index[uc_mask & (sccai >= 3)].tolist()   # active UC
        ids_b = metadata_df.index[uc_mask & (sccai < 3)].tolist()    # remission UC
        return ids_a, ids_b, "Active UC (SCCAI≥3)", "Remission UC (SCCAI<3)"

    raise ValueError(f"Unknown comparison: {comparison}")
