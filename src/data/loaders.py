"""Data loading utilities for microbiome datasets."""

import os

import pandas as pd
from biom import load_table

from .preprocess import biom_to_dataframe


def load_hmp(data_dir="data/raw/hmp"):
    """Load HMP Phase 1 OTU table and metadata.

    Args:
        data_dir: Path to the HMP raw data directory.

    Returns:
        Tuple of (otu_df, metadata_df).
    """
    biom_path = os.path.join(data_dir, "hmp1_otu_table.biom")
    meta_path = os.path.join(data_dir, "hmp1_metadata.tsv")

    table = load_table(biom_path)
    otu_df = biom_to_dataframe(table)

    metadata = pd.read_csv(meta_path, sep="\t", index_col=0)

    return otu_df, metadata


def load_agp(data_dir="data/raw/agp"):
    """Load American Gut Project OTU table and metadata.

    Args:
        data_dir: Path to the AGP raw data directory.

    Returns:
        Tuple of (otu_df, metadata_df).
    """
    biom_path = os.path.join(data_dir, "agp_otu_table.biom")
    meta_path = os.path.join(data_dir, "agp_metadata.tsv")

    table = load_table(biom_path)
    otu_df = biom_to_dataframe(table)

    metadata = pd.read_csv(meta_path, sep="\t", index_col=0)

    return otu_df, metadata
