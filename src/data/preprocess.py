"""OTU table cleaning, filtering, and normalization."""

import numpy as np
import pandas as pd
from biom import Table


def filter_low_abundance(df, min_prevalence=0.05, min_reads=1000):
    """Filter OTU table by prevalence and read depth.

    Args:
        df: DataFrame with samples as rows and OTUs as columns.
        min_prevalence: Minimum fraction of samples an OTU must appear in.
        min_reads: Minimum total reads per sample.

    Returns:
        Filtered DataFrame.
    """
    # Filter samples with low total reads
    sample_sums = df.sum(axis=1)
    df = df.loc[sample_sums >= min_reads]

    # Filter OTUs present in fewer than min_prevalence of samples
    prevalence = (df > 0).mean(axis=0)
    df = df.loc[:, prevalence >= min_prevalence]

    return df


def clr_transform(df):
    """Centered log-ratio transformation for compositional data.

    Handles zeros by adding a pseudocount of 0.5.

    Args:
        df: DataFrame with samples as rows and OTUs as columns.

    Returns:
        CLR-transformed DataFrame.
    """
    # Add pseudocount to handle zeros
    data = df.values + 0.5

    # Geometric mean per sample
    log_data = np.log(data)
    geo_mean = log_data.mean(axis=1, keepdims=True)

    # CLR = log(x) - mean(log(x))
    clr_data = log_data - geo_mean

    return pd.DataFrame(clr_data, index=df.index, columns=df.columns)


def relative_abundance(df):
    """Convert counts to relative abundances (proportions).

    Args:
        df: DataFrame with samples as rows and OTUs as columns.

    Returns:
        DataFrame with relative abundances summing to 1 per sample.
    """
    return df.div(df.sum(axis=1), axis=0)


def filter_body_site(df, metadata, body_site="stool"):
    """Filter samples to a specific body site using metadata.

    Args:
        df: OTU DataFrame with sample IDs as index.
        metadata: DataFrame with sample metadata including body_site column.
        body_site: Body site to keep.

    Returns:
        Filtered DataFrame containing only samples from the specified body site.
    """
    stool_samples = metadata.loc[
        metadata["body_site"] == body_site
    ].index
    common = df.index.intersection(stool_samples)
    return df.loc[common]


def biom_to_dataframe(table):
    """Convert a BIOM Table to a pandas DataFrame (samples x OTUs).

    Args:
        table: biom.Table object.

    Returns:
        DataFrame with samples as rows and OTU IDs as columns.
    """
    return pd.DataFrame(
        table.to_dataframe(dense=True).T
    )
