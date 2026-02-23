"""Distance matrix computation for microbiome data."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def correlation_distance(corr_df):
    """Convert a correlation matrix to a distance matrix.

    Distance = 1 - |correlation|, so strongly correlated OTUs
    (positive or negative) are close together.

    Args:
        corr_df: Correlation matrix as DataFrame.

    Returns:
        Distance matrix as DataFrame.
    """
    dist = 1 - np.abs(corr_df.values)
    np.fill_diagonal(dist, 0)
    return pd.DataFrame(dist, index=corr_df.index, columns=corr_df.columns)


def sample_distance(df, metric="braycurtis"):
    """Compute pairwise distances between samples.

    Args:
        df: DataFrame with samples as rows and OTUs as columns.
        metric: Distance metric (braycurtis, jaccard, euclidean, etc.).

    Returns:
        Distance matrix as DataFrame.
    """
    dist_condensed = pdist(df.values, metric=metric)
    dist_matrix = squareform(dist_condensed)
    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
