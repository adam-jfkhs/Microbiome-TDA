"""Vietoris-Rips filtration setup for TDA on microbiome data."""

import numpy as np


def prepare_distance_matrix(dist_df):
    """Prepare a distance matrix for Vietoris-Rips filtration.

    Ensures the matrix is symmetric, has zero diagonal, and
    contains no NaN values.

    Args:
        dist_df: Distance matrix as DataFrame.

    Returns:
        Cleaned numpy distance matrix.
    """
    dist = dist_df.values.copy()

    # Ensure symmetry
    dist = (dist + dist.T) / 2

    # Zero diagonal
    np.fill_diagonal(dist, 0)

    # Replace NaN with max distance
    max_dist = np.nanmax(dist)
    dist = np.nan_to_num(dist, nan=max_dist)

    return dist


def select_filtration_range(dist_matrix, percentiles=(5, 95)):
    """Suggest a filtration range based on distance distribution.

    Args:
        dist_matrix: Square distance matrix (numpy array).
        percentiles: Tuple of (low, high) percentiles.

    Returns:
        Tuple of (min_thresh, max_thresh) for the filtration parameter.
    """
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(dist_matrix, k=1)
    distances = dist_matrix[triu_idx]

    low = np.percentile(distances, percentiles[0])
    high = np.percentile(distances, percentiles[1])

    return low, high
