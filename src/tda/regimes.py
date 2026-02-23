"""Sliding window regime detection using topological features."""

import numpy as np

from .homology import compute_persistence
from .features import betti_curve, persistence_entropy


def sliding_window_persistence(distance_matrices, window_size=10, step=1, maxdim=2):
    """Compute persistence features over sliding windows.

    Args:
        distance_matrices: List of distance matrices (one per time point or sample group).
        window_size: Number of matrices per window.
        step: Step size between windows.
        maxdim: Maximum homological dimension.

    Returns:
        Dictionary of feature time series (betti numbers, entropy, etc.).
    """
    n = len(distance_matrices)
    features = {
        "betti_0": [],
        "betti_1": [],
        "betti_2": [],
        "entropy_1": [],
    }

    for start in range(0, n - window_size + 1, step):
        window = distance_matrices[start : start + window_size]

        # Average the distance matrices in the window
        avg_dist = np.mean(window, axis=0)

        result = compute_persistence(avg_dist, maxdim=maxdim)
        dgms = result["dgms"]

        # Betti numbers at the median filtration value
        for dim in range(min(3, len(dgms))):
            _, betti = betti_curve(dgms[dim])
            features[f"betti_{dim}"].append(np.max(betti))

        # Persistence entropy for H1
        if len(dgms) > 1:
            features["entropy_1"].append(persistence_entropy(dgms[1]))
        else:
            features["entropy_1"].append(0.0)

    return {k: np.array(v) for k, v in features.items()}


def detect_regime_change(feature_series, threshold_std=2.0):
    """Detect regime changes based on abrupt shifts in topological features.

    Args:
        feature_series: 1D array of a topological feature over time/windows.
        threshold_std: Number of standard deviations for change detection.

    Returns:
        Array of indices where regime changes are detected.
    """
    if len(feature_series) < 3:
        return np.array([])

    diffs = np.abs(np.diff(feature_series))
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    if std_diff == 0:
        return np.array([])

    change_points = np.where(diffs > mean_diff + threshold_std * std_diff)[0]
    return change_points + 1  # Offset by 1 since diff reduces length
