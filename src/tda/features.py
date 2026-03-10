"""Feature extraction from persistence diagrams."""

import numpy as np


def betti_curve(diagram, num_points=100, max_val=None):
    """Compute the Betti curve from a persistence diagram.

    The Betti curve counts the number of features alive at each
    filtration value.

    Args:
        diagram: Persistence diagram (N x 2 array of birth/death pairs).
        num_points: Number of points to sample along the filtration.
        max_val: Maximum filtration value. If None, inferred from data.

    Returns:
        Tuple of (filtration_values, betti_numbers).
    """
    if len(diagram) == 0:
        filt_vals = np.linspace(0, 1 if max_val is None else max_val, num_points)
        return filt_vals, np.zeros(num_points)

    finite_mask = np.isfinite(diagram[:, 1])
    finite_dgm = diagram[finite_mask]

    if max_val is None:
        max_val = np.max(finite_dgm[:, 1]) if len(finite_dgm) > 0 else 1.0

    filt_vals = np.linspace(0, max_val, num_points)
    betti = np.zeros(num_points)

    for birth, death in finite_dgm:
        betti += (filt_vals >= birth) & (filt_vals < death)

    return filt_vals, betti


def persistence_entropy(diagram):
    """Compute persistence entropy of a diagram.

    Measures the diversity of feature lifetimes. Higher entropy means
    more uniform distribution of lifetimes.

    Args:
        diagram: Persistence diagram (N x 2 array).

    Returns:
        Entropy value (float).
    """
    finite_mask = np.isfinite(diagram[:, 1])
    finite_dgm = diagram[finite_mask]

    if len(finite_dgm) == 0:
        return 0.0

    lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    total = np.sum(lifetimes)
    probs = lifetimes / total
    entropy = -np.sum(probs * np.log(probs))

    return float(entropy)


def persistence_landscape(diagram, num_landscapes=5, num_points=100, max_val=None):
    """Compute persistence landscapes from a diagram.

    Args:
        diagram: Persistence diagram (N x 2 array).
        num_landscapes: Number of landscape functions to compute.
        num_points: Number of sample points.
        max_val: Maximum filtration value.

    Returns:
        Array of shape (num_landscapes, num_points).
    """
    finite_mask = np.isfinite(diagram[:, 1])
    finite_dgm = diagram[finite_mask]

    if len(finite_dgm) == 0 or max_val is None and len(finite_dgm) == 0:
        filt_vals = np.linspace(0, 1 if max_val is None else max_val, num_points)
        return np.zeros((num_landscapes, num_points))

    if max_val is None:
        max_val = np.max(finite_dgm[:, 1])

    filt_vals = np.linspace(0, max_val, num_points)
    landscapes = np.zeros((num_landscapes, num_points))

    for idx, t in enumerate(filt_vals):
        tent_values = []
        for birth, death in finite_dgm:
            mid = (birth + death) / 2
            half_life = (death - birth) / 2
            val = max(0, half_life - abs(t - mid))
            tent_values.append(val)

        tent_values.sort(reverse=True)
        for k in range(min(num_landscapes, len(tent_values))):
            landscapes[k, idx] = tent_values[k]

    return landscapes
