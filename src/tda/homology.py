"""Persistent homology computation using Ripser or Giotto-TDA."""

import numpy as np

try:
    from ripser import ripser as _ripser
    _BACKEND = "ripser"
except ImportError:
    _ripser = None
    _BACKEND = "giotto"


def _compute_giotto(distance_matrix, maxdim=2, thresh=np.inf):
    from gtda.homology import VietorisRipsPersistence
    vr = VietorisRipsPersistence(
        metric="precomputed",
        homology_dimensions=list(range(maxdim + 1)),
        max_edge_length=float(thresh) if np.isfinite(thresh) else np.inf,
    )
    dm = np.asarray(distance_matrix, dtype=np.float64)[np.newaxis, :, :]
    diagrams_3d = vr.fit_transform(dm)[0]
    dgms = []
    for dim in range(maxdim + 1):
        mask = diagrams_3d[:, 2] == dim
        pairs = diagrams_3d[mask][:, :2]
        dgms.append(pairs)
    return {"dgms": dgms}


def compute_persistence(distance_matrix, maxdim=2, thresh=np.inf):
    """Compute persistent homology from a distance matrix.

    Args:
        distance_matrix: Square distance matrix (numpy array).
        maxdim: Maximum homological dimension to compute.
        thresh: Maximum filtration value.

    Returns:
        Dictionary with 'dgms' key containing persistence diagrams
        for each dimension.
    """
    if _BACKEND == "ripser":
        return _ripser(
            distance_matrix,
            maxdim=maxdim,
            thresh=thresh,
            distance_matrix=True,
        )
    return _compute_giotto(distance_matrix, maxdim=maxdim, thresh=thresh)


def filter_infinite(diagrams):
    """Remove infinite-death features from persistence diagrams.

    Args:
        diagrams: List of persistence diagrams (one per dimension).

    Returns:
        List of filtered diagrams with only finite features.
    """
    filtered = []
    for dgm in diagrams:
        dgm = np.asarray(dgm)
        if dgm.ndim != 2 or dgm.shape[0] == 0:
            filtered.append(np.empty((0, 2)))
            continue
        finite_mask = np.isfinite(dgm[:, 1])
        filtered.append(dgm[finite_mask])
    return filtered


def persistence_summary(diagrams):
    """Compute summary statistics for persistence diagrams.

    Args:
        diagrams: List of persistence diagrams.

    Returns:
        Dictionary with per-dimension statistics (count, mean lifetime,
        max lifetime).
    """
    summary = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            summary[f"H{dim}"] = {
                "count": 0,
                "mean_lifetime": 0.0,
                "max_lifetime": 0.0,
            }
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite_dgm = dgm[finite_mask]

        if len(finite_dgm) == 0:
            lifetimes = np.array([0.0])
        else:
            lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]

        summary[f"H{dim}"] = {
            "count": len(finite_dgm),
            "mean_lifetime": float(np.mean(lifetimes)),
            "max_lifetime": float(np.max(lifetimes)),
        }

    return summary
