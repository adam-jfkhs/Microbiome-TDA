"""Per-sample topological feature extraction via k-NN neighbourhood TDA.

Shared implementation used by classification benchmarks and clinical
simulations.  Centralised here so bug fixes propagate automatically.

The workflow for each sample:
  1. Find k nearest neighbours in CLR space (Euclidean distance).
  2. Compute Spearman correlation matrix of the neighbourhood.
  3. Convert to distance matrix: d = 1 - |r|.
  4. Run Ripser (H1) and extract six scalar features.
"""

import time

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

from src.tda.homology import compute_persistence, filter_infinite

# Default neighbourhood size for per-sample TDA.
#
# Scripts use K=60 (larger AGP cohort, ~3400 samples) while simulations
# use K=40 (smaller IBDMDB cohort, ~1300 samples).  Callers should pass
# the appropriate value explicitly; this default is a reasonable middle
# ground for moderate-sized datasets.
DEFAULT_K_NEIGHBOURS = 40


def h1_features(dgm_h1):
    """Extract 6 H1 scalar features from a persistence diagram.

    Parameters
    ----------
    dgm_h1 : ndarray (N, 2)
        H1 persistence diagram (birth, death) pairs.

    Returns
    -------
    list of 6 values:
        [h1_count, h1_entropy, h1_total_persistence,
         h1_mean_lifetime, h1_max_lifetime, max_betti1]
    """
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else dgm_h1
    if len(finite) == 0:
        return [0, 0.0, 0.0, 0.0, 0.0, 0]
    lifetimes = finite[:, 1] - finite[:, 0]
    total_pers = float(lifetimes.sum())
    norm_lt = lifetimes / total_pers if total_pers > 0 else lifetimes
    entropy = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))
    births, deaths = finite[:, 0], finite[:, 1]
    thresholds = np.unique(np.concatenate([births, deaths]))
    max_betti = int(max(
        np.sum((births <= t) & (deaths > t)) for t in thresholds
    )) if len(thresholds) > 0 else 0
    return [len(finite), entropy, total_pers, float(lifetimes.mean()),
            float(lifetimes.max()), max_betti]


def compute_per_sample_topology(clr_matrix, k=DEFAULT_K_NEIGHBOURS,
                                verbose=True):
    """Compute per-sample H1 features via k-NN neighbourhood TDA.

    Parameters
    ----------
    clr_matrix : ndarray (n_samples, n_taxa)
        CLR-transformed abundance matrix.
    k : int
        Number of nearest neighbours per sample.
    verbose : bool
        Print progress every 200 samples.

    Returns
    -------
    ndarray (n_samples, 6) of float32
    """
    n = clr_matrix.shape[0]
    k_actual = min(k, n - 1)
    out = np.zeros((n, 6), dtype=np.float32)
    dists = cdist(clr_matrix, clr_matrix, metric="euclidean")

    t0 = time.time()
    for i in range(n):
        if verbose and i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"    {i}/{n}  {elapsed:.0f}s elapsed")

        nn_idx = np.argsort(dists[i])[1:k_actual + 1]
        neighbourhood = clr_matrix[nn_idx]
        if neighbourhood.shape[0] < 3:
            continue
        corr_mat, _ = spearmanr(neighbourhood, axis=0)
        if corr_mat.ndim == 0:
            corr_mat = np.array([[1.0]])
        dist_mat = np.clip(1.0 - np.abs(corr_mat), 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)
        result = compute_persistence(dist_mat, maxdim=1, thresh=1.0)
        dgms = filter_infinite(result["dgms"])
        dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        out[i] = h1_features(dgm_h1)
    return out
