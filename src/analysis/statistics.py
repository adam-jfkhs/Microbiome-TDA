"""Statistical tests for topological feature analysis."""

import numpy as np
import pandas as pd
from scipy import stats


def permutation_test(group_a, group_b, n_permutations=10000, statistic="mean_diff",
                     seed=None):
    """Two-sample permutation test.

    Args:
        group_a: Feature values for group A.
        group_b: Feature values for group B.
        n_permutations: Number of permutations.
        statistic: Test statistic ('mean_diff' or 'median_diff').
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (observed_statistic, p_value).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    combined = np.concatenate([a, b])
    n_a = len(a)

    if statistic == "mean_diff":
        stat_func = lambda x, y: np.mean(x) - np.mean(y)
    else:
        stat_func = lambda x, y: np.median(x) - np.median(y)

    observed = stat_func(a, b)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_stat = stat_func(perm_a, perm_b)
        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def cohens_d(group_a, group_b):
    """Compute Cohen's d effect size (pooled standard deviation).

    Args:
        group_a, group_b: Arrays of values.

    Returns:
        Cohen's d (float). Positive means group_a > group_b.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                         / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values.
        alpha: Significance level.

    Returns:
        Tuple of (rejected, adjusted_p_values).
            rejected: Boolean array of which hypotheses are rejected.
            adjusted_p_values: FDR-adjusted p-values.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # BH adjusted p-values
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_p[i] * n / (i + 1)
        )
    adjusted = np.clip(adjusted, 0, 1)
    rejected = adjusted <= alpha

    return rejected, adjusted


def diagram_distance_permutation_test(dgm_a, dgm_b, n_permutations=1000, seed=None):
    """Permutation test on persistence diagram distance (bottleneck or wasserstein).

    Compares two persistence diagrams by computing the wasserstein distance
    and testing against a null distribution of permuted birth-death pairs.

    Args:
        dgm_a: Persistence diagram for group A (N x 2 array).
        dgm_b: Persistence diagram for group B (M x 2 array).
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Tuple of (observed_distance, p_value).
    """
    from persim import wasserstein

    dgm_a = np.asarray(dgm_a)
    dgm_b = np.asarray(dgm_b)

    # Filter to finite features
    if len(dgm_a) > 0:
        dgm_a = dgm_a[np.isfinite(dgm_a[:, 1])]
    if len(dgm_b) > 0:
        dgm_b = dgm_b[np.isfinite(dgm_b[:, 1])]

    if len(dgm_a) == 0 and len(dgm_b) == 0:
        return 0.0, 1.0

    observed = wasserstein(dgm_a, dgm_b)

    # Pool all birth-death pairs and permute
    combined = np.vstack([dgm_a, dgm_b]) if len(dgm_a) > 0 and len(dgm_b) > 0 else (
        dgm_a if len(dgm_a) > 0 else dgm_b
    )
    n_a = len(dgm_a)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_permutations):
        idx = rng.permutation(len(combined))
        perm_a = combined[idx[:n_a]]
        perm_b = combined[idx[n_a:]]
        if len(perm_a) == 0 or len(perm_b) == 0:
            continue
        perm_dist = wasserstein(perm_a, perm_b)
        if perm_dist >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def compare_topological_features(features_dict, group_labels, test="mannwhitney"):
    """Compare topological features across groups with effect sizes and FDR.

    Args:
        features_dict: Dictionary mapping feature names to arrays of values.
        group_labels: Array of group labels per sample.
        test: Statistical test ('mannwhitney', 'kruskal', or 'permutation').

    Returns:
        DataFrame with test results, effect sizes, and FDR-adjusted p-values.
    """
    results = []
    unique_groups = np.unique(group_labels)

    for feat_name, values in features_dict.items():
        values = np.asarray(values, dtype=float)
        groups = [values[group_labels == g] for g in unique_groups]

        if test == "mannwhitney" and len(unique_groups) == 2:
            stat, pval = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        elif test == "kruskal":
            stat, pval = stats.kruskal(*groups)
        elif test == "permutation" and len(unique_groups) == 2:
            stat, pval = permutation_test(groups[0], groups[1])
        else:
            stat, pval = np.nan, np.nan

        row = {
            "feature": feat_name,
            "statistic": stat,
            "p_value": pval,
            "test": test,
        }

        # Add effect size for two-group comparisons
        if len(unique_groups) == 2:
            row["cohens_d"] = cohens_d(groups[0], groups[1])
            row["mean_group_0"] = float(np.mean(groups[0]))
            row["mean_group_1"] = float(np.mean(groups[1]))

        results.append(row)

    df = pd.DataFrame(results)

    # FDR correction
    if len(df) > 1:
        rejected, adj_p = fdr_correction(df["p_value"].values)
        df["p_adjusted"] = adj_p
        df["significant"] = rejected
    else:
        df["p_adjusted"] = df["p_value"]
        df["significant"] = df["p_value"] <= 0.05

    return df
