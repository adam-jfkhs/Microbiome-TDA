"""Statistical tests for topological feature analysis."""

import numpy as np
from scipy import stats


def permutation_test(group_a, group_b, n_permutations=10000, statistic="mean_diff"):
    """Two-sample permutation test.

    Args:
        group_a: Feature values for group A.
        group_b: Feature values for group B.
        n_permutations: Number of permutations.
        statistic: Test statistic ('mean_diff' or 'median_diff').

    Returns:
        Tuple of (observed_statistic, p_value).
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    combined = np.concatenate([a, b])
    n_a = len(a)

    if statistic == "mean_diff":
        stat_func = lambda x, y: np.mean(x) - np.mean(y)
    else:
        stat_func = lambda x, y: np.median(x) - np.median(y)

    observed = stat_func(a, b)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_stat = stat_func(perm_a, perm_b)
        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def compare_topological_features(features_dict, group_labels, test="mannwhitney"):
    """Compare topological features across groups.

    Args:
        features_dict: Dictionary mapping feature names to arrays of values.
        group_labels: Array of group labels per sample.
        test: Statistical test ('mannwhitney', 'kruskal', or 'permutation').

    Returns:
        DataFrame with test results for each feature.
    """
    results = []
    unique_groups = np.unique(group_labels)

    for feat_name, values in features_dict.items():
        groups = [values[group_labels == g] for g in unique_groups]

        if test == "mannwhitney" and len(unique_groups) == 2:
            stat, pval = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        elif test == "kruskal":
            stat, pval = stats.kruskal(*groups)
        elif test == "permutation" and len(unique_groups) == 2:
            stat, pval = permutation_test(groups[0], groups[1])
        else:
            stat, pval = np.nan, np.nan

        results.append({
            "feature": feat_name,
            "statistic": stat,
            "p_value": pval,
            "test": test,
        })

    import pandas as pd
    return pd.DataFrame(results)
