"""Biomarker correlation analysis with topological features."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def correlate_features_with_metadata(topo_features, metadata, method="spearman"):
    """Correlate topological features with metadata variables.

    Args:
        topo_features: DataFrame with topological features (samples x features).
        metadata: DataFrame with metadata variables (samples x variables).
        method: Correlation method ('spearman' or 'pearson').

    Returns:
        Tuple of (correlation_df, pvalue_df).
    """
    common_idx = topo_features.index.intersection(metadata.index)
    topo = topo_features.loc[common_idx]
    meta = metadata.loc[common_idx]

    # Select only numeric metadata columns
    numeric_meta = meta.select_dtypes(include=[np.number])

    corr_func = spearmanr if method == "spearman" else pearsonr

    corr_results = pd.DataFrame(
        index=topo.columns, columns=numeric_meta.columns, dtype=float
    )
    pval_results = pd.DataFrame(
        index=topo.columns, columns=numeric_meta.columns, dtype=float
    )

    for feat in topo.columns:
        for var in numeric_meta.columns:
            mask = ~(topo[feat].isna() | numeric_meta[var].isna())
            if mask.sum() < 3:
                corr_results.loc[feat, var] = np.nan
                pval_results.loc[feat, var] = np.nan
                continue
            r, p = corr_func(topo[feat][mask], numeric_meta[var][mask])
            if method == "spearman":
                corr_results.loc[feat, var] = r.statistic if hasattr(r, 'statistic') else r
                pval_results.loc[feat, var] = r.pvalue if hasattr(r, 'pvalue') else p
            else:
                corr_results.loc[feat, var] = r
                pval_results.loc[feat, var] = p

    return corr_results, pval_results
