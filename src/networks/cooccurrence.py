"""Co-occurrence network construction from OTU tables."""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr


def spearman_correlation_matrix(df):
    """Compute pairwise Spearman correlation between OTUs.

    Args:
        df: DataFrame with samples as rows and OTUs as columns.

    Returns:
        Tuple of (correlation_df, pvalue_df).
    """
    corr, pval = spearmanr(df)
    otus = df.columns
    corr_df = pd.DataFrame(corr, index=otus, columns=otus)
    pval_df = pd.DataFrame(pval, index=otus, columns=otus)
    return corr_df, pval_df


def build_network(corr_df, threshold=0.3, pval_df=None, alpha=0.05):
    """Build a co-occurrence network from a correlation matrix.

    Args:
        corr_df: Correlation matrix as DataFrame.
        threshold: Minimum absolute correlation to include an edge.
        pval_df: Optional p-value matrix to filter by significance.
        alpha: Significance threshold for p-values.

    Returns:
        networkx.Graph with edges weighted by correlation.
    """
    G = nx.Graph()
    G.add_nodes_from(corr_df.columns)

    for i, otu_i in enumerate(corr_df.columns):
        for j, otu_j in enumerate(corr_df.columns):
            if i >= j:
                continue
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) < threshold:
                continue
            if pval_df is not None and pval_df.iloc[i, j] > alpha:
                continue
            G.add_edge(otu_i, otu_j, weight=corr_val)

    return G
