"""Standard graph-theoretic metrics from correlation matrices.

Provides a baseline for comparison with TDA features: if standard network
metrics detect the same biological signals, topology may not add value.
If TDA features detect signals that graph metrics miss, that supports the
claim that persistent homology captures genuinely novel structure.
"""

import numpy as np
import networkx as nx


def network_metrics(corr_matrix: np.ndarray, threshold: float = 0.3) -> dict:
    """Compute standard network metrics from a correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (n_taxa, n_taxa)
        Spearman correlation matrix (values in [-1, 1]).
    threshold : float
        Absolute correlation threshold for edge inclusion.

    Returns
    -------
    dict with keys: 'edge_density', 'clustering_coeff', 'transitivity',
        'mean_degree', 'modularity', 'n_components'
    """
    n = corr_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) >= threshold:
                G.add_edge(i, j, weight=float(corr_matrix[i, j]))

    # Edge density
    edge_density = nx.density(G)

    # Average clustering coefficient (0.0 for graph with no edges)
    if G.number_of_edges() == 0:
        clustering_coeff = 0.0
    else:
        clustering_coeff = nx.average_clustering(G)

    # Transitivity (global clustering coefficient)
    transitivity = nx.transitivity(G)

    # Mean degree
    degrees = [d for _, d in G.degree()]
    mean_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    # Number of connected components
    n_components = nx.number_connected_components(G)

    # Modularity via greedy modularity communities
    if G.number_of_edges() == 0:
        modularity = 0.0
    else:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)

    return {
        "edge_density": edge_density,
        "clustering_coeff": clustering_coeff,
        "transitivity": transitivity,
        "mean_degree": mean_degree,
        "modularity": modularity,
        "n_components": n_components,
    }
