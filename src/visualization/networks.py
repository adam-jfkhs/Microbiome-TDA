"""Network visualization utilities."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_cooccurrence_network(G, title="Co-occurrence Network", ax=None,
                               save_path=None, node_size_attr=None):
    """Plot a co-occurrence network with spring layout.

    Args:
        G: networkx.Graph with edge weights.
        title: Plot title.
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
        node_size_attr: Optional node attribute to scale node sizes by.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42, k=1.0 / np.sqrt(G.number_of_nodes()))

    # Edge colors by sign of correlation
    edge_colors = []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0)
        edge_colors.append("#4CAF50" if w > 0 else "#F44336")
        edge_widths.append(abs(w) * 3)

    # Node sizes
    if node_size_attr and nx.get_node_attributes(G, node_size_attr):
        sizes = [G.nodes[n].get(node_size_attr, 100) for n in G.nodes()]
    else:
        sizes = [100 + 20 * G.degree(n) for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#2196F3",
                           alpha=0.8, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
