"""Persistence diagram and barcode plot utilities."""

import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams


def plot_persistence_diagram(diagrams, title="Persistence Diagram", ax=None, save_path=None):
    """Plot persistence diagram with consistent styling.

    Args:
        diagrams: List of persistence diagrams (one per dimension).
        title: Plot title.
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    plot_diagrams(diagrams, ax=ax, show=False)
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_barcode(diagram, title="Persistence Barcode", ax=None, save_path=None, color="steelblue"):
    """Plot persistence barcode for a single dimension.

    Args:
        diagram: Persistence diagram (N x 2 array).
        title: Plot title.
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
        color: Bar color.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    finite_mask = np.isfinite(diagram[:, 1])
    finite_dgm = diagram[finite_mask]

    # Sort by birth time
    order = np.argsort(finite_dgm[:, 0])
    sorted_dgm = finite_dgm[order]

    for i, (birth, death) in enumerate(sorted_dgm):
        ax.barh(i, death - birth, left=birth, height=0.8, color=color, alpha=0.7)

    ax.set_xlabel("Filtration Value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
