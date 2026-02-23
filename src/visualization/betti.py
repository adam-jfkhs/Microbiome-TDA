"""Betti curve visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt


def plot_betti_curves(diagrams, num_points=100, max_val=None, title="Betti Curves",
                      ax=None, save_path=None):
    """Plot Betti curves for multiple homological dimensions.

    Args:
        diagrams: List of persistence diagrams.
        num_points: Number of sample points.
        max_val: Maximum filtration value.
        title: Plot title.
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
    """
    from ..tda.features import betti_curve

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    labels = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$"]

    for dim, dgm in enumerate(diagrams):
        if dim >= 3:
            break
        filt_vals, betti = betti_curve(dgm, num_points=num_points, max_val=max_val)
        ax.plot(filt_vals, betti, color=colors[dim], label=labels[dim], linewidth=2)

    ax.set_xlabel("Filtration Value", fontsize=12)
    ax.set_ylabel("Betti Number", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
