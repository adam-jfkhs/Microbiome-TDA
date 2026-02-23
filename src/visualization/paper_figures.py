"""Publication-quality figure generation utilities."""

import matplotlib.pyplot as plt
import matplotlib as mpl


def set_paper_style():
    """Set matplotlib rcParams for publication-quality figures."""
    mpl.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 300,
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def create_multi_panel_figure(nrows, ncols, figsize=None):
    """Create a multi-panel figure with consistent styling.

    Args:
        nrows: Number of rows.
        ncols: Number of columns.
        figsize: Optional figure size. Auto-calculated if None.

    Returns:
        Tuple of (fig, axes).
    """
    set_paper_style()

    if figsize is None:
        figsize = (4 * ncols + 1, 3.5 * nrows + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout(pad=3.0)

    return fig, axes
