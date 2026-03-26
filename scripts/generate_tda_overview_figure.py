#!/usr/bin/env python3
"""Generate a large multi-panel TDA overview figure for the paper.

Panels:
  A  Healthy vs IBD co-occurrence networks (spring-layout graphs)
  B  Vietoris-Rips filtration at three thresholds (healthy network)
  C  3D simplicial complex with H0/H1/H2 features annotated
  D  Persistence diagrams (H1 + H2) for healthy vs IBD
  E  Betti curves (β₁ and β₂) for healthy vs IBD
  F  Effect-size comparison bar chart (H1 vs H2 features)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Polygon, Circle
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.patheffects as pe
import networkx as nx
from ripser import ripser

from src.data.ibdmdb_loader import load_ibdmdb
from src.data.preprocess import filter_low_abundance, clr_transform
from src.analysis.bootstrap import select_global_taxa
from src.networks.cooccurrence import spearman_correlation_matrix
from src.networks.distance import correlation_distance
from src.tda.filtration import prepare_distance_matrix
from src.tda.features import betti_curve, persistence_entropy

warnings.filterwarnings("ignore")

SEED = 42
N_TAXA = 80
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_HEALTHY = "#2e7d32"      # green
C_IBD     = "#c62828"      # red
C_H0      = "#1565c0"      # blue
C_H1      = "#e65100"      # orange
C_H2      = "#6a1b9a"      # purple
C_EDGE    = "#90a4ae"      # grey
C_BG      = "#fafafa"


def load_and_compute(n_subsample=100):
    """Load IBDMDB, compute correlation networks for one bootstrap draw."""
    rng = np.random.default_rng(SEED)

    otu_df, metadata = load_ibdmdb()
    filtered = filter_low_abundance(otu_df, min_prevalence=0.05, min_reads=0)
    clr_df = clr_transform(filtered)
    meta = metadata.loc[clr_df.index.intersection(metadata.index)]
    clr_df = clr_df.loc[meta.index]

    taxa = select_global_taxa(clr_df, n=N_TAXA)

    ids_ibd = meta.loc[meta["diagnosis"].isin(["CD", "UC"])].index.intersection(clr_df.index)
    ids_healthy = meta.loc[meta["diagnosis"] == "nonIBD"].index.intersection(clr_df.index)

    boot_h = rng.choice(list(ids_healthy), size=min(n_subsample, len(ids_healthy)), replace=False)
    boot_i = rng.choice(list(ids_ibd), size=min(n_subsample, len(ids_ibd)), replace=False)

    results = {}
    for label, boot_ids in [("healthy", boot_h), ("ibd", boot_i)]:
        subset = clr_df.loc[boot_ids, taxa]
        corr, _ = spearman_correlation_matrix(subset)
        dist_df = correlation_distance(corr)
        dist_mat = prepare_distance_matrix(dist_df)

        # Persistence at maxdim=2
        pers = ripser(dist_mat, maxdim=2, distance_matrix=True)

        results[label] = {
            "corr": corr.values if hasattr(corr, "values") else corr,
            "dist": dist_mat,
            "dgms": pers["dgms"],
            "taxa": taxa,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL A: Network comparison (2D spring layout)
# ══════════════════════════════════════════════════════════════════════════════

def draw_network(ax, corr_matrix, title, color, rng, n_show=40):
    """Draw a co-occurrence network from correlation matrix."""
    n = corr_matrix.shape[0]
    # Pick top-n_show most-connected nodes
    adj = np.abs(corr_matrix) > 0.3
    np.fill_diagonal(adj, False)
    degree = adj.sum(axis=1)
    top_idx = np.argsort(degree)[-n_show:]

    G = nx.Graph()
    for i in top_idx:
        G.add_node(i)
    for ii, i in enumerate(top_idx):
        for jj, j in enumerate(top_idx):
            if jj > ii and abs(corr_matrix[i, j]) > 0.3:
                G.add_edge(i, j, weight=abs(corr_matrix[i, j]))

    pos = nx.spring_layout(G, seed=SEED, k=1.8 / np.sqrt(n_show), iterations=80)

    # Edges
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if len(edge_weights) > 0:
        max_w = max(edge_weights)
        edge_alphas = [0.15 + 0.5 * (w / max_w) for w in edge_weights]
        edge_widths = [0.3 + 1.5 * (w / max_w) for w in edge_weights]
    else:
        edge_alphas, edge_widths = [], []

    for idx, (u, v) in enumerate(G.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=C_EDGE, alpha=edge_alphas[idx],
                linewidth=edge_widths[idx], zorder=1)

    # Nodes
    node_degrees = [G.degree(n) for n in G.nodes()]
    max_deg = max(node_degrees) if node_degrees else 1
    node_sizes = [30 + 200 * (d / max_deg) for d in node_degrees]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=color, alpha=0.8, edgecolors="white",
                           linewidths=0.5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Stats annotation
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    ax.text(0.02, 0.02, f"{n_nodes} nodes, {n_edges} edges",
            transform=ax.transAxes, fontsize=8, color="grey",
            va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL B: Vietoris-Rips filtration at three scales
# ══════════════════════════════════════════════════════════════════════════════

def draw_filtration(axes, dist_matrix, taxa_list):
    """Show filtration at three threshold values."""
    n_show = 30
    # Select a tightly-connected cluster so filtration looks dense
    # Use the smallest-distance clique seed then expand
    adj_tight = dist_matrix < np.percentile(dist_matrix[dist_matrix > 0], 40)
    np.fill_diagonal(adj_tight, False)
    degree = adj_tight.sum(axis=1)
    top_idx = np.argsort(degree)[-n_show:]
    sub_dist = dist_matrix[np.ix_(top_idx, top_idx)]

    # Layout using the distance matrix directly (Kamada-Kawai for metric embedding)
    G_full = nx.Graph()
    for i in range(n_show):
        G_full.add_node(i)
    for i in range(n_show):
        for j in range(i + 1, n_show):
            G_full.add_edge(i, j, weight=sub_dist[i, j])
    pos = nx.kamada_kawai_layout(G_full, weight="weight")

    thresholds = [
        np.percentile(sub_dist[sub_dist > 0], 18),
        np.percentile(sub_dist[sub_dist > 0], 40),
        np.percentile(sub_dist[sub_dist > 0], 65),
    ]
    labels = [
        f"$\\varepsilon = {thresholds[0]:.2f}$\n(components only — $H_0$)",
        f"$\\varepsilon = {thresholds[1]:.2f}$\n(loops appear — $H_1$)",
        f"$\\varepsilon = {thresholds[2]:.2f}$\n(voids enclosed — $H_2$)",
    ]
    stage_colors = [C_H0, C_H1, C_H2]

    for panel_idx, (ax, thresh, label, scol) in enumerate(
            zip(axes, thresholds, labels, stage_colors)):
        ax.set_facecolor(C_BG)

        # Find triangles (2-simplices) for shading
        triangles = []
        for i in range(n_show):
            for j in range(i + 1, n_show):
                if sub_dist[i, j] < thresh:
                    for k in range(j + 1, n_show):
                        if sub_dist[i, k] < thresh and sub_dist[j, k] < thresh:
                            triangles.append([i, j, k])

        # Draw filled triangles
        for tri in triangles:
            verts = [pos[tri[0]], pos[tri[1]], pos[tri[2]]]
            triangle = Polygon(verts, closed=True,
                               facecolor=C_H2, alpha=0.15, edgecolor=C_H2,
                               linewidth=0.2)
            ax.add_patch(triangle)

        # Draw edges
        for i in range(n_show):
            for j in range(i + 1, n_show):
                if sub_dist[i, j] < thresh:
                    x = [pos[i][0], pos[j][0]]
                    y = [pos[i][1], pos[j][1]]
                    ax.plot(x, y, color=C_H1, alpha=0.45, linewidth=1.4, zorder=2)

        # Draw nodes with degree-scaled sizes
        for i in range(n_show):
            deg = sum(1 for j in range(n_show) if j != i and sub_dist[i, j] < thresh)
            sz = 30 + 6 * deg
            ax.scatter(pos[i][0], pos[i][1], s=sz, c=C_H0,
                       edgecolors="white", linewidths=0.8, zorder=3)

        ax.set_title(label, fontsize=10, fontweight="bold", pad=6, color=scol)
        # Fit to data extent
        xs = [pos[i][0] for i in range(n_show)]
        ys = [pos[i][1] for i in range(n_show)]
        margin = 0.25
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_aspect("equal")
        ax.axis("off")

        # Count features
        n_edges_f = sum(1 for i in range(n_show) for j in range(i+1, n_show)
                        if sub_dist[i, j] < thresh)
        n_tri = len(triangles)
        # Detect loops (cycles in the 1-skeleton not filled by triangles)
        ax.text(0.5, -0.02,
                f"{n_edges_f} edges · {n_tri} triangles",
                transform=ax.transAxes, fontsize=8, color="grey",
                ha="center", va="top")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL C: 3D simplicial complex with annotations
# ══════════════════════════════════════════════════════════════════════════════

def draw_3d_simplicial(ax, dist_matrix):
    """Draw a 3D simplicial complex highlighting H0, H1, H2."""
    n_show = 25
    # Pick a tightly connected cluster
    adj = dist_matrix < np.percentile(dist_matrix[dist_matrix > 0], 35)
    np.fill_diagonal(adj, False)
    degree = adj.sum(axis=1)
    top_idx = np.argsort(degree)[-n_show:]
    sub_dist = dist_matrix[np.ix_(top_idx, top_idx)]

    rng = np.random.default_rng(SEED + 7)
    G = nx.Graph()
    for i in range(n_show):
        G.add_node(i)

    thresh = np.percentile(sub_dist[sub_dist > 0], 55)
    for i in range(n_show):
        for j in range(i + 1, n_show):
            if sub_dist[i, j] < thresh:
                G.add_edge(i, j, weight=1.0 / (sub_dist[i, j] + 0.01))

    # True 3D spring layout
    pos_3d_raw = nx.spring_layout(G, seed=SEED, dim=3,
                                   k=1.5 / np.sqrt(n_show), iterations=150,
                                   weight="weight")
    pos_3d = {n: tuple(c) for n, c in pos_3d_raw.items()}

    # Find triangles
    triangles = []
    for i in range(n_show):
        for j in range(i + 1, n_show):
            if sub_dist[i, j] < thresh:
                for k in range(j + 1, n_show):
                    if sub_dist[i, k] < thresh and sub_dist[j, k] < thresh:
                        triangles.append([i, j, k])

    # Draw filled triangles (H2 faces) — some opaque, some translucent
    tri_verts = []
    for tri in triangles:
        verts = [pos_3d[tri[0]], pos_3d[tri[1]], pos_3d[tri[2]]]
        tri_verts.append(verts)

    if tri_verts:
        poly = Poly3DCollection(tri_verts, alpha=0.18, facecolor=C_H2,
                                edgecolor=C_H2, linewidth=0.4)
        ax.add_collection3d(poly)

    # Draw edges (H1 skeleton)
    for i in range(n_show):
        for j in range(i + 1, n_show):
            if sub_dist[i, j] < thresh:
                xs = [pos_3d[i][0], pos_3d[j][0]]
                ys = [pos_3d[i][1], pos_3d[j][1]]
                zs = [pos_3d[i][2], pos_3d[j][2]]
                ax.plot(xs, ys, zs, color=C_H1, alpha=0.45, linewidth=1.2)

    # Draw nodes (H0)
    for i in range(n_show):
        deg = G.degree(i)
        sz = 40 + 8 * deg
        ax.scatter(*pos_3d[i], s=sz, c=C_H0, edgecolors="white",
                   linewidths=0.8, zorder=5, depthshade=True)

    # Legend
    ax.text2D(0.03, 0.97, "$H_0$: components", transform=ax.transAxes,
              fontsize=9, color=C_H0, fontweight="bold",
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.text2D(0.03, 0.91, "$H_1$: loops (cycles)", transform=ax.transAxes,
              fontsize=9, color=C_H1, fontweight="bold",
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.text2D(0.03, 0.85, "$H_2$: voids (cavities)", transform=ax.transAxes,
              fontsize=9, color=C_H2, fontweight="bold",
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    n_edges = G.number_of_edges()
    n_tri = len(triangles)
    ax.text2D(0.5, 0.02, f"{n_show} nodes · {n_edges} edges · {n_tri} triangles",
              transform=ax.transAxes, fontsize=7, color="grey", ha="center",
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_title("3D Simplicial Complex\n(Healthy Network)", fontsize=12,
                 fontweight="bold", pad=8)

    # Clean 3D axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgrey")
    ax.yaxis.pane.set_edgecolor("lightgrey")
    ax.zaxis.pane.set_edgecolor("lightgrey")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, alpha=0.15)
    ax.view_init(elev=20, azim=140)


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL D: Persistence diagrams (H1 + H2) healthy vs IBD
# ══════════════════════════════════════════════════════════════════════════════

def draw_persistence_diagrams(axes, dgms_healthy, dgms_ibd):
    """Draw persistence diagrams for H1 and H2, healthy vs IBD overlaid."""
    titles = ["$H_1$ Persistence Diagram (Loops)",
              "$H_2$ Persistence Diagram (Voids)"]

    for dim_idx, (ax, title) in enumerate(zip(axes, titles)):
        dim = dim_idx + 1  # H1 = index 1, H2 = index 2

        dgm_h = dgms_healthy[dim]
        dgm_i = dgms_ibd[dim]

        # Filter finite
        mask_h = np.isfinite(dgm_h[:, 1]) if len(dgm_h) > 0 else np.array([], dtype=bool)
        mask_i = np.isfinite(dgm_i[:, 1]) if len(dgm_i) > 0 else np.array([], dtype=bool)
        fin_h = dgm_h[mask_h] if len(dgm_h) > 0 else np.empty((0, 2))
        fin_i = dgm_i[mask_i] if len(dgm_i) > 0 else np.empty((0, 2))

        # Diagonal
        all_pts = np.concatenate([fin_h, fin_i]) if len(fin_h) > 0 or len(fin_i) > 0 else np.array([[0, 1]])
        lo = all_pts[:, 0].min() * 0.95
        hi = all_pts[:, 1].max() * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)

        # Fill between diagonal and points region
        ax.fill_between([lo, hi], [lo, hi], hi, alpha=0.03, color="grey")

        # Plot points
        if len(fin_h) > 0:
            ax.scatter(fin_h[:, 0], fin_h[:, 1], s=25, c=C_HEALTHY,
                       alpha=0.6, edgecolors="white", linewidths=0.3,
                       label=f"Healthy (n={len(fin_h)})", zorder=3)
        if len(fin_i) > 0:
            ax.scatter(fin_i[:, 0], fin_i[:, 1], s=25, c=C_IBD,
                       alpha=0.6, edgecolors="white", linewidths=0.3,
                       label=f"IBD (n={len(fin_i)})", zorder=3)

        ax.set_xlabel("Birth ($\\varepsilon$)", fontsize=9)
        ax.set_ylabel("Death ($\\varepsilon$)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
        ax.set_aspect("equal")
        ax.set_facecolor(C_BG)


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL E: Betti curves
# ══════════════════════════════════════════════════════════════════════════════

def draw_betti_curves(axes, dgms_healthy, dgms_ibd):
    """Draw Betti-1 and Betti-2 curves for healthy vs IBD."""
    labels_dim = ["$\\beta_1$ (loops)", "$\\beta_2$ (voids)"]

    for dim_idx, (ax, lbl) in enumerate(zip(axes, labels_dim)):
        dim = dim_idx + 1

        eps_h, beta_h = betti_curve(dgms_healthy[dim], num_points=300)
        eps_i, beta_i = betti_curve(dgms_ibd[dim], num_points=300)

        ax.fill_between(eps_h, beta_h, alpha=0.25, color=C_HEALTHY)
        ax.plot(eps_h, beta_h, color=C_HEALTHY, linewidth=1.8,
                label="Healthy")

        ax.fill_between(eps_i, beta_i, alpha=0.25, color=C_IBD)
        ax.plot(eps_i, beta_i, color=C_IBD, linewidth=1.8,
                label="IBD")

        ax.set_xlabel("Filtration ($\\varepsilon$)", fontsize=9)
        ax.set_ylabel(f"Betti number {lbl}", fontsize=9)
        ax.set_title(f"Betti Curve — {lbl}", fontsize=11, fontweight="bold", pad=6)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.set_facecolor(C_BG)

        # Annotate peak difference
        peak_h = beta_h.max()
        peak_i = beta_i.max()
        ax.annotate(f"Peak: {int(peak_h)}",
                    xy=(eps_h[np.argmax(beta_h)], peak_h),
                    xytext=(10, 5), textcoords="offset points",
                    fontsize=7, color=C_HEALTHY, fontweight="bold")
        ax.annotate(f"Peak: {int(peak_i)}",
                    xy=(eps_i[np.argmax(beta_i)], peak_i),
                    xytext=(10, -12), textcoords="offset points",
                    fontsize=7, color=C_IBD, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL F: Effect size comparison
# ══════════════════════════════════════════════════════════════════════════════

def draw_effect_sizes(ax):
    """Bar chart comparing H1 vs H2 Cohen's d values."""
    h2_csv = os.path.join(os.path.dirname(__file__), "..", "results", "h2_exploration.csv")
    df = pd.read_csv(h2_csv)

    features_short = ["count", "entropy", "total\npersist.", "mean\nlifetime",
                       "max\nlifetime", "max\nBetti"]

    h1 = df[df["dimension"] == "H1"]["cohens_d"].abs().values
    h2 = df[df["dimension"] == "H2"]["cohens_d"].abs().values

    x = np.arange(len(features_short))
    width = 0.35

    bars1 = ax.bar(x - width/2, h1, width, label="$H_1$ (loops)",
                   color=C_H1, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, h2, width, label="$H_2$ (voids)",
                   color=C_H2, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=C_H1)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=C_H2)

    ax.set_xticks(x)
    ax.set_xticklabels(features_short, fontsize=8)
    ax.set_ylabel("|Cohen's $d$|", fontsize=10)
    ax.set_title("Effect Sizes: $H_1$ vs $H_2$ (IBD vs Healthy)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.axhline(0.8, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(5.6, 0.85, "large effect\nthreshold", fontsize=6,
            color="grey", ha="right")
    ax.set_ylim(0, 3.3)
    ax.set_facecolor(C_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FIGURE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data and computing persistence...")
    data = load_and_compute(n_subsample=100)

    print("Building figure...")
    fig = plt.figure(figsize=(24, 28))

    # Master title
    fig.suptitle(
        "Topological Data Analysis of the Gut Microbiome:\n"
        "From Co-occurrence Networks to Multi-dimensional Homology",
        fontsize=20, fontweight="bold", y=0.98,
    )

    # ── Layout: 5 rows ──────────────────────────────────────────────────────
    gs = gridspec.GridSpec(5, 4, figure=fig,
                           height_ratios=[1.0, 1.0, 1.2, 1.0, 1.0],
                           hspace=0.32, wspace=0.30,
                           left=0.05, right=0.95, top=0.94, bottom=0.03)

    # Row 1: Panel A (networks) + Panel C (3D complex)
    ax_net_h = fig.add_subplot(gs[0, 0:2])
    ax_net_i = fig.add_subplot(gs[0, 2:4])

    # Row 2: Panel B (filtration 3 stages)
    ax_filt = [fig.add_subplot(gs[1, 0]),
               fig.add_subplot(gs[1, 1]),
               fig.add_subplot(gs[1, 2])]
    ax_3d = fig.add_subplot(gs[1, 3], projection="3d")

    # Row 3: Panel D (persistence diagrams)
    ax_pd1 = fig.add_subplot(gs[2, 0:2])
    ax_pd2 = fig.add_subplot(gs[2, 2:4])

    # Row 4: Panel E (Betti curves)
    ax_bc1 = fig.add_subplot(gs[3, 0:2])
    ax_bc2 = fig.add_subplot(gs[3, 2:4])

    # Row 5: Panel F (effect sizes)
    ax_eff = fig.add_subplot(gs[4, 1:3])

    # ── Draw all panels ─────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)

    # A: Networks
    print("  Panel A: Co-occurrence networks...")
    draw_network(ax_net_h, data["healthy"]["corr"],
                 "A₁  Healthy Co-occurrence Network", C_HEALTHY, rng)
    draw_network(ax_net_i, data["ibd"]["corr"],
                 "A₂  IBD Co-occurrence Network", C_IBD, rng)

    # B: Filtration
    print("  Panel B: Vietoris-Rips filtration...")
    draw_filtration(ax_filt, data["healthy"]["dist"], data["healthy"]["taxa"])
    ax_filt[0].set_title("B₁  " + ax_filt[0].get_title(), fontsize=10, fontweight="bold")
    ax_filt[1].set_title("B₂  " + ax_filt[1].get_title(), fontsize=10, fontweight="bold")
    ax_filt[2].set_title("B₃  " + ax_filt[2].get_title(), fontsize=10, fontweight="bold")

    # C: 3D simplicial complex
    print("  Panel C: 3D simplicial complex...")
    draw_3d_simplicial(ax_3d, data["healthy"]["dist"])
    ax_3d.set_title("C  " + ax_3d.get_title(), fontsize=13, fontweight="bold")

    # D: Persistence diagrams
    print("  Panel D: Persistence diagrams...")
    draw_persistence_diagrams([ax_pd1, ax_pd2],
                              data["healthy"]["dgms"], data["ibd"]["dgms"])
    ax_pd1.set_title("D₁  " + ax_pd1.get_title(), fontsize=11, fontweight="bold")
    ax_pd2.set_title("D₂  " + ax_pd2.get_title(), fontsize=11, fontweight="bold")

    # E: Betti curves
    print("  Panel E: Betti curves...")
    draw_betti_curves([ax_bc1, ax_bc2],
                      data["healthy"]["dgms"], data["ibd"]["dgms"])
    ax_bc1.set_title("E₁  " + ax_bc1.get_title(), fontsize=11, fontweight="bold")
    ax_bc2.set_title("E₂  " + ax_bc2.get_title(), fontsize=11, fontweight="bold")

    # F: Effect sizes
    print("  Panel F: Effect size comparison...")
    draw_effect_sizes(ax_eff)
    ax_eff.set_title("F  " + ax_eff.get_title(), fontsize=11, fontweight="bold")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(FIGURE_DIR, "tda_overview_figure.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Also save a PDF version for the paper
    out_pdf = os.path.join(FIGURE_DIR, "tda_overview_figure.pdf")
    fig2 = plt.figure(figsize=(24, 28))
    # Re-render for PDF would be duplicate; just note it
    print(f"(PNG saved at 200 DPI — {os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
