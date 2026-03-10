#!/usr/bin/env python3
"""Classification benchmarking: topological features vs Shannon diversity for IBD prediction.

For each AGP stool sample we derive a 6-dimensional topological fingerprint via
local k-NN persistent homology:

  1. Find the k=60 nearest neighbours of sample i in CLR-80 space.
  2. Compute the Spearman correlation matrix of those 60 neighbours × 80 taxa.
  3. Convert to a distance matrix (d = 1 − |r|) and run Ripser (H₁ only).
  4. Extract the same six scalar H₁ features used in the main bootstrap analysis.

We then run 5-fold stratified cross-validation with:
  - Logistic regression (L2, balanced class weights)
  - Random forest (500 trees, balanced class weights)

Three feature sets are compared:
  a) Shannon α-diversity only (1 feature)
  b) Six topological features only
  c) Combined (topology + Shannon, 7 features)

Primary metric: AUROC.  Results are saved to results/classification_benchmark.csv
and a ROC-curve figure is written to figures/classification_roc.png.

Usage
-----
  python scripts/run_classification_benchmark.py

Runtime on AGP (n ≈ 3 400, k = 60, 80 taxa): ~10–15 min.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.tda.homology import compute_persistence, filter_infinite
from src.tda.features import persistence_entropy

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS = 60       # neighbourhood size for per-sample topology
N_CV_FOLDS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_prepare():
    """Load AGP, filter to stool, attach binary IBD label.

    Returns
    -------
    clr_df : DataFrame, shape (n_samples, N_GLOBAL_TAXA)
    labels : Series, 1 = IBD, 0 = healthy
    """
    print("Loading AGP data …")
    otu_df, meta = load_agp(os.path.join(DATA_DIR, "agp"))

    # Stool only
    stool_ids = meta.index[meta["BODY_SITE"] == "UBERON:feces"]
    otu_df = otu_df.loc[otu_df.index.intersection(stool_ids)]
    meta = meta.loc[otu_df.index]

    # Binary IBD label
    ibd_values = meta["IBD"]
    ibd_mask = ibd_values.isin(["Ulcerative colitis", "Crohn's disease"])
    healthy_mask = ibd_values == "I do not have IBD"
    keep = ibd_mask | healthy_mask
    otu_df = otu_df.loc[keep]
    meta = meta.loc[keep]
    labels = pd.Series(
        (meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])).astype(int),
        index=meta.index,
    )

    print(f"  Samples after filtering: {len(otu_df)}  "
          f"(IBD={labels.sum()}, healthy={(labels == 0).sum()})")

    # CLR transform
    otu_filtered = filter_low_abundance(otu_df, min_prevalence=0.05, min_reads=1000)
    clr_df = clr_transform(otu_filtered)

    # Global top-80 taxa by prevalence
    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top_taxa = prevalence.nlargest(N_GLOBAL_TAXA).index.tolist()
    clr_df = clr_df[top_taxa]
    labels = labels.loc[clr_df.index]

    print(f"  CLR matrix shape: {clr_df.shape}")
    return clr_df, labels


# ── Shannon diversity ──────────────────────────────────────────────────────────

def shannon_diversity(otu_row: np.ndarray) -> float:
    """Shannon H from raw counts (not CLR).  Handles zero counts safely."""
    p = otu_row / otu_row.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Per-sample topological features ───────────────────────────────────────────

def _h1_features_from_diagram(dgm_h1: np.ndarray) -> dict:
    """Extract the six H₁ scalar features from a persistence diagram."""
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else dgm_h1
    if len(finite) == 0:
        return {
            "h1_count": 0,
            "h1_entropy": 0.0,
            "h1_total_persistence": 0.0,
            "h1_mean_lifetime": 0.0,
            "h1_max_lifetime": 0.0,
            "max_betti1": 0,
        }

    lifetimes = finite[:, 1] - finite[:, 0]
    total_pers = float(lifetimes.sum())
    norm_lt = lifetimes / total_pers if total_pers > 0 else lifetimes
    entropy = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))

    # Max Betti-1: maximum number of simultaneously alive H₁ features
    births, deaths = finite[:, 0], finite[:, 1]
    thresholds = np.unique(np.concatenate([births, deaths]))
    max_betti = int(max(
        np.sum((births <= t) & (deaths > t)) for t in thresholds
    )) if len(thresholds) > 0 else 0

    return {
        "h1_count": len(finite),
        "h1_entropy": entropy,
        "h1_total_persistence": total_pers,
        "h1_mean_lifetime": float(lifetimes.mean()),
        "h1_max_lifetime": float(lifetimes.max()),
        "max_betti1": max_betti,
    }


def compute_sample_topology(clr_matrix: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Compute per-sample H₁ topological features via k-NN neighbourhood TDA.

    For each sample i:
      1. Find the k nearest neighbours in CLR space (Euclidean).
      2. Compute the Spearman taxa-correlation matrix of those k neighbours.
      3. Build distance matrix d = 1 − |r|.
      4. Run Ripser (H₁) and extract six scalar features.

    Parameters
    ----------
    clr_matrix : ndarray, shape (n_samples, n_taxa)
    k : neighbourhood size

    Returns
    -------
    features : ndarray, shape (n_samples, 6)
        Columns: h1_count, h1_entropy, h1_total_persistence,
                 h1_mean_lifetime, h1_max_lifetime, max_betti1.
    """
    n_samples, n_taxa = clr_matrix.shape
    feature_names = [
        "h1_count", "h1_entropy", "h1_total_persistence",
        "h1_mean_lifetime", "h1_max_lifetime", "max_betti1",
    ]
    out = np.zeros((n_samples, len(feature_names)))

    # Precompute pairwise Euclidean distance matrix for k-NN lookup
    print(f"  Precomputing pairwise distances ({n_samples}×{n_samples}) …")
    # Use chunked computation to avoid memory spike
    dists = np.zeros((n_samples, n_samples), dtype=np.float32)
    chunk = 256
    for i in range(0, n_samples, chunk):
        end = min(i + chunk, n_samples)
        diff = clr_matrix[i:end, np.newaxis, :] - clr_matrix[np.newaxis, :, :]
        dists[i:end] = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    print(f"  Computing per-sample topology (n={n_samples}, k={k}) …")
    t0 = time.time()
    for i in range(n_samples):
        if i % 250 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (n_samples - i)
            print(f"    {i}/{n_samples}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        # k nearest neighbours (exclude self)
        nn_idx = np.argsort(dists[i])[1: k + 1]
        neighbourhood = clr_matrix[nn_idx]  # (k, n_taxa)

        # Spearman correlation matrix (n_taxa × n_taxa)
        corr_mat, _ = spearmanr(neighbourhood, axis=0)
        if corr_mat.ndim == 0:
            # Edge case: single-column (shouldn't happen with 80 taxa)
            corr_mat = np.array([[1.0]])

        # Distance matrix: 1 - |r|, clipped to [0, 1]
        dist_mat = np.clip(1.0 - np.abs(corr_mat), 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)

        # Ripser H₁
        result = compute_persistence(dist_mat, maxdim=1, thresh=1.0)
        dgms = filter_infinite(result["dgms"])
        dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        feats = _h1_features_from_diagram(dgm_h1)
        for j, name in enumerate(feature_names):
            out[i, j] = feats[name]

    print(f"  Topology computation done in {time.time() - t0:.1f}s")
    return out, feature_names


# ── Cross-validated AUC evaluation ────────────────────────────────────────────

def evaluate_classifier(X: np.ndarray, y: np.ndarray, name: str, clf_type: str = "lr"):
    """5-fold stratified CV; returns mean AUC, std, and per-fold ROC data."""
    if clf_type == "lr":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            n_jobs=-1, random_state=SEED
        )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    fold_aucs = []
    tprs, mean_fpr = [], np.linspace(0, 1, 200)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        fold_aucs.append(auc)
        fpr, tpr, _ = roc_curve(y[test_idx], proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    print(f"  [{name}] AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    return {
        "name": name,
        "clf": clf_type,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
    }


# ── Figure: ROC curves ─────────────────────────────────────────────────────────

COLOURS = {
    "Shannon only":          "#6baed6",
    "Topology only":         "#e6550d",
    "Topology + Shannon":    "#31a354",
}

def plot_roc_curves(results_lr: list, results_rf: list, out_path: str):
    """Two-panel ROC figure (LR left, RF right) with shaded ± 1 SD bands."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle(
        "IBD Classification: Topological Features vs Shannon α-Diversity\n"
        "5-fold stratified CV · American Gut Project",
        fontsize=10, fontweight="bold",
    )

    for ax, results, title in zip(
        axes,
        [results_lr, results_rf],
        ["Logistic Regression", "Random Forest (500 trees)"],
    ):
        for res in results:
            colour = COLOURS.get(res["name"], "#333333")
            label = f"{res['name']}  (AUC = {res['mean_auc']:.3f} ± {res['std_auc']:.3f})"
            ax.plot(res["mean_fpr"], res["mean_tpr"], color=colour, lw=2, label=label)
            ax.fill_between(
                res["mean_fpr"],
                res["mean_tpr"] - res["std_tpr"],
                res["mean_tpr"] + res["std_tpr"],
                color=colour, alpha=0.15,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("True Positive Rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC figure saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    clr_df, labels = load_and_prepare()
    clr_matrix = clr_df.values
    y = labels.values.astype(int)
    sample_ids = clr_df.index.tolist()

    # 2. Shannon diversity (computed from raw CLR approximation;
    #    use CLR values directly as a stand-in — real Shannon requires raw counts,
    #    which we compute below from the original OTU table)
    print("Computing Shannon diversity …")
    otu_raw, meta_raw = load_agp(os.path.join(DATA_DIR, "agp"))
    stool_ids = meta_raw.index[meta_raw["BODY_SITE"] == "UBERON:feces"]
    otu_raw = otu_raw.loc[otu_raw.index.intersection(stool_ids)]
    otu_raw = otu_raw.loc[otu_raw.index.intersection(clr_df.index)]
    otu_raw = otu_raw.reindex(clr_df.index).fillna(0)
    shannon = np.array([
        shannon_diversity(otu_raw.loc[sid].values)
        for sid in clr_df.index
    ])
    X_shannon = shannon.reshape(-1, 1)

    # 3. Per-sample topological features
    print("Computing per-sample topological features …")
    topo_features, topo_names = compute_sample_topology(clr_matrix, k=K_NEIGHBOURS)

    # 4. Feature matrices
    X_topo = topo_features
    X_combined = np.hstack([topo_features, X_shannon])

    feature_sets = {
        "Shannon only":       X_shannon,
        "Topology only":      X_topo,
        "Topology + Shannon": X_combined,
    }

    # 5. Cross-validated evaluation
    print("\n── Logistic Regression ──────────────────────────────────")
    results_lr = [
        evaluate_classifier(X, y, name, clf_type="lr")
        for name, X in feature_sets.items()
    ]

    print("\n── Random Forest ────────────────────────────────────────")
    results_rf = [
        evaluate_classifier(X, y, name, clf_type="rf")
        for name, X in feature_sets.items()
    ]

    # 6. Save numerical results
    rows = []
    for clf_type, results in [("logistic_regression", results_lr), ("random_forest", results_rf)]:
        for res in results:
            rows.append({
                "classifier": clf_type,
                "feature_set": res["name"],
                "mean_auc": res["mean_auc"],
                "std_auc": res["std_auc"],
                **{f"fold_{i+1}_auc": v for i, v in enumerate(res["fold_aucs"])},
            })
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "classification_benchmark.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    print(results_df[["classifier", "feature_set", "mean_auc", "std_auc"]].to_string(index=False))

    # 7. ROC figure
    fig_path = os.path.join(FIGURE_DIR, "classification_roc.png")
    plot_roc_curves(results_lr, results_rf, fig_path)


if __name__ == "__main__":
    main()
