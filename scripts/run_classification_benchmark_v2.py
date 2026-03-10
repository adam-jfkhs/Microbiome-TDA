#!/usr/bin/env python3
"""
Classification benchmarking with β-diversity baselines.

Extends the original classification benchmark by adding:
  - Aitchison-PCoA(10): Euclidean distance on CLR-80, top-10 principal coordinates
  - BC-PCoA(10):        Bray–Curtis distance on relative abundances of top-80 taxa,
                        top-10 principal coordinates

Both PCoA embeddings use *nested* computation: for each CV fold the PCoA is
fitted on training samples only and test samples are projected via Gower's
(1968) out-of-sample formula.  This prevents any test-fold information from
leaking into the feature representation.

Feature sets compared (logistic regression + random forest, 5-fold CV):
  1.  Shannon only
  2.  Topology only           (6 H₁ scalars via k-NN neighbourhood TDA)
  3.  Topology + Shannon
  4.  Aitchison-PCoA(10)
  5.  BC-PCoA(10)
  6.  Aitchison-PCoA + Topology
  7.  BC-PCoA + Topology
  8.  Aitchison-PCoA + Topology + Shannon
  9.  BC-PCoA + Topology + Shannon

Topology features are cached to results/topo_sample_features.npz after first
computation (~10–15 min) to speed up re-runs.

Outputs
-------
  results/classification_benchmark_v2.csv
  figures/classification_roc_v2.png

Usage
-----
  python scripts/run_classification_benchmark_v2.py
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
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.loaders import load_agp
from src.data.preprocess import (
    filter_low_abundance, clr_transform, relative_abundance
)
from src.tda.homology import compute_persistence, filter_infinite

# ── Configuration ──────────────────────────────────────────────────────────────
SEED          = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS  = 60
N_CV_FOLDS    = 5
K_PCOA        = 10   # fixed PCoA rank — no tuning to keep comparison clean

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIGURE_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TOPO_CACHE  = os.path.join(RESULTS_DIR, "topo_sample_features.npz")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_prepare():
    """Load AGP, filter to stool, attach binary IBD label.

    Returns
    -------
    clr80_df   : DataFrame (n, 80)  CLR-transformed, top-80 taxa
    relabund80 : ndarray   (n, 80)  relative abundances of the same 80 taxa
    labels     : Series            1 = IBD, 0 = healthy
    """
    print("Loading AGP data …")
    otu_df, meta = load_agp(os.path.join(DATA_DIR, "agp"))

    stool_ids = meta.index[meta["BODY_SITE"] == "UBERON:feces"]
    otu_df = otu_df.loc[otu_df.index.intersection(stool_ids)]
    meta   = meta.loc[otu_df.index]

    ibd_mask     = meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
    healthy_mask = meta["IBD"] == "I do not have IBD"
    keep = ibd_mask | healthy_mask
    otu_df = otu_df.loc[keep]
    meta   = meta.loc[keep]

    labels = pd.Series(
        meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"]).astype(int),
        index=meta.index,
    )

    otu_filtered = filter_low_abundance(otu_df, min_prevalence=0.05, min_reads=1000)
    clr_df       = clr_transform(otu_filtered)

    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top_taxa   = prevalence.nlargest(N_GLOBAL_TAXA).index.tolist()
    clr80_df   = clr_df[top_taxa]
    labels     = labels.loc[clr80_df.index]

    # Relative abundances (for Bray–Curtis): compute from raw counts, same taxa
    otu_top80  = otu_filtered.loc[clr80_df.index, top_taxa]
    totals     = otu_top80.sum(axis=1)
    # Guard against any all-zero rows (shouldn't happen after filtering)
    totals     = totals.replace(0, 1)
    relabund80 = (otu_top80.div(totals, axis=0)).values.astype(np.float32)

    print(f"  n = {len(clr80_df)} | IBD = {labels.sum()} | healthy = {(labels == 0).sum()}")
    return clr80_df, relabund80, labels


# ── Shannon diversity (from filtered raw counts) ───────────────────────────────

def compute_shannon(otu_df: pd.DataFrame) -> np.ndarray:
    """Shannon H from raw counts (not CLR)."""
    raw = otu_df.values.astype(np.float64)
    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    p = raw / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -(p * logp).sum(axis=1)


# ── Per-sample topology via k-NN neighbourhood TDA ────────────────────────────

def _h1_features(dgm_h1: np.ndarray) -> list:
    """Extract 6 H₁ scalars from a persistence diagram."""
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else dgm_h1
    if len(finite) == 0:
        return [0, 0.0, 0.0, 0.0, 0.0, 0]

    lifetimes  = finite[:, 1] - finite[:, 0]
    total_pers = float(lifetimes.sum())
    norm_lt    = lifetimes / total_pers if total_pers > 0 else lifetimes
    entropy    = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))

    births, deaths = finite[:, 0], finite[:, 1]
    thresholds     = np.unique(np.concatenate([births, deaths]))
    max_betti      = int(max(
        np.sum((births <= t) & (deaths > t)) for t in thresholds
    )) if len(thresholds) > 0 else 0

    return [
        len(finite),
        entropy,
        total_pers,
        float(lifetimes.mean()),
        float(lifetimes.max()),
        max_betti,
    ]


def compute_topology_features(clr_matrix: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Per-sample H₁ features via k-NN neighbourhood TDA.

    For sample i: find k nearest neighbours → Spearman correlation on those k
    neighbours × 80 taxa → distance matrix d = 1 − |r| → Ripser H₁ → 6 scalars.

    Returns ndarray (n_samples, 6).
    """
    n_samples = clr_matrix.shape[0]
    out = np.zeros((n_samples, 6), dtype=np.float32)

    print(f"  Precomputing pairwise Euclidean distances ({n_samples}²) …")
    chunk = 256
    dists = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(0, n_samples, chunk):
        end = min(i + chunk, n_samples)
        diff = clr_matrix[i:end, np.newaxis, :] - clr_matrix[np.newaxis, :, :]
        dists[i:end] = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    print(f"  Computing per-sample topology (n={n_samples}, k={k}) …")
    t0 = time.time()
    for i in range(n_samples):
        if i % 500 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (n_samples - i)
            print(f"    {i}/{n_samples}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

        nn_idx = np.argsort(dists[i])[1: k + 1]
        neighbourhood = clr_matrix[nn_idx]

        corr_mat, _ = spearmanr(neighbourhood, axis=0)
        if corr_mat.ndim == 0:
            corr_mat = np.array([[1.0]])

        dist_mat = np.clip(1.0 - np.abs(corr_mat), 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)

        result = compute_persistence(dist_mat, maxdim=1, thresh=1.0)
        dgms   = filter_infinite(result["dgms"])
        dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        out[i] = _h1_features(dgm_h1)

    print(f"  Done in {time.time() - t0:.1f}s")
    return out


def load_or_compute_topology(clr_df: pd.DataFrame) -> np.ndarray:
    """Return cached topology features or compute and cache them."""
    if os.path.exists(TOPO_CACHE):
        print(f"Loading cached topology features from {TOPO_CACHE} …")
        data = np.load(TOPO_CACHE)
        cached_ids = data["sample_ids"]
        current_ids = np.array(clr_df.index.tolist())
        if np.array_equal(cached_ids, current_ids):
            print(f"  Cache hit: {data['features'].shape}")
            return data["features"]
        print("  Cache miss (sample IDs differ) — recomputing …")

    print("Computing per-sample topology features (first run, ~10–15 min) …")
    features = compute_topology_features(clr_df.values.astype(np.float64))
    np.savez(
        TOPO_CACHE,
        features=features,
        sample_ids=np.array(clr_df.index.tolist()),
    )
    print(f"  Cached to {TOPO_CACHE}")
    return features


# ── PCoA: nested (no-leakage) fitting and projection ──────────────────────────

def _pcoa_fit(D_train: np.ndarray, k: int):
    """Fit PCoA on a training distance matrix.

    Parameters
    ----------
    D_train : (n, n) symmetric distance matrix
    k       : number of principal coordinates to retain

    Returns
    -------
    Y_train      : (n, k_actual) training embeddings
    vecs_k       : (n, k_actual) eigenvectors
    vals_k       : (k_actual,)   positive eigenvalues
    D2_col_means : (n,)          col-means of D_train² (needed for projection)
    """
    D2 = D_train.astype(np.float64) ** 2
    D2_col_means = D2.mean(axis=0)

    A = -0.5 * D2
    row_means  = A.mean(axis=1, keepdims=True)
    col_means  = A.mean(axis=0, keepdims=True)
    grand_mean = float(A.mean())
    B = A - row_means - col_means + grand_mean

    n = len(D_train)
    n_req = min(k + 5, n)   # request a few extra to absorb numerical negatives
    vals, vecs = eigh(B, subset_by_index=[n - n_req, n - 1])

    # Descending order
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]; vecs = vecs[:, idx]

    # Discard numerical negatives
    pos  = vals > 1e-10
    vals = vals[pos]; vecs = vecs[:, pos]

    k_actual = min(k, len(vals))
    vals_k = vals[:k_actual]
    vecs_k = vecs[:, :k_actual]
    Y_train = vecs_k * np.sqrt(vals_k)

    return Y_train, vecs_k, vals_k, D2_col_means


def _pcoa_transform(D_test_train: np.ndarray, D2_col_means: np.ndarray,
                    vecs_k: np.ndarray, vals_k: np.ndarray) -> np.ndarray:
    """Project test points into PCoA space via Gower (1968).

    Parameters
    ----------
    D_test_train : (n_test, n_train) distances from test to training samples
    D2_col_means : (n_train,) column means of D_train²
    vecs_k       : (n_train, k) eigenvectors from _pcoa_fit
    vals_k       : (k,) eigenvalues

    Returns
    -------
    Y_test : (n_test, k)
    """
    d2_test  = D_test_train.astype(np.float64) ** 2
    centered = -0.5 * (d2_test - D2_col_means[np.newaxis, :])
    return centered @ vecs_k / np.sqrt(vals_k)


# ── Pairwise distance matrices ─────────────────────────────────────────────────

def compute_distance_matrices(clr80: np.ndarray, relabund80: np.ndarray):
    """Precompute full pairwise Aitchison and Bray–Curtis distance matrices.

    Aitchison distance = Euclidean distance in CLR space (proper compositional metric).
    Bray–Curtis       = computed on relative abundances (not CLR, which is signed).

    Returns
    -------
    D_ait : (n, n) float32
    D_bc  : (n, n) float32
    """
    n = clr80.shape[0]
    print(f"Computing Aitchison distance matrix ({n}×{n}) …")
    D_ait = cdist(clr80.astype(np.float64), clr80.astype(np.float64), metric="euclidean")
    D_ait = D_ait.astype(np.float32)

    print(f"Computing Bray–Curtis distance matrix ({n}×{n}) …")
    D_bc = cdist(relabund80.astype(np.float64), relabund80.astype(np.float64),
                 metric="braycurtis")
    D_bc = D_bc.astype(np.float32)

    return D_ait, D_bc


# ── Cross-validation: standard (non-PCoA) feature sets ────────────────────────

def evaluate_standard(X: np.ndarray, y: np.ndarray, name: str,
                       clf_type: str = "lr", cv: StratifiedKFold = None):
    """5-fold stratified CV for a fixed feature matrix X."""
    if clf_type == "lr":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    fold_aucs = []
    tprs, mean_fpr = [], np.linspace(0, 1, 200)

    for train_idx, test_idx in cv.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], proba))
        fpr, tpr, _ = roc_curve(y[test_idx], proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    print(f"  [{name}] AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    return {
        "name": name, "clf": clf_type,
        "mean_auc": mean_auc, "std_auc": std_auc, "fold_aucs": fold_aucs,
        "mean_fpr": mean_fpr,
        "mean_tpr": np.mean(tprs, axis=0),
        "std_tpr":  np.std(tprs, axis=0),
    }


# ── Cross-validation: PCoA-based feature sets ─────────────────────────────────

def evaluate_pcoa(D_full: np.ndarray, y: np.ndarray, name: str,
                  extra_X: np.ndarray | None = None,
                  clf_type: str = "lr", cv: StratifiedKFold = None):
    """5-fold CV where PCoA is re-fitted per fold on training samples only.

    Parameters
    ----------
    D_full   : (n, n) full pairwise distance matrix
    y        : (n,) binary labels
    name     : display name
    extra_X  : optional (n, p) array of additional features to concatenate
               with the PCoA coordinates (e.g. topology, Shannon)
    clf_type : 'lr' or 'rf'
    cv       : StratifiedKFold instance (must match splits used elsewhere)
    """
    if clf_type == "lr":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        )

    fold_aucs = []
    tprs, mean_fpr = [], np.linspace(0, 1, 200)

    for train_idx, test_idx in cv.split(D_full, y):  # D_full as proxy for X shape
        D_train  = D_full[np.ix_(train_idx, train_idx)]
        D_test_t = D_full[np.ix_(test_idx, train_idx)]

        # Fit PCoA on training distance block
        Y_tr, vecs_k, vals_k, D2_col_means = _pcoa_fit(D_train, k=K_PCOA)
        # Project test
        Y_te = _pcoa_transform(D_test_t, D2_col_means, vecs_k, vals_k)

        if extra_X is not None:
            X_tr = np.hstack([Y_tr, extra_X[train_idx]])
            X_te = np.hstack([Y_te, extra_X[test_idx]])
        else:
            X_tr = Y_tr
            X_te = Y_te

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf.fit(X_tr_s, y[train_idx])
        proba = clf.predict_proba(X_te_s)[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], proba))
        fpr, tpr, _ = roc_curve(y[test_idx], proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    print(f"  [{name}] AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    return {
        "name": name, "clf": clf_type,
        "mean_auc": mean_auc, "std_auc": std_auc, "fold_aucs": fold_aucs,
        "mean_fpr": mean_fpr,
        "mean_tpr": np.mean(tprs, axis=0),
        "std_tpr":  np.std(tprs, axis=0),
    }


# ── ROC figure ─────────────────────────────────────────────────────────────────

PALETTE = {
    "Shannon only":                    ("#a6cee3", "--"),
    "Topology only":                   ("#ff7f00", ":"),
    "Topology + Shannon":              ("#33a02c", "-"),
    "Aitchison-PCoA(10)":             ("#6a3d9a", "--"),
    "BC-PCoA(10)":                    ("#b15928", "-."),
    "Aitchison-PCoA + Topology":      ("#1f78b4", "-"),
    "BC-PCoA + Topology":             ("#e31a1c", "-"),
    "Aitchison-PCoA + Topology + Shannon": ("#b2df8a", "-"),
    "BC-PCoA + Topology + Shannon":   ("#fb9a99", "-"),
}


def plot_roc_curves(results_lr: list, results_rf: list, out_path: str):
    """Two-panel ROC figure (LR | RF) with shaded ±1 SD bands."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        "IBD Classification: Topological vs β-Diversity Features\n"
        "5-fold stratified CV · American Gut Project",
        fontsize=10, fontweight="bold",
    )

    for ax, results, title in zip(
        axes,
        [results_lr, results_rf],
        ["Logistic Regression", "Random Forest (500 trees)"],
    ):
        for res in results:
            colour, ls = PALETTE.get(res["name"], ("#555555", "-"))
            label = f"{res['name']}  (AUC={res['mean_auc']:.3f}±{res['std_auc']:.3f})"
            ax.plot(res["mean_fpr"], res["mean_tpr"],
                    color=colour, lw=1.8, ls=ls, label=label)
            ax.fill_between(
                res["mean_fpr"],
                res["mean_tpr"] - res["std_tpr"],
                res["mean_tpr"] + res["std_tpr"],
                color=colour, alpha=0.10,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("True Positive Rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC figure: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # 1. Load data
    clr80_df, relabund80, labels = load_and_prepare()
    clr80  = clr80_df.values.astype(np.float64)
    y      = labels.values.astype(int)

    # 2. Shannon (from raw filtered counts)
    print("Computing Shannon diversity …")
    otu_raw, meta_raw = load_agp(os.path.join(DATA_DIR, "agp"))
    stool_ids = meta_raw.index[meta_raw["BODY_SITE"] == "UBERON:feces"]
    otu_raw   = otu_raw.loc[otu_raw.index.intersection(stool_ids)]
    otu_raw   = filter_low_abundance(otu_raw, min_prevalence=0.05, min_reads=1000)
    otu_raw   = otu_raw.reindex(clr80_df.index).fillna(0)
    X_shannon = compute_shannon(otu_raw).reshape(-1, 1)

    # 3. Per-sample topology (cached after first run)
    X_topo = load_or_compute_topology(clr80_df).astype(np.float64)

    # 4. Pairwise distance matrices
    D_ait, D_bc = compute_distance_matrices(clr80, relabund80)

    # 5. Define composite feature matrices for standard (non-PCoA) sets
    X_topo_shannon = np.hstack([X_topo, X_shannon])

    # 6. StratifiedKFold (same random state across all evaluations)
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    # Pre-generate fold indices so all methods use identical splits
    fold_splits = list(cv.split(clr80, y))

    # Wrap cv for evaluate_standard (it calls cv.split internally)
    # We pass the same cv object — same seed → same splits every time.

    # 7. Evaluate all feature sets
    results_lr, results_rf = [], []
    t_eval = time.time()

    print("\n── Logistic Regression ──────────────────────────────────────────────")
    for name, X in [
        ("Shannon only",       X_shannon),
        ("Topology only",      X_topo),
        ("Topology + Shannon", X_topo_shannon),
    ]:
        results_lr.append(evaluate_standard(X, y, name, clf_type="lr", cv=cv))

    for name, D, extra in [
        ("Aitchison-PCoA(10)",              D_ait, None),
        ("BC-PCoA(10)",                     D_bc,  None),
        ("Aitchison-PCoA + Topology",       D_ait, X_topo),
        ("BC-PCoA + Topology",              D_bc,  X_topo),
        ("Aitchison-PCoA + Topology + Shannon", D_ait, X_topo_shannon),
        ("BC-PCoA + Topology + Shannon",    D_bc,  X_topo_shannon),
    ]:
        results_lr.append(evaluate_pcoa(D, y, name, extra_X=extra, clf_type="lr", cv=cv))

    print("\n── Random Forest ────────────────────────────────────────────────────")
    for name, X in [
        ("Shannon only",       X_shannon),
        ("Topology only",      X_topo),
        ("Topology + Shannon", X_topo_shannon),
    ]:
        results_rf.append(evaluate_standard(X, y, name, clf_type="rf", cv=cv))

    for name, D, extra in [
        ("Aitchison-PCoA(10)",              D_ait, None),
        ("BC-PCoA(10)",                     D_bc,  None),
        ("Aitchison-PCoA + Topology",       D_ait, X_topo),
        ("BC-PCoA + Topology",              D_bc,  X_topo),
        ("Aitchison-PCoA + Topology + Shannon", D_ait, X_topo_shannon),
        ("BC-PCoA + Topology + Shannon",    D_bc,  X_topo_shannon),
    ]:
        results_rf.append(evaluate_pcoa(D, y, name, extra_X=extra, clf_type="rf", cv=cv))

    print(f"\nEvaluation time: {(time.time() - t_eval):.0f}s")

    # 8. Save results
    rows = []
    for clf_label, results in [("logistic_regression", results_lr),
                                ("random_forest",       results_rf)]:
        for res in results:
            rows.append({
                "classifier":  clf_label,
                "feature_set": res["name"],
                "mean_auc":    res["mean_auc"],
                "std_auc":     res["std_auc"],
                **{f"fold_{i+1}_auc": v for i, v in enumerate(res["fold_aucs"])},
            })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "classification_benchmark_v2.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nResults: {csv_path}")
    print(df_out[["classifier", "feature_set", "mean_auc", "std_auc"]].to_string(index=False))

    # 9. ROC figure
    fig_path = os.path.join(FIGURE_DIR, "classification_roc_v2.png")
    plot_roc_curves(results_lr, results_rf, fig_path)

    print(f"\nTotal time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
