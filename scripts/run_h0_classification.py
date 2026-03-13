#!/usr/bin/env python3
"""H₀ + H₁ combined classification benchmark.

Adds four H₀ (connected-components) features to the existing six H₁ features
and tests whether the combined 10-feature topological signature improves AUC
over H₁ alone for IBD classification in the AGP cohort.

H₀ biological interpretation
------------------------------
H₀ counts how many *isolated microbial sub-communities* exist in a sample's
k-NN neighbourhood.  In a healthy microbiome, taxa quickly coalesce into one
connected ecosystem (low H₀ count, short lifetimes).  In IBD, the network
fragments — some taxa form islands that never rejoin the main community
(higher H₀ count, longer lifetimes).

Together H₀ (fragmentation) and H₁ (loss of loops) form a two-axis story:
  Healthy  → few isolated islands, many resilient cycles
  IBD      → many isolated islands, few/no cycles → ecological collapse

Feature sets evaluated
-----------------------
  A. H₁ only          (6 features — baseline, should match existing benchmark)
  B. H₀ only          (4 features — new)
  C. H₀ + H₁          (10 features — new combined)
  D. Aitchison-PCoA(10) + H₀ + H₁  (20 features — new combined with diversity)

Outputs
-------
  results/h0h1_classification.csv
  figures/h0h1_roc.png
  results/topo_sample_features_h0h1.npz   (cache, 10 features per sample)
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform
from src.tda.homology import compute_persistence, filter_infinite

# ── Config ─────────────────────────────────────────────────────────────────────
SEED          = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS  = 60
N_CV_FOLDS    = 5
K_PCOA        = 10

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIGURE_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE_H0H1  = os.path.join(RESULTS_DIR, "topo_sample_features_h0h1.npz")
OLD_CACHE   = os.path.join(RESULTS_DIR, "topo_sample_features.npz")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


# ── Feature extractors ─────────────────────────────────────────────────────────

def _h1_features(dgm_h1: np.ndarray) -> list:
    """6 H₁ scalars (unchanged from existing benchmark)."""
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else dgm_h1
    if len(finite) == 0:
        return [0, 0.0, 0.0, 0.0, 0.0, 0]
    lifetimes  = finite[:, 1] - finite[:, 0]
    total_pers = float(lifetimes.sum())
    norm_lt    = lifetimes / total_pers if total_pers > 0 else lifetimes
    entropy    = float(-np.sum(norm_lt * np.log(norm_lt + 1e-12)))
    births, deaths = finite[:, 0], finite[:, 1]
    thresholds = np.unique(np.concatenate([births, deaths]))
    max_betti  = int(max(np.sum((births <= t) & (deaths > t)) for t in thresholds)) \
                 if len(thresholds) > 0 else 0
    return [len(finite), entropy, total_pers,
            float(lifetimes.mean()), float(lifetimes.max()), max_betti]


def _h0_features(dgm_h0: np.ndarray) -> list:
    """4 H₀ scalars: finite connected-component fragmentation features.

    H₀ always has one infinite feature (the last surviving component).
    We exclude it: only finite features represent components that merge,
    i.e., *successfully* joined sub-communities.  The count of finite H₀
    = (total components - 1), measuring how many extra islands formed.
    """
    # Remove the single infinite feature
    finite = dgm_h0[np.isfinite(dgm_h0[:, 1])] if len(dgm_h0) > 0 else dgm_h0
    if len(finite) == 0:
        # Fully connected from the start — ideal healthy state
        return [0, 0.0, 0.0, 0.0]
    lifetimes = finite[:, 1] - finite[:, 0]
    return [
        len(finite),                 # h0_count: number of isolated islands formed
        float(lifetimes.sum()),      # h0_total_persistence: total isolation time
        float(lifetimes.mean()),     # h0_mean_lifetime: avg time before merging
        float(lifetimes.max()),      # h0_max_lifetime: most stubborn island
    ]


# ── Topology computation ───────────────────────────────────────────────────────

def compute_h0h1_features(clr_matrix: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Per-sample 10-feature vector: [4 H₀ | 6 H₁].

    Same k-NN neighbourhood approach as existing benchmark.
    """
    n = clr_matrix.shape[0]
    out = np.zeros((n, 10), dtype=np.float32)

    print(f"  Computing pairwise Euclidean distances ({n} samples)…")
    chunk = 256
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        diff = clr_matrix[i:end, np.newaxis, :] - clr_matrix[np.newaxis, :, :]
        dists[i:end] = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

    print(f"  Per-sample H₀+H₁ TDA (n={n}, k={k})…")
    t0 = time.time()
    for i in range(n):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (n - i)
            print(f"    {i}/{n}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

        nn_idx     = np.argsort(dists[i])[1: k + 1]
        nbhd       = clr_matrix[nn_idx]
        corr_mat, _ = spearmanr(nbhd, axis=0)
        if corr_mat.ndim == 0:
            corr_mat = np.array([[1.0]])
        dist_mat   = np.clip(1.0 - np.abs(corr_mat), 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)

        result = compute_persistence(dist_mat, maxdim=1, thresh=1.0)
        dgms   = result["dgms"]          # raw (may have inf deaths)

        dgm_h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
        dgm_h1 = filter_infinite(dgms)[1] if len(dgms) > 1 else np.empty((0, 2))

        out[i] = _h0_features(dgm_h0) + _h1_features(dgm_h1)

    print(f"  Done in {time.time()-t0:.1f}s")
    return out


def load_or_compute(clr_df: pd.DataFrame) -> np.ndarray:
    """Return cached H₀+H₁ features (10-col) or compute and cache."""
    current_ids = np.array(clr_df.index.tolist())

    if os.path.exists(CACHE_H0H1):
        d = np.load(CACHE_H0H1)
        if np.array_equal(d["sample_ids"], current_ids) and d["features"].shape[1] == 10:
            print(f"Cache hit: {CACHE_H0H1}  shape={d['features'].shape}")
            return d["features"]
        print("Cache shape mismatch — recomputing.")

    # Try to re-use old 6-feature cache for H₁ columns, only compute H₀
    if os.path.exists(OLD_CACHE):
        old = np.load(OLD_CACHE)
        if np.array_equal(old["sample_ids"], current_ids):
            print("Old 6-feature cache found. Computing H₀ features only…")
            feats_h0h1 = _augment_with_h0(clr_df.values.astype(np.float64), old["features"])
            np.savez(CACHE_H0H1, features=feats_h0h1, sample_ids=current_ids)
            return feats_h0h1

    print("Computing H₀+H₁ features from scratch (~10-15 min)…")
    feats = compute_h0h1_features(clr_df.values.astype(np.float64))
    np.savez(CACHE_H0H1, features=feats, sample_ids=current_ids)
    return feats


def _augment_with_h0(clr_matrix: np.ndarray, h1_cache: np.ndarray) -> np.ndarray:
    """Compute only H₀ (4 cols) and prepend to existing H₁ cache (6 cols)."""
    n = clr_matrix.shape[0]
    h0_out = np.zeros((n, 4), dtype=np.float32)

    print(f"  Computing pairwise Euclidean distances ({n} samples)…")
    chunk = 256
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        diff = clr_matrix[i:end, np.newaxis, :] - clr_matrix[np.newaxis, :, :]
        dists[i:end] = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

    print(f"  Computing H₀ per sample…")
    t0 = time.time()
    for i in range(n):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (n - i)
            print(f"    {i}/{n}  {elapsed:.0f}s  ETA {eta:.0f}s")
        nn_idx     = np.argsort(dists[i])[1: K_NEIGHBOURS + 1]
        nbhd       = clr_matrix[nn_idx]
        corr_mat, _ = spearmanr(nbhd, axis=0)
        if corr_mat.ndim == 0:
            corr_mat = np.array([[1.0]])
        dist_mat   = np.clip(1.0 - np.abs(corr_mat), 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)
        result = compute_persistence(dist_mat, maxdim=1, thresh=1.0)
        dgm_h0 = result["dgms"][0] if len(result["dgms"]) > 0 else np.empty((0, 2))
        h0_out[i] = _h0_features(dgm_h0)

    print(f"  H₀ done in {time.time()-t0:.1f}s")
    return np.hstack([h0_out, h1_cache])  # [H₀(4) | H₁(6)] = 10 cols


# ── Nested PCoA ────────────────────────────────────────────────────────────────

def _pcoa_fit(D_train, k):
    D2 = D_train.astype(np.float64) ** 2
    D2_col_means = D2.mean(axis=0)
    A = -0.5 * D2
    row_means  = A.mean(axis=1, keepdims=True)
    col_means  = A.mean(axis=0, keepdims=True)
    grand_mean = float(A.mean())
    B = A - row_means - col_means + grand_mean
    n = len(D_train)
    n_req = min(k + 5, n)
    vals, vecs = eigh(B, subset_by_index=[n - n_req, n - 1])
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]; vecs = vecs[:, idx]
    pos  = vals > 1e-10
    vals = vals[pos]; vecs = vecs[:, pos]
    k_actual = min(k, len(vals))
    vals = vals[:k_actual]; vecs = vecs[:, :k_actual]
    Y_train = vecs * np.sqrt(vals)
    return Y_train, vecs, vals, D2_col_means


def _pcoa_project(D_test_train, vecs, vals, D2_col_means, D2_grand_mean):
    D2 = D_test_train.astype(np.float64) ** 2
    A  = -0.5 * (D2 - D2_col_means - D2.mean(axis=1, keepdims=True) + D2_grand_mean)
    return A @ vecs / np.sqrt(vals)


# ── CV benchmark ───────────────────────────────────────────────────────────────

def cv_auc(X, y, n_folds=N_CV_FOLDS, seed=SEED):
    """5-fold stratified CV, returns mean ± std AUC for LR and RF."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    lr_aucs, rf_aucs = [], []

    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=1000, random_state=seed))])
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)

    for tr, te in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        lr.fit(Xtr, ytr)
        lr_aucs.append(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]))
        rf.fit(Xtr, ytr)
        rf_aucs.append(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))

    return (np.mean(lr_aucs), np.std(lr_aucs),
            np.mean(rf_aucs), np.std(rf_aucs))


def nested_pcoa_cv(clr_vals, y, n_folds=N_CV_FOLDS, seed=SEED):
    """Nested Aitchison-PCoA: PCoA fitted per fold to prevent leakage."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    lr_aucs, rf_aucs = [], []
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)

    for tr, te in skf.split(clr_vals, y):
        D_tr = np.sqrt(((clr_vals[tr, np.newaxis] - clr_vals[tr]) ** 2).sum(-1))
        Y_tr, vecs, vals, D2_col_means = _pcoa_fit(D_tr, K_PCOA)
        D2_grand = (D_tr.astype(np.float64) ** 2).mean()
        D_te = np.sqrt(((clr_vals[te, np.newaxis] - clr_vals[tr]) ** 2).sum(-1))
        Y_te = _pcoa_project(D_te, vecs, vals, D2_col_means, D2_grand)

        sc = StandardScaler().fit(Y_tr)
        lr.fit(sc.transform(Y_tr), y[tr])
        lr_aucs.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(Y_te))[:, 1]))
        rf.fit(Y_tr, y[tr])
        rf_aucs.append(roc_auc_score(y[te], rf.predict_proba(Y_te)[:, 1]))

    return (np.mean(lr_aucs), np.std(lr_aucs),
            np.mean(rf_aucs), np.std(rf_aucs))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("H₀ + H₁ Classification Benchmark")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────
    print("\nLoading AGP data…")
    otu_df, meta = load_agp(os.path.join(DATA_DIR, "agp"))
    stool_ids = meta.index[meta["BODY_SITE"] == "UBERON:feces"]
    otu_df = otu_df.loc[otu_df.index.intersection(stool_ids)]
    meta   = meta.loc[otu_df.index]

    from src.data.preprocess import filter_low_abundance, clr_transform
    otu_filt = filter_low_abundance(otu_df, min_prevalence=0.1)
    clr_df   = clr_transform(otu_filt)

    # Global top-80 taxa
    prevalence = (clr_df > clr_df.median()).mean()
    top80 = prevalence.nlargest(N_GLOBAL_TAXA).index
    clr80 = clr_df[top80]

    # IBD label — AGP uses "IBD" column with free-text values
    ibd_positive = {"Ulcerative colitis", "Crohn's disease",
                    "Ulcerative Colitis", "Crohn's Disease"}
    ibd_negative = {"I do not have IBD"}
    valid = meta["IBD"].isin(ibd_positive | ibd_negative)
    meta_filt = meta[valid]
    y_full = meta_filt["IBD"].isin(ibd_positive).astype(int)

    # Align: intersect filtered metadata with CLR taxa
    common   = clr80.index.intersection(y_full.index)
    clr80    = clr80.loc[common]
    y        = y_full.loc[common].values
    clr_vals = clr80.values.astype(np.float64)

    print(f"Samples: {len(y)}  IBD: {y.sum()}  Healthy: {(y==0).sum()}")

    # ── Topology features ──────────────────────────────────────────
    print("\nLoading / computing H₀+H₁ features…")
    feats10 = load_or_compute(clr80)          # (n, 10): [H₀(4) | H₁(6)]
    X_h0    = feats10[:, :4].astype(np.float64)   # H₀ only
    X_h1    = feats10[:, 4:].astype(np.float64)   # H₁ only
    X_h0h1  = feats10.astype(np.float64)           # combined

    # ── Run benchmarks ─────────────────────────────────────────────
    results = []

    print("\nRunning CV benchmarks…")

    configs = [
        ("H₁ only (6 feat)",           X_h1,   False),
        ("H₀ only (4 feat)",           X_h0,   False),
        ("H₀ + H₁ (10 feat)",          X_h0h1, False),
    ]

    for name, X, is_pcoa in configs:
        print(f"  {name}…")
        lr_m, lr_s, rf_m, rf_s = cv_auc(X, y)
        results.append({"feature_set": name,
                         "lr_auc": lr_m, "lr_auc_sd": lr_s,
                         "rf_auc": rf_m, "rf_auc_sd": rf_s})

    # Aitchison PCoA + H₀+H₁
    print("  Nested Aitchison-PCoA…")
    pcoa_lr, pcoa_lr_s, pcoa_rf, pcoa_rf_s = nested_pcoa_cv(clr_vals, y)
    results.append({"feature_set": "Aitchison-PCoA(10)",
                     "lr_auc": pcoa_lr, "lr_auc_sd": pcoa_lr_s,
                     "rf_auc": pcoa_rf, "rf_auc_sd": pcoa_rf_s})

    # Combined PCoA + H₀+H₁
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    lr_aucs, rf_aucs = [], []
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    print("  Aitchison-PCoA + H₀+H₁ (20 feat)…")
    for tr, te in skf.split(clr_vals, y):
        D_tr = np.sqrt(((clr_vals[tr, np.newaxis] - clr_vals[tr]) ** 2).sum(-1))
        Y_tr, vecs, vals, D2_col_means = _pcoa_fit(D_tr, K_PCOA)
        D2_grand = (D_tr.astype(np.float64) ** 2).mean()
        D_te = np.sqrt(((clr_vals[te, np.newaxis] - clr_vals[tr]) ** 2).sum(-1))
        Y_te = _pcoa_project(D_te, vecs, vals, D2_col_means, D2_grand)

        Xtr_comb = np.hstack([Y_tr, X_h0h1[tr]])
        Xte_comb = np.hstack([Y_te, X_h0h1[te]])
        sc = StandardScaler().fit(Xtr_comb)
        lr.fit(sc.transform(Xtr_comb), y[tr])
        lr_aucs.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(Xte_comb))[:, 1]))
        rf.fit(Xtr_comb, y[tr])
        rf_aucs.append(roc_auc_score(y[te], rf.predict_proba(Xte_comb)[:, 1]))

    results.append({"feature_set": "Aitchison-PCoA + H₀+H₁ (20 feat)",
                     "lr_auc": np.mean(lr_aucs), "lr_auc_sd": np.std(lr_aucs),
                     "rf_auc": np.mean(rf_aucs), "rf_auc_sd": np.std(rf_aucs)})

    # ── Save results ───────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "h0h1_classification.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
    print(df.to_string(index=False))

    # ── Plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1976D2", "#7B1FA2", "#388E3C", "#F57C00", "#C62828"]
    x = np.arange(len(results))
    w = 0.35
    bars_lr = ax.bar(x - w/2, df["lr_auc"], w, yerr=df["lr_auc_sd"],
                     color=colors[:len(df)], alpha=0.85, capsize=4,
                     label="Logistic Regression")
    bars_rf = ax.bar(x + w/2, df["rf_auc"], w, yerr=df["rf_auc_sd"],
                     color=colors[:len(df)], alpha=0.45, capsize=4,
                     label="Random Forest", hatch="///")

    for bar in list(bars_lr) + list(bars_rf):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature_set"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("H₀ + H₁ Topological Features — IBD Classification (AGP)\n"
                 "H₀ = ecosystem fragmentation · H₁ = loop resilience",
                 fontsize=11)
    ax.set_ylim(0.45, 0.85)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()

    out_fig = os.path.join(FIGURE_DIR, "h0h1_roc.png")
    fig.savefig(out_fig, dpi=150)
    print(f"Figure saved to {out_fig}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    h1_lr  = df.loc[df["feature_set"] == "H₁ only (6 feat)",  "lr_auc"].values[0]
    h0h1_lr = df.loc[df["feature_set"] == "H₀ + H₁ (10 feat)", "lr_auc"].values[0]
    delta   = h0h1_lr - h1_lr
    print(f"H₁ only  LR AUC: {h1_lr:.3f}")
    print(f"H₀+H₁    LR AUC: {h0h1_lr:.3f}  (Δ = {delta:+.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
