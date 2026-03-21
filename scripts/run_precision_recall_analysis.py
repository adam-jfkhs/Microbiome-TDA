#!/usr/bin/env python3
"""
Precision-recall analysis for IBD classification.

Complements the AUROC benchmark by reporting threshold-dependent metrics at two
operating points:
  A. threshold = 0.50  (standard, classifier-intrinsic)
  B. threshold calibrated to 95% specificity on the training fold, then applied
     to the held-out test fold — "how many IBD cases do we catch while keeping
     the false-positive rate at 5%?"

Both rules are applied inside the CV loop so no test-fold information is ever
used to choose the threshold (no leakage).

Feature sets evaluated (logistic regression, same 5-fold splits as benchmark):
  • Shannon only
  • Topology only
  • Aitchison-PCoA(10)
  • BC-PCoA(10)
  • Aitchison-PCoA + Topology + Shannon  (best overall from benchmark)

Outputs
-------
  results/precision_recall_metrics.csv
  figures/precision_recall_curves.png

Requires
--------
  results/topo_sample_features.npz   (written by run_classification_benchmark_v2.py)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.loaders import load_agp
from src.data.preprocess import filter_low_abundance, clr_transform

# ── Config — must match benchmark ──────────────────────────────────────────────
SEED          = 42
N_GLOBAL_TAXA = 80
K_NEIGHBOURS  = 60   # used for topology (cache is already built)
N_CV_FOLDS    = 5
K_PCOA        = 10
TARGET_SPEC   = 0.95  # 95 % specificity operating point

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURE_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
TOPO_CACHE  = os.path.join(RESULTS_DIR, "topo_sample_features.npz")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR,  exist_ok=True)


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    """Identical filtering to run_classification_benchmark_v2.py."""
    otu_df, meta = load_agp(os.path.join(DATA_DIR, "agp"))
    stool_ids = meta.index[meta["BODY_SITE"] == "UBERON:feces"]
    otu_df = otu_df.loc[otu_df.index.intersection(stool_ids)]
    meta   = meta.loc[otu_df.index]

    ibd_mask     = meta["IBD"].isin(["Ulcerative colitis", "Crohn's disease"])
    healthy_mask = meta["IBD"] == "I do not have IBD"
    keep = ibd_mask | healthy_mask
    otu_df = otu_df.loc[keep]; meta = meta.loc[keep]

    labels = (meta["IBD"].isin(["Ulcerative colitis",
                                 "Crohn's disease"])).astype(int)
    labels = pd.Series(labels.values, index=meta.index)

    otu_f    = filter_low_abundance(otu_df, min_prevalence=0.05, min_reads=1000)
    clr_df   = clr_transform(otu_f)

    prevalence = (clr_df > clr_df.median()).mean(axis=0)
    top_taxa   = prevalence.nlargest(N_GLOBAL_TAXA).index.tolist()
    clr80_df   = clr_df[top_taxa]
    labels     = labels.loc[clr80_df.index]

    otu_top80  = otu_f.loc[clr80_df.index, top_taxa]
    totals     = otu_top80.sum(axis=1).replace(0, 1)
    relabund80 = (otu_top80.div(totals, axis=0)).values.astype(np.float32)

    # Shannon from raw filtered OTU (all taxa, same as benchmark)
    raw = otu_f.reindex(clr80_df.index).fillna(0).values.astype(np.float64)
    tot = raw.sum(axis=1, keepdims=True); tot[tot == 0] = 1.0
    p   = raw / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    X_shannon = (-(p * logp).sum(axis=1)).reshape(-1, 1)

    # Topology from cache
    cache = np.load(TOPO_CACHE)
    cached_ids   = cache["sample_ids"]
    current_ids  = np.array(clr80_df.index.tolist())
    assert np.array_equal(cached_ids, current_ids), \
        "Topology cache sample IDs don't match. Re-run run_classification_benchmark_v2.py."
    X_topo = cache["features"].astype(np.float64)

    y = labels.values.astype(int)
    print(f"n={len(y)}, IBD={y.sum()}, healthy={(y==0).sum()}, "
          f"prevalence={y.mean():.3f}")
    return clr80_df.values.astype(np.float64), relabund80, X_shannon, X_topo, y


# ── PCoA utilities ─────────────────────────────────────────────────────────────

def _pcoa_fit(D_train, k):
    D2 = D_train.astype(np.float64) ** 2
    D2_col_means = D2.mean(axis=0)
    A  = -0.5 * D2
    B  = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    n  = len(D_train)
    n_req = min(k + 5, n)
    vals, vecs = eigh(B, subset_by_index=[n - n_req, n - 1])
    idx = np.argsort(vals)[::-1]; vals = vals[idx]; vecs = vecs[:, idx]
    pos = vals > 1e-10; vals = vals[pos]; vecs = vecs[:, pos]
    k_act = min(k, len(vals))
    vals_k = vals[:k_act]; vecs_k = vecs[:, :k_act]
    Y_train = vecs_k * np.sqrt(vals_k)
    return Y_train, vecs_k, vals_k, D2_col_means

def _pcoa_transform(D_test_train, D2_col_means, vecs_k, vals_k):
    d2  = D_test_train.astype(np.float64) ** 2
    cen = -0.5 * (d2 - D2_col_means[np.newaxis, :])
    return cen @ vecs_k / np.sqrt(vals_k)


# ── Threshold calibration ──────────────────────────────────────────────────────

def find_threshold_for_specificity(y_true, y_prob, target_spec=TARGET_SPEC):
    """Find the LOWEST threshold on (y_true, y_prob) that achieves target specificity.

    Scanning thresholds from high → low; specificity is monotonically
    non-increasing so we break on the first failure.

    Returns the threshold (scalar float).
    """
    healthy = y_true == 0
    n_healthy = healthy.sum()
    if n_healthy == 0:
        return 0.5

    sorted_t = np.sort(np.unique(y_prob))[::-1]   # high → low
    best_t   = float(sorted_t[0])                  # most conservative default

    for t in sorted_t:
        tn   = ((y_prob < t) & healthy).sum()
        spec = tn / n_healthy
        if spec >= target_spec:
            best_t = float(t)    # keep lowering while we still satisfy target
        else:
            break                # monotone: no lower t will satisfy it

    return best_t


# ── Per-threshold metrics ──────────────────────────────────────────────────────

def threshold_metrics(y_true, y_prob, threshold):
    """Return dict of TP/FP/TN/FN/precision/recall/specificity/F1."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, specificity=spec, f1=f1)


# ── Cross-validation with probability output ───────────────────────────────────

def cv_with_probabilities(X_train_blocks_fn, y, cv):
    """Run 5-fold CV, collecting test-fold predicted probabilities.

    Parameters
    ----------
    X_train_blocks_fn : callable(train_idx, test_idx) → (X_tr, X_te)
    y                 : (n,) labels
    cv                : StratifiedKFold

    Returns
    -------
    all_y_true : (n,) true labels in original order (reassembled from folds)
    all_y_prob : (n,) predicted probabilities
    fold_y_true : list of per-fold true label arrays (for threshold calibration)
    fold_y_prob_tr : list of per-fold TRAINING predicted probabilities
    fold_y_prob_te : list of per-fold TEST predicted probabilities
    """
    n = len(y)
    all_y_true = np.empty(n, dtype=int)
    all_y_prob = np.empty(n, dtype=float)
    fold_y_true_tr, fold_y_prob_tr = [], []
    fold_y_true_te, fold_y_prob_te = [], []

    for train_idx, test_idx in cv.split(np.zeros(n), y):
        X_tr, X_te = X_train_blocks_fn(train_idx, test_idx)
        y_tr = y[train_idx]; y_te = y[test_idx]

        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
        )
        sc  = StandardScaler().fit(X_tr)
        clf.fit(sc.transform(X_tr), y_tr)

        prob_tr = clf.predict_proba(sc.transform(X_tr))[:, 1]
        prob_te = clf.predict_proba(sc.transform(X_te))[:, 1]

        all_y_true[test_idx] = y_te
        all_y_prob[test_idx] = prob_te
        fold_y_true_tr.append(y_tr)
        fold_y_prob_tr.append(prob_tr)
        fold_y_true_te.append(y_te)
        fold_y_prob_te.append(prob_te)

    return (all_y_true, all_y_prob,
            fold_y_true_tr, fold_y_prob_tr,
            fold_y_true_te, fold_y_prob_te)


def evaluate_feature_set(name, X_blocks_fn, y, cv,
                          D_ait=None, D_bc=None,
                          use_ait=False, use_bc=False,
                          extra_X=None):
    """Wrap cv_with_probabilities for PCoA-based or static feature sets.

    For PCoA sets, `X_blocks_fn` is None and (use_ait or use_bc) is True.
    """
    n = len(y)

    def blocks_fn(tr, te):
        parts_tr, parts_te = [], []
        # Static features
        if extra_X is not None:
            parts_tr.append(extra_X[tr])
            parts_te.append(extra_X[te])
        # Aitchison PCoA — nested fit/project
        if use_ait:
            D_tr = D_ait[np.ix_(tr, tr)]
            D_te = D_ait[np.ix_(te, tr)]
            Y_tr, vecs, vals, D2c = _pcoa_fit(D_tr, K_PCOA)
            Y_te = _pcoa_transform(D_te, D2c, vecs, vals)
            parts_tr.append(Y_tr); parts_te.append(Y_te)
        # BC PCoA — nested fit/project
        if use_bc:
            D_tr = D_bc[np.ix_(tr, tr)]
            D_te = D_bc[np.ix_(te, tr)]
            Y_tr, vecs, vals, D2c = _pcoa_fit(D_tr, K_PCOA)
            Y_te = _pcoa_transform(D_te, D2c, vecs, vals)
            parts_tr.append(Y_tr); parts_te.append(Y_te)
        return np.hstack(parts_tr), np.hstack(parts_te)

    print(f"  Evaluating: {name} …")
    (all_y, all_p,
     ftr_y, ftr_p,
     fte_y, fte_p) = cv_with_probabilities(blocks_fn, y, cv)

    # AP score (area under PR curve, pooled across folds)
    ap = average_precision_score(all_y, all_p)

    # ── Threshold A: 0.5 ──────────────────────────────────────────────────────
    row_a = threshold_metrics(all_y, all_p, threshold=0.5)
    row_a["ap_score"] = ap
    row_a["threshold_rule"] = "0.50"
    row_a["threshold_value"] = 0.50

    # ── Threshold B: 95 % specificity, calibrated per fold ────────────────────
    # Find threshold on each training fold; apply to test fold; aggregate.
    pooled_y_te, pooled_p_te_B = [], []
    thresholds_B = []
    for y_tr, p_tr, y_te, p_te in zip(ftr_y, ftr_p, fte_y, fte_p):
        t_star = find_threshold_for_specificity(y_tr, p_tr, TARGET_SPEC)
        thresholds_B.append(t_star)
        pooled_y_te.append(y_te)
        pooled_p_te_B.append(p_te)

    pooled_y   = np.concatenate(pooled_y_te)
    pooled_p_B = np.concatenate(pooled_p_te_B)
    t_bar      = float(np.mean(thresholds_B))

    row_b = threshold_metrics(pooled_y, pooled_p_B,
                               threshold=np.mean(thresholds_B))
    # Re-apply each fold's own threshold for exact pooling
    tp_b = fp_b = tn_b = fn_b = 0
    for y_te, p_te, t_s in zip(pooled_y_te, pooled_p_te_B, thresholds_B):
        m = threshold_metrics(y_te, p_te, t_s)
        tp_b += m["tp"]; fp_b += m["fp"]
        tn_b += m["tn"]; fn_b += m["fn"]

    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
    rec_b  = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
    spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
    f1_b   = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0

    row_b = dict(tp=tp_b, fp=fp_b, tn=tn_b, fn=fn_b,
                 precision=prec_b, recall=rec_b, specificity=spec_b, f1=f1_b,
                 ap_score=ap,
                 threshold_rule=f"95%-spec",
                 threshold_value=round(t_bar, 3))

    # ── PR curve data (pooled) ────────────────────────────────────────────────
    prec_arr, rec_arr, _ = precision_recall_curve(all_y, all_p)

    for row in (row_a, row_b):
        row["name"] = name

    print(f"    t=0.50  prec={row_a['precision']:.3f} rec={row_a['recall']:.3f} "
          f"spec={row_a['specificity']:.3f} f1={row_a['f1']:.3f}")
    print(f"    95%spec prec={row_b['precision']:.3f} rec={row_b['recall']:.3f} "
          f"spec={row_b['specificity']:.3f} f1={row_b['f1']:.3f}  "
          f"(mean threshold={t_bar:.3f})")

    return row_a, row_b, prec_arr, rec_arr


# ── PR curve figure ─────────────────────────────────────────────────────────────

PALETTE = {
    "Shannon only":                   ("#a6cee3", "--"),
    "Topology only":                  ("#ff7f00", ":"),
    "BC-PCoA(10)":                    ("#b15928", "-."),
    "Aitchison-PCoA(10)":            ("#1f78b4", "-"),
    "Aitchison-PCoA + Topo + Shannon": ("#e31a1c", "-"),
}


def plot_pr_curves(curve_data, prevalence, out_path):
    """Single-panel precision-recall figure.

    curve_data : list of (canonical_name, label_with_ap, prec_array, rec_array)
    prevalence : scalar — 'no-skill' baseline (horizontal line)
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for canonical, label, prec, rec in curve_data:
        colour, ls = PALETTE.get(canonical, ("#555555", "-"))
        ax.plot(rec, prec, color=colour, lw=1.8, ls=ls, label=label)

    # No-skill baseline
    ax.axhline(prevalence, color="#aaaaaa", lw=0.9, ls=":",
               label=f"No-skill baseline (prevalence = {prevalence:.2f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision (PPV)", fontsize=10)
    ax.set_title(
        "Precision–Recall Curves for IBD Classification\n"
        "Logistic Regression · 5-fold CV · AGP ($n=3{,}249$, prevalence ≈ 5\\%)",
        fontsize=9.5,
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"PR figure: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print("Loading data …")
    clr80, relabund80, X_shannon, X_topo, y = load_data()
    prevalence = float(y.mean())

    # 2. Precompute distance matrices
    print("Computing Aitchison distances …")
    D_ait = cdist(clr80, clr80, "euclidean").astype(np.float32)
    print("Computing Bray–Curtis distances …")
    D_bc  = cdist(relabund80.astype(np.float64),
                  relabund80.astype(np.float64), "braycurtis").astype(np.float32)

    # 3. Composite static feature matrices
    X_topo_sh = np.hstack([X_topo, X_shannon])

    # 4. CV splitter — identical seed to benchmark
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    # 5. Feature set definitions
    feature_sets = [
        dict(name="Shannon only",
             extra_X=X_shannon, use_ait=False, use_bc=False),
        dict(name="Topology only",
             extra_X=X_topo,    use_ait=False, use_bc=False),
        dict(name="BC-PCoA(10)",
             extra_X=None,      use_ait=False, use_bc=True),
        dict(name="Aitchison-PCoA(10)",
             extra_X=None,      use_ait=True,  use_bc=False),
        dict(name="Aitchison-PCoA + Topo + Shannon",
             extra_X=X_topo_sh, use_ait=True,  use_bc=False),
    ]

    rows_a, rows_b, curve_data = [], [], []

    for fs in feature_sets:
        name = fs["name"]
        ra, rb, prec_arr, rec_arr = evaluate_feature_set(
            name=name,
            X_blocks_fn=None,
            y=y,
            cv=cv,
            D_ait=D_ait,
            D_bc=D_bc,
            use_ait=fs["use_ait"],
            use_bc=fs["use_bc"],
            extra_X=fs["extra_X"],
        )
        rows_a.append(ra); rows_b.append(rb)
        curve_data.append((name, prec_arr, rec_arr))

    # 6. Assemble table with AP score labels for figure
    # Re-label with AP for figure legend
    labelled_curves = []
    for (name, prec, rec), ra in zip(curve_data, rows_a):
        label = f"{name}  (AP={ra['ap_score']:.3f})"
        labelled_curves.append((label, prec, rec))

    # 7. Save CSV
    all_rows = []
    for ra, rb in zip(rows_a, rows_b):
        for row in (ra, rb):
            all_rows.append({
                "feature_set":    row["name"],
                "threshold_rule": row["threshold_rule"],
                "threshold_value": row["threshold_value"],
                "precision":      row["precision"],
                "recall":         row["recall"],
                "specificity":    row["specificity"],
                "f1":             row["f1"],
                "ap_score":       row["ap_score"],
                "tp": row["tp"], "fp": row["fp"],
                "tn": row["tn"], "fn": row["fn"],
            })
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "precision_recall_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics table: {csv_path}")

    cols = ["feature_set", "threshold_rule", "precision", "recall",
            "specificity", "f1", "ap_score"]
    print(df[cols].to_string(index=False))

    # 8. PR curve figure
    # Build (canonical_name, ap_label, prec_arr, rec_arr) tuples
    full_curve_data = [
        (name, f"{name}  (AP={ra['ap_score']:.3f})", prec, rec)
        for (name, prec, rec), ra in zip(curve_data, rows_a)
    ]
    fig_path = os.path.join(FIGURE_DIR, "precision_recall_curves.png")
    plot_pr_curves(full_curve_data, prevalence, fig_path)


if __name__ == "__main__":
    main()
