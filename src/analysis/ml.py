"""ML classifiers on topological features."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def classify_with_topological_features(X, y, classifier="rf", n_splits=5):
    """Classify samples using topological features with cross-validation.

    Args:
        X: Feature matrix (samples x features).
        y: Target labels.
        classifier: Classifier type ('rf' for Random Forest, 'svm' for SVM).
        n_splits: Number of cross-validation folds.

    Returns:
        Dictionary with mean accuracy, std, and per-fold scores.
    """
    if classifier == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier == "svm":
        clf = SVC(kernel="rbf", random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "fold_scores": scores.tolist(),
    }
