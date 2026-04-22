"""Tests for src.analysis.ml — ML classifiers on topological features."""

import numpy as np
import pytest

from src.analysis.ml import classify_with_topological_features


@pytest.fixture
def classification_data():
    """Simple separable binary classification dataset."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(0, 1, (30, 4)),
        rng.normal(2, 1, (30, 4)),
    ])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


def test_rf_returns_dict(classification_data):
    """Random Forest classifier should return a dict with expected keys."""
    X, y = classification_data
    result = classify_with_topological_features(X, y, classifier="rf", n_splits=3)
    assert "mean_accuracy" in result
    assert "std_accuracy" in result
    assert "fold_scores" in result


def test_svm_returns_dict(classification_data):
    """SVM classifier should return a dict with expected keys."""
    X, y = classification_data
    result = classify_with_topological_features(X, y, classifier="svm", n_splits=3)
    assert "mean_accuracy" in result


def test_accuracy_range(classification_data):
    """Accuracy should be in [0, 1]."""
    X, y = classification_data
    result = classify_with_topological_features(X, y, classifier="rf", n_splits=3)
    assert 0 <= result["mean_accuracy"] <= 1
    assert result["std_accuracy"] >= 0


def test_fold_count(classification_data):
    """Number of fold scores should match n_splits."""
    X, y = classification_data
    result = classify_with_topological_features(X, y, classifier="rf", n_splits=5)
    assert len(result["fold_scores"]) == 5


def test_separable_data_high_accuracy(classification_data):
    """Well-separated data should give reasonable accuracy."""
    X, y = classification_data
    result = classify_with_topological_features(X, y, classifier="rf", n_splits=3)
    assert result["mean_accuracy"] > 0.7


def test_unknown_classifier():
    """Should raise ValueError for unknown classifier type."""
    X = np.random.default_rng(0).random((20, 3))
    y = np.array([0] * 10 + [1] * 10)
    with pytest.raises(ValueError, match="Unknown classifier"):
        classify_with_topological_features(X, y, classifier="xgboost")
