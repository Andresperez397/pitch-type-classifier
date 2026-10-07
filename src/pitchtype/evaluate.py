"""Models and the label-ambiguity check (ANALYSIS_PLAN.md sections 4-5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "gbm":
        # Fixed settings from the plan. early_stopping=False matters: scikit-learn otherwise
        # turns on early stopping for n > 10,000 using a random 10% of *pitches*, which would
        # tune the number of trees on pitchers the model has already seen.
        return HistGradientBoostingClassifier(
            max_depth=8, learning_rate=0.1, max_iter=300, early_stopping=False, random_state=0
        )
    raise ValueError(f"unknown model kind: {kind}")


def classify_errors(y_true: np.ndarray, y_pred: np.ndarray, neighbor_labels: np.ndarray) -> pd.DataFrame:
    """Label each misclassified pitch by what its nearest other-pitcher neighbours are called.

    neighbor_labels has one row per misclassified pitch and one column per neighbour.
    - model_error: most neighbours carry MLB's label (the model missed a clear case)
    - ambiguous_label: most neighbours carry the model's label (the pitch looks like what
      other pitchers call the predicted type)
    - mixed: neither label has a majority
    """
    share_true = (neighbor_labels == y_true[:, None]).mean(1)
    share_pred = (neighbor_labels == y_pred[:, None]).mean(1)
    kind = np.select([share_true > 0.5, share_pred > 0.5], ["model_error", "ambiguous_label"], "mixed")
    return pd.DataFrame(
        {"true": y_true, "pred": y_pred, "share_true": share_true, "share_pred": share_pred, "kind": kind}
    )


def neighbor_labels(X_ref: np.ndarray, y_ref: np.ndarray, X_query: np.ndarray, k: int = 50) -> np.ndarray:
    """Labels of the k nearest reference pitches (standardized on the reference set)."""
    scaler = StandardScaler().fit(X_ref)
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(X_ref))
    _, idx = nn.kneighbors(scaler.transform(X_query))
    return y_ref[idx]
