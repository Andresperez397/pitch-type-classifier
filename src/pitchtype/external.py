"""Out-of-season check: train on one season, score the next, splitting pitchers by prior history."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def split_by_history(test: pd.DataFrame, train_pitchers: set) -> dict[str, pd.Series]:
    """Row masks for the later season: pitchers absent ('new') or present ('returning') before."""
    seen = test["pitcher"].isin(train_pitchers)
    return {
        "new_pitchers": ~seen,
        "returning_pitchers": seen,
        "all_pitchers": pd.Series(True, index=test.index),
    }


def score(y_true: np.ndarray, proba: np.ndarray, classes: list[str], labels: list[str]) -> dict:
    pred = np.array(classes)[proba.argmax(1)]
    return {
        "n_pitches": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
        "per_class_f1": {
            lab: float(v)
            for lab, v in zip(
                labels, f1_score(y_true, pred, average=None, labels=labels, zero_division=0), strict=True
            )
        },
    }
