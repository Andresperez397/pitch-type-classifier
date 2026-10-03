import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchtype import evaluate as E  # noqa: E402


def test_classify_errors_majority_rules():
    y_true = np.array(["SL", "SL", "SL"])
    y_pred = np.array(["FC", "FC", "FC"])
    neigh = np.array([["SL"] * 30 + ["FC"] * 20,   # MLB label has the majority -> model error
                      ["FC"] * 40 + ["SL"] * 10,   # model label has the majority -> ambiguous
                      ["CU"] * 20 + ["SL"] * 15 + ["FC"] * 15])  # neither -> mixed
    out = E.classify_errors(y_true, y_pred, neigh)
    assert list(out["kind"]) == ["model_error", "ambiguous_label", "mixed"]


def test_neighbor_labels_returns_nearest():
    X_ref = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    y_ref = np.array(["A", "A", "B", "B"])
    got = E.neighbor_labels(X_ref, y_ref, np.array([[0.05, 0.0], [10.05, 10.0]]), k=2)
    assert (got[0] == "A").all() and (got[1] == "B").all()


def test_gbm_has_no_hidden_early_stopping():
    assert E.make_model("gbm").get_params()["early_stopping"] is False
