import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchtype import evaluate as E  # noqa: E402


def test_classify_errors_majority_rules():
    y_true = np.array(["SL", "SL", "SL"])
    y_pred = np.array(["FC", "FC", "FC"])
    neigh = np.array(
        [
            ["SL"] * 30 + ["FC"] * 20,  # MLB label has the majority -> model error
            ["FC"] * 40 + ["SL"] * 10,  # model label has the majority -> ambiguous
            ["CU"] * 20 + ["SL"] * 15 + ["FC"] * 15,
        ]
    )  # neither -> mixed
    out = E.classify_errors(y_true, y_pred, neigh)
    assert list(out["kind"]) == ["model_error", "ambiguous_label", "mixed"]


def test_neighbor_labels_returns_nearest():
    X_ref = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    y_ref = np.array(["A", "A", "B", "B"])
    got = E.neighbor_labels(X_ref, y_ref, np.array([[0.05, 0.0], [10.05, 10.0]]), k=2)
    assert (got[0] == "A").all() and (got[1] == "B").all()


def test_gbm_has_no_hidden_early_stopping():
    assert E.make_model("gbm").get_params()["early_stopping"] is False


def test_split_by_history_separates_new_from_returning_pitchers():
    import pandas as pd

    from pitchtype.external import split_by_history

    test = pd.DataFrame({"pitcher": [1, 1, 2, 3, 3, 3]})
    masks = split_by_history(test, {1, 9})
    assert masks["returning_pitchers"].tolist() == [True, True, False, False, False, False]
    assert masks["new_pitchers"].sum() == 4 and masks["all_pitchers"].all()
    assert (masks["new_pitchers"] ^ masks["returning_pitchers"]).all()


def test_score_returns_accuracy_and_per_class_f1_for_the_given_labels():
    import numpy as np

    from pitchtype.external import score

    y = np.array(["FF", "FF", "SL", "SL"])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])  # classes: FF, SL
    r = score(y, proba, ["FF", "SL"], ["FF", "SL"])
    assert r["accuracy"] == 0.75 and r["n_pitches"] == 4
    assert abs(r["per_class_f1"]["SL"] - 0.8) < 1e-9
