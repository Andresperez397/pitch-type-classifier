"""EXPLORATORY (not in the frozen plan; see DEVIATIONS.md D5).

Is the gradient-boosted model behind logistic regression only because its fixed settings overfit?
Same 5 pitcher splits and training subsample as run_analysis.py, relative features (set B). Here
the number of boosting rounds is chosen by early stopping on a validation set of *held-out
training pitchers* (15% of training pitchers), never on pitches from pitchers the model trains on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_analysis import N_REPEATS, SEASON, SETS, TRAIN_CAP, features_for  # noqa: E402

from pitchtype import features as F  # noqa: E402


def main() -> None:
    df, _ = F.clean(pd.read_parquet(ROOT / "data" / "raw" / f"statcast_{SEASON}.parquet"))
    cols = SETS["B_relative"]
    rows = []
    splitter = GroupShuffleSplit(n_splits=N_REPEATS, test_size=0.2, random_state=0)
    for seed, (tr, te) in enumerate(splitter.split(df, groups=df["pitcher"])):
        train, test = df.iloc[tr], df.iloc[te]
        Xtr_all, Xte = features_for(train), features_for(test)
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(train), size=min(TRAIN_CAP, len(train)), replace=False)  # as main run
        tr_sub, X_sub = train.iloc[sub], Xtr_all[cols].to_numpy()[sub]
        y_sub = tr_sub["label"].to_numpy()
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        fit_i, val_i = next(inner.split(X_sub, groups=tr_sub["pitcher"]))
        model = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.1,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            scoring="loss",
            random_state=0,
        )
        model.fit(X_sub[fit_i], y_sub[fit_i], X_val=X_sub[val_i], y_val=y_sub[val_i])
        pred = model.predict(Xte[cols].to_numpy())
        yte = test["label"].to_numpy()
        rows.append(
            {
                "seed": seed,
                "n_iter": int(model.n_iter_),
                "accuracy": accuracy_score(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro"),
            }
        )
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports" / "tables" / "exploratory_tuned_gbm.csv", index=False)
    summary = out[["n_iter", "accuracy", "macro_f1"]].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    with open(ROOT / "reports" / "tables" / "exploratory_tuned_gbm_summary.json", "w") as f:
        json.dump(summary.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
