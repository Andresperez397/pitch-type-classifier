"""Run the pre-specified analysis in ANALYSIS_PLAN.md (Q1-Q3) and write tables to reports/tables/."""
from __future__ import annotations


import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pitchtype import features as F  # noqa: E402

OUT = ROOT / "reports" / "tables"
SEASON = 2025
N_REPEATS = 5
TRAIN_CAP = 250_000
LABELS = F.KEEP_LABELS
SETS = {"A_absolute": F.ABSOLUTE, "B_relative": F.ABSOLUTE + F.RELATIVE}


def make(kind: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=8, learning_rate=0.1, max_iter=300,
                                          random_state=0)


def features_for(df: pd.DataFrame) -> pd.DataFrame:
    return F.relative_features(F.absolute_features(df), df["pitcher"])


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ROOT / "data" / "raw" / f"statcast_{SEASON}.parquet")
    df, log = F.clean(raw)
    log["label_counts"] = df["label"].value_counts().to_dict()
    print(json.dumps(log, indent=2))

    rows, per_class, pitcher_rows = [], [], []
    seed0 = {}
    splitter = GroupShuffleSplit(n_splits=N_REPEATS, test_size=0.2, random_state=0)
    for seed, (tr, te) in enumerate(splitter.split(df, groups=df["pitcher"])):
        train, test = df.iloc[tr], df.iloc[te]
        Xtr_all, Xte_all = features_for(train), features_for(test)  # reference within each split
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(train), size=min(TRAIN_CAP, len(train)), replace=False)
        ytr, yte = train["label"].to_numpy()[sub], test["label"].to_numpy()
        for set_name, cols in SETS.items():
            for kind in ("logistic", "gbm"):
                model = make(kind).fit(Xtr_all[cols].to_numpy()[sub], ytr)
                proba = model.predict_proba(Xte_all[cols].to_numpy())
                classes = list(model.classes_)
                pred = np.array(classes)[proba.argmax(1)]
                rows.append({"seed": seed, "features": set_name, "model": kind,
                             "accuracy": accuracy_score(yte, pred),
                             "macro_f1": f1_score(yte, pred, average="macro"),
                             "log_loss": log_loss(yte, proba, labels=classes),
                             "n_train": len(sub), "n_test": len(yte),
                             "test_pitchers": test["pitcher"].nunique()})
                f1s = f1_score(yte, pred, average=None, labels=LABELS)
                per_class += [{"seed": seed, "features": set_name, "model": kind, "label": l,
                               "f1": f} for l, f in zip(LABELS, f1s)]
                acc_p = pd.Series(pred == yte).groupby(test["pitcher"].to_numpy()).mean()
                pitcher_rows.append({"seed": seed, "features": set_name, "model": kind,
                                     "share_pitchers_ge95": float((acc_p >= 0.95).mean()),
                                     "median_pitcher_acc": float(acc_p.median())})
                if seed == 0:
                    seed0[(set_name, kind)] = (pred, proba, classes)
                print(f"seed {seed} {set_name} {kind}: acc {rows[-1]['accuracy']:.3f} "
                      f"macroF1 {rows[-1]['macro_f1']:.3f}", flush=True)
        if seed == 0:
            test0, Xtr0, Xte0, sub0, ytr0 = test, Xtr_all, Xte_all, sub, ytr

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "repeats.csv", index=False)
    summary = res.groupby(["features", "model"])[["accuracy", "macro_f1", "log_loss"]].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    pc = pd.DataFrame(per_class)
    pc.groupby(["features", "model", "label"])["f1"].mean().unstack("label")[LABELS].to_csv(OUT / "per_class_f1.csv")
    pd.DataFrame(pitcher_rows).groupby(["features", "model"]).mean(numeric_only=True).drop(columns="seed").to_csv(OUT / "pitcher_level.csv")

    # Q2 pre-declared rule: relative wins macro-F1 in all 5 repeats (GBM and logistic separately).
    q2 = {}
    for kind in ("logistic", "gbm"):
        a = res[(res.model == kind) & (res.features == "A_absolute")].set_index("seed")["macro_f1"]
        b = res[(res.model == kind) & (res.features == "B_relative")].set_index("seed")["macro_f1"]
        q2[kind] = {"wins": int((b > a).sum()), "mean_gain": float((b - a).mean()),
                    "relative_better": bool((b > a).all())}

    # Confusion matrix (seed 0, main model, relative features).
    pred, proba, classes = seed0[("B_relative", "gbm")]
    yte0 = test0["label"].to_numpy()
    cm = confusion_matrix(yte0, pred, labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT / "confusion_seed0.csv")

    # Q3 label ambiguity: 50 nearest training neighbours (other pitchers) in standardized set B.
    cols = SETS["B_relative"]
    scaler = StandardScaler().fit(Xtr0[cols].to_numpy()[sub0])
    nn = NearestNeighbors(n_neighbors=50).fit(scaler.transform(Xtr0[cols].to_numpy()[sub0]))
    wrong = np.flatnonzero(pred != yte0)
    _, idx = nn.kneighbors(scaler.transform(Xte0[cols].to_numpy()[wrong]))
    neigh = ytr0[idx]
    share_true = (neigh == yte0[wrong][:, None]).mean(1)
    share_pred = (neigh == pred[wrong][:, None]).mean(1)
    amb = pd.DataFrame({"true": yte0[wrong], "pred": pred[wrong], "share_true": share_true,
                        "share_pred": share_pred})
    amb["kind"] = np.select([amb.share_true > 0.5, amb.share_pred > 0.5],
                            ["model_error", "ambiguous_label"], "mixed")
    pairs = (amb.groupby(["true", "pred"]).agg(n=("kind", "size"),
             model_error=("kind", lambda k: (k == "model_error").mean()),
             ambiguous=("kind", lambda k: (k == "ambiguous_label").mean()))
             .sort_values("n", ascending=False))
    pairs.to_csv(OUT / "error_pairs_seed0.csv")
    overall = amb["kind"].value_counts(normalize=True).to_dict()

    # Save seed-0 test predictions for the app and figures.
    keep = ["game_date", "pitcher", "player_name", "p_throws", "label"]
    app = pd.concat([test0[keep].reset_index(drop=True),
                     Xte0[["velo", "hb_in", "ivb_in", "spin", "d_velo"]].reset_index(drop=True)], axis=1)
    app["pred"] = pred
    app["confidence"] = proba.max(1)
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    app.to_parquet(proc / "test_predictions_seed0.parquet", index=False)

    out = {"cleaning": log, "q2": q2, "q3_error_kinds": overall, "n_errors_seed0": int(len(wrong)),
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT / "results.json", "w"), indent=2, default=float)
    print(summary.round(4).to_string())
    print(json.dumps(out, indent=2, default=float))
    print(pairs.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
