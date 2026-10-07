"""Train on 2025, score 2026: does the result hold up in a season the model has never seen?

    python scripts/external_season.py

Needs data/raw/statcast_2025.parquet and statcast_2026.parquet (scripts/fetch_statcast.py --season YEAR).
The model is the plan's best one (logistic regression on pitcher-relative features), fit on a 250,000-pitch
sample of 2025. The headline number is for pitchers who did not pitch in 2025 at all; returning pitchers are
shown for contrast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pitchtype import evaluate as E  # noqa: E402
from pitchtype import external as X  # noqa: E402
from pitchtype import features as F  # noqa: E402

OUT = ROOT / "reports" / "tables"
LABELS = F.KEEP_LABELS
COLS = F.ABSOLUTE + F.RELATIVE
TRAIN_CAP = 250_000


def prepare(season: int):
    raw = pd.read_parquet(ROOT / "data" / "raw" / f"statcast_{season}.parquet")
    df, log = F.clean(raw)
    feats = F.relative_features(F.absolute_features(df), df["pitcher"])
    return df.reset_index(drop=True), feats.reset_index(drop=True), log


def main() -> None:
    d25, f25, log25 = prepare(2025)
    d26, f26, log26 = prepare(2026)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(d25), size=min(TRAIN_CAP, len(d25)), replace=False)
    model = E.make_model("logistic").fit(f25[COLS].to_numpy()[sub], d25["label"].to_numpy()[sub])
    classes = list(model.classes_)
    proba = model.predict_proba(f26[COLS].to_numpy())
    masks = X.split_by_history(d26, set(d25["pitcher"].unique()))
    out = {"train_season": 2025, "test_season": 2026, "cleaning_2026": log26, "groups": {}}
    for name, mask in masks.items():
        m = mask.to_numpy()
        res = X.score(d26["label"].to_numpy()[m], proba[m], classes, LABELS)
        res["pitchers"] = int(d26.loc[m, "pitcher"].nunique())
        out["groups"][name] = res
    # Label mix by season, to see whether the reference labels themselves moved.
    mix = pd.DataFrame(
        {"2025": d25["label"].value_counts(normalize=True), "2026": d26["label"].value_counts(normalize=True)}
    ).fillna(0)
    out["label_share"] = {k: {y: round(float(v), 4) for y, v in row.items()} for k, row in mix.iterrows()}
    out["pitches"] = {"2025": int(len(d25)), "2026": int(len(d26))}
    out["pitchers"] = {"2025": int(d25["pitcher"].nunique()), "2026": int(d26["pitcher"].nunique())}
    (OUT / "external_season_2026.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({g: r["per_class_f1"] for g, r in out["groups"].items()}).round(3).to_csv(
        OUT / "external_season_2026_per_class_f1.csv"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "cleaning_2026"}, indent=2))
    _ = log25


if __name__ == "__main__":
    main()
