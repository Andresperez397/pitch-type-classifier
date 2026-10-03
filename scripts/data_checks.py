"""Descriptive data checks quoted in the README (written to reports/tables/data_checks.json).

- coverage: games and pitches in the downloaded season
- handedness frame: four-seam medians by hand after mirroring (should match)
- reference composition: which MLB labels make up each pitcher's top-10% velocity reference
- majority-class baseline: accuracy of always predicting the most common training label on the
  same 5 held-out pitcher splits used in run_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pitchtype import features as F  # noqa: E402

SEASON = 2025


def main() -> None:
    raw = pd.read_parquet(ROOT / "data" / "raw" / f"statcast_{SEASON}.parquet")
    df, _ = F.clean(raw)
    a = F.absolute_features(df)
    out: dict = {"coverage": {"games": int(raw["game_pk"].nunique()), "pitches_raw": len(raw),
                              "pitches_clean": len(df), "pitchers_clean": int(df["pitcher"].nunique())}}

    ff = df["label"] == "FF"
    med = a[ff].groupby(df.loc[ff, "p_throws"])[["rel_side", "hb_in", "axis_deg"]].median()
    out["four_seam_medians_by_hand_after_mirroring"] = med.round(2).to_dict(orient="index")

    cut = df.groupby("pitcher")["release_speed"].transform(lambda v: v.quantile(F.REF_QUANTILE))
    ref = df[df["release_speed"] >= cut]
    comp = ref["label"].value_counts(normalize=True)
    top = ref.groupby("pitcher")["label"].agg(lambda s: s.value_counts().index[0])
    out["reference_composition"] = {k: round(float(v), 4) for k, v in comp.items()}
    out["reference_fastball_share"] = round(float(comp.reindex(["FF", "SI", "FC"]).fillna(0).sum()), 4)
    out["pitchers_with_non_fastball_reference"] = int((~top.isin(["FF", "SI", "FC"])).sum())

    accs = []
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for tr, te in splitter.split(df, groups=df["pitcher"]):
        majority = df.iloc[tr]["label"].value_counts().index[0]
        accs.append(float((df.iloc[te]["label"] == majority).mean()))
    out["majority_baseline_accuracy"] = {"per_repeat": [round(x, 4) for x in accs],
                                         "mean": round(float(np.mean(accs)), 4)}

    path = ROOT / "reports" / "tables" / "data_checks.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
