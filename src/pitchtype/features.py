"""Cleaning, handedness mirroring and label-free pitcher-relative features (ANALYSIS_PLAN.md 2-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEEP_LABELS = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]
MERGE = {"CS": "CU"}
MIN_PITCHES = 100
REF_QUANTILE = 0.90

RAW_NEEDED = [
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "arm_angle",
]
ABSOLUTE = [
    "velo",
    "spin",
    "axis_sin",
    "axis_cos",
    "hb_in",
    "ivb_in",
    "rel_side",
    "rel_height",
    "extension",
    "arm_angle",
]
RELATIVE = ["d_velo", "d_hb", "d_ivb", "d_spin", "d_axis"]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df["label"] = df["pitch_type"].replace(MERGE)
    log = {"rows_in": len(df)}
    df = df[df["label"].isin(KEEP_LABELS)]
    log["dropped_label"] = log["rows_in"] - len(df)
    n = len(df)
    df = df.dropna(subset=RAW_NEEDED)
    log["dropped_missing"] = n - len(df)
    counts = df.groupby("pitcher")["label"].transform("size")
    n = len(df)
    df = df[counts >= MIN_PITCHES]
    log["dropped_low_volume_pitchers"] = n - len(df)
    log["rows_out"] = len(df)
    log["pitchers_out"] = int(df["pitcher"].nunique())
    return df.reset_index(drop=True), log


def absolute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Describe every pitch from the pitcher's side, with arm side positive.

    Statcast's x axis is the catcher's view (third-base side negative), so a right-hander's
    arm-side movement and release side are negative. Multiplying by -1 for right-handers and +1
    for left-handers puts both hands in one frame where arm side is positive. Spin axis is
    mirrored for left-handers (360 - axis) into the right-hander frame.
    """
    lefty = (df["p_throws"] == "L").to_numpy()
    sign = np.where(lefty, 1.0, -1.0)
    axis = np.where(lefty, 360.0 - df["spin_axis"], df["spin_axis"])
    out = pd.DataFrame(index=df.index)
    out["velo"] = df["release_speed"]
    out["spin"] = df["release_spin_rate"]
    out["axis_deg"] = axis
    out["axis_sin"] = np.sin(np.deg2rad(axis))
    out["axis_cos"] = np.cos(np.deg2rad(axis))
    out["hb_in"] = df["pfx_x"] * 12 * sign
    out["ivb_in"] = df["pfx_z"] * 12
    out["rel_side"] = df["release_pos_x"] * sign
    out["rel_height"] = df["release_pos_z"]
    out["extension"] = df["release_extension"]
    out["arm_angle"] = df["arm_angle"]
    return out


def relative_features(feat: pd.DataFrame, pitcher: pd.Series) -> pd.DataFrame:
    """Gap from each pitcher's own hardest pitches (at or above that pitcher's 90th-percentile velocity).

    Uses no labels, so it is available for a pitcher the model has never seen. Call it on one
    split at a time so the reference only uses pitches from that split.
    """
    f = feat.copy()
    f["pitcher"] = pitcher.to_numpy()
    cut = f.groupby("pitcher")["velo"].transform(lambda v: v.quantile(REF_QUANTILE))
    hard = f[f["velo"] >= cut]
    ref = hard.groupby("pitcher").agg(
        r_velo=("velo", "mean"),
        r_hb=("hb_in", "mean"),
        r_ivb=("ivb_in", "mean"),
        r_spin=("spin", "mean"),
        r_sin=("axis_sin", "mean"),
        r_cos=("axis_cos", "mean"),
    )
    r = ref.loc[f["pitcher"]].set_index(f.index)
    ref_axis = np.rad2deg(np.arctan2(r["r_sin"], r["r_cos"])) % 360
    out = feat.copy()
    out["d_velo"] = f["velo"] - r["r_velo"]
    out["d_hb"] = f["hb_in"] - r["r_hb"]
    out["d_ivb"] = f["ivb_in"] - r["r_ivb"]
    out["d_spin"] = f["spin"] - r["r_spin"]
    out["d_axis"] = (f["axis_deg"] - ref_axis + 180) % 360 - 180  # signed circular gap
    return out
