import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchtype import features as F  # noqa: E402


def _raw(n=120, hand="R", pitcher=1, label="FF"):
    rng = np.random.default_rng(pitcher)
    return pd.DataFrame({
        "pitcher": pitcher, "p_throws": hand, "pitch_type": label,
        "release_speed": rng.normal(94, 1, n), "release_spin_rate": rng.normal(2300, 50, n),
        "spin_axis": np.full(n, 200.0), "pfx_x": np.full(n, -0.7), "pfx_z": np.full(n, 1.3),
        "release_pos_x": np.full(n, -2.0), "release_pos_z": np.full(n, 5.8),
        "release_extension": np.full(n, 6.3), "arm_angle": np.full(n, 40.0)})


def test_mirroring_makes_lefty_match_righty():
    r = F.absolute_features(_raw(hand="R"))
    lraw = _raw(hand="L")
    lraw["pfx_x"] *= -1
    lraw["release_pos_x"] *= -1
    lraw["spin_axis"] = 360 - lraw["spin_axis"]
    l = F.absolute_features(lraw)
    for c in ["hb_in", "rel_side", "axis_deg", "axis_sin", "axis_cos"]:
        assert np.allclose(r[c], l[c]), c


def test_arm_side_is_positive():
    # A right-hander's fastball runs toward third base from the catcher's view (pfx_x < 0).
    f = F.absolute_features(_raw(hand="R"))
    assert (f["hb_in"] > 0).all() and (f["rel_side"] > 0).all()


def test_clean_merges_drops_and_logs():
    raw = pd.concat([_raw(label="FF"), _raw(n=5, label="PO"), _raw(n=5, label="CS"),
                     _raw(n=50, pitcher=2)])
    df, log = F.clean(raw)
    assert set(df["label"]) == {"FF", "CU"}
    assert log["dropped_label"] == 5
    assert log["dropped_low_volume_pitchers"] == 50
    assert log["rows_out"] == 125


def test_relative_features_ignore_labels_and_wrap_axis():
    raw = _raw()
    raw.loc[:9, "spin_axis"] = 10.0  # near the 0/360 wrap
    feat = F.absolute_features(raw)
    a = F.relative_features(feat, raw["pitcher"])
    raw2 = raw.assign(pitch_type="SL")
    b = F.relative_features(F.absolute_features(raw2), raw2["pitcher"])
    pd.testing.assert_frame_equal(a, b)
    assert a["d_axis"].between(-180, 180).all()
    assert a["d_velo"].abs().max() < 10
