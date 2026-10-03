"""Download one regular season of public Statcast pitch data from Baseball Savant, one day per request.

Savant caps a single CSV export at 25,000 rows, so multi-day windows silently truncate;
daily requests stay well under the cap, and each day is checked against that cap.
"""
from __future__ import annotations

import argparse
import io
import ssl
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd

URL = "https://baseballsavant.mlb.com/statcast_search/csv"
KEEP = ["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher", "player_name",
        "p_throws", "pitch_type", "release_speed", "release_spin_rate", "spin_axis",
        "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
        "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "game_type"]
CAP = 25_000
SSL = ssl.create_default_context(cafile=certifi.where())


def fetch_day(d: date) -> pd.DataFrame:
    q = {"all": "true", "type": "details", "player_type": "pitcher", "hfGT": "R|",
         "game_date_gt": d.isoformat(), "game_date_lt": d.isoformat()}
    req = Request(f"{URL}?{urlencode(q)}", headers={"User-Agent": "Mozilla/5.0 (research)"})
    for attempt in range(4):
        try:
            raw = urlopen(req, timeout=120, context=SSL).read().decode("utf-8-sig")
            break
        except Exception:  # noqa: BLE001 - retry transient network errors
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"failed {d}")
    if not raw.strip() or raw.lstrip().startswith("<"):
        return pd.DataFrame(columns=KEEP)
    df = pd.read_csv(io.StringIO(raw), low_memory=False)
    if len(df) >= CAP:
        raise RuntimeError(f"{d}: {len(df)} rows hit the Savant cap; split the request")
    return df[[c for c in KEEP if c in df.columns]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    start = date.fromisoformat(a.start or f"{a.season}-03-15")
    end = date.fromisoformat(a.end or f"{a.season}-10-01")
    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / f"days_{a.season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = start
    while d <= end:
        f = out_dir / f"{d.isoformat()}.parquet"
        if not f.exists():
            day = fetch_day(d)
            day.to_parquet(f, index=False)
            print(d, len(day), flush=True)
            time.sleep(1.0)
        d += timedelta(days=1)
    full = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))],
                     ignore_index=True)
    full = full[full["game_type"] == "R"]
    full.to_parquet(out_dir.parent / f"statcast_{a.season}.parquet", index=False)
    print("total", len(full), "games", full["game_pk"].nunique(), "pitchers", full["pitcher"].nunique())


if __name__ == "__main__":
    main()
