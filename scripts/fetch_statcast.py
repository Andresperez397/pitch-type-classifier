"""Download one regular season of public Statcast pitch data from Baseball Savant, one day per request.

Savant caps a single CSV export at 25,000 rows, so multi-day windows silently truncate;
daily requests stay well under the cap, and each day is checked against that cap. After the
download, the set of games is checked against MLB's official schedule (every final regular-season
game must be present, and no others).
"""

from __future__ import annotations

import argparse
import io
import json
import ssl
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd

URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
FINAL_STATES = {"Final", "Completed Early", "Game Over"}
KEEP = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "player_name",
    "p_throws",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "arm_angle",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "game_type",
]
CAP = 25_000
SSL = ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    last = None
    for attempt in range(4):
        try:
            return urlopen(req, timeout=120, context=SSL).read().decode("utf-8-sig")
        except Exception as e:  # noqa: BLE001 - retry transient network errors
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"request failed after 4 attempts: {url}") from last


def fetch_day(d: date) -> pd.DataFrame:
    q = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "hfGT": "R|",
        "game_date_gt": d.isoformat(),
        "game_date_lt": d.isoformat(),
    }
    raw = _get(f"{URL}?{urlencode(q)}")
    if raw.lstrip().startswith("<"):
        # An HTML page is an error or rate-limit response, never "no games": fail loudly.
        raise RuntimeError(f"{d}: Savant returned HTML instead of CSV")
    if not raw.strip():
        return pd.DataFrame(columns=KEEP)
    df = pd.read_csv(io.StringIO(raw), low_memory=False)
    if len(df) >= CAP:
        raise RuntimeError(f"{d}: {len(df)} rows hit the Savant cap; split the request")
    missing = set(KEEP) - set(df.columns)
    if len(df) and missing:
        raise RuntimeError(f"{d}: expected columns missing: {sorted(missing)}")
    return df.reindex(columns=KEEP)


def official_final_games(season: int) -> set[int]:
    q = {"sportId": 1, "season": season, "gameType": "R"}
    sched = json.loads(_get(f"{SCHEDULE}?{urlencode(q)}"))
    return {
        g["gamePk"]
        for day in sched["dates"]
        for g in day["games"]
        if g["status"]["detailedState"] in FINAL_STATES
    }


def verify_complete(full: pd.DataFrame, season: int) -> None:
    want, have = official_final_games(season), set(full["game_pk"].unique())
    missing, extra = want - have, have - want
    print(
        f"schedule check: {len(want)} final games, {len(have)} downloaded, "
        f"{len(missing)} missing, {len(extra)} extra"
    )
    if missing or extra:
        raise RuntimeError(
            f"incomplete download; missing e.g. {sorted(missing)[:5]}, extra e.g. {sorted(extra)[:5]}"
        )


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
    full = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))], ignore_index=True)
    full = full[full["game_type"] == "R"]
    verify_complete(full, a.season)
    full.to_parquet(out_dir.parent / f"statcast_{a.season}.parquet", index=False)
    print("total", len(full), "games", full["game_pk"].nunique(), "pitchers", full["pitcher"].nunique())


if __name__ == "__main__":
    main()
