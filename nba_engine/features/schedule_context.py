"""Rest, back-to-backs, travel and time-zone features, computed only from past games."""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Approximate arena coordinates and local time zones (current arenas).
ARENAS: dict[str, tuple[float, float, str]] = {
    "ATL": (33.757, -84.396, "America/New_York"),
    "BOS": (42.366, -71.062, "America/New_York"),
    "BKN": (40.683, -73.976, "America/New_York"),
    "CHA": (35.225, -80.839, "America/New_York"),
    "CHI": (41.881, -87.674, "America/Chicago"),
    "CLE": (41.496, -81.688, "America/New_York"),
    "DAL": (32.790, -96.810, "America/Chicago"),
    "DEN": (39.749, -105.008, "America/Denver"),
    "DET": (42.341, -83.055, "America/New_York"),
    "GSW": (37.768, -122.388, "America/Los_Angeles"),
    "HOU": (29.751, -95.362, "America/Chicago"),
    "IND": (39.764, -86.155, "America/Indiana/Indianapolis"),
    "LAC": (33.945, -118.341, "America/Los_Angeles"),
    "LAL": (34.043, -118.267, "America/Los_Angeles"),
    "MEM": (35.138, -90.051, "America/Chicago"),
    "MIA": (25.781, -80.188, "America/New_York"),
    "MIL": (43.045, -87.917, "America/Chicago"),
    "MIN": (44.979, -93.276, "America/Chicago"),
    "NOP": (29.949, -90.082, "America/Chicago"),
    "NYK": (40.751, -73.993, "America/New_York"),
    "OKC": (35.463, -97.515, "America/Chicago"),
    "ORL": (28.539, -81.384, "America/New_York"),
    "PHI": (39.901, -75.172, "America/New_York"),
    "PHX": (33.446, -112.071, "America/Phoenix"),
    "POR": (45.532, -122.667, "America/Los_Angeles"),
    "SAC": (38.580, -121.500, "America/Los_Angeles"),
    "SAS": (29.427, -98.437, "America/Chicago"),
    "TOR": (43.643, -79.379, "America/Toronto"),
    "UTA": (40.768, -111.901, "America/Denver"),
    "WAS": (38.898, -77.021, "America/New_York"),
}
HIGH_ALTITUDE = {"DEN", "UTA"}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def _utc_offset_h(tz: str, ts: pd.Timestamp) -> float:
    return ts.tz_convert(ZoneInfo(tz)).utcoffset().total_seconds() / 3600


def schedule_features(games: pd.DataFrame) -> pd.DataFrame:
    """Per game: rest days, back-to-back, 3-in-4, travel km and time-zone shift for each side.
    Neutral-site games have no known arena, so travel is 0 and location is unchanged."""
    g = games.sort_values("tip_utc").reset_index(drop=True)
    last_date: dict[str, pd.Timestamp] = {}
    last_loc: dict[str, str | None] = {}
    recent: dict[str, list[pd.Timestamp]] = {}
    rows = []
    for r in g.itertuples(index=False):
        local_day = r.tip_utc.tz_convert("America/New_York").normalize()
        loc = None if r.neutral else r.home
        feats = {"game_id": r.game_id}
        for side, team in (("home", r.home), ("away", r.away)):
            prev = last_date.get(team)
            rest = 7 if prev is None else min(7, (local_day - prev).days)
            window = [d for d in recent.get(team, []) if (local_day - d).days <= 3]
            prev_loc = last_loc.get(team, team)
            travel = tz_shift = 0.0
            if loc is not None and prev_loc is not None and prev is not None:
                travel = haversine_km(ARENAS[prev_loc][:2], ARENAS[loc][:2])
                tz_shift = (_utc_offset_h(ARENAS[loc][2], r.tip_utc)
                            - _utc_offset_h(ARENAS[prev_loc][2], r.tip_utc))
            feats |= {
                f"{side}_rest": rest,
                f"{side}_b2b": int(rest == 1),
                f"{side}_3in4": int(len(window) >= 2),
                f"{side}_travel_km": travel,
                f"{side}_tz_shift": tz_shift,
            }
            last_date[team] = local_day
            if loc is not None:
                last_loc[team] = loc
            recent[team] = window + [local_day]
        feats["altitude_home"] = int(not r.neutral and r.home in HIGH_ALTITUDE)
        rows.append(feats)
    out = pd.DataFrame(rows)
    out["rest_diff"] = np.clip(out["home_rest"], 0, 4) - np.clip(out["away_rest"], 0, 4)
    return out
