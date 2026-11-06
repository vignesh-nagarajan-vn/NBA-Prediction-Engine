"""Live play-by-play from ESPN, with new-play detection and dead-ball classification.

Dead balls matter for in-game trading (strategy S3): with a 10-30s feed lag we only act while
the game state is frozen (timeouts, period breaks, free throws, reviews), so a stale feed is
still an accurate picture of the game.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DEAD_BALL_PATTERNS = [
    r"timeout",
    r"end period",
    r"end game",
    r"free throw",
    r"review",
    r"challenge",
    r"technical",
    r"flagrant",
    r"ejection",
    r"delay",
    r"jumpball",
    r"jump ball",
    r"injury",
]
_DEAD_BALL_RE = re.compile("|".join(_DEAD_BALL_PATTERNS), re.IGNORECASE)

# Stoppages long enough to act in (seconds of expected frozen state).
_LONG_STOPPAGE_RE = re.compile(r"timeout|end period|review|challenge|injury", re.IGNORECASE)


def is_dead_ball(play_type: str) -> bool:
    return bool(_DEAD_BALL_RE.search(play_type or ""))


def is_long_stoppage(play_type: str) -> bool:
    return bool(_LONG_STOPPAGE_RE.search(play_type or ""))


@dataclass
class Play:
    espn_game_id: str
    play_id: str
    sequence: int
    period: int
    clock: str
    type: str
    text: str
    home_score: int
    away_score: int
    scoring: bool
    score_value: int
    team_id: str | None
    wallclock: str | None
    recv_ms: int

    @property
    def dead_ball(self) -> bool:
        return is_dead_ball(self.type)

    @property
    def long_stoppage(self) -> bool:
        return is_long_stoppage(self.type)

    def to_record(self) -> dict:
        d = self.__dict__.copy()
        d["dead_ball"] = self.dead_ball
        d["long_stoppage"] = self.long_stoppage
        return d


def parse_plays(espn_game_id: str, summary: dict, recv_ms: int) -> list[Play]:
    plays = []
    for p in summary.get("plays", []) or []:
        plays.append(Play(
            espn_game_id=espn_game_id,
            play_id=str(p.get("id")),
            sequence=int(p.get("sequenceNumber") or 0),
            period=int((p.get("period") or {}).get("number") or 0),
            clock=str((p.get("clock") or {}).get("displayValue") or ""),
            type=str((p.get("type") or {}).get("text") or "").replace("\n", " "),
            text=str(p.get("text") or ""),
            home_score=int(p.get("homeScore") or 0),
            away_score=int(p.get("awayScore") or 0),
            scoring=bool(p.get("scoringPlay")),
            score_value=int(p.get("scoreValue") or 0),
            team_id=str((p.get("team") or {}).get("id")) if p.get("team") else None,
            wallclock=p.get("wallclock"),
            recv_ms=recv_ms,
        ))
    return plays


class PlayTracker:
    """Remembers which plays were already seen per game; returns only new ones."""

    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}

    def new_plays(self, plays: list[Play]) -> list[Play]:
        out = []
        for p in plays:
            seen = self._seen.setdefault(p.espn_game_id, set())
            if p.play_id not in seen:
                seen.add(p.play_id)
                out.append(p)
        return out
