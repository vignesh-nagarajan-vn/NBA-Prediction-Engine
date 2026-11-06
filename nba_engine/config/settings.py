"""Runtime settings, loaded from environment variables / .env (prefix NBA_)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

KALSHI_HOSTS = {
    "prod": {
        "rest": "https://external-api.kalshi.com/trade-api/v2",
        "ws": "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
    },
    "demo": {
        "rest": "https://external-api.demo.kalshi.co/trade-api/v2",
        "ws": "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2",
    },
}

NBA_GAME_SERIES = "KXNBAGAME"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NBA_", env_file=".env", extra="ignore")

    kalshi_env: Literal["prod", "demo"] = "prod"
    kalshi_key_id: str = ""
    kalshi_private_key_path: Path | None = None
    kalshi_max_rps: float = Field(default=4.0, gt=0)  # public limit is ~5/s

    data_dir: Path = Path("data")

    # Recorder cadence (seconds)
    book_poll_live_s: float = 2.0  # REST-mode book polling while any game is live
    book_poll_idle_s: float = 30.0  # REST-mode book polling otherwise
    trades_poll_s: float = 5.0  # REST-mode trade polling for live games
    espn_live_poll_s: float = 4.0  # play-by-play polling for live games
    schedule_refresh_s: float = 300.0  # re-discover games and markets
    pregame_window_h: float = 36.0  # record markets for games tipping within this window

    @property
    def rest_base(self) -> str:
        return KALSHI_HOSTS[self.kalshi_env]["rest"]

    @property
    def ws_url(self) -> str:
        return KALSHI_HOSTS[self.kalshi_env]["ws"]

    @property
    def has_kalshi_auth(self) -> bool:
        return bool(self.kalshi_key_id and self.kalshi_private_key_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()
