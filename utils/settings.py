"""
Local configuration loader.
Reads config/tickers.json for ticker list and scheduler metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Settings:
    tickers: List[str]
    schedule_time: str = "06:00"
    timezone: str = "Asia/Seoul"


def _default_settings() -> Settings:
    return Settings(tickers=["NVDA"], schedule_time="06:00", timezone="Asia/Seoul")


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load ticker/settings config from config/tickers.json.
    """
    if config_path is None:
        root_dir = Path(__file__).resolve().parents[2]
        config_path = root_dir / "config" / "tickers.json"
    path_obj = Path(config_path)
    if not path_obj.exists():
        return _default_settings()
    try:
        data = json.loads(path_obj.read_text(encoding="utf-8"))
        tickers = data.get("tickers") or []
        schedule = data.get("schedule_time", "06:00")
        timezone = data.get("timezone", "Asia/Seoul")
        return Settings(tickers=tickers, schedule_time=schedule, timezone=timezone)
    except Exception:
        return _default_settings()
