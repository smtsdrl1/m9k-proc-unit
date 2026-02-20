"""
Economic Calendar — News Kill Zone filter.
ICT concept: avoid trading during high-impact news events.
Static FOMC dates (2025-2026) + CoinGlass async fetch.
"""
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("matrix_trader.data.economic_calendar")

# ─── FOMC Meeting Dates 2025-2026 (UTC) ──────────────────────────────────────
FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
ALL_FOMC = FOMC_DATES_2025 + FOMC_DATES_2026

NEWS_KILL_WINDOW_MINUTES = 30
_CACHE: dict = {}


def _parse_fomc() -> list:
    events = []
    for d in ALL_FOMC:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").replace(
                hour=18, minute=0, tzinfo=timezone.utc  # FOMC decision ~14:00 ET = 18:00 UTC
            )
            events.append({"title": "FOMC Decision", "date": dt, "impact": "HIGH"})
        except ValueError:
            pass
    return events


def check_news_kill_zone(window_minutes: int = None) -> dict:
    """Check if current time is within a news kill zone."""
    if window_minutes is None:
        window_minutes = NEWS_KILL_WINDOW_MINUTES
    now = datetime.now(timezone.utc)
    events = _parse_fomc()

    for ev in events:
        delta = abs((ev["date"] - now).total_seconds() / 60)
        if delta <= window_minutes:
            return {
                "in_kill_zone": True,
                "event": ev["title"],
                "minutes_until": int((ev["date"] - now).total_seconds() / 60),
                "should_avoid": True,
                "confidence_modifier": -25.0,
            }

    return {
        "in_kill_zone": False,
        "event": None,
        "should_avoid": False,
        "confidence_modifier": 0.0,
    }


def get_upcoming_events(hours_ahead: int = 24) -> list:
    """Return upcoming high-impact events within timeframe."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    return [
        ev for ev in _parse_fomc()
        if now <= ev["date"] <= cutoff
    ]


async def fetch_calendar_events(session: aiohttp.ClientSession = None) -> list:
    """Attempt to fetch live events from CoinGlass (fallback to static)."""
    try:
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        try:
            url = "https://open-api.coinglass.com/api/calendar/events"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
        except Exception:
            pass
        finally:
            if close_session:
                await session.close()
    except Exception as e:
        logger.debug(f"Calendar fetch error: {e}")
    return _parse_fomc()
