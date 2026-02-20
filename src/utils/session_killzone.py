"""
ICT Session Killzone Filter for Matrix Trader AI.
Only trade during high-liquidity market hours.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("matrix_trader.utils.session_killzone")

try:
    from src.config import (
        SESSION_FILTER_ENABLED, LONDON_KILLZONE_START, LONDON_KILLZONE_END,
        NY_KILLZONE_START, NY_KILLZONE_END, ASIA_KILLZONE_START,
        ASIA_KILLZONE_END, SESSION_MIN_QUALITY,
    )
except ImportError:
    SESSION_FILTER_ENABLED = True
    LONDON_KILLZONE_START = 2; LONDON_KILLZONE_END = 5
    NY_KILLZONE_START = 13; NY_KILLZONE_END = 16
    ASIA_KILLZONE_START = 0; ASIA_KILLZONE_END = 2
    SESSION_MIN_QUALITY = 3


SESSIONS = {
    "LONDON":   {"start": LONDON_KILLZONE_START, "end": LONDON_KILLZONE_END,   "quality": 5},
    "NY":       {"start": NY_KILLZONE_START,      "end": NY_KILLZONE_END,       "quality": 5},
    "ASIA":     {"start": ASIA_KILLZONE_START,    "end": ASIA_KILLZONE_END,     "quality": 3},
    "OVERLAP":  {"start": NY_KILLZONE_START,      "end": NY_KILLZONE_START + 3, "quality": 6},
    "OFF":      {"start": 0, "end": 24, "quality": 1},
}


def get_current_session(dt: datetime = None) -> dict:
    if dt is None:
        dt = datetime.now(timezone.utc)
    hour = dt.hour + dt.minute / 60.0

    # London-NY Overlap check first (highest quality)
    lo_start = NY_KILLZONE_START
    lo_end = LONDON_KILLZONE_END + 12  # e.g., 13-17 UTC overlap scenario
    if LONDON_KILLZONE_END <= hour < NY_KILLZONE_START:
        pass  # no overlap here
    if NY_KILLZONE_START <= hour < min(LONDON_KILLZONE_END + 12, NY_KILLZONE_END):
        if LONDON_KILLZONE_START <= LONDON_KILLZONE_END:
            # Check actual overlap window
            pass

    if NY_KILLZONE_START <= hour < NY_KILLZONE_END and LONDON_KILLZONE_START <= hour:
        overlap_end = min(NY_KILLZONE_END, LONDON_KILLZONE_END + 12)
        if hour < overlap_end:
            return {
                "session": "OVERLAP", "quality": 6,
                "description": "London/NY Overlap — Maximum Liquidity",
                "is_active": True,
            }

    if LONDON_KILLZONE_START <= hour < LONDON_KILLZONE_END:
        return {
            "session": "LONDON", "quality": 5,
            "description": "London Killzone — High Liquidity",
            "is_active": True,
        }
    if NY_KILLZONE_START <= hour < NY_KILLZONE_END:
        return {
            "session": "NY", "quality": 5,
            "description": "New York Killzone — High Liquidity",
            "is_active": True,
        }
    if ASIA_KILLZONE_START <= hour < ASIA_KILLZONE_END:
        return {
            "session": "ASIA", "quality": 3,
            "description": "Asia Session — Moderate Liquidity",
            "is_active": True,
        }
    return {
        "session": "OFF", "quality": 1,
        "description": "Off-Session — Low Liquidity",
        "is_active": False,
    }


def is_tradeable_session(min_quality: int = None, dt: datetime = None) -> tuple[bool, dict]:
    if min_quality is None:
        min_quality = SESSION_MIN_QUALITY
    sess = get_current_session(dt)
    tradeable = sess["quality"] >= min_quality
    if not tradeable and SESSION_FILTER_ENABLED:
        logger.debug(f"Session filter blocked: {sess['session']} (Q{sess['quality']} < {min_quality})")
    return tradeable, sess


def session_score_modifier(session_info: dict) -> float:
    """Return confidence score modifier: OVERLAP=+5, LONDON/NY=+3, ASIA=0, OFF=-10"""
    q = session_info.get("quality", 1)
    if q >= 6:   return 5.0
    elif q >= 5: return 3.0
    elif q >= 3: return 0.0
    else:        return -10.0
