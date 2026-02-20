"""
Target Time Estimator — Predicts how long each TP level might take.
Uses ATR-based daily velocity, trend strength (ADX), and historical volatility
to estimate arrival time for each target price.

Logic:
- Daily price movement ≈ ATR (1d) per day for stocks, ATR(4h)*6 for crypto
- Adjust by ADX: strong trend (ADX>25) moves faster, weak trend slower
- Adjust by volume: high volume accelerates moves
- Adjust by market type: BIST trades ~8h/day, crypto 24h
"""
import math
import logging
from typing import Optional

logger = logging.getLogger("matrix_trader.signals.time_estimator")


def estimate_target_times(
    price: float,
    targets: dict,
    atr: float,
    adx: float = 20.0,
    volume_ratio: float = 1.0,
    is_bist: bool = False,
    direction: str = "BUY",
    timeframe_atr: str = "1d",
) -> dict:
    """
    Estimate time-to-target for each TP level.

    Args:
        price: Current price
        targets: {"t1": float, "t2": float, "t3": float}
        atr: ATR value from primary timeframe
        adx: ADX value (trend strength)
        volume_ratio: Current volume / avg volume
        is_bist: BIST stock (8h trading day) vs crypto (24h)
        direction: BUY or SELL
        timeframe_atr: Timeframe of the ATR value

    Returns:
        {"t1": {"days": int, "label": str}, "t2": {...}, "t3": {...}}
    """
    if not targets or not atr or atr <= 0 or price <= 0:
        return {}

    # Convert ATR to daily movement estimate
    daily_atr = _atr_to_daily(atr, timeframe_atr, is_bist)

    if daily_atr <= 0:
        return {}

    # ADX adjustment: strong trends (>25) move ~30% faster, weak (<15) ~40% slower
    adx = max(adx or 20, 5)
    if adx >= 40:
        adx_factor = 0.65   # Very strong trend — arrives ~35% faster
    elif adx >= 25:
        adx_factor = 0.80   # Strong trend — arrives ~20% faster
    elif adx >= 15:
        adx_factor = 1.0    # Normal
    else:
        adx_factor = 1.4    # Weak/ranging — takes ~40% longer

    # Volume adjustment: high volume = faster moves
    vol_ratio = max(volume_ratio or 1.0, 0.3)
    if vol_ratio >= 2.0:
        vol_factor = 0.75   # Very high volume — faster
    elif vol_ratio >= 1.3:
        vol_factor = 0.90   # Above average
    elif vol_ratio >= 0.7:
        vol_factor = 1.0    # Normal
    else:
        vol_factor = 1.3    # Low volume — slower

    # Net daily expected move toward target
    # Price doesn't move in straight line — assume ~40% of ATR is "trend progress"
    trend_efficiency = 0.40
    effective_daily_move = daily_atr * trend_efficiency / adx_factor / vol_factor

    result = {}
    for tname, tval in targets.items():
        distance = abs(tval - price)
        if distance <= 0 or effective_daily_move <= 0:
            result[tname] = {"days": 0, "hours": 0, "label": "—"}
            continue

        raw_days = distance / effective_daily_move

        # Apply some randomness factor for uncertainty — farther targets are less certain
        if tname == "t1":
            uncertainty = 1.0
        elif tname == "t2":
            uncertainty = 1.15
        else:
            uncertainty = 1.3

        est_days = raw_days * uncertainty

        # Convert to business days for BIST
        if is_bist:
            est_days = est_days  # Already in trading days
            label = _format_bist_time(est_days)
        else:
            label = _format_crypto_time(est_days)

        result[tname] = {
            "days": round(est_days, 1),
            "hours": round(est_days * 24, 0) if not is_bist else round(est_days * 8, 0),
            "label": label,
        }

    return result


def _atr_to_daily(atr: float, timeframe: str, is_bist: bool) -> float:
    """Convert ATR from any timeframe to daily equivalent."""
    # Mapping: how many candles of this TF fit in one trading day
    if is_bist:
        # BIST: 8h trading day
        tf_to_daily = {
            "15m": 32,   # 32 candles per day
            "1h": 8,     # 8 candles per day
            "4h": 2,     # 2 candles per day
            "1d": 1,     # 1 candle per day
            "1wk": 0.2,  # 1/5 of a week
        }
    else:
        # Crypto: 24h trading
        tf_to_daily = {
            "15m": 96,   # 96 candles per day
            "1h": 24,    # 24 candles per day
            "4h": 6,     # 6 candles per day
            "1d": 1,
            "1wk": 0.143,
        }

    multiplier = tf_to_daily.get(timeframe, 1)

    if multiplier >= 1:
        # Intraday: daily ATR ≈ ATR * sqrt(n) (not linear — overlapping ranges)
        return atr * math.sqrt(multiplier)
    else:
        # Weekly/Monthly: scale down
        return atr * multiplier


def _format_bist_time(days: float) -> str:
    """Format estimated time for BIST (trading days)."""
    if days < 0.5:
        return "Gün içi"
    elif days < 1.5:
        return "~1 gün"
    elif days < 3:
        return f"~{round(days)} gün"
    elif days < 7:
        return f"~{round(days)} iş günü"
    elif days < 15:
        weeks = round(days / 5, 1)
        if weeks <= 1:
            return "~1 hafta"
        return f"~{weeks:.0f} hafta"
    elif days < 60:
        weeks = round(days / 5)
        return f"~{weeks} hafta"
    else:
        months = round(days / 22, 1)
        if months <= 1:
            return "~1 ay"
        return f"~{months:.0f} ay"


def _format_crypto_time(days: float) -> str:
    """Format estimated time for crypto (24/7)."""
    hours = days * 24
    if hours < 4:
        return f"~{max(1, round(hours))} saat"
    elif hours < 24:
        return f"~{round(hours)} saat"
    elif days < 2:
        return "~1 gün"
    elif days < 7:
        return f"~{round(days)} gün"
    elif days < 21:
        weeks = round(days / 7, 1)
        if weeks <= 1:
            return "~1 hafta"
        return f"~{weeks:.0f} hafta"
    elif days < 90:
        weeks = round(days / 7)
        return f"~{weeks} hafta"
    else:
        months = round(days / 30, 1)
        if months <= 1:
            return "~1 ay"
        return f"~{months:.0f} ay"
