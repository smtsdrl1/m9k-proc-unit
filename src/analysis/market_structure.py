"""
Market Structure Analysis — BOS (Break of Structure) & CHoCH (Change of Character)
ICT methodology adapted for Matrix Trader AI.
"""
import logging
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np

logger = logging.getLogger("matrix_trader.analysis.market_structure")


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "HIGH" or "LOW"
    label: str = ""  # HH, LH, HL, LL


def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    if len(df) < lookback * 2 + 1:
        return []
    swings = []
    for i in range(lookback, len(df) - lookback):
        hi = df["high"].iloc[i]
        lo = df["low"].iloc[i]
        if hi == df["high"].iloc[i - lookback: i + lookback + 1].max():
            swings.append(SwingPoint(i, hi, "HIGH"))
        elif lo == df["low"].iloc[i - lookback: i + lookback + 1].min():
            swings.append(SwingPoint(i, lo, "LOW"))
    return swings


def classify_swing_structure(swings: List[SwingPoint]) -> List[SwingPoint]:
    highs = [s for s in swings if s.type == "HIGH"]
    lows  = [s for s in swings if s.type == "LOW"]

    for i in range(1, len(highs)):
        if highs[i].price > highs[i - 1].price:
            highs[i].label = "HH"
        else:
            highs[i].label = "LH"

    for i in range(1, len(lows)):
        if lows[i].price > lows[i - 1].price:
            lows[i].label = "HL"
        else:
            lows[i].label = "LL"

    all_swings = sorted(swings, key=lambda s: s.index)
    return all_swings


def detect_structure_breaks(df: pd.DataFrame, swings: List[SwingPoint]) -> list:
    breaks = []
    recent_price = df["close"].iloc[-1]
    high_swings = [s for s in swings if s.type == "HIGH" and s.label in ("HH", "LH")]
    low_swings  = [s for s in swings if s.type == "LOW"  and s.label in ("HL", "LL")]

    if high_swings:
        last_high = high_swings[-1]
        if recent_price > last_high.price:
            breaks.append({
                "type": "BOS_BULLISH" if last_high.label == "HH" else "CHoCH_BULLISH",
                "price": last_high.price, "direction": "BUY",
            })

    if low_swings:
        last_low = low_swings[-1]
        if recent_price < last_low.price:
            breaks.append({
                "type": "BOS_BEARISH" if last_low.label == "LL" else "CHoCH_BEARISH",
                "price": last_low.price, "direction": "SELL",
            })

    return breaks


def analyze_market_structure(df: pd.DataFrame) -> dict:
    """Full BOS+CHoCH analysis. Returns score_boost for confidence scorer."""
    if len(df) < 30:
        return {"trend": "UNKNOWN", "score_boost": 0, "bos_detected": False, "choch_detected": False}

    swings = detect_swing_points(df)
    if not swings:
        return {"trend": "UNKNOWN", "score_boost": 0, "bos_detected": False, "choch_detected": False}

    classified = classify_swing_structure(swings)
    breaks = detect_structure_breaks(df, classified)

    # Determine trend from recent labels
    recent = classified[-8:]
    hh = sum(1 for s in recent if s.label == "HH")
    hl = sum(1 for s in recent if s.label == "HL")
    lh = sum(1 for s in recent if s.label == "LH")
    ll = sum(1 for s in recent if s.label == "LL")

    trend = "NEUTRAL"
    if hh + hl > lh + ll:
        trend = "BULLISH"
    elif lh + ll > hh + hl:
        trend = "BEARISH"

    bos  = any("BOS" in b["type"] for b in breaks)
    choch = any("CHoCH" in b["type"] for b in breaks)

    # Score boost for the scorer (+10 for CHoCH, +6 for BOS, direction-aware)
    score_boost = 0
    for b in breaks:
        if b["direction"] == "BUY":
            score_boost += 10 if "CHoCH" in b["type"] else 6
        else:
            score_boost -= 10 if "CHoCH" in b["type"] else 6

    return {
        "trend": trend, "bos_detected": bos, "choch_detected": choch,
        "breaks": breaks, "score_boost": score_boost,
        "hh_count": hh, "hl_count": hl, "lh_count": lh, "ll_count": ll,
    }
