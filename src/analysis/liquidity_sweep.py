"""
Liquidity Sweep / Stop Hunt Detection — ICT methodology.
Detects smart money sweeping retail stop losses.
"""
import logging
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np

logger = logging.getLogger("matrix_trader.analysis.liquidity_sweep")


@dataclass
class LiquiditySweep:
    type: str          # "BULLISH_SWEEP" or "BEARISH_SWEEP"
    swept_level: float
    recovery_pct: float
    bar_index: int


def detect_equal_levels(df: pd.DataFrame, tolerance: float = 0.002) -> dict:
    if len(df) < 20:
        return {"equal_highs": [], "equal_lows": []}
    subset = df.tail(30)
    equal_highs, equal_lows = [], []
    highs = subset["high"].values
    lows  = subset["low"].values
    for i in range(len(highs) - 1):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                equal_highs.append(round((highs[i] + highs[j]) / 2, 6))
    for i in range(len(lows) - 1):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                equal_lows.append(round((lows[i] + lows[j]) / 2, 6))
    return {"equal_highs": list(set(equal_highs)), "equal_lows": list(set(equal_lows))}


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 30,
                            min_recovery_pct: float = 0.3) -> List[LiquiditySweep]:
    if len(df) < lookback + 5:
        return []

    sweeps = []
    subset = df.tail(lookback + 5).reset_index(drop=True)

    for i in range(5, len(subset) - 1):
        bar = subset.iloc[i]
        prev_lows  = subset["low"].iloc[:i].values
        prev_highs = subset["high"].iloc[:i].values

        # Bullish Sweep: bar goes below previous low and closes back above
        local_low = np.min(prev_lows[-20:]) if len(prev_lows) >= 20 else np.min(prev_lows)
        if bar["low"] < local_low:
            recovery = (bar["close"] - bar["low"]) / max(bar["high"] - bar["low"], 1e-10) * 100
            if recovery >= min_recovery_pct * 100:
                sweeps.append(LiquiditySweep(
                    type="BULLISH_SWEEP", swept_level=local_low,
                    recovery_pct=recovery, bar_index=i
                ))

        # Bearish Sweep: bar goes above previous high and closes back below
        local_high = np.max(prev_highs[-20:]) if len(prev_highs) >= 20 else np.max(prev_highs)
        if bar["high"] > local_high:
            recovery = (bar["high"] - bar["close"]) / max(bar["high"] - bar["low"], 1e-10) * 100
            if recovery >= min_recovery_pct * 100:
                sweeps.append(LiquiditySweep(
                    type="BEARISH_SWEEP", swept_level=local_high,
                    recovery_pct=recovery, bar_index=i
                ))

    # Keep only recent sweeps
    return sweeps[-5:]


def get_sweep_score(sweeps: List[LiquiditySweep], direction: str) -> float:
    """Return score contribution from liquidity sweeps (up to ±15)."""
    if not sweeps:
        return 0.0
    latest = sweeps[-1]
    boost = min(15.0, latest.recovery_pct / 10)
    if direction == "BUY" and latest.type == "BULLISH_SWEEP":
        return boost
    elif direction == "SELL" and latest.type == "BEARISH_SWEEP":
        return boost
    elif direction == "BUY" and latest.type == "BEARISH_SWEEP":
        return -boost * 0.5
    elif direction == "SELL" and latest.type == "BULLISH_SWEEP":
        return -boost * 0.5
    return 0.0
