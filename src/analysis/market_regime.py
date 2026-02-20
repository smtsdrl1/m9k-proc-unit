"""
Market Regime Detector — ADX + ATR based.
Classifies current market as TRENDING / RANGING / VOLATILE / QUIET.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("matrix_trader.analysis.market_regime")

try:
    from src.config import ADX_TREND_THRESHOLD, ATR_VOLATILE_THRESHOLD, REGIME_DETECTION_ENABLED
except ImportError:
    ADX_TREND_THRESHOLD = 25
    ATR_VOLATILE_THRESHOLD = 2.5
    REGIME_DETECTION_ENABLED = True

REGIME_TRENDING = "TRENDING"
REGIME_RANGING  = "RANGING"
REGIME_VOLATILE = "VOLATILE"
REGIME_QUIET    = "QUIET"
REGIME_VOLATILE_TREND = "VOLATILE_TREND"
REGIME_TRANSITION     = "TRANSITION"


class MarketRegimeDetector:
    def __init__(self):
        self.cache: dict = {}
        self.cache_ttl_minutes = 15

    def detect(self, df: pd.DataFrame, symbol: str = "") -> dict:
        now = datetime.now()
        if symbol in self.cache:
            cached = self.cache[symbol]
            if now - cached.get("_ts", now) < timedelta(minutes=self.cache_ttl_minutes):
                return cached

        result = self._compute(df, symbol)
        result["_ts"] = now
        if symbol:
            self.cache[symbol] = result
        return result

    def _compute(self, df: pd.DataFrame, symbol: str) -> dict:
        if len(df) < 30:
            return {"regime": REGIME_TRANSITION, "position_multiplier": 0.8,
                    "adx": 0, "atr_pct": 0, "symbol": symbol}
        try:
            import pandas_ta as ta
            # ADX
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            adx = float(adx_df.iloc[-1, 0]) if adx_df is not None else 20

            # ATR %
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
            atr = float(atr_series.iloc[-1]) if atr_series is not None else 0
            price = float(df["close"].iloc[-1])
            atr_pct = atr / price * 100 if price > 0 else 1.0

        except Exception:
            high  = df["high"].values
            low   = df["low"].values
            close = df["close"].values
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
            )
            atr = float(np.mean(tr[-14:]))
            price = float(close[-1])
            atr_pct = atr / price * 100 if price > 0 else 1.0
            adx = 20.0  # fallback

        is_trending = adx > ADX_TREND_THRESHOLD
        is_volatile = atr_pct > ATR_VOLATILE_THRESHOLD
        is_quiet    = atr_pct < 0.5

        if is_quiet:
            regime = REGIME_QUIET
            pos_mult = 0.5
        elif is_trending and is_volatile:
            regime = REGIME_VOLATILE_TREND
            pos_mult = 0.7
        elif is_trending:
            regime = REGIME_TRENDING
            pos_mult = 1.0
        elif is_volatile:
            regime = REGIME_VOLATILE
            pos_mult = 0.6
        else:
            regime = REGIME_RANGING
            pos_mult = 0.8

        return {
            "regime": regime, "symbol": symbol, "adx": round(adx, 2),
            "atr_pct": round(atr_pct, 3), "position_multiplier": pos_mult,
        }

    def get_confidence_modifier(self, df: pd.DataFrame, symbol: str = "") -> float:
        """Return confidence score modifier based on regime."""
        info = self.detect(df, symbol)
        regime = info.get("regime", REGIME_TRANSITION)
        modifiers = {
            REGIME_TRENDING:       5.0,
            REGIME_RANGING:        0.0,
            REGIME_VOLATILE:      -5.0,
            REGIME_QUIET:        -15.0,
            REGIME_VOLATILE_TREND: 2.0,
            REGIME_TRANSITION:     0.0,
        }
        return modifiers.get(regime, 0.0)


# Singleton
market_regime_detector = MarketRegimeDetector()
