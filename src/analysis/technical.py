"""
Technical Analysis Engine — RSI, MACD, Bollinger, ATR, Stochastic, ADX, EMAs, OBV, S/R.
FVG (Fair Value Gap) + Fibonacci Confluence dahil — Alper INCE (@alper3968) metodolojisi.
Kaynak: https://x.com/alper3968/status/1862990567153557955  #xu100 #fibonacci #fvg
"""
import logging
from typing import Optional
import pandas as pd
import pandas_ta as ta
import numpy as np
from src.utils.helpers import safe_float, safe_positive
from src.analysis.fvg_fibonacci import analyze_fvg_fibonacci

logger = logging.getLogger("matrix_trader.analysis.technical")


def calculate_indicators(df: pd.DataFrame) -> Optional[dict]:
    """
    Calculate all technical indicators from OHLCV DataFrame.
    Returns a dict with all indicator values or None on failure.
    """
    if df is None or len(df) < 30:
        return None

    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ─── RSI (14) ────────────────────────────────────────
        rsi_series = ta.rsi(close, length=14)
        rsi = safe_float(rsi_series.iloc[-1]) if rsi_series is not None else 50.0

        # ─── MACD (12, 26, 9) ────────────────────────────────
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and len(macd_df.columns) >= 3:
            macd_line = safe_float(macd_df.iloc[-1, 0])
            macd_signal = safe_float(macd_df.iloc[-1, 1])
            macd_hist = safe_float(macd_df.iloc[-1, 2])
            # Previous histogram for crossover detection
            macd_hist_prev = safe_float(macd_df.iloc[-2, 2]) if len(macd_df) >= 2 else 0.0
        else:
            macd_line = macd_signal = macd_hist = macd_hist_prev = 0.0

        # ─── Bollinger Bands (20, 2) ─────────────────────────
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None and len(bb_df.columns) >= 5:
            bb_upper = safe_float(bb_df.iloc[-1, 0])
            bb_middle = safe_float(bb_df.iloc[-1, 1])
            bb_lower = safe_float(bb_df.iloc[-1, 2])
            bb_bandwidth = safe_float(bb_df.iloc[-1, 3])
            bb_pctb = safe_float(bb_df.iloc[-1, 4])
        else:
            price = safe_float(close.iloc[-1])
            bb_upper = price * 1.02
            bb_middle = price
            bb_lower = price * 0.98
            bb_bandwidth = 4.0
            bb_pctb = 0.5

        # ─── ATR (14) ────────────────────────────────────────
        atr_series = ta.atr(high, low, close, length=14)
        atr = safe_float(atr_series.iloc[-1]) if atr_series is not None else safe_positive(close.iloc[-1]) * 0.02

        # ─── Stochastic RSI (14, 14, 3, 3) ───────────────────
        stoch_df = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
        if stoch_df is not None and len(stoch_df.columns) >= 2:
            stoch_k = safe_float(stoch_df.iloc[-1, 0]) * 100
            stoch_d = safe_float(stoch_df.iloc[-1, 1]) * 100
        else:
            stoch_k = stoch_d = 50.0

        # ─── ADX (14) ────────────────────────────────────────
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and len(adx_df.columns) >= 3:
            adx = safe_float(adx_df.iloc[-1, 0])
            plus_di = safe_float(adx_df.iloc[-1, 1])
            minus_di = safe_float(adx_df.iloc[-1, 2])
        else:
            adx = 20.0
            plus_di = minus_di = 20.0

        # ─── EMAs ────────────────────────────────────────────
        ema9 = safe_float(ta.ema(close, length=9).iloc[-1]) if len(close) >= 9 else safe_float(close.iloc[-1])
        ema21 = safe_float(ta.ema(close, length=21).iloc[-1]) if len(close) >= 21 else safe_float(close.iloc[-1])
        ema50 = safe_float(ta.ema(close, length=50).iloc[-1]) if len(close) >= 50 else safe_float(close.iloc[-1])
        ema200 = safe_float(ta.ema(close, length=200).iloc[-1]) if len(close) >= 200 else safe_float(close.iloc[-1])

        # ─── SMA (20, 50) ────────────────────────────────────
        sma20 = safe_float(ta.sma(close, length=20).iloc[-1]) if len(close) >= 20 else safe_float(close.iloc[-1])
        sma50 = safe_float(ta.sma(close, length=50).iloc[-1]) if len(close) >= 50 else safe_float(close.iloc[-1])

        # ─── Volume Analysis ─────────────────────────────────
        vol_sma20 = safe_float(ta.sma(volume, length=20).iloc[-1]) if len(volume) >= 20 else safe_float(volume.mean())
        current_volume = safe_float(volume.iloc[-1])
        volume_ratio = current_volume / safe_positive(vol_sma20) if vol_sma20 > 0 else 1.0

        # ─── OBV (On Balance Volume) ─────────────────────────
        obv_series = ta.obv(close, volume)
        obv = safe_float(obv_series.iloc[-1]) if obv_series is not None else 0.0
        obv_prev = safe_float(obv_series.iloc[-5]) if obv_series is not None and len(obv_series) >= 5 else obv
        obv_trend = "UP" if obv > obv_prev else "DOWN"

        # ─── Support / Resistance ────────────────────────────
        sr = calculate_support_resistance(df)

        # ─── MACD Crossover ──────────────────────────────────
        macd_crossover = "BULLISH" if macd_hist > 0 and macd_hist_prev <= 0 else \
                         "BEARISH" if macd_hist < 0 and macd_hist_prev >= 0 else "NONE"

        # ─── Golden / Death Cross ────────────────────────────
        if len(close) >= 200:
            sma50_series = ta.sma(close, length=50)
            sma200_series = ta.sma(close, length=200)
            if sma50_series is not None and sma200_series is not None:
                sma50_now = safe_float(sma50_series.iloc[-1])
                sma200_now = safe_float(sma200_series.iloc[-1])
                sma50_prev = safe_float(sma50_series.iloc[-2])
                sma200_prev = safe_float(sma200_series.iloc[-2])
                if sma50_now > sma200_now and sma50_prev <= sma200_prev:
                    cross = "GOLDEN_CROSS"
                elif sma50_now < sma200_now and sma50_prev >= sma200_prev:
                    cross = "DEATH_CROSS"
                else:
                    cross = "NONE"
            else:
                cross = "NONE"
        else:
            cross = "NONE"

        current_price = safe_float(close.iloc[-1])

        # ─── FVG + Fibonacci Confluence (Alper INCE metodu) ──────────────────
        # #xu100 #fibonacci #fvg — FVG bölgesi + Fib seviyesi örtüşümü = sniper giriş
        try:
            fvg_fib_result = analyze_fvg_fibonacci(df)
        except Exception as _fvg_err:
            logger.debug(f"FVG+Fib analiz hatası: {_fvg_err}")
            fvg_fib_result = {
                "has_confluence": False, "signal": "NEUTRAL",
                "score_boost": 0, "active_fvgs": [],
                "bullish_fvg_count": 0, "bearish_fvg_count": 0,
                "confluence": None, "summary": "hata"
            }

        return {
            "currentPrice": current_price,
            "rsi": round(rsi, 1),
            "macd": round(macd_line, 6),
            "macd_signal": round(macd_signal, 6),
            "macd_hist": round(macd_hist, 6),
            "macd_crossover": macd_crossover,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "bb_bandwidth": round(bb_bandwidth, 2),
            "bb_pctb": round(bb_pctb, 2),
            "atr": atr,
            "stoch_k": round(stoch_k, 1),
            "stoch_d": round(stoch_d, 1),
            "adx": round(adx, 1),
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "ema200": ema200,
            "sma20": sma20,
            "sma50": sma50,
            "volume": current_volume,
            "volume_sma20": vol_sma20,
            "volume_ratio": round(volume_ratio, 2),
            "obv": obv,
            "obv_trend": obv_trend,
            "sr": sr,
            "cross": cross,
            # FVG + Fibonacci Confluence — Alper INCE @alper3968
            "fvg_fib_confluence":   fvg_fib_result["has_confluence"],
            "fvg_fib_signal":       fvg_fib_result["signal"],
            "fvg_fib_score_boost":  fvg_fib_result["score_boost"],
            "fvg_fib_confluence_detail": fvg_fib_result["confluence"],
            "fvg_bullish_count":    fvg_fib_result["bullish_fvg_count"],
            "fvg_bearish_count":    fvg_fib_result["bearish_fvg_count"],
            "fvg_fib_summary":      fvg_fib_result["summary"],
        }

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None


def calculate_support_resistance(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Calculate support and resistance levels using pivot points."""
    try:
        recent = df.tail(lookback)
        high = recent["high"]
        low = recent["low"]
        close = recent["close"]

        # Pivot Points
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        r1 = 2 * pivot - low.iloc[-1]
        r2 = pivot + (high.iloc[-1] - low.iloc[-1])
        s1 = 2 * pivot - high.iloc[-1]
        s2 = pivot - (high.iloc[-1] - low.iloc[-1])

        # Also find recent swing highs/lows
        rolling_high = high.rolling(window=10).max()
        rolling_low = low.rolling(window=10).min()

        resistance1 = safe_float(max(r1, rolling_high.iloc[-1]))
        resistance2 = safe_float(r2)
        support1 = safe_float(min(s1, rolling_low.iloc[-1]))
        support2 = safe_float(s2)

        return {
            "pivot": safe_float(pivot),
            "resistance1": resistance1,
            "resistance2": resistance2,
            "support1": support1,
            "support2": support2,
        }
    except Exception as e:
        price = safe_float(df["close"].iloc[-1])
        return {
            "pivot": price,
            "resistance1": price * 1.02,
            "resistance2": price * 1.04,
            "support1": price * 0.98,
            "support2": price * 0.96,
        }


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30) -> Optional[str]:
    """
    Detect bullish/bearish RSI divergence.
    Bullish: Price makes lower low, RSI makes higher low.
    Bearish: Price makes higher high, RSI makes lower high.
    """
    try:
        if len(df) < lookback + 14:
            return None

        close = df["close"].iloc[-lookback:]
        rsi_series = ta.rsi(df["close"], length=14)
        if rsi_series is None:
            return None
        rsi = rsi_series.iloc[-lookback:]

        # Find two recent swing lows/highs
        mid = lookback // 2

        price_low1 = close.iloc[:mid].min()
        price_low2 = close.iloc[mid:].min()
        rsi_low1 = rsi.iloc[:mid].min()
        rsi_low2 = rsi.iloc[mid:].min()

        price_high1 = close.iloc[:mid].max()
        price_high2 = close.iloc[mid:].max()
        rsi_high1 = rsi.iloc[:mid].max()
        rsi_high2 = rsi.iloc[mid:].max()

        # Bullish divergence: lower price low, higher RSI low
        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            return "BULLISH_DIVERGENCE"

        # Bearish divergence: higher price high, lower RSI high
        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
            return "BEARISH_DIVERGENCE"

        return None
    except Exception:
        return None


def calculate_cvd(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Cumulative Volume Delta — measures buy vs sell pressure.

    Returns:
        dict with cvd_value, cvd_trend, buy_pressure, sell_pressure,
              cvd_divergence, cvd_signal, score_boost
    """
    if df is None or len(df) < 10:
        return {"cvd_signal": "NEUTRAL", "score_boost": 0, "cvd_trend": "FLAT"}

    try:
        subset = df.tail(lookback).copy()
        high = subset["high"].values
        low  = subset["low"].values
        close = subset["close"].values
        volume = subset["volume"].values

        # Buy volume = proportional to position within candle range
        ranges = high - low
        ranges = np.where(ranges < 1e-10, 1e-10, ranges)
        buy_ratio  = (close - low)  / ranges
        sell_ratio = (high - close) / ranges

        buy_volume  = buy_ratio  * volume
        sell_volume = sell_ratio * volume

        delta = buy_volume - sell_volume
        cvd   = np.cumsum(delta)

        cvd_value   = float(cvd[-1])
        buy_press  = float(np.sum(buy_volume[-10:]))
        sell_press = float(np.sum(sell_volume[-10:]))

        # Trend: compare recent half to first half
        mid = len(cvd) // 2
        cvd_trend = "UP" if cvd[-1] > cvd[mid] else ("DOWN" if cvd[-1] < cvd[mid] else "FLAT")

        # Divergence: price up but CVD down (or vice versa)
        price_up = close[-1] > close[mid]
        cvd_up   = cvd[-1] > cvd[mid]
        divergence = (price_up and not cvd_up) or (not price_up and cvd_up)

        # Signal
        cvd_ratio = buy_press / (buy_press + sell_press + 1e-10)
        if cvd_ratio > 0.65 and cvd_trend == "UP":
            signal = "STRONG_BUY"
            boost  = 8.0
        elif cvd_ratio > 0.55:
            signal = "MILD_BUY"
            boost  = 4.0
        elif cvd_ratio < 0.35 and cvd_trend == "DOWN":
            signal = "STRONG_SELL"
            boost  = -8.0
        elif cvd_ratio < 0.45:
            signal = "MILD_SELL"
            boost  = -4.0
        else:
            signal = "NEUTRAL"
            boost  = 0.0

        if divergence:
            boost *= -1  # Reverse when divergence detected

        return {
            "cvd_value":     round(cvd_value, 2),
            "cvd_trend":     cvd_trend,
            "buy_pressure":  round(buy_press, 2),
            "sell_pressure": round(sell_press, 2),
            "cvd_divergence": divergence,
            "cvd_signal":    signal,
            "score_boost":   round(boost, 2),
        }
    except Exception as e:
        logger.debug(f"CVD calculation error: {e}")
        return {"cvd_signal": "NEUTRAL", "score_boost": 0, "cvd_trend": "FLAT"}
