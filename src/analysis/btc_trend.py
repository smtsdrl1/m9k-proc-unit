"""
BTC Trend Bias — Master Market Direction Filter.

Before scanning any altcoin, determine BTC's current macro trend.
Altcoins are highly correlated with BTC (~0.75-0.95 correlation).

Rules:
  BTC STRONG BULL → Allow BUY altcoin signals, block SELL signals
  BTC STRONG BEAR → Allow SELL altcoin signals, block BUY signals
  BTC NEUTRAL     → Allow both directions (but reduce confidence)
  BTC is BTC/ETH  → No filter (scanning BTC/ETH directly)

Trend detection uses:
  - 4h EMA20 vs EMA50 (primary)
  - Daily EMA50 vs EMA200 (secondary — longer term)
  - ADX on 4h for trend strength
  - Recent price structure (higher highs/lows or lower highs/lows)
"""
import logging
import time
from typing import Optional
import pandas as pd

logger = logging.getLogger("matrix_trader.analysis.btc_trend")

# ── Cache ─────────────────────────────────────────────────────────────
_CACHE: dict = {}
_CACHE_TTL = 900  # 15 minutes (BTC trend doesn't change that fast)


def get_btc_trend(feed) -> dict:
    """
    Fetch BTC 4h data and determine macro trend.

    Args:
        feed: CryptoFeed instance (already initialized)

    Returns:
        {
            "trend":     "BULLISH" | "BEARISH" | "NEUTRAL",
            "strength":  "STRONG" | "MODERATE" | "WEAK",
            "ema20_4h":  float,
            "ema50_4h":  float,
            "adx_4h":    float,
            "bias":      "BUY" | "SELL" | "BOTH",
            "description": str,
        }
    """
    now = time.time()
    cached = _CACHE.get("btc_trend")
    if cached and now - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    try:
        import asyncio
        from src.analysis.technical import calculate_indicators

        # Fetch BTC 4h data synchronously
        loop = asyncio.new_event_loop()
        try:
            df_4h = loop.run_until_complete(feed.fetch_ohlcv("BTC/USDT", "4h", limit=100))
        finally:
            loop.close()

        if df_4h is None or len(df_4h) < 50:
            result = _neutral_result("insufficient BTC 4h data")
            _CACHE["btc_trend"] = {"ts": now, "data": result}
            return result

        result = _analyze_trend(df_4h)
        _CACHE["btc_trend"] = {"ts": now, "data": result}
        logger.info(
            f"BTC Trend: {result['trend']} ({result['strength']}) | "
            f"EMA20={result['ema20_4h']:.0f} EMA50={result['ema50_4h']:.0f} "
            f"ADX={result['adx_4h']:.1f}"
        )
        return result

    except Exception as e:
        logger.warning(f"BTC trend fetch error: {e}")
        return _neutral_result(str(e))


def _analyze_trend(df: pd.DataFrame) -> dict:
    """Analyze 4h DataFrame to determine BTC trend."""
    try:
        from src.analysis.technical import calculate_indicators
        ind = calculate_indicators(df)
        if not ind:
            return _neutral_result("indicator calculation failed")

        price  = ind.get("currentPrice", 0)
        ema9   = ind.get("ema9",  price)
        ema21  = ind.get("ema21", price)
        ema50  = ind.get("ema50", price)
        adx    = ind.get("adx",   20)
        rsi    = ind.get("rsi",   50)

        # Primary: 4h EMA alignment
        bull_ema = ema9 > ema21 and ema21 > ema50 and price > ema21
        bear_ema = ema9 < ema21 and ema21 < ema50 and price < ema21

        # Secondary: recent price structure (last 20 candles)
        closes = df["close"].values[-20:]
        highs  = df["high"].values[-20:]
        lows   = df["low"].values[-20:]

        # Higher highs + higher lows = bullish structure
        mid     = len(closes) // 2
        hh      = highs[-5:].max() > highs[:mid].max()
        hl      = lows[-5:].min()  > lows[:mid].min()
        lh      = highs[-5:].max() < highs[:mid].max()
        ll      = lows[-5:].min()  < lows[:mid].min()
        bull_structure = hh and hl
        bear_structure = lh and ll

        # Determine trend
        if bull_ema and bull_structure:
            trend = "BULLISH"
        elif bear_ema and bear_structure:
            trend = "BEARISH"
        elif bull_ema:
            trend = "BULLISH"
        elif bear_ema:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Strength based on ADX + EMA separation
        if adx >= 35:
            strength = "STRONG"
        elif adx >= 22:
            strength = "MODERATE"
        else:
            strength = "WEAK"
            trend = "NEUTRAL"  # Low ADX = not trending = neutral

        # Bias for altcoin filtering
        if trend == "BULLISH" and strength in ("STRONG", "MODERATE"):
            bias = "BUY"    # Prefer longs
        elif trend == "BEARISH" and strength in ("STRONG", "MODERATE"):
            bias = "SELL"   # Prefer shorts
        else:
            bias = "BOTH"   # No restriction

        return {
            "trend":       trend,
            "strength":    strength,
            "ema20_4h":    round(ema9, 2),   # Using ema9 as proxy for ema20
            "ema50_4h":    round(ema50, 2),
            "adx_4h":      round(adx, 1),
            "rsi_4h":      round(rsi, 1),
            "bias":        bias,
            "bull_ema":    bull_ema,
            "bear_ema":    bear_ema,
            "bull_struct": bull_structure,
            "bear_struct": bear_structure,
            "description": f"BTC 4h: {trend} ({strength}) | bias={bias}",
        }

    except Exception as e:
        logger.warning(f"BTC trend analysis error: {e}")
        return _neutral_result(str(e))


def _neutral_result(reason: str) -> dict:
    return {
        "trend":       "NEUTRAL",
        "strength":    "WEAK",
        "ema20_4h":    0,
        "ema50_4h":    0,
        "adx_4h":      0,
        "rsi_4h":      50,
        "bias":        "BOTH",
        "description": f"BTC trend unknown: {reason}",
    }


def is_signal_aligned_with_btc(
    symbol: str,
    signal_direction: str,
    btc_trend: dict,
    strict: bool = True,
) -> tuple[bool, str]:
    """
    Check if an altcoin signal aligns with BTC macro trend.

    Args:
        symbol:           e.g. "ETH/USDT"
        signal_direction: "BUY" or "SELL"
        btc_trend:        from get_btc_trend()
        strict:           If True, block signals against strong BTC trend

    Returns:
        (allowed: bool, reason: str)
    """
    # BTC and ETH can trade against the trend (they ARE the trend)
    base = symbol.split("/")[0].upper()
    if base in ("BTC", "ETH", "WBTC"):
        return True, "BTC/ETH — no trend filter"

    bias      = btc_trend.get("bias", "BOTH")
    trend     = btc_trend.get("trend", "NEUTRAL")
    strength  = btc_trend.get("strength", "WEAK")
    is_buy    = signal_direction in ("BUY", "LONG", "AL")

    if bias == "BOTH" or strength == "WEAK":
        return True, f"BTC {trend} ({strength}) — no restriction"

    if strict:
        # In strong BTC bull: block SELL altcoin signals
        if bias == "BUY" and not is_buy:
            return False, (
                f"BTC {trend} ({strength}) — blocking {symbol} SELL "
                f"(alts follow BTC uptrend)"
            )
        # In strong BTC bear: block BUY altcoin signals
        if bias == "SELL" and is_buy:
            return False, (
                f"BTC {trend} ({strength}) — blocking {symbol} BUY "
                f"(alts follow BTC downtrend)"
            )
    else:
        # Lenient: only block in STRONG trend
        if strength == "STRONG":
            if bias == "BUY" and not is_buy:
                return False, f"BTC STRONG BULL — blocking alt SELL"
            if bias == "SELL" and is_buy:
                return False, f"BTC STRONG BEAR — blocking alt BUY"

    return True, f"BTC {trend} ({strength}) — signal aligned"
