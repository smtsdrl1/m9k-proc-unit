"""
Multi-Timeframe Confluence Analysis.
Checks alignment across 15m, 1h, 4h, 1d (or BIST equivalents).
Professional traders never trade on a single timeframe.
"""
import logging
from typing import Optional
from src.analysis.technical import calculate_indicators
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.analysis.multi_timeframe")


# Timeframe weights — higher TF = more weight
TF_WEIGHTS = {
    "15m": 0.10,
    "1h":  0.20,
    "4h":  0.30,
    "1d":  0.40,
    "1wk": 0.40,
}


def analyze_timeframe(indicators: dict) -> dict:
    """
    Analyze a single timeframe and return direction + strength.
    Returns: {"direction": "BUY"/"SELL"/"NEUTRAL", "strength": 0-100, "signals": [...]}
    """
    if not indicators:
        return {"direction": "NEUTRAL", "strength": 0, "signals": []}

    buy_signals = []
    sell_signals = []

    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd_hist", 0)
    macd_crossover = indicators.get("macd_crossover", "NONE")
    bb_pctb = indicators.get("bb_pctb", 0.5)
    stoch_k = indicators.get("stoch_k", 50)
    adx = indicators.get("adx", 20)
    plus_di = indicators.get("plus_di", 20)
    minus_di = indicators.get("minus_di", 20)
    price = indicators.get("currentPrice", 0)
    ema9 = indicators.get("ema9", price)
    ema21 = indicators.get("ema21", price)
    ema50 = indicators.get("ema50", price)
    volume_ratio = indicators.get("volume_ratio", 1.0)

    # RSI
    if rsi < 30:
        buy_signals.append(("RSI_OVERSOLD", 15))
    elif rsi < 40:
        buy_signals.append(("RSI_LOW", 8))
    elif rsi > 70:
        sell_signals.append(("RSI_OVERBOUGHT", 15))
    elif rsi > 60:
        sell_signals.append(("RSI_HIGH", 8))

    # MACD
    if macd_crossover == "BULLISH":
        buy_signals.append(("MACD_CROSS_UP", 20))
    elif macd_crossover == "BEARISH":
        sell_signals.append(("MACD_CROSS_DOWN", 20))
    elif macd_hist > 0:
        buy_signals.append(("MACD_POSITIVE", 8))
    elif macd_hist < 0:
        sell_signals.append(("MACD_NEGATIVE", 8))

    # Bollinger Bands
    if bb_pctb < 0.0:
        buy_signals.append(("BB_BELOW_LOWER", 15))
    elif bb_pctb < 0.2:
        buy_signals.append(("BB_NEAR_LOWER", 8))
    elif bb_pctb > 1.0:
        sell_signals.append(("BB_ABOVE_UPPER", 15))
    elif bb_pctb > 0.8:
        sell_signals.append(("BB_NEAR_UPPER", 8))

    # Stochastic
    if stoch_k < 20:
        buy_signals.append(("STOCH_OVERSOLD", 10))
    elif stoch_k > 80:
        sell_signals.append(("STOCH_OVERBOUGHT", 10))

    # EMA Alignment
    if price > ema9 > ema21 > ema50:
        buy_signals.append(("EMA_BULLISH_ALIGN", 12))
    elif price < ema9 < ema21 < ema50:
        sell_signals.append(("EMA_BEARISH_ALIGN", 12))

    # ADX Trend Strength
    if adx > 25:
        if plus_di > minus_di:
            buy_signals.append(("ADX_STRONG_TREND_UP", 10))
        else:
            sell_signals.append(("ADX_STRONG_TREND_DOWN", 10))

    # Volume confirmation
    if volume_ratio > 1.5:
        if len(buy_signals) > len(sell_signals):
            buy_signals.append(("VOLUME_CONFIRM_BUY", 8))
        elif len(sell_signals) > len(buy_signals):
            sell_signals.append(("VOLUME_CONFIRM_SELL", 8))

    buy_score = sum(s[1] for s in buy_signals)
    sell_score = sum(s[1] for s in sell_signals)

    if buy_score > sell_score and buy_score >= 15:
        direction = "BUY"
        strength = min(buy_score, 100)
        signals = [s[0] for s in buy_signals]
    elif sell_score > buy_score and sell_score >= 15:
        direction = "SELL"
        strength = min(sell_score, 100)
        signals = [s[0] for s in sell_signals]
    else:
        direction = "NEUTRAL"
        strength = 0
        signals = []

    return {"direction": direction, "strength": strength, "signals": signals}


def multi_timeframe_confluence(tf_data: dict[str, dict]) -> dict:
    """
    Analyze multiple timeframes and compute confluence score.

    Args:
        tf_data: {"15m": indicators_dict, "1h": indicators_dict, ...}

    Returns:
        {
            "direction": "BUY"/"SELL"/"NEUTRAL",
            "confluence_score": 0-100,
            "aligned_count": int,
            "total_count": int,
            "timeframes": {tf: analysis_result, ...},
            "recommendation": str,
        }
    """
    if not tf_data:
        return {
            "direction": "NEUTRAL",
            "confluence_score": 0,
            "aligned_count": 0,
            "total_count": 0,
            "timeframes": {},
            "recommendation": "Yetersiz veri",
        }

    analyses = {}
    for tf, indicators in tf_data.items():
        analyses[tf] = analyze_timeframe(indicators)

    # Count directions
    buy_count = sum(1 for a in analyses.values() if a["direction"] == "BUY")
    sell_count = sum(1 for a in analyses.values() if a["direction"] == "SELL")
    total = len(analyses)

    # Weighted score
    weighted_buy = 0.0
    weighted_sell = 0.0
    for tf, analysis in analyses.items():
        weight = TF_WEIGHTS.get(tf, 0.2)
        if analysis["direction"] == "BUY":
            weighted_buy += weight * analysis["strength"]
        elif analysis["direction"] == "SELL":
            weighted_sell += weight * analysis["strength"]

    # Determine overall direction
    if weighted_buy > weighted_sell and buy_count >= total * 0.5:
        direction = "BUY"
        aligned = buy_count
        confluence_score = min(round(weighted_buy), 100)
    elif weighted_sell > weighted_buy and sell_count >= total * 0.5:
        direction = "SELL"
        aligned = sell_count
        confluence_score = min(round(weighted_sell), 100)
    else:
        direction = "NEUTRAL"
        aligned = 0
        confluence_score = 0

    # Generate recommendation
    if aligned == total and direction != "NEUTRAL":
        recommendation = f"TÜM ZAMAN DİLİMLERİ {direction} YÖNLERİNDE HİZALI — Güçlü sinyal"
    elif aligned >= total - 1 and direction != "NEUTRAL":
        recommendation = f"{aligned}/{total} zaman dilimi hizalı — İyi sinyal"
    elif direction != "NEUTRAL":
        # Check if higher TFs disagree
        higher_tfs = [tf for tf in ["4h", "1d", "1wk"] if tf in analyses]
        higher_direction = None
        for tf in higher_tfs:
            if analyses[tf]["direction"] != "NEUTRAL":
                higher_direction = analyses[tf]["direction"]
                break

        if higher_direction and higher_direction != direction:
            recommendation = f"Kısa vade {direction} ama uzun vade {higher_direction} — Düzeltme bekle"
        else:
            recommendation = f"{aligned}/{total} hizalı — Dikkatli ol"
    else:
        recommendation = "Kararsız piyasa — İşlem açma"

    return {
        "direction": direction,
        "confluence_score": confluence_score,
        "aligned_count": aligned,
        "total_count": total,
        "timeframes": analyses,
        "recommendation": recommendation,
    }
