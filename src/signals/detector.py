"""
6-Tier Signal Detector.
Combines technical indicators, MTF confluence, volume, and divergence.
"""
import logging
from src.analysis.technical import detect_rsi_divergence
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.signals.detector")


def detect_signal(indicators: dict, mtf_result: dict = None, smart_money: dict = None) -> dict:
    """
    Detect trade signal with 6-tier classification.

    Tiers:
        1. EXTREME: 5+ indicators aligned + MTF + Volume
        2. STRONG: 4 indicators + Volume confirmation
        3. MODERATE: 3 indicators aligned
        4. SPECULATIVE: 2 indicators, some alignment
        5. DIVERGENCE: RSI divergence detected
        6. CONTRARIAN: Against prevailing trend (reversal setup)

    Returns:
        {
            "direction": "BUY"/"SELL"/"NEUTRAL",
            "tier": 1-6,
            "tier_name": str,
            "reasons": [str],
            "indicator_count": int,
        }
    """
    if not indicators:
        return _neutral()

    buy_reasons = []
    sell_reasons = []

    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd_hist", 0)
    macd_crossover = indicators.get("macd_crossover", "NONE")
    bb_pctb = indicators.get("bb_pctb", 0.5)
    stoch_k = indicators.get("stoch_k", 50)
    adx = indicators.get("adx", 20)
    plus_di = indicators.get("plus_di", 20)
    minus_di = indicators.get("minus_di", 20)
    volume_ratio = indicators.get("volume_ratio", 1.0)
    cross = indicators.get("cross", "NONE")
    obv_trend = indicators.get("obv_trend", "NEUTRAL")
    price = indicators.get("currentPrice", 0)
    ema9 = indicators.get("ema9", price)
    ema21 = indicators.get("ema21", price)
    ema50 = indicators.get("ema50", price)

    # ─── RSI ─────────────────────────────────────────────
    if rsi <= 30:
        buy_reasons.append("RSI AŞIRI SATIM (≤30)")
    elif rsi <= 40:
        buy_reasons.append("RSI Düşük (≤40)")
    elif rsi >= 70:
        sell_reasons.append("RSI AŞIRI ALIM (≥70)")
    elif rsi >= 60:
        sell_reasons.append("RSI Yüksek (≥60)")

    # ─── MACD ────────────────────────────────────────────
    if macd_crossover == "BULLISH":
        buy_reasons.append("MACD Yukarı Kesişim")
    elif macd_crossover == "BEARISH":
        sell_reasons.append("MACD Aşağı Kesişim")
    elif macd_hist > 0:
        buy_reasons.append("MACD Pozitif")
    elif macd_hist < 0:
        sell_reasons.append("MACD Negatif")

    # ─── Bollinger Bands ─────────────────────────────────
    if bb_pctb < 0.0:
        buy_reasons.append("BB Alt Bandın Altında")
    elif bb_pctb < 0.20:
        buy_reasons.append("BB Alt Banda Yakın")
    elif bb_pctb > 1.0:
        sell_reasons.append("BB Üst Bandın Üstünde")
    elif bb_pctb > 0.80:
        sell_reasons.append("BB Üst Banda Yakın")

    # ─── Stochastic ──────────────────────────────────────
    if stoch_k <= 25:
        buy_reasons.append("Stochastic Aşırı Satım")
    elif stoch_k >= 75:
        sell_reasons.append("Stochastic Aşırı Alım")

    # ─── ADX + DI ────────────────────────────────────────
    if adx > 20:
        if plus_di > minus_di:
            buy_reasons.append(f"ADX Güçlü Yükseliş ({adx:.0f})")
        else:
            sell_reasons.append(f"ADX Güçlü Düşüş ({adx:.0f})")

    # ─── EMA Alignment ───────────────────────────────────
    if price > ema9 > ema21 > ema50:
        buy_reasons.append("EMA Yükseliş Dizilimi")
    elif price > ema21:
        buy_reasons.append("Fiyat EMA21 Üzerinde")
    elif price < ema9 < ema21 < ema50:
        sell_reasons.append("EMA Düşüş Dizilimi")
    elif price < ema21:
        sell_reasons.append("Fiyat EMA21 Altında")

    # ─── Volume ──────────────────────────────────────────
    has_volume = volume_ratio >= 1.2

    # ─── Golden/Death Cross ──────────────────────────────
    if cross == "GOLDEN_CROSS":
        buy_reasons.append("GOLDEN CROSS 🌟")
    elif cross == "DEATH_CROSS":
        sell_reasons.append("DEATH CROSS 💀")

    # ─── OBV Trend ───────────────────────────────────────
    if obv_trend == "UP" and len(buy_reasons) > len(sell_reasons):
        buy_reasons.append("OBV Yükseliş")
    elif obv_trend == "DOWN" and len(sell_reasons) > len(buy_reasons):
        sell_reasons.append("OBV Düşüş")

    # ─── MTF Confluence ──────────────────────────────────
    mtf_aligned = False
    if mtf_result:
        mtf_dir = mtf_result.get("direction", "NEUTRAL")
        mtf_score = mtf_result.get("confluence_score", 0)
        if mtf_dir == "BUY" and mtf_score >= 20:
            buy_reasons.append(f"MTF Yükseliş Hizası ({mtf_result.get('aligned_count', 0)}/{mtf_result.get('total_count', 0)})")
            mtf_aligned = True
        elif mtf_dir == "SELL" and mtf_score >= 20:
            sell_reasons.append(f"MTF Düşüş Hizası ({mtf_result.get('aligned_count', 0)}/{mtf_result.get('total_count', 0)})")
            mtf_aligned = True

    # ─── Smart Money ─────────────────────────────────────
    if smart_money:
        sm_dir = smart_money.get("direction", "NEUTRAL")
        if sm_dir == "BUY":
            buy_reasons.append("Akıllı Para Alım Sinyali")
        elif sm_dir == "SELL":
            sell_reasons.append("Akıllı Para Satış Sinyali")

    # ─── FVG + Fibonacci Confluence — Alper INCE @alper3968 ───────
    # Kaynak: https://x.com/alper3968/status/1862990567153557955
    # FVG bölgesi + Fibonacci retracement confluence u = sniper giriş noktası
    fvg_fib_confluence = indicators.get("fvg_fib_confluence", False)
    fvg_fib_signal     = indicators.get("fvg_fib_signal", "NEUTRAL")
    fvg_fib_detail     = indicators.get("fvg_fib_confluence_detail") or {}
    fvg_score_boost    = indicators.get("fvg_fib_score_boost", 0)

    if fvg_fib_confluence:
        fib_lvl   = fvg_fib_detail.get("fib_level", "?")
        fvg_type  = fvg_fib_detail.get("fvg_type", "?").upper()
        is_golden = fvg_fib_detail.get("is_golden", False)
        golden_tag = " 🏆 ALTIN ORAN" if is_golden else ""
        strength   = fvg_fib_detail.get("strength", 0)

        if fvg_fib_signal == "BUY":
            label = f"FVG+Fib Confluence{golden_tag}: {fvg_type} FVG × Fib {fib_lvl} (güç: {strength:.2f})"
            buy_reasons.append(label)
            # Altin oran confluence → ekstra STRONG sinyal
            if is_golden:
                buy_reasons.append("FVG × 0.618 Sniper Giriş 🎯")
        elif fvg_fib_signal == "SELL":
            label = f"FVG+Fib Confluence{golden_tag}: {fvg_type} FVG × Fib {fib_lvl} (güç: {strength:.2f})"
            sell_reasons.append(label)
            if is_golden:
                sell_reasons.append("FVG × 0.618 Sniper Giriş 🎯")

    # ─── Determine Direction & Tier ──────────────────────
    buy_count = len(buy_reasons)
    sell_count = len(sell_reasons)

    if buy_count <= 1 and sell_count <= 1:
        return _neutral()

    if buy_count > sell_count:
        direction = "BUY"
        reasons = buy_reasons
        count = buy_count
    elif sell_count > buy_count:
        direction = "SELL"
        reasons = sell_reasons
        count = sell_count
    else:
        return _neutral()

    # Assign tier
    # FVG+Fib golden ratio confluence = otomatik tier upgrade
    has_fvg_golden = (
        fvg_fib_confluence
        and fvg_fib_signal == direction
        and fvg_fib_detail.get("is_golden", False)
    )
    has_fvg_normal = (
        fvg_fib_confluence
        and fvg_fib_signal == direction
        and not fvg_fib_detail.get("is_golden", False)
    )

    if count >= 5 and has_volume and mtf_aligned:
        tier = 1
        tier_name = "🔥 EXTREME"
    elif count >= 4 and (has_volume or mtf_aligned):
        tier = 2
        tier_name = "💪 STRONG"
    elif count >= 3 or (count >= 2 and has_fvg_golden):
        # FVG + 0.618 confluence varsa 2 indikatör de MODERATE'e yükseltir
        tier = 3
        tier_name = "📊 MODERATE" + (" + FVG🎯" if has_fvg_golden else "")
    elif count >= 2 or (count >= 1 and has_fvg_normal):
        tier = 4
        tier_name = "🎲 SPECULATIVE" + (" + FVG" if has_fvg_normal else "")
    else:
        tier = 5
        tier_name = "🔀 WEAK"

    return {
        "direction":       direction,
        "tier":            tier,
        "tier_name":       tier_name,
        "reasons":         reasons,
        "indicator_count": count,
        "fvg_fib_boost":   fvg_score_boost,
        "fvg_fib_present": fvg_fib_confluence,
    }


def check_divergence(df, indicators: dict) -> dict:
    """Check for RSI divergence — can override tier to DIVERGENCE."""
    if df is None:
        return {"divergence": None}

    from src.analysis.technical import detect_rsi_divergence
    div = detect_rsi_divergence(df)

    if div == "BULLISH_DIVERGENCE":
        return {
            "divergence": "BULLISH",
            "direction": "BUY",
            "tier": 5,
            "tier_name": "🔀 DIVERGENCE",
            "reasons": ["RSI Yükseliş Diverjansı — Dip yapıyor olabilir"],
        }
    elif div == "BEARISH_DIVERGENCE":
        return {
            "divergence": "BEARISH",
            "direction": "SELL",
            "tier": 5,
            "tier_name": "🔀 DIVERGENCE",
            "reasons": ["RSI Düşüş Diverjansı — Tepe yapıyor olabilir"],
        }
    return {"divergence": None}


def _neutral():
    return {
        "direction": "NEUTRAL",
        "tier": 0,
        "tier_name": "NEUTRAL",
        "reasons": [],
        "indicator_count": 0,
    }


def apply_pre_trade_filters(signal: dict, df=None, symbol: str = "") -> dict:
    """Apply Session Killzone + Market Regime filters to a signal.

    Args:
        signal: Signal dict from detect_signal()
        df: OHLCV DataFrame (required for regime detection)
        symbol: Symbol name (for regime cache)

    Returns:
        Modified signal dict; may flip direction to NEUTRAL if filtered.
    """
    if signal.get("direction", "NEUTRAL") == "NEUTRAL":
        return signal

    result = dict(signal)
    filtered_by = []

    # ── Session Killzone ─────────────────────────────────
    try:
        from src.utils.session_killzone import is_tradeable_session, get_current_session
        from src.config import SESSION_FILTER_ENABLED, SESSION_MIN_QUALITY
        if SESSION_FILTER_ENABLED:
            tradeable, sess_info = is_tradeable_session(SESSION_MIN_QUALITY)
            result["session"] = sess_info
            if not tradeable:
                filtered_by.append(
                    f"Session filter: {sess_info['session']} (Q{sess_info['quality']})"
                )
    except Exception as e:
        logger.debug(f"Session filter skipped: {e}")

    # ── Market Regime ─────────────────────────────────────
    if df is not None and len(df) >= 30:
        try:
            from src.analysis.market_regime import market_regime_detector
            from src.config import REGIME_DETECTION_ENABLED
            if REGIME_DETECTION_ENABLED:
                regime_info = market_regime_detector.detect(df, symbol)
                result["market_regime"] = regime_info
                if regime_info.get("regime") == "QUIET":
                    filtered_by.append("Market regime: QUIET (no trading)")
        except Exception as e:
            logger.debug(f"Regime filter skipped: {e}")

    # ── Economic Calendar (News Kill) ─────────────────────
    try:
        from src.data.economic_calendar import check_news_kill_zone
        news = check_news_kill_zone()
        result["news_kill"] = news
        if news.get("should_avoid"):
            filtered_by.append(f"News kill: {news.get('event', 'High impact event')}")
    except Exception as e:
        logger.debug(f"News kill skipped: {e}")

    # Apply filters
    if filtered_by:
        result["direction"] = "NEUTRAL"
        result["tier"] = 0
        result["tier_name"] = "FILTERED"
        result["filtered_by"] = filtered_by
        logger.debug(f"{symbol} filtered: {'; '.join(filtered_by)}")

    return result
