"""
Signal Confidence Scorer.
Combines all analysis modules into a single 0-100 confidence score.
Weights: Technical 40, MTF 20, Volume 15, Momentum 5, Sentiment 5, Smart Money 10, Macro 5.
ML model adjusts final score based on learned patterns from historical outcomes.
"""
import logging
from src.config import SCORE_WEIGHTS
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.signals.scorer")

# Lazy-loaded ML predictor (singleton)
_ml_predictor = None


def _get_ml_predictor():
    """Lazy-load ML predictor to avoid circular imports and slow startup."""
    global _ml_predictor
    if _ml_predictor is None:
        try:
            from src.ml.model import SignalPredictor
            _ml_predictor = SignalPredictor()
        except Exception as e:
            logger.warning(f"ML predictor not available: {e}")
    return _ml_predictor


def calculate_confidence(
    indicators: dict,
    direction: str,
    mtf_result: dict = None,
    sentiment: dict = None,
    smart_money: dict = None,
    macro: dict = None,
    fear_greed: int = 50,
    is_crypto: bool = True,
    funding_rate: dict = None,
    df=None,             # pd.DataFrame for advanced analysis (CVD, MS, OB, Sweep, VPVR)
    symbol: str = "",    # for regime cache
    order_book: dict = None,   # from CryptoFeed.fetch_order_book()
    onchain: dict = None,      # from onchain_feed.get_onchain_confidence_boost()
) -> dict:
    """
    Calculate comprehensive confidence score.

    Returns:
        {
            "total": 0-100,
            "breakdown": {component: score, ...},
            "grade": "A"/"B"/"C"/"D"/"F",
        }
    """
    breakdown = {}

    # ─── Technical Score (0-40) ───────────────────────────
    tech_score = _score_technical(indicators, direction)
    breakdown["technical"] = round(tech_score * SCORE_WEIGHTS["technical"] / 100)

    # ─── MTF Confluence (0-20) ────────────────────────────
    mtf_score = _score_mtf(mtf_result, direction) if mtf_result else 40
    breakdown["mtf_confluence"] = round(mtf_score * SCORE_WEIGHTS["mtf_confluence"] / 100)

    # ─── Volume Profile (0-15) ────────────────────────────
    vol_score = _score_volume(indicators)
    breakdown["volume_profile"] = round(vol_score * SCORE_WEIGHTS["volume_profile"] / 100)

    # ─── Momentum (0-5) ──────────────────────────────────
    mom_score = _score_momentum(indicators, direction)
    breakdown["momentum"] = round(mom_score * SCORE_WEIGHTS["momentum"] / 100)

    # ─── Sentiment (0-5) ─────────────────────────────────
    sent_score = _score_sentiment(sentiment, direction, fear_greed, is_crypto) if sentiment else 50
    breakdown["sentiment"] = round(sent_score * SCORE_WEIGHTS["sentiment"] / 100)

    # ─── Smart Money (0-10) ──────────────────────────────
    sm_score = _score_smart_money(smart_money, direction) if smart_money else 50
    breakdown["smart_money"] = round(sm_score * SCORE_WEIGHTS["smart_money"] / 100)

    # ─── Macro (0-5) ─────────────────────────────────────
    macro_score = _score_macro(macro, direction, is_crypto) if macro else 50
    breakdown["macro"] = round(macro_score * SCORE_WEIGHTS["macro"] / 100)

    total = sum(breakdown.values())
    total = max(0, min(100, total))

    # ─── Order Book Imbalance (Crypto Only) ─────────────
    if is_crypto and order_book:
        try:
            from src.analysis.orderbook import analyze_order_book
            ob_result = analyze_order_book(order_book, direction)
            ob_boost = ob_result.get("confidence_boost", 0)
            if ob_boost != 0:
                total = max(0, min(100, total + ob_boost))
                breakdown["order_book"] = ob_boost
                logger.debug(
                    f"OrderBook boost: {ob_boost:+d} "
                    f"({ob_result.get('description', '')})"
                )
        except Exception as e:
            logger.debug(f"OrderBook analysis skipped: {e}")

    # ─── On-Chain Data Boost ─────────────────────────────
    if is_crypto and onchain:
        try:
            onchain_boost = onchain.get("boost", 0)
            if onchain_boost != 0:
                total = max(0, min(100, total + onchain_boost))
                breakdown["onchain"] = onchain_boost
                logger.debug(
                    f"OnChain boost: {onchain_boost:+d} "
                    f"({onchain.get('reason', '')})"
                )
        except Exception as e:
            logger.debug(f"OnChain boost skipped: {e}")

    # ─── Funding Rate Adjustment (Crypto Only) ──────────
    funding_adjustment = 0
    if is_crypto and funding_rate:
        funding_adjustment = _score_funding_rate(funding_rate, direction)
        if funding_adjustment != 0:
            total = max(0, min(100, total + funding_adjustment))
            breakdown["funding_rate"] = funding_adjustment
            logger.info(f"Funding rate adjustment: {funding_adjustment:+d}")

    # ─── Bull/Bear Market Adaptive Bias ─────────────────
    market_bias = 0
    if is_crypto and fear_greed > 0:
        market_bias = _market_regime_bias(fear_greed, direction)
        if market_bias != 0:
            total = max(0, min(100, total + market_bias))
            breakdown["market_regime"] = market_bias
            regime = "FEAR" if fear_greed < 30 else "GREED" if fear_greed > 70 else "NEUTRAL"
            logger.info(f"Market regime bias: {market_bias:+d} ({regime})")

    # ─── ML Model Adjustment ────────────────────────────
    ml_adjustment = 0
    ml_prediction = None
    ml_predictor = _get_ml_predictor()
    if ml_predictor and ml_predictor.is_loaded:
        try:
            features = ml_predictor.extract_features(
                indicators=indicators,
                mtf_result=mtf_result,
                sentiment=sentiment,
                smart_money=smart_money,
                macro=macro,
                confidence=total,
                is_crypto=is_crypto,
            )
            ml_prediction = ml_predictor.predict(features)
            if ml_prediction:
                ml_adjustment = ml_prediction["confidence_adjustment"]
                total = max(0, min(100, total + ml_adjustment))
                logger.info(
                    f"ML adjustment: {ml_adjustment:+d} "
                    f"(win_prob={ml_prediction['win_probability']:.1%})"
                )
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")

    # ─── Advanced Analysis (df required) ────────────────
    if df is not None and len(df) >= 30:

        # CVD — Cumulative Volume Delta
        try:
            from src.analysis.technical import calculate_cvd
            cvd = calculate_cvd(df)
            cvd_boost = cvd.get("score_boost", 0)
            if direction == "SELL":
                cvd_boost = -cvd_boost  # flip for sell direction
            if cvd_boost != 0:
                total = max(0, min(100, total + cvd_boost))
                breakdown["cvd"] = round(cvd_boost, 1)
        except Exception as e:
            logger.debug(f"CVD analysis skipped: {e}")

        # Market Structure BOS+CHoCH
        try:
            from src.analysis.market_structure import analyze_market_structure
            ms = analyze_market_structure(df)
            ms_boost = ms.get("score_boost", 0)
            if direction == "SELL":
                ms_boost = -ms_boost
            if ms_boost != 0:
                total = max(0, min(100, total + ms_boost))
                breakdown["market_structure"] = round(ms_boost, 1)
        except Exception as e:
            logger.debug(f"Market structure skipped: {e}")

        # Order Block Detection
        try:
            from src.analysis.order_blocks import detect_order_blocks, get_order_block_score
            obs = detect_order_blocks(df)
            price = float(df["close"].iloc[-1])
            ob_boost = get_order_block_score(price, obs, direction)
            if ob_boost != 0:
                total = max(0, min(100, total + ob_boost))
                breakdown["order_blocks"] = round(ob_boost, 1)
        except Exception as e:
            logger.debug(f"Order blocks skipped: {e}")

        # Liquidity Sweep
        try:
            from src.analysis.liquidity_sweep import detect_liquidity_sweeps, get_sweep_score
            sweeps = detect_liquidity_sweeps(df)
            sweep_boost = get_sweep_score(sweeps, direction)
            if sweep_boost != 0:
                total = max(0, min(100, total + sweep_boost))
                breakdown["liquidity_sweep"] = round(sweep_boost, 1)
        except Exception as e:
            logger.debug(f"Liquidity sweep skipped: {e}")

        # Market Regime
        try:
            from src.analysis.market_regime import market_regime_detector
            regime_mod = market_regime_detector.get_confidence_modifier(df, symbol)
            if regime_mod != 0:
                total = max(0, min(100, total + regime_mod))
                breakdown["market_regime_adv"] = round(regime_mod, 1)
            # QUIET regime → zero signal
            regime_info = market_regime_detector.detect(df, symbol)
            if regime_info.get("regime") == "QUIET":
                total = min(total, 40)  # Cap at 40 in quiet market
        except Exception as e:
            logger.debug(f"Market regime skipped: {e}")

        # VPVR
        try:
            from src.analysis.vpvr import calculate_vpvr, get_vpvr_confidence_modifier
            vpvr = calculate_vpvr(df)
            vpvr_mod = get_vpvr_confidence_modifier(vpvr, direction)
            if vpvr_mod != 0:
                total = max(0, min(100, total + vpvr_mod))
                breakdown["vpvr"] = round(vpvr_mod, 1)
        except Exception as e:
            logger.debug(f"VPVR skipped: {e}")

        # Session Killzone
        try:
            from src.utils.session_killzone import get_current_session, session_score_modifier
            sess = get_current_session()
            sess_mod = session_score_modifier(sess)
            if sess_mod != 0:
                total = max(0, min(100, total + sess_mod))
                breakdown["session"] = round(sess_mod, 1)
        except Exception as e:
            logger.debug(f"Session killzone skipped: {e}")

    total = max(0, min(100, total))

    # Grade assignment
    if total >= 80:
        grade = "A"
    elif total >= 65:
        grade = "B"
    elif total >= 50:
        grade = "C"
    elif total >= 35:
        grade = "D"
    else:
        grade = "F"

    # Build feature snapshot for ML training (from real data only)
    feature_snapshot = None
    if ml_predictor:
        try:
            feature_snapshot = ml_predictor.extract_features(
                indicators=indicators,
                mtf_result=mtf_result,
                sentiment=sentiment,
                smart_money=smart_money,
                macro=macro,
                confidence=total,
                is_crypto=is_crypto,
            )
        except Exception:
            pass

    return {
        "total": total,
        "breakdown": breakdown,
        "grade": grade,
        "ml_adjustment": ml_adjustment,
        "ml_prediction": ml_prediction,
        "features": feature_snapshot,
    }


def _score_technical(ind: dict, direction: str) -> float:
    """Score technical indicators 0-100 with granular RSI scoring."""
    score = 50  # Neutral baseline
    rsi = ind.get("rsi", 50)
    macd_hist = ind.get("macd_hist", 0)
    macd_crossover = ind.get("macd_crossover", "NONE")
    bb_pctb = ind.get("bb_pctb", 0.5)
    stoch_k = ind.get("stoch_k", 50)
    adx = ind.get("adx", 20)

    if direction == "BUY":
        # Granular RSI scoring
        if rsi <= 20:
            score += 25
        elif rsi <= 30:
            score += 20
        elif rsi <= 40:
            score += 12
        elif rsi <= 50:
            score += 5
        elif rsi >= 70:
            score -= 15

        if macd_crossover == "BULLISH":
            score += 15
        elif macd_hist > 0:
            score += 8

        if bb_pctb < 0.1:
            score += 15
        elif bb_pctb < 0.2:
            score += 12
        elif bb_pctb < 0.3:
            score += 6

        if stoch_k < 20:
            score += 12
        elif stoch_k < 30:
            score += 8

    elif direction == "SELL":
        if rsi >= 80:
            score += 25
        elif rsi >= 70:
            score += 20
        elif rsi >= 60:
            score += 12
        elif rsi >= 50:
            score += 5
        elif rsi <= 30:
            score -= 15

        if macd_crossover == "BEARISH":
            score += 15
        elif macd_hist < 0:
            score += 8

        if bb_pctb > 0.9:
            score += 15
        elif bb_pctb > 0.8:
            score += 12
        elif bb_pctb > 0.7:
            score += 6

        if stoch_k > 80:
            score += 12
        elif stoch_k > 70:
            score += 8

    # ADX bonus — trend strength
    if adx > 30:
        score += 10
    elif adx > 20:
        score += 6
    elif adx > 15:
        score += 3

    return max(0, min(100, score))


def _score_mtf(mtf: dict, direction: str) -> float:
    """Score multi-timeframe alignment 0-100."""
    if not mtf:
        return 40  # Baseline when no MTF data (not penalize)
    mtf_dir = mtf.get("direction", "NEUTRAL")
    if mtf_dir == direction:
        aligned = mtf.get("aligned_count", 0)
        total = mtf.get("total_count", 1)
        return min(50 + (aligned / max(total, 1)) * 50, 100)
    elif mtf_dir != "NEUTRAL":
        return 20  # Against alignment = penalty
    return 40


def _score_momentum(ind: dict, direction: str) -> float:
    """Score price momentum 0-100."""
    score = 50
    price = ind.get("currentPrice", 0)
    ema9 = ind.get("ema9", price)
    ema21 = ind.get("ema21", price)
    ema50 = ind.get("ema50", price)
    price_change = ind.get("price_change_pct", 0)

    if direction == "BUY":
        if price > ema9 > ema21:
            score += 20
        elif price > ema21:
            score += 10
        if price_change > 0:
            score += min(price_change * 5, 15)
    else:
        if price < ema9 < ema21:
            score += 20
        elif price < ema21:
            score += 10
        if price_change < 0:
            score += min(abs(price_change) * 5, 15)

    return max(0, min(100, score))


def _score_volume(ind: dict) -> float:
    """Score volume profile 0-100."""
    vol_ratio = ind.get("volume_ratio", 1.0)
    obv_trend = ind.get("obv_trend", "NEUTRAL")
    score = 50

    if vol_ratio >= 2.0:
        score += 30
    elif vol_ratio >= 1.5:
        score += 20
    elif vol_ratio >= 1.2:
        score += 10
    elif vol_ratio < 0.5:
        score -= 20

    if obv_trend == "UP":
        score += 10
    elif obv_trend == "DOWN":
        score -= 10

    return max(0, min(100, score))


def _score_sentiment(sent: dict, direction: str, fng: int, is_crypto: bool) -> float:
    """Score sentiment 0-100."""
    score = 50
    if sent:
        sentiment_score = sent.get("score", 0)  # -100 to +100
        if direction == "BUY":
            score += sentiment_score * 0.25
        else:
            score -= sentiment_score * 0.25

    # Fear & Greed (crypto only)
    if is_crypto:
        if direction == "BUY" and fng < 25:
            score += 15  # "Be greedy when others are fearful"
        elif direction == "SELL" and fng > 75:
            score += 15  # "Be fearful when others are greedy"
        elif direction == "BUY" and fng > 80:
            score -= 10  # Too greedy
        elif direction == "SELL" and fng < 20:
            score -= 10  # Too fearful to short

    return max(0, min(100, score))


def _score_smart_money(sm: dict, direction: str) -> float:
    """Score smart money signals 0-100."""
    sm_dir = sm.get("direction", "NEUTRAL")
    if sm_dir == direction:
        return 80
    elif sm_dir != "NEUTRAL":
        return 20  # Against smart money = bad sign
    return 50


def _score_macro(macro: dict, direction: str, is_crypto: bool) -> float:
    """Score macro alignment 0-100."""
    if not macro:
        return 50

    filter_key = "crypto_filter" if is_crypto else "bist_filter"
    filter_level = macro.get(filter_key, "ALLOW")

    if filter_level == "BLOCK":
        return 15
    elif filter_level == "CAUTION":
        return 40
    else:
        return 65


def _score_funding_rate(funding: dict, direction: str) -> int:
    """Score funding rate impact on confidence.
    Positive funding = longs pay shorts → bearish when extreme.
    Negative funding = shorts pay longs → bullish when extreme.

    Returns confidence adjustment: -15 to +10
    """
    if not funding:
        return 0

    rate = funding.get("funding_rate", 0)

    if direction in ("BUY", "LONG", "AL"):
        if rate > 0.05:     # Very high positive = longs overcrowded
            return -15       # Strong penalty for long
        elif rate > 0.01:   # High positive
            return -8
        # Fix: check < -0.05 BEFORE < -0.01, otherwise < -0.05 is never reached
        elif rate < -0.05:  # Very negative = shorts overcrowded → strong buy signal
            return 10
        elif rate < -0.01:  # Negative = shorts paying, bullish for longs
            return 8
    else:  # SELL / SHORT
        if rate < -0.05:    # Very negative = shorts overcrowded
            return -15
        elif rate < -0.01:
            return -8
        # Fix: check > 0.05 BEFORE > 0.01, otherwise > 0.05 is never reached
        elif rate > 0.05:   # Very high positive = longs overcrowded → strong sell signal
            return 10
        elif rate > 0.01:   # Positive = longs paying, good for shorts
            return 8

    return 0


def _market_regime_bias(fear_greed: int, direction: str) -> int:
    """Bull/Bear market adaptive scoring.
    In fear markets → favor SHORT, penalize LONG
    In greed markets → favor LONG, penalize SHORT
    Extreme readings create stronger bias.

    Inspired by CAI: "Fear & Greed Index: 5 (Extreme Fear).
    Piyasa bearish. SHORT mantıklı."

    Returns confidence adjustment: -10 to +10
    """
    if direction in ("BUY", "LONG", "AL"):
        if fear_greed <= 10:       # Extreme fear
            return -8              # Very risky to long in extreme fear
        elif fear_greed <= 25:     # Fear
            return -3              # Slight penalty, but contrarian can work
        elif fear_greed >= 90:     # Extreme greed
            return -5              # Too late to long
        elif fear_greed >= 70:     # Greed
            return 5               # Momentum with longs
        elif 40 <= fear_greed <= 60:
            return 0               # Neutral zone
    else:  # SELL / SHORT
        if fear_greed >= 90:       # Extreme greed
            return -8              # Risky to short at extreme greed
        elif fear_greed >= 75:
            return -3
        elif fear_greed <= 10:     # Extreme fear
            return -5              # Too late to short
        elif fear_greed <= 30:     # Fear
            return 5               # Momentum with shorts
        elif 40 <= fear_greed <= 60:
            return 0

    return 0
