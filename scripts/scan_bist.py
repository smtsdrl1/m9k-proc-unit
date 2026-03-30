"""
BIST Scanner — GitHub Actions entry point.
Scans all BIST_100 symbols during market hours, detects signals, sends to Telegram.
Run via: python -m scripts.scan_bist
"""
import asyncio
import logging
import sys
import os
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BIST_100, BIST_TIMEFRAMES, CAPITAL, RISK_PERCENT,
    MIN_CONFIDENCE, SIGNAL_COOLDOWN_MINUTES,
    CIRCUIT_BREAKER_ENABLED,
    MAX_SIGNALS_PER_BIST_RUN, SL_HIT_CONFIDENCE_BOOST, SL_HIT_LOOKBACK_HOURS,
    PAPER_TRADING_ENABLED, KAP_FILTER_ENABLED,
)
from src.data.bist_feed import BistFeed
from src.data.macro_feed import MacroFeed
from src.analysis.technical import calculate_indicators
from src.analysis.multi_timeframe import multi_timeframe_confluence
from src.analysis.smart_money import smart_money_analysis
from src.analysis.sentiment import fetch_bist_news, keyword_sentiment_score
from src.analysis.macro_filter import analyze_macro, should_filter_signal
from src.signals.detector import detect_signal, apply_pre_trade_filters, check_divergence
from src.signals.risk_manager import calculate_risk
from src.signals.scorer import calculate_confidence
from src.signals.validator import validate_signal
from src.signals.circuit_breaker import CircuitBreaker
from src.signals.time_estimator import estimate_target_times
from src.ai.groq_engine import GroqEngine
from src.telegram.formatter import format_signal_message
from src.telegram.sender import TelegramSender
from src.database.db import Database
from src.utils.helpers import setup_logging, is_bist_market_hours

logger = logging.getLogger("matrix_trader.scan_bist")


async def scan_symbol(
    symbol: str,
    feed: BistFeed,
    groq: GroqEngine,
    db: Database,
    macro_result: dict,
    circuit_breaker: CircuitBreaker = None,
    min_confidence: int = None,
) -> dict:
    """Scan a single BIST symbol through full pipeline. Returns signal info or None."""
    result = {"symbol": symbol, "signal_data": None, "error": None}

    effective_min_confidence = min_confidence if min_confidence is not None else MIN_CONFIDENCE

    try:
        # KAP filter — suppress signals around financial disclosures
        if KAP_FILTER_ENABLED:
            try:
                from src.data.kap_feed import should_suppress_signal
                kap = should_suppress_signal(symbol)
                if kap.get("suppress"):
                    logger.info(f"[{symbol}] KAP suppression: {kap.get('reason', '')}")
                    result["error"] = "kap_suppressed"
                    return result
            except Exception:
                pass

        # Circuit breaker check
        if circuit_breaker and CIRCUIT_BREAKER_ENABLED:
            can_trade, cb_reason = circuit_breaker.can_trade()
            if not can_trade:
                result["error"] = f"circuit_breaker: {cb_reason}"
                return result

        # Cooldown check
        if db.check_cooldown(symbol, SIGNAL_COOLDOWN_MINUTES):
            result["error"] = "cooldown"
            return result

        # Fetch multi-timeframe data
        tf_data = feed.fetch_multi_timeframe(symbol, BIST_TIMEFRAMES)
        if not tf_data:
            result["error"] = "no_data"
            return result

        # Calculate indicators per timeframe
        tf_indicators = {}
        for tf, df in tf_data.items():
            ind = calculate_indicators(df)
            if ind:
                tf_indicators[tf] = ind

        if not tf_indicators:
            result["error"] = "no_indicators"
            return result

        primary_tf = list(tf_indicators.keys())[-1]
        indicators = tf_indicators[primary_tf]
        primary_df = tf_data[primary_tf]

        # MTF confluence
        mtf_result = multi_timeframe_confluence(tf_indicators)

        # Smart money
        sm_result = smart_money_analysis(primary_df, indicators["atr"])

        # Signal detection
        signal = detect_signal(indicators, mtf_result, sm_result)

        # Divergence fallback — when main signal is NEUTRAL, try RSI divergence
        if signal["direction"] == "NEUTRAL":
            div = check_divergence(primary_df, indicators)
            if div.get("direction") and div["direction"] != "NEUTRAL":
                signal = div
                logger.debug(f"[{symbol}] Divergence signal: {signal['direction']} ({signal['tier_name']})")
            else:
                return result

        # Pre-trade filters — Session Killzone, Market Regime, News Kill
        signal = apply_pre_trade_filters(signal, primary_df, symbol)
        if signal["direction"] == "NEUTRAL":
            if signal.get("filtered_by"):
                logger.info(f"[{symbol}] Pre-trade filtered: {'; '.join(signal['filtered_by'])}")
            return result

        # Circuit breaker — direction limit
        if circuit_breaker and CIRCUIT_BREAKER_ENABLED:
            can_open, dir_reason = circuit_breaker.can_open_direction(signal["direction"])
            if not can_open:
                logger.info(f"[{symbol}] ⚡ Direction blocked: {dir_reason}")
                return result

        # Macro filter
        filter_result = should_filter_signal(macro_result, signal["direction"], is_bist=True, symbol=symbol)
        if filter_result["action"] == "BLOCK":
            logger.info(f"[{symbol}] Blocked by macro: {filter_result['reason']}")
            return result

        # Risk management
        risk_mgmt = calculate_risk(
            indicators["currentPrice"], indicators["atr"],
            indicators["sr"], signal["direction"],
            is_bist=True, capital=CAPITAL, risk_pct=RISK_PERCENT,
        )

        # Fundamental data
        fundamental = None
        try:
            fundamental = feed.fetch_fundamental(symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Fundamental error: {e}")

        # Pre-check confidence WITHOUT sentiment/AI — skip Groq if base score too low
        pre_score = calculate_confidence(
            indicators, signal["direction"],
            mtf_result, None, sm_result, macro_result,
            is_crypto=False,
            df=primary_df,
            symbol=symbol,
        )
        if pre_score["total"] < effective_min_confidence - 15:
            return result

        # Sentiment (keyword-based — saves Groq budget for AI analysis)
        sentiment_result = None
        news_headlines = None
        try:
            news_headlines = await fetch_bist_news(symbol)
            if news_headlines:
                sentiment_result = keyword_sentiment_score(news_headlines)
        except Exception as e:
            logger.warning(f"[{symbol}] Sentiment error: {e}")

        # Confidence scoring (with ML adjustment + advanced df analysis)
        score_result = calculate_confidence(
            indicators, signal["direction"],
            mtf_result, sentiment_result, sm_result, macro_result,
            is_crypto=False,
            df=primary_df,
            symbol=symbol,
        )
        confidence = score_result["total"]
        grade = score_result["grade"]
        ml_features = score_result.get("features")  # Feature snapshot for ML training

        # Fix tier_numeric in ML features — tier only known after detect_signal()
        if ml_features and signal.get("tier_name"):
            _tier_map = {"EXTREME": 6, "STRONG": 5, "MODERATE": 4,
                         "SPECULATIVE": 3, "DIVERGENCE": 2, "CONTRARIAN": 1, "WEAK": 1}
            _tn = 0
            for _k, _v in _tier_map.items():
                if _k in signal["tier_name"].upper():
                    _tn = _v
                    break
            ml_features["tier_numeric"] = _tn

        if confidence < effective_min_confidence:
            return result

        # SL hit recently? BIST requires higher confidence for re-entry.
        if db.was_sl_hit_recently(symbol, SL_HIT_LOOKBACK_HOURS):
            required = effective_min_confidence + SL_HIT_CONFIDENCE_BOOST
            if confidence < required:
                logger.info(
                    f"[{symbol}] SL hit recently → require confidence ≥{required} "
                    f"(got {confidence})"
                )
                return result

        # Validation
        valid, errors = validate_signal(
            symbol, indicators["currentPrice"], risk_mgmt,
            confidence, signal["direction"],
            is_bist=True, min_confidence=effective_min_confidence,
        )
        if not valid:
            logger.warning(f"[{symbol}] Validation failed: {errors}")
            return result

        # AI analysis — try Groq first, fallback to rule-based
        ai_analysis = None
        if groq.available:
            try:
                ai_analysis = groq.get_investment_analysis(
                    symbol, signal["direction"], indicators, risk_mgmt,
                    confidence, mtf_result, sentiment_result, sm_result,
                    macro_result, fundamental, news=news_headlines, is_bist=True,
                )
            except Exception as e:
                logger.warning(f"[{symbol}] AI error: {e}")

        # Fallback: rule-based analysis from real data
        if not ai_analysis:
            ai_analysis = GroqEngine.generate_fallback_analysis(
                symbol, signal["direction"], indicators, risk_mgmt,
                confidence, sentiment_result, sm_result, macro_result,
            )
            logger.info(f"[{symbol}] Using fallback AI analysis (rule-based)")

        # AI veto
        if ai_analysis and ai_analysis.get("karar") == "REDDET":
            logger.info(f"[{symbol}] Vetoed by AI")
            return result

        # Risk budget check
        if circuit_breaker and CIRCUIT_BREAKER_ENABLED:
            sl = risk_mgmt.get("stop_loss", 0)
            price = indicators["currentPrice"]
            if sl and price:
                new_risk_pct = abs(price - sl) / price * 100
                can_risk, risk_reason = circuit_breaker.check_risk_budget(new_risk_pct)
                if not can_risk:
                    logger.info(f"[{symbol}] ⚡ Risk budget exceeded: {risk_reason}")
                    return result

        # Time estimates for targets
        time_estimates = estimate_target_times(
            price=indicators["currentPrice"],
            targets=risk_mgmt.get("targets", {}),
            atr=indicators["atr"],
            adx=indicators.get("adx", 20),
            volume_ratio=indicators.get("volume_ratio", 1.0),
            is_bist=True,
            direction=signal["direction"],
            timeframe_atr=primary_tf,
        )

        # Macro caution note
        caution_note = ""
        if filter_result.get("action") == "CAUTION":
            caution_note = f"\n⚠️ {filter_result.get('reason', '')}"

        # Package signal data
        result["signal_data"] = {
            "symbol": symbol,
            "direction": signal["direction"],
            "tier_name": signal["tier_name"],
            "confidence": confidence,
            "grade": grade,
            "indicators": indicators,
            "risk_mgmt": risk_mgmt,
            "ai_analysis": ai_analysis,
            "mtf_result": mtf_result,
            "sentiment": sentiment_result,
            "smart_money": sm_result,
            "macro": macro_result,
            "reasons": signal.get("reasons", []),
            "caution_note": caution_note,
            "ml_features": ml_features,
            "time_estimates": time_estimates,
        }

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{symbol}] Error: {e}\n{traceback.format_exc()}")

    return result


async def main():
    """Main BIST scanner loop."""
    setup_logging()

    # Market hours check (skip if manual dispatch via FORCE_SCAN)
    force = os.getenv("FORCE_SCAN", "false").lower() == "true"
    if not force and not is_bist_market_hours():
        logger.info("BIST market is closed. Skipping scan.")
        return

    logger.info("=" * 60)
    logger.info("🏛 Matrix Trader AI — BIST Scanner Starting")
    logger.info(f"   Scanning {len(BIST_100)} symbols")
    logger.info("=" * 60)

    feed = BistFeed()
    groq = GroqEngine()
    sender = TelegramSender()
    db = Database()
    circuit_breaker = CircuitBreaker(db) if CIRCUIT_BREAKER_ENABLED else None

    signals_found = 0
    errors = 0

    # Pre-fetch macro data
    macro_result = {}
    try:
        macro_feed = MacroFeed()
        macro_data = macro_feed.fetch_all_current()
        macro_result = analyze_macro(macro_data, None, is_bist=True)
    except Exception as e:
        logger.warning(f"Macro fetch error: {e}")

    # Adaptive confidence threshold based on recent performance
    from src.signals.adaptive_threshold import get_adaptive_threshold
    adaptive_threshold = get_adaptive_threshold(db, is_crypto=False)
    if adaptive_threshold != MIN_CONFIDENCE:
        logger.info(f"BIST adaptive threshold: {MIN_CONFIDENCE} → {adaptive_threshold}")

    # Scan each symbol
    for i, symbol in enumerate(BIST_100):
        try:
            result = await scan_symbol(
                symbol, feed, groq, db, macro_result, circuit_breaker,
                min_confidence=adaptive_threshold,
            )
            sig = result.get("signal_data")

            if sig:
                # Format and send
                message = format_signal_message(
                    symbol=sig["symbol"],
                    direction=sig["direction"],
                    tier_name=sig["tier_name"],
                    confidence=sig["confidence"],
                    grade=sig["grade"],
                    indicators=sig["indicators"],
                    risk_mgmt=sig["risk_mgmt"],
                    is_bist=True,
                    ai_analysis=sig["ai_analysis"],
                    mtf_result=sig["mtf_result"],
                    sentiment=sig["sentiment"] if isinstance(sig["sentiment"], dict) else (sig["sentiment"].__dict__ if sig["sentiment"] and hasattr(sig["sentiment"], '__dict__') else None),
                    smart_money=sig["smart_money"],
                    macro=sig["macro"],
                    reasons=sig["reasons"],
                    time_estimates=sig.get("time_estimates"),
                )
                if sig["caution_note"]:
                    message += sig["caution_note"]

                sent = await sender.send_message(message)

                if sent:
                    signal_id = db.record_signal(
                        symbol=sig["symbol"],
                        direction=sig["direction"],
                        tier=sig["tier_name"],
                        confidence=sig["confidence"],
                        entry_price=sig["indicators"]["currentPrice"],
                        stop_loss=sig["risk_mgmt"].get("stop_loss", 0),
                        targets=sig["risk_mgmt"].get("targets", {}),
                        is_crypto=False,
                        features=sig.get("ml_features"),
                    )
                    db.set_cooldown(sig["symbol"], sig["direction"])
                    signals_found += 1
                    logger.info(f"✅ [{symbol}] {sig['direction']} signal sent ({sig['confidence']}%)")

                    # Open paper trade using live price right now
                    if PAPER_TRADING_ENABLED and signal_id:
                        try:
                            from src.paper_trading.executor import PaperTradeExecutor
                            from src.paper_trading.drawdown_guard import DrawdownGuard

                            guard = DrawdownGuard(db)
                            guard_info = guard.get_mode()

                            if not guard_info["can_trade"]:
                                logger.warning(
                                    f"[{symbol}] Paper trade BLOCKED by drawdown guard "
                                    f"(mode={guard_info['mode']})"
                                )
                            elif not guard.is_tier_allowed(sig["tier_name"], guard_info["mode"]):
                                logger.info(
                                    f"[{symbol}] Paper trade skipped: tier {sig['tier_name']} "
                                    f"not allowed in {guard_info['mode']} mode"
                                )
                            else:
                                executor = PaperTradeExecutor(db)
                                targets = sig["risk_mgmt"].get("targets", {})
                                trade = executor.open_trade(
                                    signal_id=signal_id,
                                    symbol=sig["symbol"],
                                    direction=sig["direction"],
                                    is_crypto=False,
                                    signal_tier=sig["tier_name"],
                                    signal_confidence=sig["confidence"],
                                    signal_sent_at=datetime.utcnow().isoformat(),
                                    signal_entry_price=sig["indicators"]["currentPrice"],
                                    stop_loss=sig["risk_mgmt"].get("stop_loss", 0),
                                    target1=targets.get("t1", 0),
                                    target2=targets.get("t2", 0),
                                    target3=targets.get("t3", 0),
                                    drawdown_position_mult=guard_info["position_mult"],
                                )
                                if trade:
                                    paper_msg = executor.format_trade_open_message(trade)
                                    await sender.send_message(paper_msg)
                        except Exception as e:
                            logger.error(f"[{symbol}] Paper trade open error: {e}")

            if result.get("error") and result["error"] not in ("cooldown", "no_data"):
                errors += 1

        except Exception as e:
            errors += 1
            logger.error(f"[{symbol}] Unhandled: {e}")

        # Early exit when max signals reached
        if signals_found >= MAX_SIGNALS_PER_BIST_RUN:
            logger.info(f"🛑 Max {MAX_SIGNALS_PER_BIST_RUN} BIST signals reached — stopping scan early")
            break

        # Progress
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i + 1}/{len(BIST_100)} ({signals_found} signals)")

    # Summary
    logger.info("=" * 60)
    logger.info(f"✅ BIST Scan Complete: {signals_found} signals, {errors} errors")
    logger.info("=" * 60)

    if signals_found > 0:
        await sender.send_message(
            f"🏛 <b>BIST Tarama Tamamlandı</b>\n\n"
            f"Taranan: {len(BIST_100)} sembol\n"
            f"Sinyal: {signals_found}\n"
            f"Hata: {errors}"
        )


if __name__ == "__main__":
    asyncio.run(main())
