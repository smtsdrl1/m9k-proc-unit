"""
Crypto Scanner — GitHub Actions entry point.
Scans all CRYPTO_SYMBOLS, detects signals, sends to Telegram.
Run via: python -m scripts.scan_crypto
"""
import asyncio
import logging
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CRYPTO_SYMBOLS, CRYPTO_TIMEFRAMES, CAPITAL, RISK_PERCENT,
    MIN_CONFIDENCE, SIGNAL_COOLDOWN_MINUTES,
    CIRCUIT_BREAKER_ENABLED, FUNDING_RATE_ENABLED,
    MAX_SIGNALS_PER_CRYPTO_RUN, SL_HIT_CONFIDENCE_BOOST, SL_HIT_LOOKBACK_HOURS,
)
from src.data.crypto_feed import CryptoFeed
from src.data.macro_feed import MacroFeed
from src.analysis.technical import calculate_indicators
from src.analysis.multi_timeframe import multi_timeframe_confluence
from src.analysis.smart_money import smart_money_analysis
from src.analysis.sentiment import fetch_crypto_news, keyword_sentiment_score
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
from src.utils.helpers import setup_logging

logger = logging.getLogger("matrix_trader.scan_crypto")


async def scan_symbol(
    symbol: str,
    feed: CryptoFeed,
    groq: GroqEngine,
    sender: TelegramSender,
    db: Database,
    macro_result: dict,
    circuit_breaker: CircuitBreaker = None,
    fear_greed: int = 50,
) -> dict:
    """Scan a single crypto symbol through the full pipeline."""
    result = {"symbol": symbol, "signal": False, "error": None}

    try:
        # 0. Circuit breaker check — skip scanning if system is paused
        if circuit_breaker and CIRCUIT_BREAKER_ENABLED:
            can_trade, cb_reason = circuit_breaker.can_trade()
            if not can_trade:
                result["error"] = f"circuit_breaker: {cb_reason}"
                logger.info(f"[{symbol}] ⚡ Circuit breaker active: {cb_reason}")
                return result

        # 1. Cooldown check
        if db.check_cooldown(symbol, SIGNAL_COOLDOWN_MINUTES):
            result["error"] = "cooldown"
            return result

        # 2. Fetch multi-timeframe data
        tf_data = await feed.fetch_multi_timeframe(symbol, CRYPTO_TIMEFRAMES)
        if not tf_data:
            result["error"] = "no_data"
            return result

        # 3. Calculate indicators for each timeframe
        tf_indicators = {}
        for tf, df in tf_data.items():
            ind = calculate_indicators(df)
            if ind:
                tf_indicators[tf] = ind

        if not tf_indicators:
            result["error"] = "no_indicators"
            return result

        # Use the highest timeframe for primary analysis
        primary_tf = list(tf_indicators.keys())[-1]
        indicators = tf_indicators[primary_tf]
        primary_df = tf_data[primary_tf]

        # 4. Multi-timeframe confluence
        mtf_result = multi_timeframe_confluence(tf_indicators)

        # 5. Smart money analysis
        sm_result = smart_money_analysis(primary_df, indicators["atr"])

        # 6. Signal detection
        signal = detect_signal(indicators, mtf_result, sm_result)

        # 6.1. Divergence fallback — when main signal is NEUTRAL, try RSI divergence
        if signal["direction"] == "NEUTRAL":
            div = check_divergence(primary_df, indicators)
            if div.get("direction") and div["direction"] != "NEUTRAL":
                signal = div
                logger.debug(f"[{symbol}] Divergence signal: {signal['direction']} ({signal['tier_name']})")
            else:
                return result

        # 6.2. Pre-trade filters — Session Killzone, Market Regime, News Kill
        signal = apply_pre_trade_filters(signal, primary_df, symbol)
        if signal["direction"] == "NEUTRAL":
            if signal.get("filtered_by"):
                logger.info(f"[{symbol}] Pre-trade filtered: {'; '.join(signal['filtered_by'])}")
            return result

        # 6.5. Circuit breaker — check direction limit
        if circuit_breaker and CIRCUIT_BREAKER_ENABLED:
            can_open, dir_reason = circuit_breaker.can_open_direction(signal["direction"])
            if not can_open:
                logger.info(f"[{symbol}] ⚡ Direction blocked: {dir_reason}")
                return result

        # 7. Macro filter — block if macro conditions are adverse
        filter_result = should_filter_signal(macro_result, signal["direction"], is_bist=False)
        if filter_result["action"] == "BLOCK":
            logger.info(f"[{symbol}] Signal blocked by macro filter: {filter_result['reason']}")
            return result

        # 8. Risk management
        risk_mgmt = calculate_risk(
            indicators["currentPrice"], indicators["atr"],
            indicators["sr"], signal["direction"],
            is_bist=False, capital=CAPITAL, risk_pct=RISK_PERCENT,
        )

        # 8.5. Pre-check confidence WITHOUT sentiment/AI — skip Groq if base score too low
        pre_score = calculate_confidence(
            indicators, signal["direction"],
            mtf_result, None, sm_result, macro_result,
            fear_greed=fear_greed,
            is_crypto=True,
            funding_rate=None,
            df=primary_df,
            symbol=symbol,
        )
        if pre_score["total"] < MIN_CONFIDENCE - 15:
            # Even with max sentiment boost, won't reach MIN_CONFIDENCE
            return result

        # 9. Sentiment (keyword-based — saves Groq budget for AI analysis)
        sentiment_result = None
        news_headlines = None
        try:
            news_headlines = await fetch_crypto_news(symbol.split("/")[0])
            if news_headlines:
                sentiment_result = keyword_sentiment_score(news_headlines)
        except Exception as e:
            logger.warning(f"[{symbol}] Sentiment error: {e}")

        # 9.5. Funding rate (crypto only)
        funding_rate = None
        if FUNDING_RATE_ENABLED:
            try:
                funding_rate = await feed.fetch_funding_rate(symbol)
            except Exception as e:
                logger.warning(f"[{symbol}] Funding rate error: {e}")

        # 10. Confidence scoring (with ML adjustment + funding rate + advanced df analysis)
        score_result = calculate_confidence(
            indicators, signal["direction"],
            mtf_result, sentiment_result, sm_result, macro_result,
            fear_greed=fear_greed,
            is_crypto=True,
            funding_rate=funding_rate,
            df=primary_df,
            symbol=symbol,
        )
        confidence = score_result["total"]
        grade = score_result["grade"]
        ml_features = score_result.get("features")  # Feature snapshot for ML training

        # 10.1. Fix tier_numeric in ML features — tier is known only after detect_signal()
        if ml_features and signal.get("tier_name"):
            _tier_map = {"EXTREME": 6, "STRONG": 5, "MODERATE": 4,
                         "SPECULATIVE": 3, "DIVERGENCE": 2, "CONTRARIAN": 1, "WEAK": 1}
            _tn = 0
            for _k, _v in _tier_map.items():
                if _k in signal["tier_name"].upper():
                    _tn = _v
                    break
            ml_features["tier_numeric"] = _tn

        if confidence < MIN_CONFIDENCE:
            return result

        # 10.5. SL hit recently? Raise confidence bar for re-entry.
        if db.was_sl_hit_recently(symbol, SL_HIT_LOOKBACK_HOURS):
            required = MIN_CONFIDENCE + SL_HIT_CONFIDENCE_BOOST
            if confidence < required:
                logger.info(
                    f"[{symbol}] SL hit recently → require confidence ≥{required} "
                    f"(got {confidence})"
                )
                return result
        valid, errors = validate_signal(
            symbol, indicators["currentPrice"], risk_mgmt,
            confidence, signal["direction"],
            is_bist=False, min_confidence=MIN_CONFIDENCE,
        )
        if not valid:
            logger.warning(f"[{symbol}] Validation failed: {errors}")
            return result

        # 12. AI Analysis — try Groq first, fallback to rule-based
        ai_analysis = None
        if groq.available:
            try:
                ai_analysis = groq.get_investment_analysis(
                    symbol, signal["direction"], indicators, risk_mgmt,
                    confidence, mtf_result, sentiment_result, sm_result,
                    macro_result, None, news=news_headlines, is_bist=False,
                )
            except Exception as e:
                logger.warning(f"[{symbol}] AI analysis error: {e}")
        
        # Fallback: rule-based analysis from real data (no random/fake data)
        if not ai_analysis:
            ai_analysis = GroqEngine.generate_fallback_analysis(
                symbol, signal["direction"], indicators, risk_mgmt,
                confidence, sentiment_result, sm_result, macro_result,
            )
            logger.info(f"[{symbol}] Using fallback AI analysis (rule-based)")

        # AI veto check
        if ai_analysis and ai_analysis.get("karar") == "REDDET":
            logger.info(f"[{symbol}] Signal vetoed by AI: {ai_analysis.get('yorum', '')[:100]}")
            return result

        # 12.5. Risk budget check
        if circuit_breaker and CIRCUIT_BREAKER_ENABLED:
            sl = risk_mgmt.get("stop_loss", 0)
            price = indicators["currentPrice"]
            if sl and price:
                new_risk_pct = abs(price - sl) / price * 100
                can_risk, risk_reason = circuit_breaker.check_risk_budget(new_risk_pct)
                if not can_risk:
                    logger.info(f"[{symbol}] ⚡ Risk budget exceeded: {risk_reason}")
                    return result

        # 13. Format message
        time_estimates = estimate_target_times(
            price=indicators["currentPrice"],
            targets=risk_mgmt.get("targets", {}),
            atr=indicators["atr"],
            adx=indicators.get("adx", 20),
            volume_ratio=indicators.get("volume_ratio", 1.0),
            is_bist=False,
            direction=signal["direction"],
            timeframe_atr=primary_tf,
        )
        message = format_signal_message(
            symbol=symbol,
            direction=signal["direction"],
            tier_name=signal["tier_name"],
            confidence=confidence,
            grade=grade,
            indicators=indicators,
            risk_mgmt=risk_mgmt,
            is_bist=False,
            ai_analysis=ai_analysis,
            mtf_result=mtf_result,
            sentiment=sentiment_result if isinstance(sentiment_result, dict) else (sentiment_result.__dict__ if sentiment_result and hasattr(sentiment_result, '__dict__') else None),
            smart_money=sm_result,
            macro=macro_result,
            reasons=signal.get("reasons", []),
            funding_rate=funding_rate,
            time_estimates=time_estimates,
        )

        # 14. Send to Telegram (text only — no chart photos)
        sent = await sender.send_message(message)

        if sent:
            # Record signal with ML feature snapshot
            db.record_signal(
                symbol=symbol,
                direction=signal["direction"],
                tier=signal["tier_name"],
                confidence=confidence,
                entry_price=indicators["currentPrice"],
                stop_loss=risk_mgmt.get("stop_loss", 0),
                targets=risk_mgmt.get("targets", {}),
                is_crypto=True,
                features=ml_features,
            )
            db.set_cooldown(symbol, signal["direction"])
            result["signal"] = True
            logger.info(f"✅ [{symbol}] {signal['direction']} signal sent (confidence: {confidence}%)")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{symbol}] Error: {e}\n{traceback.format_exc()}")

    return result


async def main():
    """Main scanner loop."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("🚀 Matrix Trader AI — Crypto Scanner Starting")
    logger.info(f"   Scanning {len(CRYPTO_SYMBOLS)} symbols")
    logger.info("=" * 60)

    feed = CryptoFeed()
    groq = GroqEngine()
    sender = TelegramSender()
    db = Database()
    circuit_breaker = CircuitBreaker(db) if CIRCUIT_BREAKER_ENABLED else None

    signals_found = 0
    errors = 0

    try:
        # Pre-fetch macro data
        macro_result = {}
        fear_greed_val = 50  # default neutral
        try:
            macro_feed = MacroFeed()
            macro_data = macro_feed.fetch_all_current()
            fear_greed_data = await macro_feed.fetch_fear_greed()
            if fear_greed_data:
                fear_greed_val = fear_greed_data.get("value", 50)
            macro_result = analyze_macro(macro_data, fear_greed_data, is_bist=False)
        except Exception as e:
            logger.warning(f"Macro fetch error: {e}")

        # Scan each symbol
        for i, symbol in enumerate(CRYPTO_SYMBOLS):
            try:
                result = await scan_symbol(symbol, feed, groq, sender, db, macro_result, circuit_breaker, fear_greed=fear_greed_val)
                if result["signal"]:
                    signals_found += 1
                if result["error"] and result["error"] not in ("cooldown", "no_data"):
                    errors += 1
            except Exception as e:
                errors += 1
                logger.error(f"[{symbol}] Unhandled error: {e}")

            # Early exit when max signals reached
            if signals_found >= MAX_SIGNALS_PER_CRYPTO_RUN:
                logger.info(f"🛑 Max {MAX_SIGNALS_PER_CRYPTO_RUN} signals reached — stopping scan early")
                break

            # Rate limiting
            if (i + 1) % 5 == 0:
                await asyncio.sleep(2)

            # Progress
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i + 1}/{len(CRYPTO_SYMBOLS)} ({signals_found} signals)")

    finally:
        await feed.close()

    # Summary
    logger.info("=" * 60)
    logger.info(f"✅ Scan Complete: {signals_found} signals, {errors} errors")
    logger.info("=" * 60)

    # Send summary if signals found
    if signals_found > 0:
        await sender.send_message(
            f"📊 <b>Kripto Tarama Tamamlandı</b>\n\n"
            f"Taranan: {len(CRYPTO_SYMBOLS)} sembol\n"
            f"Sinyal: {signals_found}\n"
            f"Hata: {errors}"
        )


if __name__ == "__main__":
    asyncio.run(main())
