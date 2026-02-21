"""
Crypto Scanner — Precision Mode (Target: ~95% Directional Accuracy).

Architecture: SNIPER — very few signals, each one near-certain.
- Ultra-strict 10-gate mandatory filter (ALL must pass)
- 12-system consensus engine (8+ FOR votes required)
- BTC trend bias (altcoins follow BTC macro direction)
- Restricted to Top 20 liquid symbols (less noise)
- BIST disabled — crypto-only focus
"""
import asyncio
import logging
import sys
import os
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ULTRA_CRYPTO_SYMBOLS, CRYPTO_SYMBOLS, CRYPTO_TIMEFRAMES, CAPITAL, RISK_PERCENT,
    MIN_CONFIDENCE, SIGNAL_COOLDOWN_MINUTES,
    CIRCUIT_BREAKER_ENABLED, FUNDING_RATE_ENABLED,
    MAX_SIGNALS_PER_CRYPTO_RUN, SL_HIT_CONFIDENCE_BOOST, SL_HIT_LOOKBACK_HOURS,
    PAPER_TRADING_ENABLED, ORDERBOOK_ENABLED, ONCHAIN_FEED_ENABLED,
    ULTRA_FILTER_ENABLED, ULTRA_CONFIDENCE_MIN,
    CONSENSUS_ENGINE_ENABLED, BTC_TREND_FILTER_ENABLED, BTC_TREND_STRICT_MODE,
    SCALP_MODE_ENABLED, SCALP_CONFIDENCE_MIN, SCALP_CONSENSUS_REQUIRED,
    SCALP_ADX_MIN, SCALP_VOLUME_RATIO_MIN, SCALP_RR_MIN,
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
from src.signals.ultra_filter import UltraFilter
from src.signals.consensus_engine import ConsensusEngine
from src.ai.groq_engine import GroqEngine
from src.telegram.formatter import format_signal_message
from src.telegram.sender import TelegramSender
from src.database.db import Database
from src.utils.helpers import setup_logging

logger = logging.getLogger("matrix_trader.scan_crypto")

# Active symbol list — top 20 liquid in precision mode
_ACTIVE_SYMBOLS = ULTRA_CRYPTO_SYMBOLS if ULTRA_FILTER_ENABLED else CRYPTO_SYMBOLS

_ultra_filter   = UltraFilter()
_consensus      = ConsensusEngine()


async def scan_symbol(
    symbol: str,
    feed: CryptoFeed,
    groq: GroqEngine,
    sender: TelegramSender,
    db: Database,
    macro_result: dict,
    circuit_breaker: CircuitBreaker = None,
    fear_greed: int = 50,
    min_confidence: int = None,
    onchain_data: dict = None,
    btc_trend: dict = None,
    mode: str = "SWING",                   # "SWING" = precision 4h | "SCALP" = fast 15m
    prefetched_tf_data: dict = None,       # pre-fetched OHLCV (avoids double fetch)
    prefetched_tf_indicators: dict = None, # pre-computed indicators
) -> dict:
    """Scan a single crypto symbol — SWING (precision 4h) or SCALP (fast 15m) mode."""
    result = {"symbol": symbol, "signal": False, "error": None}

    # Threshold depends on mode: scalp is more lenient, swing uses ultra/adaptive
    if mode == "SCALP":
        effective_min_confidence = SCALP_CONFIDENCE_MIN
    elif ULTRA_FILTER_ENABLED:
        effective_min_confidence = ULTRA_CONFIDENCE_MIN
    else:
        effective_min_confidence = min_confidence if min_confidence is not None else MIN_CONFIDENCE

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

        # 2. Fetch multi-timeframe data (use prefetched if available — avoids double fetch)
        if prefetched_tf_data is not None:
            tf_data = prefetched_tf_data
            tf_indicators = prefetched_tf_indicators or {}
        else:
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

        # Select primary timeframe based on mode
        if mode == "SCALP":
            # Prefer shorter TFs for scalp (15m → 5m → 1h fallback)
            for preferred_tf in ["15m", "5m", "1h"]:
                if preferred_tf in tf_indicators:
                    primary_tf = preferred_tf
                    break
            else:
                primary_tf = list(tf_indicators.keys())[0]
        else:
            # SWING: use highest available TF for macro picture
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
        if pre_score["total"] < effective_min_confidence - 15:
            # Even with max sentiment boost, won't reach effective threshold
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

        # 9.6. Order book analysis (crypto only)
        ob_data = None
        if ORDERBOOK_ENABLED:
            try:
                ob_data = await feed.fetch_order_book(symbol, limit=20)
            except Exception as e:
                logger.debug(f"[{symbol}] Order book error: {e}")

        # 9.7. On-chain confidence boost from pre-fetched global data
        onchain_boost = None
        if ONCHAIN_FEED_ENABLED and onchain_data:
            try:
                from src.data.onchain_feed import get_onchain_confidence_boost
                onchain_boost = get_onchain_confidence_boost(
                    onchain_data, symbol, signal["direction"], is_crypto=True
                )
            except Exception as e:
                logger.debug(f"[{symbol}] OnChain boost error: {e}")

        # 10. Confidence scoring (with ML adjustment + funding rate + OB + on-chain + advanced df)
        score_result = calculate_confidence(
            indicators, signal["direction"],
            mtf_result, sentiment_result, sm_result, macro_result,
            fear_greed=fear_greed,
            is_crypto=True,
            funding_rate=funding_rate,
            df=primary_df,
            symbol=symbol,
            order_book=ob_data,
            onchain=onchain_boost,
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

        if confidence < effective_min_confidence:
            return result

        # ── Precision Gate A: BTC Trend Alignment ────────────────
        if BTC_TREND_FILTER_ENABLED and btc_trend:
            from src.analysis.btc_trend import is_signal_aligned_with_btc
            btc_ok, btc_reason = is_signal_aligned_with_btc(
                symbol, signal["direction"], btc_trend,
                strict=BTC_TREND_STRICT_MODE,
            )
            if not btc_ok:
                logger.info(f"[{symbol}] BTC trend block: {btc_reason}")
                result["error"] = "btc_trend_block"
                return result

        # ── Precision Gate B: Mode-based mandatory filter ─────────
        if mode == "SWING" and ULTRA_FILTER_ENABLED:
            # SWING: full 10-gate ultra filter (strict)
            uf_result = _ultra_filter.check(
                indicators=indicators,
                signal_direction=signal["direction"],
                risk_mgmt=risk_mgmt,
                mtf_result=mtf_result,
                sm_result=sm_result,
                funding_rate=funding_rate,
                confidence=confidence,
                ob_data=ob_data,
            )
            if not uf_result["passes"]:
                logger.info(_ultra_filter.format_rejection_log(
                    uf_result, symbol, signal["direction"]
                ))
                result["error"] = "ultra_filter"
                return result
            logger.debug(
                f"[{symbol}][SWING] UltraFilter PASS "
                f"({uf_result['gates_passed']}/{uf_result['gates_total']}) "
                f"entry={uf_result['entry_quality']}"
            )
        elif mode == "SCALP":
            # SCALP: lighter manual checks (ADX + volume + R:R)
            adx_val  = indicators.get("adx", 0)
            vol_rat  = indicators.get("volume_ratio", 1.0)
            rr_val   = risk_mgmt.get("risk_reward_ratio", 0) or 0
            if adx_val < SCALP_ADX_MIN:
                result["error"] = f"scalp_adx_low"
                return result
            if vol_rat < SCALP_VOLUME_RATIO_MIN:
                result["error"] = f"scalp_vol_low"
                return result
            if rr_val < SCALP_RR_MIN:
                result["error"] = f"scalp_rr_low"
                return result

        # ── Precision Gate C: 12-System Consensus Engine ──────────
        cv_result = None
        if CONSENSUS_ENGINE_ENABLED:
            cv_result = _consensus.vote(
                indicators=indicators,
                direction=signal["direction"],
                mtf_result=mtf_result,
                sm_result=sm_result,
                funding_rate=funding_rate,
                fear_greed=fear_greed,
                order_book=ob_data,
                df=primary_df,
            )
            # Mode-specific consensus threshold
            if mode == "SCALP":
                consensus_ok = (
                    cv_result["net_for"] >= SCALP_CONSENSUS_REQUIRED
                    and cv_result["net_against"] <= 1
                )
            else:
                consensus_ok = cv_result["passes"]  # CONSENSUS_REQUIRED=8, max_against=1

            if not consensus_ok:
                need = SCALP_CONSENSUS_REQUIRED if mode == "SCALP" else 8
                logger.info(
                    f"[{symbol}][{mode}] Consensus FAIL: "
                    f"{cv_result['net_for']}✅ {cv_result['net_against']}❌ "
                    f"({cv_result['agreement_pct']:.0f}%) — need {need}✅ max 1❌"
                )
                result["error"] = "consensus_fail"
                return result

        # 10.5. SL hit recently? Raise confidence bar for re-entry.
        if db.was_sl_hit_recently(symbol, SL_HIT_LOOKBACK_HOURS):
            required = effective_min_confidence + SL_HIT_CONFIDENCE_BOOST
            if confidence < required:
                logger.info(
                    f"[{symbol}] SL hit recently → require confidence ≥{required} "
                    f"(got {confidence})"
                )
                return result
        valid, errors = validate_signal(
            symbol, indicators["currentPrice"], risk_mgmt,
            confidence, signal["direction"],
            is_bist=False, min_confidence=effective_min_confidence,
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

        # 13.5 Append consensus info + mode tag to message
        if CONSENSUS_ENGINE_ENABLED and cv_result is not None:
            try:
                consensus_line = _consensus.format_consensus_line(cv_result)
                mode_tag = "⚡ SCALP (15m)" if mode == "SCALP" else "📊 SWING (4h)"
                message += f"\n{consensus_line}\n🏷 Mod: {mode_tag}"
            except Exception:
                pass

        # 14. Send to Telegram (text only — no chart photos)
        sent = await sender.send_message(message)

        if sent:
            # Record signal with ML feature snapshot
            signal_id = db.record_signal(
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
            logger.info(f"✅ [{symbol}][{mode}] {signal['direction']} signal sent (confidence: {confidence}%)")

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
                            f"(mode={guard_info['mode']}, DD={guard_info['drawdown_pct']}%)"
                        )
                    else:
                        # Check if tier is allowed under current drawdown mode
                        if not guard.is_tier_allowed(signal["tier_name"], guard_info["mode"]):
                            logger.info(
                                f"[{symbol}] Paper trade skipped: tier {signal['tier_name']} "
                                f"not allowed in {guard_info['mode']} mode"
                            )
                        else:
                            executor = PaperTradeExecutor(db)
                            targets = risk_mgmt.get("targets", {})
                            trade = executor.open_trade(
                                signal_id=signal_id,
                                symbol=symbol,
                                direction=signal["direction"],
                                is_crypto=True,
                                signal_tier=signal["tier_name"],
                                signal_confidence=confidence,
                                signal_sent_at=datetime.utcnow().isoformat(),
                                signal_entry_price=indicators["currentPrice"],
                                stop_loss=risk_mgmt.get("stop_loss", 0),
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

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{symbol}] Error: {e}\n{traceback.format_exc()}")

    return result


async def main():
    """Main scanner loop — Precision Mode."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("🎯 Matrix Trader AI — Multi-Mode Crypto Scanner")
    logger.info(f"   Symbols: {len(_ACTIVE_SYMBOLS)} (top-20 liquid)")
    logger.info(f"   Modes: {'SCALP(15m)+SWING(4h)' if SCALP_MODE_ENABLED else 'SWING(4h) only'}")
    logger.info(f"   TFs: {CRYPTO_TIMEFRAMES}")
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

        # Pre-fetch on-chain data (single call for all symbols)
        onchain_global = None
        if ONCHAIN_FEED_ENABLED:
            try:
                from src.data.onchain_feed import fetch_global_data
                onchain_global = fetch_global_data()
                if onchain_global:
                    logger.info(
                        f"OnChain: BTC.D={onchain_global['btc_dominance']}%, "
                        f"market_chg={onchain_global['market_cap_change_24h']}%"
                    )
            except Exception as e:
                logger.warning(f"OnChain fetch error: {e}")

        # Pre-fetch BTC trend for altcoin bias filter
        btc_trend_data = None
        if BTC_TREND_FILTER_ENABLED:
            try:
                from src.analysis.btc_trend import get_btc_trend
                btc_trend_data = get_btc_trend(feed)
                logger.info(f"BTC Trend: {btc_trend_data['description']}")
            except Exception as e:
                logger.warning(f"BTC trend fetch error: {e}")

        # Adaptive confidence threshold (only used when ultra filter is OFF)
        adaptive_threshold = MIN_CONFIDENCE
        if not ULTRA_FILTER_ENABLED:
            try:
                from src.signals.adaptive_threshold import get_adaptive_threshold
                adaptive_threshold = get_adaptive_threshold(db, is_crypto=True)
                if adaptive_threshold != MIN_CONFIDENCE:
                    logger.info(f"Adaptive threshold: {MIN_CONFIDENCE} → {adaptive_threshold}")
            except Exception:
                pass

        # Scan each symbol — fetch data ONCE, then run SCALP→SWING in sequence
        _skip_errors = {
            "cooldown", "no_data", "circuit_breaker", "btc_trend_block",
            "ultra_filter", "consensus_fail",
            "scalp_adx_low", "scalp_vol_low", "scalp_rr_low",
        }
        for i, symbol in enumerate(_ACTIVE_SYMBOLS):
            # ── Pre-fetch all TF data once per symbol ───────────────
            try:
                pf_data = await feed.fetch_multi_timeframe(symbol, CRYPTO_TIMEFRAMES)
                pf_ind = {}
                if pf_data:
                    for tf, df in pf_data.items():
                        ind = calculate_indicators(df)
                        if ind:
                            pf_ind[tf] = ind
            except Exception as e:
                logger.error(f"[{symbol}] Data fetch error: {e}")
                errors += 1
                continue

            if not pf_ind:
                continue

            signal_sent = False

            # ── Pass 1: SCALP mode (15m primary, lighter thresholds) ─
            if SCALP_MODE_ENABLED:
                try:
                    r = await scan_symbol(
                        symbol, feed, groq, sender, db, macro_result,
                        circuit_breaker,
                        fear_greed=fear_greed_val,
                        min_confidence=adaptive_threshold,
                        onchain_data=onchain_global,
                        btc_trend=btc_trend_data,
                        mode="SCALP",
                        prefetched_tf_data=pf_data,
                        prefetched_tf_indicators=pf_ind,
                    )
                    if r.get("signal"):
                        signals_found += 1
                        signal_sent = True
                    elif r.get("error") and r["error"] not in _skip_errors:
                        errors += 1
                except Exception as e:
                    logger.error(f"[{symbol}][SCALP] Error: {e}")
                    errors += 1

            # ── Pass 2: SWING mode (4h primary, full precision) ──────
            if not signal_sent:
                try:
                    r = await scan_symbol(
                        symbol, feed, groq, sender, db, macro_result,
                        circuit_breaker,
                        fear_greed=fear_greed_val,
                        min_confidence=adaptive_threshold,
                        onchain_data=onchain_global,
                        btc_trend=btc_trend_data,
                        mode="SWING",
                        prefetched_tf_data=pf_data,
                        prefetched_tf_indicators=pf_ind,
                    )
                    if r.get("signal"):
                        signals_found += 1
                    elif r.get("error") and r["error"] not in _skip_errors:
                        errors += 1
                except Exception as e:
                    logger.error(f"[{symbol}][SWING] Error: {e}")
                    errors += 1

            # Early exit when max signals reached
            if signals_found >= MAX_SIGNALS_PER_CRYPTO_RUN:
                logger.info(f"🛑 Max {MAX_SIGNALS_PER_CRYPTO_RUN} signals reached — stopping scan early")
                break

            # Rate limiting
            if (i + 1) % 5 == 0:
                await asyncio.sleep(2)

            # Progress
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(_ACTIVE_SYMBOLS)} | signals={signals_found}")

    finally:
        await feed.close()

    # Summary
    logger.info("=" * 60)
    logger.info(f"✅ Scan Complete: {signals_found} signals, {errors} errors")
    logger.info("=" * 60)

    # Send summary if signals found
    if signals_found > 0:
        mode_tag = "🎯 PRECISION" if ULTRA_FILTER_ENABLED else "📊 STANDARD"
        await sender.send_message(
            f"{mode_tag} <b>Kripto Tarama Tamamlandı</b>\n\n"
            f"Taranan: {len(_ACTIVE_SYMBOLS)} sembol\n"
            f"Sinyal: {signals_found}\n"
            f"BTC Trend: {btc_trend_data.get('trend','?') if btc_trend_data else 'N/A'}\n"
            f"Hata: {errors}"
        )


if __name__ == "__main__":
    asyncio.run(main())
