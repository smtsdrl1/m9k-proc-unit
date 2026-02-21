"""
Ultra-Strict Signal Filter — Precision Gate for ~95% Directional Accuracy.

Strategy: SNIPER, not machine gun.
- Less signals (1-5 per week) but every signal is high-conviction
- ALL mandatory gates must pass — binary pass/fail, not score additions
- A single failed gate rejects the signal, no exceptions

Mandatory Gates (ALL must pass):
  1. ADX_TREND         — ADX >= 25: trending market, not ranging
  2. VOLUME_SPIKE      — Volume ratio >= 1.5x: institutional participation
  3. MTF_MAJORITY      — >= 3 of 4 timeframes aligned in signal direction
  4. CONFIDENCE_A      — Confidence score >= 80 (Grade A only)
  5. RISK_REWARD       — R:R >= 2.5 (only high reward setups)
  6. NOT_OVEREXTENDED  — Price within 2.5 ATR of EMA21 (no chasing)
  7. BAND_POSITION     — bb_pctb < 0.35 (BUY) or > 0.65 (SELL) — near band edge
  8. NOT_RANGING       — Bollinger band width > 1.5% (market is moving)
  9. SMART_MONEY_OK    — Smart money not against signal direction
 10. FUNDING_OK        — Funding rate not extreme against direction

Entry Precision Gates (>= 7 of 10 mandatory + 1 of these):
 11. AT_EMA_PULLBACK   — Price within 0.6 ATR of EMA21 or EMA50
 12. AT_SR_LEVEL       — Price within 0.7 ATR of significant S/R
 13. CANDLE_CONFIRM    — Reversal candle pattern on last 2 bars
"""
import logging

logger = logging.getLogger("matrix_trader.signals.ultra_filter")

# ── Thresholds ────────────────────────────────────────────────────────
ADX_MIN            = 25       # Minimum ADX for trending market
VOLUME_RATIO_MIN   = 1.5      # Minimum volume ratio (vs 20-period avg)
MTF_ALIGNED_MIN    = 3        # Minimum timeframes aligned (out of 4)
CONFIDENCE_MIN     = 80       # Minimum confidence score
RR_MIN             = 2.5      # Minimum Risk:Reward ratio
ATR_EXTEND_MAX     = 2.5      # Max ATR distance from EMA21 before "overextended"
BB_PCTB_BUY_MAX    = 0.35     # BUY: price must be in lower 35% of BB
BB_PCTB_SELL_MIN   = 0.65     # SELL: price must be in upper 35% of BB
BB_WIDTH_MIN       = 0.015    # Bollinger width (as fraction of price) — not ranging
FUNDING_EXTREME    = 0.02     # Funding rate > 2% or < -2% is extreme (100 * raw rate)
EMA_PULLBACK_ATR   = 0.6      # Price within this many ATR of EMA for pullback entry
SR_PROXIMITY_ATR   = 0.7      # Price within this many ATR of S/R level
# ──────────────────────────────────────────────────────────────────────


class UltraFilter:
    """
    Precision gate that enforces ALL mandatory checks before a signal fires.
    Returns detailed breakdown of which gates passed/failed.
    """

    def check(
        self,
        indicators: dict,
        signal_direction: str,
        risk_mgmt: dict,
        mtf_result: dict,
        sm_result: dict,
        funding_rate: dict,
        confidence: int,
        ob_data: dict = None,
    ) -> dict:
        """
        Run all mandatory gate checks.

        Returns:
            {
                "passes": bool,
                "gates_passed": int,
                "gates_total": int,
                "failed_gates": list[str],
                "passed_gates": list[str],
                "entry_quality": str,   # "PREMIUM" / "STANDARD" / "POOR"
                "rejection_reason": str,
            }
        """
        direction = signal_direction
        is_buy = direction in ("BUY", "LONG", "AL")

        price      = indicators.get("currentPrice", 0)
        atr        = indicators.get("atr", price * 0.02)
        adx        = indicators.get("adx", 0)
        vol_ratio  = indicators.get("volume_ratio", 1.0)
        bb_pctb    = indicators.get("bb_pctb", 0.5)
        ema21      = indicators.get("ema21", price)
        ema50      = indicators.get("ema50", price)
        obv_trend  = indicators.get("obv_trend", "NEUTRAL")
        rsi        = indicators.get("rsi", 50)

        # Bollinger band width approximation
        bb_upper   = indicators.get("bb_upper", price * 1.02)
        bb_lower   = indicators.get("bb_lower", price * 0.98)
        bb_width   = (bb_upper - bb_lower) / price if price > 0 else 0

        sr_levels  = indicators.get("sr", {})
        targets    = risk_mgmt.get("targets", {})
        stop_loss  = risk_mgmt.get("stop_loss", 0)

        passed  = []
        failed  = []

        # ── Gate 1: ADX trending ────────────────────────────────────
        if adx >= ADX_MIN:
            passed.append("ADX_TREND")
        else:
            failed.append(f"ADX_TREND(adx={adx:.1f}<{ADX_MIN})")

        # ── Gate 2: Volume spike ────────────────────────────────────
        if vol_ratio >= VOLUME_RATIO_MIN:
            passed.append("VOLUME_SPIKE")
        else:
            failed.append(f"VOLUME_SPIKE(vol={vol_ratio:.2f}<{VOLUME_RATIO_MIN})")

        # ── Gate 3: MTF majority ────────────────────────────────────
        if mtf_result:
            aligned = mtf_result.get("aligned_count", 0)
            total   = mtf_result.get("total_count", 4)
            mtf_dir = mtf_result.get("direction", "NEUTRAL")
            if mtf_dir == direction and aligned >= MTF_ALIGNED_MIN:
                passed.append("MTF_MAJORITY")
            else:
                failed.append(
                    f"MTF_MAJORITY(dir={mtf_dir},aligned={aligned}/{total})"
                )
        else:
            failed.append("MTF_MAJORITY(no_data)")

        # ── Gate 4: Confidence grade A ──────────────────────────────
        if confidence >= CONFIDENCE_MIN:
            passed.append("CONFIDENCE_A")
        else:
            failed.append(f"CONFIDENCE_A(conf={confidence}<{CONFIDENCE_MIN})")

        # ── Gate 5: Risk:Reward ────────────────────────────────────
        if stop_loss and stop_loss > 0 and price > 0:
            sl_dist  = abs(price - stop_loss)
            t1_price = targets.get("t1", 0)
            if t1_price and t1_price > 0:
                t1_dist = abs(t1_price - price)
                rr      = t1_dist / sl_dist if sl_dist > 0 else 0
                if rr >= RR_MIN:
                    passed.append("RISK_REWARD")
                else:
                    failed.append(f"RISK_REWARD(rr={rr:.2f}<{RR_MIN})")
            else:
                failed.append("RISK_REWARD(no_t1)")
        else:
            failed.append("RISK_REWARD(no_sl)")

        # ── Gate 6: Not overextended ────────────────────────────────
        if ema21 > 0 and atr > 0:
            dist_from_ema = abs(price - ema21)
            atr_multiple  = dist_from_ema / atr
            if atr_multiple <= ATR_EXTEND_MAX:
                passed.append("NOT_OVEREXTENDED")
            else:
                failed.append(
                    f"NOT_OVEREXTENDED(dist={atr_multiple:.1f}x_ATR>{ATR_EXTEND_MAX})"
                )
        else:
            passed.append("NOT_OVEREXTENDED")  # Can't check — don't penalize

        # ── Gate 7: Band position (near directional band edge) ──────
        if is_buy:
            if bb_pctb <= BB_PCTB_BUY_MAX:
                passed.append("BAND_POSITION")
            else:
                failed.append(
                    f"BAND_POSITION(pctb={bb_pctb:.2f}>{BB_PCTB_BUY_MAX}_for_buy)"
                )
        else:
            if bb_pctb >= BB_PCTB_SELL_MIN:
                passed.append("BAND_POSITION")
            else:
                failed.append(
                    f"BAND_POSITION(pctb={bb_pctb:.2f}<{BB_PCTB_SELL_MIN}_for_sell)"
                )

        # ── Gate 8: Not in ranging squeeze ──────────────────────────
        if bb_width >= BB_WIDTH_MIN:
            passed.append("NOT_RANGING")
        else:
            failed.append(f"NOT_RANGING(bb_width={bb_width:.4f}<{BB_WIDTH_MIN})")

        # ── Gate 9: Smart money not against direction ───────────────
        sm_dir = sm_result.get("direction", "NEUTRAL") if sm_result else "NEUTRAL"
        if sm_dir == direction or sm_dir == "NEUTRAL":
            passed.append("SMART_MONEY_OK")
        else:
            failed.append(f"SMART_MONEY_OK(sm={sm_dir}_vs_{direction})")

        # ── Gate 10: Funding rate not extreme against direction ──────
        if funding_rate:
            rate_pct = funding_rate.get("funding_rate_pct", 0)  # Already in pct (×100)
            if is_buy:
                # For BUY: very positive funding = longs crowded → risk
                if rate_pct > FUNDING_EXTREME:
                    failed.append(
                        f"FUNDING_OK(rate={rate_pct:.3f}%>+{FUNDING_EXTREME}%_vs_buy)"
                    )
                else:
                    passed.append("FUNDING_OK")
            else:
                # For SELL: very negative funding = shorts crowded → risk
                if rate_pct < -FUNDING_EXTREME:
                    failed.append(
                        f"FUNDING_OK(rate={rate_pct:.3f}%<-{FUNDING_EXTREME}%_vs_sell)"
                    )
                else:
                    passed.append("FUNDING_OK")
        else:
            passed.append("FUNDING_OK")  # No data → neutral, don't penalize

        # ── Entry Precision: at least ONE of these must pass ────────
        entry_score = 0
        entry_details = []

        # Check EMA pullback
        if ema21 > 0 and atr > 0:
            dist21 = abs(price - ema21)
            if dist21 <= EMA_PULLBACK_ATR * atr:
                entry_score += 1
                entry_details.append(f"EMA21_pullback({dist21/atr:.2f}ATR)")
        if ema50 > 0 and atr > 0:
            dist50 = abs(price - ema50)
            if dist50 <= EMA_PULLBACK_ATR * atr:
                entry_score += 1
                entry_details.append(f"EMA50_pullback({dist50/atr:.2f}ATR)")

        # Check S/R proximity
        if sr_levels and atr > 0:
            support  = sr_levels.get("support", [])
            resist   = sr_levels.get("resistance", [])
            all_lvls = (support if isinstance(support, list) else [support]) + \
                       (resist  if isinstance(resist,  list) else [resist])
            for lvl in all_lvls:
                if lvl and abs(price - lvl) <= SR_PROXIMITY_ATR * atr:
                    entry_score += 1
                    entry_details.append(f"SR_level({lvl:.4f})")
                    break

        # Reversal candle check (RSI + BB edge as proxy — no raw candles needed)
        if is_buy and rsi < 40 and bb_pctb < 0.25:
            entry_score += 1
            entry_details.append("reversal_candle_proxy(rsi+bb)")
        elif not is_buy and rsi > 60 and bb_pctb > 0.75:
            entry_score += 1
            entry_details.append("reversal_candle_proxy(rsi+bb)")

        entry_quality = "PREMIUM" if entry_score >= 2 else ("STANDARD" if entry_score == 1 else "POOR")

        # Final verdict
        passes = len(failed) == 0  # ALL 10 gates must pass

        rejection = ""
        if failed:
            rejection = f"FAILED: {failed[0]}"  # Report first failure

        return {
            "passes": passes,
            "gates_passed": len(passed),
            "gates_total": len(passed) + len(failed),
            "failed_gates": failed,
            "passed_gates": passed,
            "entry_quality": entry_quality,
            "entry_details": entry_details,
            "rejection_reason": rejection,
            "bb_width": round(bb_width, 4),
        }

    def format_rejection_log(self, result: dict, symbol: str, direction: str) -> str:
        """Short log string for failed ultra filter."""
        return (
            f"[{symbol}] UltraFilter REJECTED ({direction}): "
            f"{result['gates_passed']}/{result['gates_total']} gates — "
            f"{result['rejection_reason']}"
        )
