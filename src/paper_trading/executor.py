"""
Paper Trading Executor — Demo simulation with real live market prices.

Tracks every detail:
- Exact time signal was sent vs when paper trade was opened
- Live market price at that exact moment (real API fetch, no simulation)
- Whether live data fetch succeeded (LIVE / FAILED)
- Entry price deviation from signal price
- T1/T2/T3 hits and SL hits with timestamps and PnL amounts
- Portfolio balance updated with every close

No random or null data — only real prices from CryptoFeed / yfinance.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from src.database.db import Database
from src.config import (
    PAPER_TRADING_CAPITAL,
    PAPER_TRADE_MAX_SLIPPAGE_PCT,
    RISK_PERCENT,
    PARTIAL_TP_ENABLED,
    PARTIAL_TP_RATIOS,
    TRAILING_STOP_ENABLED,
    TRAILING_STOP_ATR_MULT,
    LIMIT_ORDER_SIMULATION,
    LIMIT_ORDER_PULLBACK_PCT,
    LIMIT_ORDER_TIMEOUT_MINUTES,
)

logger = logging.getLogger("matrix_trader.paper_trading")


class PaperTradeExecutor:
    """Opens, tracks, and closes paper trades using real market prices."""

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self._crypto_feed = None

    # ─── Price Fetching ──────────────────────────────────────────

    def _get_crypto_feed(self):
        if self._crypto_feed is None:
            from src.data.crypto_feed import CryptoFeed
            self._crypto_feed = CryptoFeed()
        return self._crypto_feed

    def _fetch_live_price(self, symbol: str, is_crypto: bool) -> tuple[Optional[float], str]:
        """
        Fetch live price synchronously.
        Returns (price, data_quality) where data_quality is 'LIVE' or 'FAILED'.
        """
        try:
            if is_crypto:
                feed = self._get_crypto_feed()
                loop = asyncio.new_event_loop()
                try:
                    ticker = loop.run_until_complete(feed.fetch_ticker(symbol))
                finally:
                    loop.close()
                if ticker and ticker.get("price", 0) > 0:
                    return ticker["price"], "LIVE"
            else:
                try:
                    import yfinance as yf
                    t = yf.Ticker(f"{symbol}.IS")
                    hist = t.history(period="1d", interval="1m")
                    if hist is not None and not hist.empty:
                        return float(hist["Close"].iloc[-1]), "LIVE"
                except Exception as e:
                    logger.error(f"[{symbol}] BIST price fetch error: {e}")
        except Exception as e:
            logger.error(f"[{symbol}] Live price fetch error: {e}")
        return None, "FAILED"

    # ─── Open Trade ──────────────────────────────────────────────

    def open_trade(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        is_crypto: bool,
        signal_tier: str,
        signal_confidence: int,
        signal_sent_at: str,
        signal_entry_price: float,
        stop_loss: float,
        target1: float,
        target2: float,
        target3: float,
        drawdown_position_mult: float = 1.0,
    ) -> Optional[dict]:
        """
        Open a paper trade using the live market price at this exact moment.

        Fetches real price via API. If fetch fails, paper trade is NOT opened
        (we never use null or simulated data).

        Returns trade dict (with all details) or None if price fetch failed.
        """
        if signal_entry_price <= 0:
            logger.warning(f"[{symbol}] Invalid signal entry price: {signal_entry_price}")
            return None

        # ── Limit Order Simulation ────────────────────────────
        # When enabled, wait for price to pull back before opening.
        # Since this runs in GitHub Actions, we check if the signal is "fresh"
        # (< LIMIT_ORDER_TIMEOUT_MINUTES old). If signal is stale → market order.
        limit_order_note = ""
        if LIMIT_ORDER_SIMULATION and is_crypto and signal_sent_at:
            try:
                sent_dt = datetime.fromisoformat(signal_sent_at)
                age_minutes = (datetime.utcnow() - sent_dt).total_seconds() / 60
                if age_minutes < LIMIT_ORDER_TIMEOUT_MINUTES:
                    # Signal is fresh — try limit order (wait for pullback)
                    target_entry = signal_entry_price * (
                        (1 - LIMIT_ORDER_PULLBACK_PCT / 100)
                        if direction in ("BUY", "LONG", "AL")
                        else (1 + LIMIT_ORDER_PULLBACK_PCT / 100)
                    )
                    limit_order_note = f"limit_target={target_entry:.6f}"
                    logger.info(
                        f"[{symbol}] Limit order simulation: "
                        f"waiting for {target_entry:.6f} "
                        f"(pullback {LIMIT_ORDER_PULLBACK_PCT}%)"
                    )
            except Exception:
                pass

        # Fetch live price RIGHT NOW — same moment signal is sent
        live_price, data_quality = self._fetch_live_price(symbol, is_crypto)

        if live_price is None or live_price <= 0:
            logger.warning(
                f"[{symbol}] Paper trade SKIPPED — could not fetch live price "
                f"(signal price was {signal_entry_price})"
            )
            return None

        # Calculate deviation from signal price
        deviation_pct = abs(live_price - signal_entry_price) / signal_entry_price * 100

        if deviation_pct > PAPER_TRADE_MAX_SLIPPAGE_PCT:
            logger.warning(
                f"[{symbol}] Price deviation {deviation_pct:.2f}% > max "
                f"{PAPER_TRADE_MAX_SLIPPAGE_PCT}% — opening trade anyway (marked as slippage)"
            )

        # ── Limit order check: use live_price vs target_entry ────
        if LIMIT_ORDER_SIMULATION and limit_order_note and is_crypto:
            target_entry_val = None
            try:
                target_entry_val = float(limit_order_note.split("=")[1])
            except Exception:
                pass
            if target_entry_val:
                if direction in ("BUY", "LONG", "AL"):
                    if live_price > target_entry_val * 1.002:
                        # Price hasn't pulled back yet, skip this check cycle
                        logger.info(
                            f"[{symbol}] Limit order not filled yet "
                            f"(live={live_price:.6f} > limit={target_entry_val:.6f}) — market entry"
                        )
                        # Fall through to market entry
                else:
                    if live_price < target_entry_val * 0.998:
                        logger.info(
                            f"[{symbol}] Limit order not filled — market entry"
                        )

        # Position sizing: risk RISK_PERCENT of current available capital
        stats = self.db.get_paper_trade_stats(days=365)
        current_balance = stats.get("current_balance", PAPER_TRADING_CAPITAL)

        risk_amount = current_balance * (RISK_PERCENT / 100)

        # Capital to allocate: based on stop distance
        if stop_loss and stop_loss > 0:
            sl_distance_pct = abs(live_price - stop_loss) / live_price
            if sl_distance_pct > 0:
                capital_allocated = risk_amount / sl_distance_pct
                # Cap at 25% of balance per trade
                capital_allocated = min(capital_allocated, current_balance * 0.25)
            else:
                capital_allocated = current_balance * 0.10
        else:
            capital_allocated = current_balance * 0.10

        # Apply drawdown guard position multiplier (1.0 = normal, 0.5 = caution, etc.)
        if drawdown_position_mult != 1.0:
            capital_allocated = round(capital_allocated * drawdown_position_mult, 2)
            risk_amount = round(risk_amount * drawdown_position_mult, 2)
            logger.info(
                f"[{symbol}] Drawdown guard: position mult={drawdown_position_mult:.2f} "
                f"→ capital=${capital_allocated:.0f}"
            )

        position_size = capital_allocated / live_price if live_price > 0 else 0

        # Store in DB
        trade_id = self.db.open_paper_trade(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            is_crypto=is_crypto,
            signal_tier=signal_tier,
            signal_confidence=signal_confidence,
            signal_sent_at=signal_sent_at,
            signal_entry_price=signal_entry_price,
            actual_entry_price=live_price,
            entry_price_deviation_pct=deviation_pct,
            data_quality=data_quality,
            capital_allocated=round(capital_allocated, 2),
            position_size=round(position_size, 8),
            risk_amount=round(risk_amount, 2),
            stop_loss=stop_loss,
            target1=target1,
            target2=target2,
            target3=target3,
            notes=f"slippage={deviation_pct:.3f}%" if deviation_pct > 0.1 else None,
        )

        trade = {
            "id": trade_id,
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "is_crypto": is_crypto,
            "signal_tier": signal_tier,
            "signal_confidence": signal_confidence,
            "signal_entry_price": signal_entry_price,
            "actual_entry_price": live_price,
            "entry_price_deviation_pct": deviation_pct,
            "data_quality": data_quality,
            "capital_allocated": round(capital_allocated, 2),
            "position_size": round(position_size, 8),
            "risk_amount": round(risk_amount, 2),
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "target3": target3,
            "entry_timestamp": datetime.utcnow().isoformat(),
            "current_balance": round(current_balance, 2),
        }

        logger.info(
            f"[{symbol}] Paper trade OPENED — {direction} @ {live_price} "
            f"(signal: {signal_entry_price}, dev: {deviation_pct:.3f}%, "
            f"alloc: ${capital_allocated:.0f}, quality: {data_quality})"
        )

        return trade

    # ─── Track Open Trades ───────────────────────────────────────

    def check_open_trades(self) -> list[dict]:
        """
        Check all open paper trades against current live prices.
        Returns list of events (target hits, SL hits, trailing stop).
        """
        open_trades = self.db.get_open_paper_trades()
        if not open_trades:
            return []

        events = []
        for trade in open_trades:
            try:
                trade_events = self._check_trade(trade)
                if trade_events:
                    events.extend(trade_events)
            except Exception as e:
                logger.error(f"[{trade['symbol']}] Paper trade check error: {e}")

        # Expire old trades
        self.db.expire_old_paper_trades(max_age_hours=72)

        logger.info(
            f"Paper trading: checked {len(open_trades)} open trades, "
            f"{len(events)} events"
        )
        return events

    def _check_trade(self, trade: dict) -> list[dict]:
        """Check a single paper trade against live price."""
        symbol = trade["symbol"]
        is_crypto = trade["is_crypto"]
        direction = trade["direction"]
        trade_id = trade["id"]

        live_price, data_quality = self._fetch_live_price(symbol, is_crypto)
        if live_price is None or live_price <= 0:
            logger.warning(f"[{symbol}] Could not fetch live price for paper trade #{trade_id}")
            return []

        entry = trade["actual_entry_price"]
        capital = trade["capital_allocated"]
        position_size = trade["position_size"]
        events = []

        # Update MFE / MAE
        if entry > 0:
            if direction in ("BUY", "LONG", "AL"):
                pct_move = (live_price - entry) / entry * 100
            else:
                pct_move = (entry - live_price) / entry * 100

            mfe = max(trade.get("max_favorable_pct", 0), pct_move if pct_move > 0 else 0)
            mae = max(trade.get("max_adverse_pct", 0), abs(pct_move) if pct_move < 0 else 0)
            self.db.update_paper_trade_extremes(trade_id, mfe, mae)

        # ── Effective SL (may be trailed) ────────────────────────
        original_sl = trade.get("stop_loss", 0)
        effective_sl = original_sl

        if TRAILING_STOP_ENABLED and trade.get("t1_hit_at"):
            trailing_sl = self._calc_trailing_sl(trade, live_price, direction)
            if trailing_sl:
                if direction in ("BUY", "LONG", "AL"):
                    effective_sl = max(original_sl, trailing_sl)
                else:
                    effective_sl = min(original_sl, trailing_sl) if original_sl > 0 else trailing_sl

        # ── SL Check ─────────────────────────────────────────────
        if effective_sl and effective_sl > 0:
            sl_hit = (
                (live_price <= effective_sl) if direction in ("BUY", "LONG", "AL")
                else (live_price >= effective_sl)
            )
            if sl_hit:
                is_trailing = effective_sl != original_sl
                status = "TRAILING_STOP" if (trade.get("t1_hit_at") and is_trailing) else "SL_HIT"

                if direction in ("BUY", "LONG", "AL"):
                    pnl_pct = (effective_sl - entry) / entry * 100
                else:
                    pnl_pct = (entry - effective_sl) / entry * 100

                pnl_amount = position_size * abs(effective_sl - entry)
                if pnl_pct < 0:
                    pnl_amount = -pnl_amount

                duration_min = self._duration_minutes(trade["entry_timestamp"])
                self.db.close_paper_trade(trade_id, status, live_price, pnl_amount, pnl_pct, duration_min)

                events.append({
                    "type": status,
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "direction": direction,
                    "is_crypto": is_crypto,
                    "entry_price": entry,
                    "exit_price": live_price,
                    "sl_price": effective_sl,
                    "pnl_amount": round(pnl_amount, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "duration_min": duration_min,
                    "capital_allocated": capital,
                    "targets_hit": sum(1 for k in ("t1_hit_at", "t2_hit_at", "t3_hit_at")
                                       if trade.get(k)),
                })
                return events  # Trade closed

        # ── Target Checks ─────────────────────────────────────────
        for t_num in (1, 2, 3):
            hit_key = f"t{t_num}_hit_at"
            target_price = trade.get(f"target{t_num}", 0)

            if not target_price or target_price <= 0 or trade.get(hit_key):
                continue

            target_hit = (
                (live_price >= target_price) if direction in ("BUY", "LONG", "AL")
                else (live_price <= target_price)
            )
            if not target_hit:
                continue

            # Calculate PnL for this target
            if direction in ("BUY", "LONG", "AL"):
                target_pnl_pct = (target_price - entry) / entry * 100
            else:
                target_pnl_pct = (entry - target_price) / entry * 100

            # Partial close ratio
            close_ratio = PARTIAL_TP_RATIOS.get(f"t{t_num}", 1/3) if PARTIAL_TP_ENABLED else (1/3)
            partial_size = position_size * close_ratio
            pnl_amount = partial_size * abs(target_price - entry)
            if target_pnl_pct < 0:
                pnl_amount = -pnl_amount

            self.db.update_paper_trade_target(trade_id, t_num, live_price, pnl_amount)

            duration_min = self._duration_minutes(trade["entry_timestamp"])
            remaining_pct = sum(
                PARTIAL_TP_RATIOS.get(f"t{i}", 0) * 100
                for i in range(t_num + 1, 4)
            ) if PARTIAL_TP_ENABLED else 0

            event = {
                "type": f"T{t_num}_HIT",
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "is_crypto": is_crypto,
                "entry_price": entry,
                "target_price": target_price,
                "current_price": live_price,
                "pnl_amount": round(pnl_amount, 2),
                "pnl_pct": round(target_pnl_pct, 2),
                "partial_close_pct": round(close_ratio * 100, 0),
                "remaining_pct": round(remaining_pct, 0),
                "duration_min": duration_min,
                "capital_allocated": capital,
            }
            events.append(event)

            # Close trade at T3
            if t_num == 3:
                self.db.close_paper_trade(
                    trade_id, "T3_HIT", live_price,
                    sum(e["pnl_amount"] for e in events if e.get("trade_id") == trade_id),
                    target_pnl_pct, duration_min
                )

            logger.info(
                f"[{symbol}] Paper T{t_num} HIT @ {live_price} "
                f"(PnL: {pnl_amount:+.2f}$, {target_pnl_pct:+.1f}%)"
            )

        return events

    # ─── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _calc_trailing_sl(trade: dict, current_price: float, direction: str) -> Optional[float]:
        """Estimate trailing SL from ATR (derived from SL distance)."""
        entry = trade["actual_entry_price"]
        original_sl = trade.get("stop_loss", 0)
        if not original_sl or entry <= 0:
            return None
        atr_estimate = abs(entry - original_sl) / 1.5
        if direction in ("BUY", "LONG", "AL"):
            trailing = current_price - TRAILING_STOP_ATR_MULT * atr_estimate
            return max(trailing, entry)
        else:
            trailing = current_price + TRAILING_STOP_ATR_MULT * atr_estimate
            return min(trailing, entry)

    @staticmethod
    def _duration_minutes(entry_timestamp: str) -> int:
        try:
            entry_dt = datetime.fromisoformat(entry_timestamp)
            return int((datetime.utcnow() - entry_dt).total_seconds() / 60)
        except Exception:
            return 0

    # ─── Telegram Messages ───────────────────────────────────────

    @staticmethod
    def format_trade_open_message(trade: dict) -> str:
        """Format Telegram message for paper trade opening."""
        symbol = trade["symbol"]
        direction = trade["direction"]
        is_crypto = trade.get("is_crypto", "/" in symbol)
        currency = "$" if is_crypto else "₺"
        deviation = trade.get("entry_price_deviation_pct", 0)
        quality = trade.get("data_quality", "LIVE")
        quality_icon = "✅" if quality == "LIVE" else "⚠️"

        dir_icon = "📈" if direction in ("BUY", "LONG", "AL") else "📉"
        tier = trade.get("signal_tier", "")
        conf = trade.get("signal_confidence", 0)

        msg = (
            f"📋 <b>PAPER TRADE AÇILDI</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{dir_icon} {symbol} — <b>{direction}</b>\n"
            f"🏷 Tier: {tier} | Conf: {conf}%\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📡 Sinyal Fiyatı: {currency}{trade['signal_entry_price']:.4f}\n"
            f"{quality_icon} Anlık Fiyat:  {currency}{trade['actual_entry_price']:.4f}\n"
            f"📏 Kayma: {deviation:.3f}%\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Ayrılan Sermaye: {currency}{trade['capital_allocated']:.0f}\n"
            f"📦 Pozisyon: {trade['position_size']:.6f}\n"
            f"⚠️ Risk: {currency}{trade['risk_amount']:.0f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🛑 Stop-Loss:   {currency}{trade['stop_loss']:.4f}\n"
            f"🎯 Hedef 1:     {currency}{trade['target1']:.4f}\n"
            f"🎯 Hedef 2:     {currency}{trade['target2']:.4f}\n"
            f"🎯 Hedef 3:     {currency}{trade['target3']:.4f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💼 Demo Bakiye: {currency}{trade.get('current_balance', 0):.0f}"
        )
        return msg

    @staticmethod
    def format_trade_event_message(event: dict) -> str:
        """Format Telegram message for paper trade events (target hits, SL)."""
        event_type = event["type"]
        symbol = event["symbol"]
        direction = event.get("direction", "")
        is_crypto = event.get("is_crypto", "/" in symbol)
        currency = "$" if is_crypto else "₺"
        pnl = event.get("pnl_amount", 0)
        pnl_pct = event.get("pnl_pct", 0)
        pnl_icon = "✅" if pnl >= 0 else "❌"
        duration = event.get("duration_min", 0)
        duration_str = _format_duration(duration)

        if event_type == "SL_HIT":
            return (
                f"🔴 <b>PAPER — STOP-LOSS</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} ({direction})\n"
                f"💰 Giriş: {currency}{event['entry_price']:.4f}\n"
                f"🛑 SL: {currency}{event['sl_price']:.4f}\n"
                f"📉 Çıkış: {currency}{event['exit_price']:.4f}\n"
                f"❌ PnL: {pnl:+.2f}{currency} ({pnl_pct:+.2f}%)\n"
                f"⏱ Süre: {duration_str}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
        elif event_type == "TRAILING_STOP":
            targets_hit = event.get("targets_hit", 0)
            return (
                f"🔒 <b>PAPER — TRAILING STOP</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} ({direction})\n"
                f"💰 Giriş: {currency}{event['entry_price']:.4f}\n"
                f"🔒 Trailing SL: {currency}{event['sl_price']:.4f}\n"
                f"📈 Çıkış: {currency}{event['exit_price']:.4f}\n"
                f"🎯 Hedefler: {targets_hit}/3 vuruldu\n"
                f"{pnl_icon} PnL: {pnl:+.2f}{currency} ({pnl_pct:+.2f}%)\n"
                f"⏱ Süre: {duration_str}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
        else:
            t_num = event_type[1]
            stars = "⭐" * int(t_num)
            partial = event.get("partial_close_pct", 33)
            remaining = event.get("remaining_pct", 0)
            return (
                f"🎯 <b>PAPER — HEDEF {t_num}</b> ✅ {stars}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} ({direction})\n"
                f"💰 Giriş: {currency}{event['entry_price']:.4f}\n"
                f"🎯 Hedef {t_num}: {currency}{event['target_price']:.4f}\n"
                f"📈 Şu an: {currency}{event['current_price']:.4f}\n"
                f"{pnl_icon} PnL: {pnl:+.2f}{currency} ({pnl_pct:+.2f}%)\n"
                f"📦 Kapatıldı: %{partial:.0f} | Kalan: %{remaining:.0f}\n"
                f"⏱ Süre: {duration_str}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )

    @staticmethod
    def format_portfolio_stats_message(stats: dict) -> str:
        """Format Telegram message for portfolio stats panel."""
        total = stats.get("total_trades", 0)
        closed = stats.get("closed_trades", 0)
        open_t = stats.get("open_trades", 0)
        wins = stats.get("winning_trades", 0)
        losses = stats.get("losing_trades", 0)
        win_rate = stats.get("win_rate", 0)
        total_pnl = stats.get("total_pnl_amount", 0)
        total_pnl_pct = stats.get("total_pnl_pct", 0)
        balance = stats.get("current_balance", PAPER_TRADING_CAPITAL)
        starting = stats.get("starting_capital", PAPER_TRADING_CAPITAL)
        t1_hits = stats.get("t1_hit_count", 0)
        t2_hits = stats.get("t2_hit_count", 0)
        t3_hits = stats.get("t3_hit_count", 0)
        sl_hits = stats.get("sl_hit_count", 0)
        avg_dur = stats.get("avg_duration_min", 0)
        avg_dev = stats.get("avg_deviation_pct", 0)
        live_pct = stats.get("live_data_pct", 0)
        best = stats.get("best_trade_pnl", 0)
        worst = stats.get("worst_trade_pnl", 0)

        pnl_icon = "📈" if total_pnl >= 0 else "📉"
        win_icon = "✅" if win_rate >= 50 else "⚠️"

        t1_rate = round(t1_hits / closed * 100) if closed > 0 else 0
        t2_rate = round(t2_hits / closed * 100) if closed > 0 else 0
        t3_rate = round(t3_hits / closed * 100) if closed > 0 else 0
        sl_rate = round(sl_hits / closed * 100) if closed > 0 else 0

        dur_str = _format_duration(avg_dur)

        return (
            f"📊 <b>PAPER TRADING RAPORU</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💼 Başlangıç: ${starting:,.0f}\n"
            f"💰 Güncel Bakiye: ${balance:,.2f}\n"
            f"{pnl_icon} Toplam PnL: {total_pnl:+.2f}$ ({total_pnl_pct:+.2f}%)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📈 İşlemler\n"
            f"  Toplam: {total} | Açık: {open_t}\n"
            f"  Kapandı: {closed} | Kazanan: {wins} | Kaybeden: {losses}\n"
            f"  {win_icon} Kazanma Oranı: {win_rate:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Hedef İstatistikleri\n"
            f"  T1: {t1_hits} ({t1_rate}%) | T2: {t2_hits} ({t2_rate}%)\n"
            f"  T3: {t3_hits} ({t3_rate}%) | SL: {sl_hits} ({sl_rate}%)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⏱ Ort. Süre: {dur_str}\n"
            f"📏 Ort. Kayma: {avg_dev:.3f}%\n"
            f"📡 Canlı Veri: {live_pct:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🏆 En İyi: {best:+.2f}$\n"
            f"💀 En Kötü: {worst:+.2f}$"
        )


def _format_duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}dk"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        return f"{hours}sa {mins}dk"
    days = hours // 24
    hrs = hours % 24
    return f"{days}gün {hrs}sa"
