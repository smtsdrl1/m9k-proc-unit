"""
Signal Performance Tracker — Real-time outcome tracking.
Checks PENDING signals against live market prices to detect:
- T1/T2/T3 target hits with exact timing
- Stop-loss hits (including trailing stop)
- Max favorable / adverse excursion (MFE / MAE)
- Kademeli kar alma (partial close at each target)
Sends Telegram notifications on target achievements.
"""
import logging
from datetime import datetime
from typing import Optional

from src.database.db import Database
from src.config import (
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR_MULT, TRAILING_STOP_ACTIVATION,
    PARTIAL_TP_ENABLED, PARTIAL_TP_RATIOS,
)

logger = logging.getLogger("matrix_trader.tracker")


class SignalTracker:
    """Tracks pending signals and records outcomes with trailing stop."""

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self._crypto_feed = None

    def _get_crypto_feed(self):
        """Lazy-init CryptoFeed to avoid circular imports."""
        if self._crypto_feed is None:
            from src.data.crypto_feed import CryptoFeed
            self._crypto_feed = CryptoFeed()
        return self._crypto_feed

    def track_all_pending(self) -> list[dict]:
        """Check all pending signals against current prices.
        Returns list of events (target hits, SL hits, trailing stop updates).
        """
        pending = self.db.get_pending_signals()
        if not pending:
            logger.info("No pending signals to track")
            return []

        events = []
        for signal in pending:
            try:
                event = self._check_signal(signal)
                if event:
                    events.extend(event)
            except Exception as e:
                logger.error(f"Error tracking {signal['symbol']}: {e}")

        # Expire old signals (>72h)
        self.db.expire_old_signals(max_age_hours=72)

        logger.info(f"Tracked {len(pending)} signals, {len(events)} events detected")
        return events

    def _check_signal(self, signal: dict) -> list[dict]:
        """Check a single signal against current price with trailing stop logic."""
        symbol = signal["symbol"]
        is_crypto = signal.get("is_crypto", True)
        direction = signal.get("direction", "BUY")
        entry = signal["entry_price"]
        signal_id = signal["id"]

        # Fetch current price from real market data
        current_price = self._get_current_price(symbol, is_crypto)
        if current_price is None or current_price <= 0:
            logger.warning(f"Could not get price for {symbol}")
            return []

        events = []

        # Calculate MFE / MAE
        pct_move = 0
        if entry > 0:
            if direction in ("BUY", "LONG", "AL"):
                pct_move = (current_price - entry) / entry * 100
            else:
                pct_move = (entry - current_price) / entry * 100

            max_favorable = max(signal.get("max_favorable", 0), pct_move if pct_move > 0 else 0)
            max_adverse = max(signal.get("max_adverse", 0), abs(pct_move) if pct_move < 0 else 0)
            self.db.update_signal_extremes(signal_id, max_favorable, max_adverse)

        # Determine effective SL (original or trailing)
        original_sl = signal.get("stop_loss", 0)
        effective_sl = original_sl

        # Trailing stop: activate after T1 hit, move SL in profit direction
        if TRAILING_STOP_ENABLED and signal.get("t1_hit"):
            trailing_sl = self._calculate_trailing_sl(signal, current_price, direction)
            if trailing_sl:
                # Only tighten, never loosen
                if direction in ("BUY", "LONG", "AL"):
                    effective_sl = max(original_sl, trailing_sl)
                else:
                    effective_sl = min(original_sl, trailing_sl) if original_sl > 0 else trailing_sl

                if effective_sl != original_sl:
                    self.db.update_signal_trailing_sl(signal_id, effective_sl)

        # Check stop-loss (original or trailing)
        if effective_sl and effective_sl > 0 and not signal.get("sl_hit"):
            if self._is_sl_hit(current_price, effective_sl, direction):
                is_trailing = effective_sl != original_sl

                # If T1 already hit, this is a trailing stop close (still profitable)
                if signal.get("t1_hit") and is_trailing:
                    # Trailing stop after profit = partial win
                    if direction in ("BUY", "LONG", "AL"):
                        exit_pnl = (effective_sl - entry) / entry * 100
                    else:
                        exit_pnl = (entry - effective_sl) / entry * 100

                    self.db.update_signal_pnl(signal_id, current_price, exit_pnl, "TRAILING_STOP")
                    events.append({
                        "type": "TRAILING_STOP",
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "direction": direction,
                        "is_crypto": is_crypto,
                        "entry_price": entry,
                        "exit_price": current_price,
                        "sl_price": effective_sl,
                        "pnl_pct": round(exit_pnl, 2),
                        "targets_hit": sum(1 for t in ("t1_hit", "t2_hit", "t3_hit") if signal.get(t)),
                    })
                    logger.info(f"🔒 TRAILING STOP: {symbol} @ {current_price} (PnL: {exit_pnl:+.1f}%)")
                else:
                    self.db.update_signal_sl_hit(signal_id, current_price)
                    events.append({
                        "type": "SL_HIT",
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "direction": direction,
                        "is_crypto": is_crypto,
                        "entry_price": entry,
                        "exit_price": current_price,
                        "sl_price": effective_sl,
                        "pnl_pct": round(pct_move, 2),
                    })
                    logger.info(f"🔴 SL HIT: {symbol} @ {current_price}")
                return events  # Signal closed

        # Check targets in order: T1 → T2 → T3
        for t_num in (1, 2, 3):
            target_key = f"target{t_num}"
            hit_key = f"t{t_num}_hit"
            target_price = signal.get(target_key, 0)

            if target_price and target_price > 0 and not signal.get(hit_key):
                if self._is_target_hit(current_price, target_price, direction):
                    self.db.update_signal_target(signal_id, t_num, current_price)

                    # Calculate PnL for this target
                    if direction in ("BUY", "LONG", "AL"):
                        target_pnl = (target_price - entry) / entry * 100
                    else:
                        target_pnl = (entry - target_price) / entry * 100

                    event = {
                        "type": f"T{t_num}_HIT",
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "direction": direction,
                        "is_crypto": is_crypto,
                        "entry_price": entry,
                        "target_price": target_price,
                        "current_price": current_price,
                        "pnl_pct": round(target_pnl, 2),
                    }

                    # Kademeli kar alma info
                    if PARTIAL_TP_ENABLED:
                        close_pct = PARTIAL_TP_RATIOS.get(f"t{t_num}", 0) * 100
                        event["partial_close_pct"] = close_pct
                        remaining = sum(
                            PARTIAL_TP_RATIOS.get(f"t{i}", 0) * 100
                            for i in range(t_num + 1, 4)
                        )
                        event["remaining_pct"] = remaining

                    # Calculate duration
                    sent_at = datetime.fromisoformat(signal["sent_at"])
                    duration_min = int((datetime.utcnow() - sent_at).total_seconds() / 60)
                    event["duration_min"] = duration_min
                    event["duration_str"] = self._format_duration(duration_min)

                    events.append(event)
                    logger.info(
                        f"🎯 T{t_num} HIT: {symbol} @ {current_price} "
                        f"(+{target_pnl:.1f}%) [{event['duration_str']}]"
                    )

        return events

    def _calculate_trailing_sl(self, signal: dict, current_price: float, direction: str) -> Optional[float]:
        """Calculate trailing stop level based on ATR and price movement."""
        entry = signal["entry_price"]
        # Estimate ATR from entry and stop loss distance
        original_sl = signal.get("stop_loss", 0)
        if not original_sl or entry <= 0:
            return None

        atr_estimate = abs(entry - original_sl) / 1.5  # We used 1.5*ATR for initial SL

        if direction in ("BUY", "LONG", "AL"):
            # Trail below current price
            trailing = current_price - TRAILING_STOP_ATR_MULT * atr_estimate
            # Never below entry (after T1 hit, lock in breakeven minimum)
            return max(trailing, entry)
        else:
            # Trail above current price
            trailing = current_price + TRAILING_STOP_ATR_MULT * atr_estimate
            return min(trailing, entry)

    def _get_current_price(self, symbol: str, is_crypto: bool) -> Optional[float]:
        """Fetch real current price from market API."""
        import asyncio
        try:
            if is_crypto:
                feed = self._get_crypto_feed()
                loop = asyncio.new_event_loop()
                try:
                    ticker = loop.run_until_complete(feed.fetch_ticker(symbol))
                finally:
                    loop.close()
                if ticker and ticker.get("price", 0) > 0:
                    return ticker["price"]
            else:
                # BIST: use yfinance for price
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(f"{symbol}.IS")
                    hist = ticker.history(period="1d", interval="1m")
                    if hist is not None and not hist.empty:
                        return float(hist["Close"].iloc[-1])
                except Exception as e:
                    logger.error(f"BIST price fetch error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return None

    @staticmethod
    def _is_target_hit(current_price: float, target: float, direction: str) -> bool:
        if direction in ("BUY", "LONG", "AL"):
            return current_price >= target
        else:
            return current_price <= target

    @staticmethod
    def _is_sl_hit(current_price: float, sl: float, direction: str) -> bool:
        if direction in ("BUY", "LONG", "AL"):
            return current_price <= sl
        else:
            return current_price >= sl

    @staticmethod
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

    def format_event_message(self, event: dict) -> str:
        """Format a tracking event as Telegram message."""
        event_type = event["type"]
        symbol = event["symbol"]
        direction = event.get("direction", "")
        # Crypto symbols contain "/"; BIST symbols are plain tickers
        currency = "$" if event.get("is_crypto", "/" in symbol) else "₺"

        if event_type == "SL_HIT":
            return (
                f"🔴 STOP-LOSS\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} ({direction})\n"
                f"💰 Giriş: {currency}{event['entry_price']:.4f}\n"
                f"🛑 SL: {currency}{event['sl_price']:.4f}\n"
                f"📉 Çıkış: {currency}{event['exit_price']:.4f}\n"
                f"❌ PnL: {event['pnl_pct']:+.2f}%\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
        elif event_type == "TRAILING_STOP":
            targets_hit = event.get("targets_hit", 0)
            return (
                f"🔒 TRAILING STOP\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} ({direction})\n"
                f"💰 Giriş: {currency}{event['entry_price']:.4f}\n"
                f"🔒 Trailing SL: {currency}{event['sl_price']:.4f}\n"
                f"📈 Çıkış: {currency}{event['exit_price']:.4f}\n"
                f"🎯 Hedefler: {targets_hit}/3 vuruldu\n"
                f"{'✅' if event['pnl_pct'] > 0 else '❌'} PnL: {event['pnl_pct']:+.2f}%\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
        else:
            t_num = event_type[1]
            stars = "⭐" * int(t_num)
            msg = (
                f"🎯 HEDEF {t_num} ✅ {stars}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} ({direction})\n"
                f"💰 Giriş: {currency}{event['entry_price']:.4f}\n"
                f"🎯 Hedef {t_num}: {currency}{event['target_price']:.4f}\n"
                f"📈 Şu an: {currency}{event['current_price']:.4f}\n"
                f"✅ PnL: {event['pnl_pct']:+.2f}%\n"
                f"⏱ Süre: {event.get('duration_str', 'N/A')}\n"
            )
            # Kademeli kar alma info
            if event.get("partial_close_pct"):
                msg += (
                    f"📦 Kapat: %{event['partial_close_pct']:.0f} pozisyon\n"
                    f"📦 Kalan: %{event.get('remaining_pct', 0):.0f} trailing SL ile\n"
                )
            msg += "━━━━━━━━━━━━━━━━━━"
            return msg
