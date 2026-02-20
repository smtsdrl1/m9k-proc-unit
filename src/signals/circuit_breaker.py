"""
Circuit Breaker — Automatic trading pause on consecutive losses or large drawdown.
Inspired by CAI Compass system: "ne zaman YAPMAMALIYIM bilgisi".

Activates when:
- N consecutive SL hits detected
- Daily loss exceeds threshold
- Too many open positions

When active: rejects all new signals for cooldown period.
"""
import logging
from datetime import datetime, timedelta

from src.config import (
    CIRCUIT_BREAKER_ENABLED,
    CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES,
    CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT,
    CIRCUIT_BREAKER_COOLDOWN_HOURS,
    MAX_OPEN_SIGNALS,
    MAX_TOTAL_RISK_PCT,
    MAX_SINGLE_RISK_PCT,
    MAX_CORRELATED_POSITIONS,
)

from src.config import (
    CIRCUIT_BREAKER_ENABLED,
    CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES,
    CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT,
    CIRCUIT_BREAKER_COOLDOWN_HOURS,
    MAX_OPEN_SIGNALS,
    MAX_TOTAL_RISK_PCT,
    MAX_SINGLE_RISK_PCT,
    MAX_CORRELATED_POSITIONS,
    CB_CONSECUTIVE_LOSSES, CB_HOURLY_LOSS_LIMIT,
    CB_BTC_DUMP_THRESHOLD, NEWS_KILL_WINDOW_MINUTES,
)

logger = logging.getLogger("matrix_trader.signals.circuit_breaker")


class CircuitBreaker:
    """Trading circuit breaker — protects against cascading losses."""

    def __init__(self, db=None):
        from src.database.db import Database
        self.db = db or Database()
        # Enhanced state
        self.manual_stopped = False
        self.news_kill_active = False
        self.btc_dump_active  = False

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed. Returns (allowed, reason)."""
        if not CIRCUIT_BREAKER_ENABLED:
            return True, "Circuit breaker disabled"

        # Manual stop
        if self.manual_stopped:
            return False, "Manuel durdurma aktif — /cbresume ile devam et"

        # News kill zone
        if self.news_kill_active:
            return False, "Haber kill zone aktif — yüksek etkili haber"

        # BTC market dump
        if self.btc_dump_active:
            return False, "BTC sert düşüş algılandı — piyasa koruması"

        # Check consecutive losses
        allowed, reason = self._check_consecutive_losses()
        if not allowed:
            return False, reason

        # Check daily loss limit
        allowed, reason = self._check_daily_loss()
        if not allowed:
            return False, reason

        # Check open position limit
        allowed, reason = self._check_open_positions()
        if not allowed:
            return False, reason

        return True, "OK"

    def manual_stop(self, reason: str = "Kullanıcı tarafından durduruldu"):
        """Manually halt all trading."""
        self.manual_stopped = True
        logger.warning(f"🔴 Manuel circuit breaker: {reason}")

    def manual_resume(self):
        """Manually resume trading."""
        self.manual_stopped = False
        self.btc_dump_active = False
        logger.info("🟢 Circuit breaker sıfırlandı — trading devam ediyor")

    def set_news_kill(self, active: bool, event: str = ""):
        """Activate/deactivate news kill zone."""
        self.news_kill_active = active
        if active:
            logger.warning(f"📰 Haber kill zone aktif: {event}")
        else:
            logger.info("📰 Haber kill zone pasif")

    def check_btc_market_dump(self, btc_change_pct: float):
        """Check if BTC has dumped hard — halt all trading."""
        if btc_change_pct <= CB_BTC_DUMP_THRESHOLD:
            self.btc_dump_active = True
            logger.critical(
                f"⚠️ BTC sert düşüş: {btc_change_pct:.2f}% "
                f"(eşik: {CB_BTC_DUMP_THRESHOLD:.2f}%)"
            )
        else:
            self.btc_dump_active = False

    def can_open_direction(self, direction: str) -> tuple[bool, str]:
        """Check if a position in given direction is allowed (correlation limit)."""
        pending = self.db.get_pending_signals()
        same_dir = sum(1 for s in pending if s.get("direction") == direction)

        if same_dir >= MAX_CORRELATED_POSITIONS:
            reason = (
                f"Aynı yön limiti: {same_dir}/{MAX_CORRELATED_POSITIONS} "
                f"{direction} pozisyon açık"
            )
            logger.warning(f"🚫 {reason}")
            return False, reason

        return True, "OK"

    def check_risk_budget(self, new_risk_pct: float) -> tuple[bool, str]:
        """Check if adding a new position would exceed total risk budget."""
        # Single position risk check
        if new_risk_pct > MAX_SINGLE_RISK_PCT:
            reason = (
                f"Tekil risk limiti: {new_risk_pct:.1f}% > {MAX_SINGLE_RISK_PCT}% "
            )
            logger.warning(f"🚫 Single risk exceeded: {reason}")
            return False, reason

        pending = self.db.get_pending_signals()

        # Calculate current total risk
        total_risk = 0.0
        for sig in pending:
            entry = sig.get("entry_price", 0)
            sl = sig.get("stop_loss", 0)
            if entry > 0 and sl > 0:
                risk_pct = abs(entry - sl) / entry * 100
                total_risk += risk_pct

        projected_risk = total_risk + new_risk_pct

        if projected_risk > MAX_TOTAL_RISK_PCT:
            reason = (
                f"Toplam risk limiti: {projected_risk:.1f}% > {MAX_TOTAL_RISK_PCT}% "
                f"(mevcut: {total_risk:.1f}%, yeni: {new_risk_pct:.1f}%)"
            )
            logger.warning(f"🚫 Risk budget exceeded: {reason}")
            return False, reason

        return True, f"Risk OK: {projected_risk:.1f}% / {MAX_TOTAL_RISK_PCT}%"

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        """Check for consecutive SL hits."""
        recent = self.db.get_recent_signals(CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES + 2)
        closed = [s for s in recent if s.get("outcome") not in ("PENDING", None)]

        consecutive_losses = 0
        for sig in closed:
            if sig.get("outcome") == "SL_HIT":
                consecutive_losses += 1
            else:
                break  # First non-loss breaks the streak

        if consecutive_losses >= CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES:
            # Check if cooldown has passed since last loss
            last_loss = closed[0] if closed else None
            if last_loss:
                closed_at = last_loss.get("closed_at") or last_loss.get("sent_at", "")
                if closed_at:
                    loss_time = datetime.fromisoformat(closed_at)
                    cooldown_end = loss_time + timedelta(hours=CIRCUIT_BREAKER_COOLDOWN_HOURS)
                    if datetime.utcnow() < cooldown_end:
                        remaining = (cooldown_end - datetime.utcnow()).total_seconds() / 60
                        reason = (
                            f"🔴 CIRCUIT BREAKER AKTİF: {consecutive_losses} art arda kayıp! "
                            f"Kalan bekleme: {int(remaining)}dk"
                        )
                        logger.warning(reason)
                        return False, reason

        return True, "OK"

    def _check_daily_loss(self) -> tuple[bool, str]:
        """Check if daily loss limit is exceeded."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        recent = self.db.get_recent_signals(50)
        today_closed = [
            s for s in recent
            if (s.get("closed_at") or "").startswith(today)
            and s.get("outcome") not in ("PENDING", None)
        ]

        if not today_closed:
            return True, "OK"

        total_pnl = sum(s.get("pnl_pct", 0) for s in today_closed)

        if total_pnl < -CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT:
            reason = (
                f"🔴 GÜNLÜK KAYIP LİMİTİ: {total_pnl:.1f}% "
                f"(limit: -{CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT}%)"
            )
            logger.warning(reason)
            return False, reason

        return True, "OK"

    def _check_open_positions(self) -> tuple[bool, str]:
        """Check if max open position limit is reached."""
        pending = self.db.get_pending_signals()

        if len(pending) >= MAX_OPEN_SIGNALS:
            reason = (
                f"Maksimum açık pozisyon: {len(pending)}/{MAX_OPEN_SIGNALS}"
            )
            logger.info(f"🚫 {reason}")
            return False, reason

        return True, "OK"

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        pending = self.db.get_pending_signals()
        recent = self.db.get_recent_signals(10)
        closed = [s for s in recent if s.get("outcome") not in ("PENDING", None)]

        # Current streak
        consecutive_losses = 0
        for sig in closed:
            if sig.get("outcome") == "SL_HIT":
                consecutive_losses += 1
            else:
                break

        # Today's PnL
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_closed = [
            s for s in recent
            if (s.get("closed_at") or "").startswith(today)
            and s.get("pnl_pct") is not None
        ]
        daily_pnl = sum(s.get("pnl_pct", 0) for s in today_closed)

        # Total risk
        total_risk = 0.0
        for sig in pending:
            entry = sig.get("entry_price", 0)
            sl = sig.get("stop_loss", 0)
            if entry > 0 and sl > 0:
                total_risk += abs(entry - sl) / entry * 100

        can_trade, reason = self.can_trade()

        return {
            "can_trade": can_trade,
            "reason": reason,
            "open_positions": len(pending),
            "max_positions": MAX_OPEN_SIGNALS,
            "consecutive_losses": consecutive_losses,
            "max_consecutive_losses": CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES,
            "daily_pnl_pct": round(daily_pnl, 2),
            "max_daily_loss_pct": CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT,
            "total_risk_pct": round(total_risk, 2),
            "max_total_risk_pct": MAX_TOTAL_RISK_PCT,
            "manual_stopped": self.manual_stopped,
            "news_kill_active": self.news_kill_active,
            "btc_dump_active": self.btc_dump_active,
        }
