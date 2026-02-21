"""
Drawdown Recovery Guard.

Monitors paper portfolio drawdown and automatically adjusts system behavior
to protect capital during losing streaks.

Modes:
  NORMAL    — DD < 10%: full operation
  CAUTION   — DD 10-20%: half position size, STRONG+ tier only, +5 confidence required
  DEFENSIVE — DD 20-30%: quarter position size, EXTREME tier only, +15 confidence required
  HALT      — DD > 30%: paper trading stopped (no new trades)
"""
import logging
from src.config import (
    PAPER_TRADING_CAPITAL,
    DRAWDOWN_CAUTION_PCT,
    DRAWDOWN_DEFENSIVE_PCT,
    DRAWDOWN_HALT_PCT,
)

logger = logging.getLogger("matrix_trader.paper_trading.drawdown_guard")

# Tier rank used to enforce minimum tier in non-NORMAL modes
_TIER_RANK = {
    "EXTREME": 6,
    "STRONG": 5,
    "MODERATE": 4,
    "SPECULATIVE": 3,
    "DIVERGENCE": 2,
    "CONTRARIAN": 1,
}

_MODE_CONFIG = {
    "NORMAL": {
        "position_mult": 1.0,
        "min_tier": "ALL",
        "confidence_boost_required": 0,
        "emoji": "✅",
    },
    "CAUTION": {
        "position_mult": 0.5,
        "min_tier": "STRONG",
        "confidence_boost_required": 5,
        "emoji": "⚠️",
    },
    "DEFENSIVE": {
        "position_mult": 0.25,
        "min_tier": "EXTREME",
        "confidence_boost_required": 15,
        "emoji": "🛡️",
    },
    "HALT": {
        "position_mult": 0.0,
        "min_tier": "NONE",
        "confidence_boost_required": 999,
        "emoji": "🔴",
    },
}


class DrawdownGuard:
    """Monitors portfolio drawdown and returns operational mode."""

    def __init__(self, db):
        self.db = db

    def get_mode(self) -> dict:
        """
        Calculate current drawdown mode from paper portfolio stats.

        Returns dict with: mode, drawdown_pct, balance, position_mult,
        min_tier, confidence_boost_required, can_trade, emoji
        """
        try:
            stats = self.db.get_paper_trade_stats(days=365)
            balance = stats.get("current_balance", PAPER_TRADING_CAPITAL)
            starting = PAPER_TRADING_CAPITAL

            if balance >= starting:
                drawdown_pct = 0.0
            else:
                drawdown_pct = (starting - balance) / starting * 100

            if drawdown_pct >= DRAWDOWN_HALT_PCT:
                mode = "HALT"
            elif drawdown_pct >= DRAWDOWN_DEFENSIVE_PCT:
                mode = "DEFENSIVE"
            elif drawdown_pct >= DRAWDOWN_CAUTION_PCT:
                mode = "CAUTION"
            else:
                mode = "NORMAL"

            cfg = _MODE_CONFIG[mode]

            if mode != "NORMAL":
                logger.info(
                    f"DrawdownGuard: mode={mode} "
                    f"(DD={drawdown_pct:.1f}%, balance=${balance:.0f})"
                )

            return {
                "mode": mode,
                "drawdown_pct": round(drawdown_pct, 2),
                "balance": round(balance, 2),
                "starting": starting,
                "can_trade": mode != "HALT",
                **cfg,
            }

        except Exception as e:
            logger.warning(f"DrawdownGuard error, defaulting to NORMAL: {e}")
            return {
                "mode": "NORMAL",
                "drawdown_pct": 0.0,
                "balance": PAPER_TRADING_CAPITAL,
                "starting": PAPER_TRADING_CAPITAL,
                "can_trade": True,
                **_MODE_CONFIG["NORMAL"],
            }

    def is_tier_allowed(self, tier_name: str, mode: str) -> bool:
        """Check if a signal tier is allowed in the current drawdown mode."""
        min_tier = _MODE_CONFIG.get(mode, {}).get("min_tier", "ALL")

        if min_tier == "ALL":
            return True
        if min_tier == "NONE":
            return False

        tier_upper = tier_name.upper()
        signal_rank = 0
        for key, rank in _TIER_RANK.items():
            if key in tier_upper:
                signal_rank = rank
                break

        min_rank = _TIER_RANK.get(min_tier, 0)
        return signal_rank >= min_rank

    def format_mode_message(self, guard_info: dict) -> str:
        """Format Telegram alert when mode changes to non-NORMAL."""
        mode = guard_info["mode"]
        dd = guard_info["drawdown_pct"]
        balance = guard_info["balance"]
        emoji = guard_info["emoji"]
        mult = guard_info["position_mult"]
        min_tier = guard_info["min_tier"]
        boost = guard_info["confidence_boost_required"]

        return (
            f"{emoji} <b>DRAWDOWN GUARD: {mode}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📉 Drawdown: {dd:.1f}%\n"
            f"💼 Bakiye: ${balance:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📦 Pozisyon Katsayısı: x{mult:.2f}\n"
            f"🏷 Min Tier: {min_tier}\n"
            f"📊 Ekstra Confidence: +{boost}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{'🔴 Yeni paper trade açılmıyor!' if mode == 'HALT' else '⚠️ Koruma modu aktif'}"
        )
