"""
Adaptive Confidence Threshold.

Adjusts signal confidence thresholds based on recent system performance.
- Win rate > HIGH_WR → relax threshold (system performing well, capture more signals)
- Win rate < LOW_WR  → tighten threshold (system struggling, be more selective)
- Not enough data    → use default threshold

Uses last 7 days of resolved signals for fast adaptation.
"""
import logging
from src.config import (
    CRYPTO_CONFIDENCE_THRESHOLD,
    BIST_CONFIDENCE_THRESHOLD,
    ADAPTIVE_THRESHOLD_ENABLED,
    ADAPTIVE_THRESHOLD_HIGH_WR,
    ADAPTIVE_THRESHOLD_LOW_WR,
    ADAPTIVE_THRESHOLD_RELAX,
    ADAPTIVE_THRESHOLD_TIGHTEN,
)

logger = logging.getLogger("matrix_trader.signals.adaptive_threshold")

# Hard limits — never go below/above these regardless of performance
_MIN_THRESHOLD = 40
_MAX_THRESHOLD = 80


def get_adaptive_threshold(db, is_crypto: bool) -> int:
    """
    Returns a dynamically adjusted confidence threshold.

    Args:
        db: Database instance
        is_crypto: True for crypto, False for BIST

    Returns:
        Adjusted integer threshold (clamped to [40, 80])
    """
    base = CRYPTO_CONFIDENCE_THRESHOLD if is_crypto else BIST_CONFIDENCE_THRESHOLD

    if not ADAPTIVE_THRESHOLD_ENABLED:
        return base

    try:
        # Use last 7 days for fast adaptation, require minimum 5 resolved signals
        stats = db.get_accuracy_stats(days=7)
        total = stats.get("total", 0)

        if total < 5:
            logger.debug(f"Adaptive threshold: insufficient data ({total} signals), using base {base}")
            return base

        win_rate = stats.get("win_rate", 50.0)

        if win_rate > ADAPTIVE_THRESHOLD_HIGH_WR:
            adjustment = -ADAPTIVE_THRESHOLD_RELAX
            reason = f"win_rate={win_rate:.1f}% > {ADAPTIVE_THRESHOLD_HIGH_WR}% → relax -{ADAPTIVE_THRESHOLD_RELAX}"
        elif win_rate < ADAPTIVE_THRESHOLD_LOW_WR:
            adjustment = ADAPTIVE_THRESHOLD_TIGHTEN
            reason = f"win_rate={win_rate:.1f}% < {ADAPTIVE_THRESHOLD_LOW_WR}% → tighten +{ADAPTIVE_THRESHOLD_TIGHTEN}"
        else:
            adjustment = 0
            reason = f"win_rate={win_rate:.1f}% normal range → no adjustment"

        new_threshold = max(_MIN_THRESHOLD, min(_MAX_THRESHOLD, base + adjustment))

        if adjustment != 0:
            label = "CRYPTO" if is_crypto else "BIST"
            logger.info(
                f"Adaptive threshold [{label}]: {base} → {new_threshold} "
                f"(n={total}, {reason})"
            )

        return new_threshold

    except Exception as e:
        logger.warning(f"Adaptive threshold error, using base {base}: {e}")
        return base


def get_threshold_status(db) -> dict:
    """Get adaptive threshold status for both markets (used in reports)."""
    try:
        stats_7d = db.get_accuracy_stats(days=7)
        win_rate = stats_7d.get("win_rate", 0)
        total = stats_7d.get("total", 0)

        crypto_thresh = get_adaptive_threshold(db, is_crypto=True)
        bist_thresh = get_adaptive_threshold(db, is_crypto=False)

        adjustment = 0
        if total >= 5:
            if win_rate > ADAPTIVE_THRESHOLD_HIGH_WR:
                adjustment = -ADAPTIVE_THRESHOLD_RELAX
            elif win_rate < ADAPTIVE_THRESHOLD_LOW_WR:
                adjustment = ADAPTIVE_THRESHOLD_TIGHTEN

        return {
            "win_rate_7d": round(win_rate, 1),
            "signals_7d": total,
            "crypto_threshold": crypto_thresh,
            "bist_threshold": bist_thresh,
            "adjustment": adjustment,
            "mode": (
                "RELAX" if adjustment < 0
                else "TIGHTEN" if adjustment > 0
                else "NORMAL"
            ),
        }
    except Exception:
        return {
            "win_rate_7d": 0,
            "signals_7d": 0,
            "crypto_threshold": CRYPTO_CONFIDENCE_THRESHOLD,
            "bist_threshold": BIST_CONFIDENCE_THRESHOLD,
            "adjustment": 0,
            "mode": "NORMAL",
        }
