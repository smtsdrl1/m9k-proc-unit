"""
Order Book Imbalance Analyzer.

Analyzes L2 order book depth to detect buying/selling pressure, large walls,
and bid/ask imbalances. Uses data from CryptoFeed.fetch_order_book().

Score contribution: -8 to +8 confidence boost.
"""
import logging
from src.config import ORDERBOOK_ENABLED, ORDERBOOK_IMBALANCE_THRESHOLD, ORDERBOOK_MAX_BOOST

logger = logging.getLogger("matrix_trader.analysis.orderbook")


def analyze_order_book(ob_data: dict, direction: str) -> dict:
    """
    Analyze order book imbalance and return confidence adjustment.

    Args:
        ob_data: dict from CryptoFeed.fetch_order_book()
                 Keys: bid_volume, ask_volume, bid_ask_ratio, spread_pct
        direction: "BUY" or "SELL"

    Returns:
        {
            "confidence_boost": int (-8 to +8),
            "imbalance_ratio": float,
            "spread_pct": float,
            "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
            "wall_detected": bool,
            "description": str,
        }
    """
    if not ORDERBOOK_ENABLED or not ob_data:
        return {"confidence_boost": 0, "signal": "NEUTRAL", "description": "disabled/no data"}

    bid_vol = ob_data.get("bid_volume", 0)
    ask_vol = ob_data.get("ask_volume", 0)
    ratio = ob_data.get("bid_ask_ratio", 1.0)
    spread = ob_data.get("spread_pct", 0)

    boost = 0
    signal = "NEUTRAL"
    wall_detected = False
    description = ""

    # ── Spread penalty — wide spread = low liquidity ──────────
    spread_penalty = 0
    if spread > 0.5:     # Very wide spread
        spread_penalty = -4
        description += f"wide_spread({spread:.2f}%) "
    elif spread > 0.1:   # Moderately wide
        spread_penalty = -2

    # ── Imbalance detection ────────────────────────────────────
    if ratio >= ORDERBOOK_IMBALANCE_THRESHOLD * 1.5:    # Very strong buy pressure (≥3x)
        signal = "BULLISH"
        boost = ORDERBOOK_MAX_BOOST
        wall_detected = True
        description += f"very_strong_bid_wall(ratio={ratio:.2f})"
    elif ratio >= ORDERBOOK_IMBALANCE_THRESHOLD:         # Strong buy pressure (≥2x)
        signal = "BULLISH"
        boost = round(ORDERBOOK_MAX_BOOST * 0.6)
        description += f"bid_dominant(ratio={ratio:.2f})"
    elif ratio >= 1.5:                                   # Moderate buy pressure
        signal = "BULLISH"
        boost = round(ORDERBOOK_MAX_BOOST * 0.25)
        description += f"slight_bid_pressure(ratio={ratio:.2f})"
    elif ratio <= 1 / (ORDERBOOK_IMBALANCE_THRESHOLD * 1.5):  # Very strong sell pressure
        signal = "BEARISH"
        boost = ORDERBOOK_MAX_BOOST
        wall_detected = True
        description += f"very_strong_ask_wall(ratio={ratio:.2f})"
    elif ratio <= 1 / ORDERBOOK_IMBALANCE_THRESHOLD:           # Strong sell pressure
        signal = "BEARISH"
        boost = round(ORDERBOOK_MAX_BOOST * 0.6)
        description += f"ask_dominant(ratio={ratio:.2f})"
    elif ratio <= 1 / 1.5:                              # Moderate sell pressure
        signal = "BEARISH"
        boost = round(ORDERBOOK_MAX_BOOST * 0.25)
        description += f"slight_ask_pressure(ratio={ratio:.2f})"
    else:
        description += "balanced_book"

    # Flip boost sign based on direction alignment
    if signal == "BULLISH":
        if direction not in ("BUY", "LONG", "AL"):
            boost = -boost   # Order book contradicts signal direction
    elif signal == "BEARISH":
        if direction in ("BUY", "LONG", "AL"):
            boost = -boost   # Order book contradicts signal direction

    # Apply spread penalty
    boost += spread_penalty

    # Clamp
    boost = max(-ORDERBOOK_MAX_BOOST, min(ORDERBOOK_MAX_BOOST, boost))

    result = {
        "confidence_boost": boost,
        "imbalance_ratio": round(ratio, 3),
        "spread_pct": round(spread, 4),
        "bid_volume": round(bid_vol, 2),
        "ask_volume": round(ask_vol, 2),
        "signal": signal,
        "wall_detected": wall_detected,
        "description": description.strip(),
    }

    if boost != 0:
        logger.debug(
            f"OrderBook: boost={boost:+d}, signal={signal}, "
            f"ratio={ratio:.2f}, {description}"
        )

    return result
