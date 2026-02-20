"""
Order Block Detection — ICT methodology.
Bullish/Bearish order blocks with impulse-based filtering.
"""
import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import numpy as np

logger = logging.getLogger("matrix_trader.analysis.order_blocks")


@dataclass
class OrderBlock:
    type: str        # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    index: int
    strength: float  # impulse %
    active: bool = True


def detect_order_blocks(df: pd.DataFrame, lookback: int = 50,
                        min_impulse_pct: float = 0.5) -> List[OrderBlock]:
    """Detect active order blocks."""
    if len(df) < lookback + 5:
        return []

    blocks = []
    current_price = df["close"].iloc[-1]
    subset = df.tail(lookback + 5).reset_index(drop=True)

    for i in range(2, len(subset) - 3):
        candle = subset.iloc[i]
        # Bullish OB: bearish candle followed by strong bullish impulse
        if candle["close"] < candle["open"]:
            impulse = 0.0
            for j in range(i + 1, min(i + 4, len(subset))):
                chg = (subset["close"].iloc[j] - candle["close"]) / candle["close"] * 100
                impulse = max(impulse, chg)
            if impulse >= min_impulse_pct:
                ob_top    = max(candle["open"], candle["close"])
                ob_bottom = min(candle["open"], candle["close"])
                # Only keep if price above the OB (still valid)
                if current_price > ob_top:
                    blocks.append(OrderBlock(
                        type="BULLISH", top=ob_top, bottom=ob_bottom,
                        index=i, strength=impulse
                    ))

        # Bearish OB: bullish candle followed by strong bearish impulse
        elif candle["close"] > candle["open"]:
            impulse = 0.0
            for j in range(i + 1, min(i + 4, len(subset))):
                chg = (candle["close"] - subset["close"].iloc[j]) / candle["close"] * 100
                impulse = max(impulse, chg)
            if impulse >= min_impulse_pct:
                ob_top    = max(candle["open"], candle["close"])
                ob_bottom = min(candle["open"], candle["close"])
                if current_price < ob_bottom:
                    blocks.append(OrderBlock(
                        type="BEARISH", top=ob_top, bottom=ob_bottom,
                        index=i, strength=impulse
                    ))

    return blocks


def get_order_block_score(price: float, order_blocks: List[OrderBlock],
                          direction: str, tolerance: float = 0.003) -> float:
    """Return score contribution from order blocks (+/-  up to 12)."""
    best = 0.0
    for ob in order_blocks:
        in_zone = (ob.bottom * (1 - tolerance) <= price <= ob.top * (1 + tolerance))
        if in_zone:
            boost = min(12.0, ob.strength * 2)
            if direction == "BUY" and ob.type == "BULLISH":
                best = max(best, boost)
            elif direction == "SELL" and ob.type == "BEARISH":
                best = max(best, boost)
            elif ob.type != ("BULLISH" if direction == "BUY" else "BEARISH"):
                best = min(best, -boost)
    return best
