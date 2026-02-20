"""
Volume Profile Visible Range (VPVR) for Matrix Trader AI.
Identifies POC, VAH, VAL, HVN, LVN from price-volume distribution.
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("matrix_trader.analysis.vpvr")


def calculate_vpvr(df: pd.DataFrame, num_bins: int = 50,
                   value_area_pct: float = 0.70) -> Optional[dict]:
    """Calculate Volume Profile. Returns POC, VAH, VAL, HVN, LVN."""
    if len(df) < 30:
        return None
    try:
        price_min = df["low"].min()
        price_max = df["high"].max()
        if price_max <= price_min:
            return None

        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        vol_by_bin = np.zeros(num_bins)

        for _, row in df.iterrows():
            lo, hi, vol = row["low"], row["high"], row["volume"]
            if hi <= lo or vol <= 0:
                continue
            span = hi - lo
            for b_idx in range(num_bins):
                b_lo = bins[b_idx]; b_hi = bins[b_idx + 1]
                overlap = max(0.0, min(hi, b_hi) - max(lo, b_lo))
                vol_by_bin[b_idx] += vol * (overlap / span)

        poc_idx = int(np.argmax(vol_by_bin))
        poc = bin_centers[poc_idx]

        # Value Area
        total_vol = vol_by_bin.sum()
        target_vol = total_vol * value_area_pct
        cum = vol_by_bin[poc_idx]
        lo_idx = hi_idx = poc_idx
        while cum < target_vol:
            expand_lo = (lo_idx > 0 and
                         (vol_by_bin[lo_idx - 1] >=
                          (vol_by_bin[hi_idx + 1] if hi_idx < num_bins - 1 else 0)))
            if expand_lo:
                lo_idx -= 1; cum += vol_by_bin[lo_idx]
            elif hi_idx < num_bins - 1:
                hi_idx += 1; cum += vol_by_bin[hi_idx]
            else:
                break

        val = bin_centers[lo_idx]
        vah = bin_centers[hi_idx]

        # HVN / LVN
        avg_vol = np.mean(vol_by_bin)
        hvn = [bin_centers[i] for i in range(num_bins) if vol_by_bin[i] > avg_vol * 1.5]
        lvn = [bin_centers[i] for i in range(num_bins) if 0 < vol_by_bin[i] < avg_vol * 0.5]

        current = df["close"].iloc[-1]
        if current > vah:
            zone = "ABOVE_VALUE_AREA"
        elif current < val:
            zone = "BELOW_VALUE_AREA"
        elif abs(current - poc) / poc < 0.005:
            zone = "AT_POC"
        else:
            zone = "INSIDE_VALUE_AREA"

        return {
            "poc": round(poc, 6), "vah": round(vah, 6), "val": round(val, 6),
            "hvn_levels": [round(x, 6) for x in hvn[-5:]],
            "lvn_levels": [round(x, 6) for x in lvn[:5]],
            "current_zone": zone, "current_price": round(current, 6),
        }
    except Exception as e:
        logger.warning(f"VPVR calculation error: {e}")
        return None


def get_vpvr_confidence_modifier(vpvr: Optional[dict], direction: str) -> float:
    """Return confidence modifier based on VPVR zone (±6 points)."""
    if not vpvr:
        return 0.0
    zone = vpvr.get("current_zone", "")
    if direction == "BUY":
        if zone in ("BELOW_VALUE_AREA", "AT_POC"):
            return 6.0
        if zone == "ABOVE_VALUE_AREA":
            return -3.0
    elif direction == "SELL":
        if zone in ("ABOVE_VALUE_AREA", "AT_POC"):
            return 6.0
        if zone == "BELOW_VALUE_AREA":
            return -3.0
    return 0.0
