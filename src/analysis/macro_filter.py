"""
Macro Correlation Filter.
DXY, USDTRY, VIX tracking — filter signals that go against macro trend.
Rule: "DXY sert yükseliyorsa Kripto Long açma. Dolar/TL düşüyorsa ihracatçı BIST önerme."
"""
import logging
from typing import Optional
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.analysis.macro_filter")

# Known BIST exporter companies that benefit from weak TRY
BIST_EXPORTERS = {
    "FROTO", "TOASO", "ARCLK", "VESTL", "VESBE", "OTKAR",
    "BRISA", "DOAS", "EGEEN", "KRDMD", "EREGL", "PETKM",
}


def analyze_macro(macro_data: dict, fear_greed: dict = None, is_bist: bool = False) -> dict:
    """
    Analyze macro indicators and return filters.

    Args:
        macro_data: Output of MacroFeed.fetch_all_current()
        fear_greed: Fear & Greed index data (optional)
        is_bist: Whether analyzing for BIST context

    Returns:
        {
            "dxy": {"value": x, "trend": "UP"/"DOWN"/"FLAT", "impact": str},
            "usdtry": {...},
            "vix": {...},
            "crypto_filter": "ALLOW"/"CAUTION"/"BLOCK",
            "bist_filter": "ALLOW"/"CAUTION"/"BLOCK",
            "exporter_boost": bool,
            "summary": str,
            "alerts": [],
        }
    """
    result = {
        "crypto_filter": "ALLOW",
        "bist_filter": "ALLOW",
        "exporter_boost": False,
        "details": {},
        "summary": "",
        "alerts": [],
    }

    warnings = []

    # ─── DXY Analysis ──────────────────────────────────────
    dxy = macro_data.get("DXY")
    if dxy:
        change = dxy.get("change_pct", 0)
        if change > 1.0:
            trend = "VERY_STRONG_UP"
            result["crypto_filter"] = "BLOCK"
            warnings.append("DXY çok sert yükseliyor — Kripto LONG AÇMA")
        elif change > 0.5:
            trend = "STRONG_UP"
            result["crypto_filter"] = "CAUTION"
            warnings.append("DXY güçlü yükseliş — Kripto için dikkatli ol")
        elif change < -0.5:
            trend = "DOWN"
            warnings.append("DXY düşüşte — Kripto için olumlu")
        else:
            trend = "FLAT"

        result["details"]["dxy"] = {
            "value": dxy.get("value", 0),
            "change_pct": change,
            "trend": trend,
        }

    # ─── USDTRY Analysis ──────────────────────────────────
    usdtry = macro_data.get("USDTRY")
    if usdtry:
        change = usdtry.get("change_pct", 0)
        if change < -0.3:
            trend = "TRY_STRENGTH"
            result["exporter_boost"] = False
            warnings.append("TL güçleniyor — İhracatçılar etkilenebilir")
        elif change > 0.5:
            trend = "TRY_WEAKNESS"
            result["exporter_boost"] = True
            warnings.append("TL zayıflıyor — İhracatçı hisseler avantajlı")
        else:
            trend = "STABLE"

        result["details"]["usdtry"] = {
            "value": usdtry.get("value", 0),
            "change_pct": change,
            "trend": trend,
        }

    # ─── VIX Analysis ──────────────────────────────────────
    vix = macro_data.get("VIX")
    if vix:
        val = vix.get("value", 20)
        change = vix.get("change_pct", 0)
        if val > 30 or change > 10:
            # High fear = risk-off
            result["crypto_filter"] = "CAUTION" if result["crypto_filter"] == "ALLOW" else result["crypto_filter"]
            result["bist_filter"] = "CAUTION"
            warnings.append(f"VIX yüksek ({val:.1f}) — Risk-off ortam, dikkatli ol")
        elif val > 40:
            result["crypto_filter"] = "BLOCK"
            result["bist_filter"] = "BLOCK"
            warnings.append(f"VIX çok yüksek ({val:.1f}) — PANIK modu, işlem AÇMA")

        result["details"]["vix"] = {
            "value": val,
            "change_pct": change,
        }

    result["summary"] = " | ".join(warnings) if warnings else "Makro ortam normal"
    result["alerts"] = warnings

    # Add Fear & Greed to analysis if available
    if fear_greed:
        fg_val = fear_greed.get("value", 50)
        if fg_val < 20:
            result["alerts"].append(f"Extreme Fear ({fg_val}) — Contrarian fırsat olabilir")
        elif fg_val > 80:
            result["alerts"].append(f"Extreme Greed ({fg_val}) — Dikkatli ol, düzeltme gelebilir")

    return result


def should_filter_signal(macro: dict, direction: str, is_bist: bool = False, symbol: str = None) -> dict:
    """
    Check if a signal should be filtered out based on macro conditions.

    Returns:
        {"action": "ALLOW"/"CAUTION"/"BLOCK", "reason": str}
    """
    is_crypto = not is_bist

    if is_crypto:
        filter_level = macro.get("crypto_filter", "ALLOW")
        if filter_level == "BLOCK" and direction == "BUY":
            return {"action": "BLOCK", "reason": "Makro filtre: DXY/VIX nedeniyle kripto LONG engellendi"}
        if filter_level == "CAUTION":
            return {"action": "CAUTION", "reason": "Makro dikkat: DXY/VIX yükseliyor"}
    else:
        filter_level = macro.get("bist_filter", "ALLOW")
        if filter_level == "BLOCK":
            return {"action": "BLOCK", "reason": "Makro filtre: VIX çok yüksek, BIST işlemi engellendi"}

        # Exporter boost
        if symbol and macro.get("exporter_boost") and symbol in BIST_EXPORTERS and direction == "BUY":
            return {"action": "ALLOW", "reason": f"İhracatçı boost: {symbol} TL zayıflığından faydalanır"}

    return {"action": "ALLOW", "reason": ""}
