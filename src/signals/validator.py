"""
Pre-send Signal Validator.
Ensures no broken signals (0-price, infinite position size, etc.) reach Telegram.
Learned from sniper_v2 REZ/USDT 0-price bug.
"""
import logging
from src.utils.helpers import format_price, safe_float

logger = logging.getLogger("matrix_trader.signals.validator")


def validate_signal(
    symbol: str,
    price: float,
    risk_mgmt: dict,
    confidence: int,
    direction: str,
    is_bist: bool = False,
    min_confidence: int = 35,
) -> tuple[bool, str]:
    """
    Validate a signal before sending to Telegram.
    Returns (is_valid, reason).
    """
    # Price must be positive
    if price is None or price <= 0:
        return False, f"{symbol}: Fiyat sıfır veya negatif ({price})"

    # Formatted price should not be all zeros
    formatted = format_price(price, is_bist)
    clean = formatted.replace(".", "").replace("0", "").replace(",", "").strip()
    if not clean or formatted == "—":
        return False, f"{symbol}: Formatlanmış fiyat sıfır gösteriliyor ({formatted})"

    # Stop loss validation
    sl = risk_mgmt.get("stop_loss", 0)
    if sl <= 0:
        return False, f"{symbol}: Stop Loss <= 0"

    # Target validation
    targets = risk_mgmt.get("targets", {})
    for tname, tval in targets.items():
        if tval <= 0:
            return False, f"{symbol}: Hedef {tname} <= 0"

    # Risk amount validation
    risk_amount = risk_mgmt.get("risk_amount", 0)
    if risk_amount <= 0:
        return False, f"{symbol}: Risk tutarı <= 0"

    # Position size sanity
    pos_size = risk_mgmt.get("position_size", 0)
    if pos_size <= 0:
        return False, f"{symbol}: Pozisyon boyutu <= 0"
    if pos_size > 100000:
        return False, f"{symbol}: Pozisyon boyutu çok büyük ({pos_size})"

    # R:R must be reasonable
    rr = risk_mgmt.get("reward_risk", 0)
    if rr < 0.5:
        return False, f"{symbol}: R:R çok düşük ({rr})"

    # Confidence check
    if confidence < min_confidence:
        return False, f"{symbol}: Güven çok düşük ({confidence}% < {min_confidence}%)"

    # Direction check
    if direction not in ("BUY", "SELL"):
        return False, f"{symbol}: Geçersiz yön ({direction})"

    # Target direction check
    if direction == "BUY":
        if targets.get("t1", 0) <= price:
            return False, f"{symbol}: BUY sinyali ama T1 fiyatın altında"
        if sl >= price:
            return False, f"{symbol}: BUY sinyali ama SL fiyatın üstünde"
    else:
        if targets.get("t1", 0) >= price:
            return False, f"{symbol}: SELL sinyali ama T1 fiyatın üstünde"
        if sl <= price:
            return False, f"{symbol}: SELL sinyali ama SL fiyatın altında"

    return True, "OK"
