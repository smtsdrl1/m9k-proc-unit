"""
ATR-based Dynamic Risk Management.
Every asset gets custom SL/TP based on its own volatility — no fixed %.
"""
import logging
from src.utils.helpers import safe_positive, smart_round

logger = logging.getLogger("matrix_trader.signals.risk_manager")


def calculate_risk(
    price: float,
    atr: float,
    sr: dict,
    direction: str,
    is_bist: bool = False,
    capital: float = 10000,
    risk_pct: float = 2.0,
) -> dict:
    """
    Calculate ATR-based stop loss, take profit targets, and position size.

    SL = 1.5 * ATR from entry
    T1 = 1 * ATR, T2 = 2 * ATR, T3 = 3 * ATR

    Returns:
        {
            "stop_loss": float,
            "targets": {"t1": float, "t2": float, "t3": float},
            "risk_amount": float,
            "reward_risk": float,
            "position_size": float,
            "position_value": float,
        }
    """
    p = safe_positive(price, 100)
    a = safe_positive(atr, p * 0.02)

    s1 = safe_positive(sr.get("support1", p * 0.98), p * 0.98)
    s2 = safe_positive(sr.get("support2", p * 0.96), p * 0.96)
    r1 = safe_positive(sr.get("resistance1", p * 1.02), p * 1.02)
    r2 = safe_positive(sr.get("resistance2", p * 1.04), p * 1.04)

    if direction == "BUY":
        # Stop Loss: below support or 1.5 ATR below entry
        sl = max(min(s1, p - 1.5 * a), p - 3 * a)
        sl = min(sl, p * 0.95)  # Never more than 5% below

        # Targets: ATR multiples or resistance levels
        t1 = max(p + 1.0 * a, r1 * 0.99)
        t2 = max(p + 2.0 * a, r1)
        t3 = max(p + 3.0 * a, r2)
    else:  # SELL
        # Stop Loss: above resistance or 1.5 ATR above entry
        sl = min(max(r1, p + 1.5 * a), p + 3 * a)
        sl = max(sl, p * 1.05)  # Never more than 5% above

        # Targets: ATR multiples below
        t1 = min(p - 1.0 * a, s1 * 1.01)
        t2 = min(p - 2.0 * a, s1)
        t3 = min(p - 3.0 * a, s2)

    # Ensure SL and targets are valid
    sl = safe_positive(sl, p * (0.95 if direction == "BUY" else 1.05))
    t1 = safe_positive(t1, p * (1.02 if direction == "BUY" else 0.98))
    t2 = safe_positive(t2, p * (1.04 if direction == "BUY" else 0.96))
    t3 = safe_positive(t3, p * (1.06 if direction == "BUY" else 0.94))

    # Risk calculation
    risk_amount = abs(p - sl)
    risk_amount = safe_positive(risk_amount, p * 0.02)

    avg_reward = (abs(t1 - p) + abs(t2 - p) + abs(t3 - p)) / 3
    avg_reward = safe_positive(avg_reward, p * 0.02)
    reward_risk = round(avg_reward / risk_amount, 2)

    # Position sizing: risk_pct of capital / risk_amount per unit
    pos_size = (capital * risk_pct / 100) / risk_amount
    pos_size = min(pos_size, 100000)  # Cap at 100K units

    # Kademeli kar alma (partial take profit)
    from src.config import PARTIAL_TP_ENABLED, PARTIAL_TP_RATIOS
    partial_tp = None
    if PARTIAL_TP_ENABLED:
        partial_tp = {
            "t1_close_pct": PARTIAL_TP_RATIOS["t1"] * 100,  # 33%
            "t2_close_pct": PARTIAL_TP_RATIOS["t2"] * 100,  # 33%
            "t3_close_pct": PARTIAL_TP_RATIOS["t3"] * 100,  # 34% (trailing SL)
            "t1_size": round(pos_size * PARTIAL_TP_RATIOS["t1"], 2),
            "t2_size": round(pos_size * PARTIAL_TP_RATIOS["t2"], 2),
            "t3_size": round(pos_size * PARTIAL_TP_RATIOS["t3"], 2),
        }

    # Trailing stop initial value
    from src.config import TRAILING_STOP_ENABLED, TRAILING_STOP_ATR_MULT
    trailing_sl = None
    if TRAILING_STOP_ENABLED:
        trailing_sl = calculate_trailing_stop(p, p, a, direction, TRAILING_STOP_ATR_MULT)

    # Smart rounding based on price magnitude
    return {
        "stop_loss": smart_round(sl, p),
        "targets": {
            "t1": smart_round(t1, p),
            "t2": smart_round(t2, p),
            "t3": smart_round(t3, p),
        },
        "risk_amount": smart_round(risk_amount, p),
        "reward_risk": reward_risk,
        "position_size": round(pos_size, 2),
        "position_value": round(pos_size * p, 2),
        "partial_tp": partial_tp,
        "trailing_sl": smart_round(trailing_sl, p) if trailing_sl else None,
    }


def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    atr: float,
    direction: str,
    multiplier: float = 2.0,
) -> float:
    """
    ATR Trailing Stop — dynamic stop that follows the price.
    Moves with the trend, locks in profits as price moves favorably.
    """
    a = safe_positive(atr, entry_price * 0.02)

    if direction == "BUY":
        trailing_sl = current_price - multiplier * a
        # Never below entry minus initial SL
        initial_sl = entry_price - 1.5 * a
        return max(trailing_sl, initial_sl)
    else:
        trailing_sl = current_price + multiplier * a
        initial_sl = entry_price + 1.5 * a
        return min(trailing_sl, initial_sl)


def calculate_kelly_size(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    capital: float,
    price: float,
) -> dict:
    """Kelly Criterion position sizing.

    Args:
        win_rate: Historical win rate 0-1
        avg_win_pct: Average win as fraction (0.04 = 4%)
        avg_loss_pct: Average loss as fraction (positive, 0.02 = 2%)
        capital: Total capital
        price: Current asset price

    Returns:
        {"kelly_full": float, "kelly_half": float, "position_value": float,
         "position_size": float, "notes": str}
    """
    try:
        from src.config import KELLY_SIZING_ENABLED, KELLY_FRACTION, KELLY_MAX_PCT
    except ImportError:
        KELLY_SIZING_ENABLED = True; KELLY_FRACTION = 0.5; KELLY_MAX_PCT = 5.0

    if not KELLY_SIZING_ENABLED or avg_win_pct <= 0 or avg_loss_pct <= 0:
        return {"notes": "Kelly disabled or insufficient data"}

    if win_rate <= 0 or win_rate >= 1:
        return {"notes": "Invalid win rate"}

    # Kelly: f* = (b*p - q) / b, where b = avg_win/avg_loss
    b = avg_win_pct / avg_loss_pct
    p = win_rate
    q = 1 - win_rate
    kelly_full = (b * p - q) / b

    if kelly_full <= 0:
        return {"kelly_full": 0, "kelly_half": 0,
                "position_value": 0, "position_size": 0,
                "notes": "Kelly negative — skip trade"}

    kelly_half = kelly_full * KELLY_FRACTION
    kelly_capped = min(kelly_half, KELLY_MAX_PCT / 100)

    position_value = capital * kelly_capped
    position_size  = position_value / price if price > 0 else 0

    return {
        "kelly_full":      round(kelly_full * 100, 2),   # %
        "kelly_half":      round(kelly_half * 100, 2),   # %
        "kelly_applied":   round(kelly_capped * 100, 2), # % (after cap)
        "position_value":  round(position_value, 2),
        "position_size":   round(position_size, 4),
        "notes": f"WR={win_rate:.1%} W={avg_win_pct:.2%} L={avg_loss_pct:.2%}",
    }


def check_correlation(new_symbol: str, open_symbols: list,
                      price_histories: dict) -> tuple[bool, float]:
    """Check if new position is highly correlated with existing positions.

    Args:
        new_symbol: Symbol to open
        open_symbols: Currently open symbols
        price_histories: {symbol: list[float]} close prices

    Returns:
        (can_open: bool, max_correlation: float)
    """
    try:
        from src.config import CORRELATION_ENABLED, MAX_CORRELATION_THRESHOLD
    except ImportError:
        CORRELATION_ENABLED = True; MAX_CORRELATION_THRESHOLD = 0.75

    if not CORRELATION_ENABLED or not open_symbols:
        return True, 0.0

    import numpy as np

    new_prices = price_histories.get(new_symbol)
    if not new_prices or len(new_prices) < 10:
        return True, 0.0

    max_corr = 0.0
    for sym in open_symbols:
        if sym == new_symbol:
            continue
        existing = price_histories.get(sym)
        if not existing or len(existing) < 10:
            continue
        try:
            min_len = min(len(new_prices), len(existing))
            p1 = np.array(new_prices[-min_len:], dtype=float)
            p2 = np.array(existing[-min_len:],   dtype=float)
            r1 = np.diff(p1) / p1[:-1]
            r2 = np.diff(p2) / p2[:-1]
            corr = abs(float(np.corrcoef(r1, r2)[0, 1]))
            if corr > max_corr:
                max_corr = corr
        except Exception:
            continue

    can_open = max_corr < MAX_CORRELATION_THRESHOLD
    if not can_open:
        logger.warning(
            f"🔗 Correlation block: {new_symbol} vs open positions "
            f"corr={max_corr:.2f} > {MAX_CORRELATION_THRESHOLD}"
        )
    return can_open, round(max_corr, 3)
