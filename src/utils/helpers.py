"""
Common utility functions — safe math, formatting, logging.
Lessons learned from sniper_v2 bugs applied here.
"""
import math
import logging
from datetime import datetime
import pytz

logger = logging.getLogger("matrix_trader")


def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float, never crash."""
    try:
        result = float(val)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_positive(val, default: float = 1.0) -> float:
    """Return val if positive, else default. Never 0 or negative."""
    v = safe_float(val, default)
    return v if v > 0 else default


def format_price(price: float, is_bist: bool = False) -> str:
    """
    Smart price formatting — micro-cap coins won't display as 0.00.
    BIST always 2 decimals. Crypto adapts to price magnitude.
    """
    if price is None or (isinstance(price, float) and (math.isnan(price) or math.isinf(price))):
        return "—"
    if is_bist:
        return f"{price:.2f}"
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


def smart_round(value: float, reference_price: float) -> float:
    """
    Round value based on price magnitude.
    Prevents micro-cap prices from rounding to 0.00.
    """
    if reference_price >= 1000:
        return round(value, 2)
    elif reference_price >= 1:
        return round(value, 4)
    elif reference_price >= 0.01:
        return round(value, 6)
    else:
        return round(value, 8)


def format_pct(value: float) -> str:
    """Format percentage with sign."""
    return f"{'+' if value > 0 else ''}{value:.1f}%"


def format_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return f"{n:.2f}"


def is_bist_market_hours() -> bool:
    """Check if BIST is currently open (10:00-18:00 Istanbul time, Mon-Fri)."""
    ist = pytz.timezone("Europe/Istanbul")
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Weekend
        return False
    return 10 <= now.hour < 18


def get_istanbul_time() -> datetime:
    """Get current Istanbul time."""
    return datetime.now(pytz.timezone("Europe/Istanbul"))


def calculate_change_pct(current: float, reference: float) -> float:
    """Calculate percentage change safely."""
    if reference == 0:
        return 0.0
    return round((current - reference) / reference * 100, 2)
