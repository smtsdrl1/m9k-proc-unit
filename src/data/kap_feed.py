"""
BIST KAP (Public Disclosure Platform) Calendar.

Checks for recent financial disclosures from KAP to suppress signals
within a blackout window around announcements.

Uses KAP's public search endpoint (no API key required).
Gracefully falls back to time-based heuristics if unavailable.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
import requests

logger = logging.getLogger("matrix_trader.data.kap")

_KAP_API = "https://www.kap.org.tr/tr/api/contents/notifications"
_CACHE: dict = {}
_CACHE_TTL = 300   # 5 minutes


def fetch_recent_disclosures(symbol: str, hours: int = 2) -> list[dict]:
    """
    Fetch recent KAP disclosures for a BIST symbol.

    Returns list of dicts: [{symbol, title, type, published_at}, ...]
    Returns empty list on any error (non-blocking).
    """
    cache_key = f"{symbol}:{hours}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    try:
        params = {
            "stock": symbol.upper(),
            "pageName": "ALL",
            "pageSize": 10,
        }
        r = requests.get(_KAP_API, params=params, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (compatible; MatrixTrader/1.0)"
        })

        if r.status_code != 200:
            logger.debug(f"KAP API returned {r.status_code} for {symbol}")
            _CACHE[cache_key] = {"ts": now, "data": []}
            return []

        items = r.json() if isinstance(r.json(), list) else r.json().get("content", [])
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = []

        for item in items[:20]:
            # KAP returns publishedAt in ISO format
            pub_str = item.get("publishedAt") or item.get("published_at") or ""
            if not pub_str:
                continue
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00").replace("+03:00", ""))
                # Convert Turkey time (UTC+3) to UTC
                pub_dt_utc = pub_dt - timedelta(hours=3)
            except Exception:
                continue

            if pub_dt_utc >= cutoff:
                recent.append({
                    "symbol": symbol,
                    "title": item.get("title", ""),
                    "type": item.get("type", ""),
                    "published_at": pub_dt_utc.isoformat(),
                    "is_financial": _is_financial_disclosure(item.get("type", ""), item.get("title", "")),
                })

        _CACHE[cache_key] = {"ts": now, "data": recent}
        if recent:
            logger.info(f"[{symbol}] KAP: {len(recent)} recent disclosure(s) in last {hours}h")
        return recent

    except Exception as e:
        logger.debug(f"KAP fetch error for {symbol}: {e}")
        _CACHE[cache_key] = {"ts": now, "data": []}
        return []


def _is_financial_disclosure(disc_type: str, title: str) -> bool:
    """Check if disclosure is a high-impact financial event."""
    financial_keywords = [
        "finansal", "bilanço", "kâr", "kar", "zarar", "temettü", "sermaye",
        "faaliyet", "yönetim kurulu", "genel kurul", "bağımsız denetim",
        "financial", "balance", "dividend", "capital", "earnings",
    ]
    combined = (disc_type + " " + title).lower()
    return any(kw in combined for kw in financial_keywords)


def should_suppress_signal(symbol: str, blackout_minutes: int = 30) -> dict:
    """
    Determine if a signal should be suppressed due to recent KAP activity.

    Returns:
        {"suppress": bool, "reason": str, "disclosures": list}
    """
    from src.config import KAP_FILTER_ENABLED, KAP_BLACKOUT_MINUTES

    if not KAP_FILTER_ENABLED:
        return {"suppress": False, "reason": "disabled"}

    # Use configurable blackout window
    bm = blackout_minutes or KAP_BLACKOUT_MINUTES

    # ── Time-based heuristics (always active) ─────────────────
    now_utc = datetime.utcnow()
    # BIST market: 09:00-18:00 Istanbul (UTC+3) = 06:00-15:00 UTC
    bist_open_utc = now_utc.replace(hour=6, minute=0, second=0, microsecond=0)
    bist_close_utc = now_utc.replace(hour=15, minute=0, second=0, microsecond=0)

    # First 30 minutes after market open — volatile, suppress speculative signals
    if bist_open_utc <= now_utc <= bist_open_utc + timedelta(minutes=bm):
        return {
            "suppress": False,  # Don't suppress, but note
            "reason": "market_open_window",
            "caution": True,
        }

    # Last 15 minutes before close
    if bist_close_utc - timedelta(minutes=15) <= now_utc <= bist_close_utc:
        return {"suppress": False, "reason": "pre_close", "caution": True}

    # ── KAP API check ──────────────────────────────────────────
    try:
        disclosures = fetch_recent_disclosures(symbol, hours=int(bm / 60) + 1)
        financial_discs = [d for d in disclosures if d.get("is_financial")]

        if financial_discs:
            latest = financial_discs[0]
            return {
                "suppress": True,
                "reason": f"KAP finansal açıklama: {latest['title'][:60]}",
                "disclosures": financial_discs,
            }

        return {"suppress": False, "reason": "no_kap_activity", "disclosures": disclosures}

    except Exception:
        return {"suppress": False, "reason": "kap_check_failed"}
