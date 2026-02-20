"""
On-Chain Data Feed — CoinGecko proxy + Fear & Greed Index.
Free APIs only, no API key required.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("matrix_trader.data.onchain")

_CACHE: dict = {}
_CACHE_TTL = 300  # 5 minutes


def _cached(key: str, ttl: int = _CACHE_TTL):
    entry = _CACHE.get(key)
    if entry and (datetime.now() - entry["ts"]).total_seconds() < ttl:
        return entry["data"]
    return None


def _store(key: str, data):
    _CACHE[key] = {"data": data, "ts": datetime.now()}
    return data


async def get_fear_greed_index(session: aiohttp.ClientSession = None) -> dict:
    """Fetch Fear & Greed Index from alternative.me (free)."""
    cached = _cached("fear_greed")
    if cached:
        return cached

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        url = "https://api.alternative.me/fng/?limit=1"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status == 200:
                data = await resp.json()
                entry = data["data"][0]
                value = int(entry["value"])
                label = entry["value_classification"]

                if value <= 20:
                    signal = "EXTREME_FEAR"
                    modifier = 8.0
                elif value <= 40:
                    signal = "FEAR"
                    modifier = 4.0
                elif value >= 80:
                    signal = "EXTREME_GREED"
                    modifier = -8.0
                elif value >= 60:
                    signal = "GREED"
                    modifier = -4.0
                else:
                    signal = "NEUTRAL"
                    modifier = 0.0

                result = {
                    "value": value, "label": label, "signal": signal,
                    "confidence_modifier_buy": modifier,
                    "confidence_modifier_sell": -modifier,
                    "timestamp": entry.get("timestamp", ""),
                }
                return _store("fear_greed", result)
    except Exception as e:
        logger.debug(f"Fear & Greed fetch error: {e}")
    finally:
        if close_session:
            await session.close()

    return {"value": 50, "label": "Neutral", "signal": "NEUTRAL",
            "confidence_modifier_buy": 0, "confidence_modifier_sell": 0}


async def get_exchange_flows(symbol: str = "bitcoin",
                             session: aiohttp.ClientSession = None) -> dict:
    """Estimate exchange flow via CoinGecko vol/mcap ratio (free proxy)."""
    key = f"flows_{symbol}"
    cached = _cached(key)
    if cached:
        return cached

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}"
        params = {"localization": "false", "tickers": "false",
                  "community_data": "false", "developer_data": "false"}
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data  = await resp.json()
                mkt   = data.get("market_data", {})
                vol   = mkt.get("total_volume",    {}).get("usd", 0)
                mcap  = mkt.get("market_cap",      {}).get("usd", 1)
                ratio = vol / mcap if mcap > 0 else 0

                if ratio > 0.15:
                    flow_signal = "HIGH_ACTIVITY"
                    modifier    = 3.0
                elif ratio < 0.03:
                    flow_signal = "LOW_ACTIVITY"
                    modifier    = -2.0
                else:
                    flow_signal = "NORMAL"
                    modifier    = 0.0

                result = {
                    "vol_mcap_ratio": round(ratio, 4),
                    "flow_signal": flow_signal,
                    "confidence_modifier": modifier,
                }
                return _store(key, result)
    except Exception as e:
        logger.debug(f"Exchange flow fetch error ({symbol}): {e}")
    finally:
        if close_session:
            await session.close()

    return {"vol_mcap_ratio": 0, "flow_signal": "UNKNOWN", "confidence_modifier": 0}


async def get_onchain_composite(symbol: str = "bitcoin",
                                direction: str = "BUY") -> dict:
    """Combine Fear/Greed + exchange flows into single confidence modifier."""
    async with aiohttp.ClientSession() as session:
        fg, flows = await asyncio.gather(
            get_fear_greed_index(session),
            get_exchange_flows(symbol, session),
        )

    fg_mod = fg.get("confidence_modifier_buy" if direction == "BUY"
                    else "confidence_modifier_sell", 0)
    flow_mod = flows.get("confidence_modifier", 0)
    if direction == "SELL":
        flow_mod = -flow_mod

    total = max(-15.0, min(15.0, fg_mod + flow_mod))
    return {
        "fear_greed": fg, "exchange_flows": flows,
        "total_modifier": round(total, 2), "direction": direction,
    }
