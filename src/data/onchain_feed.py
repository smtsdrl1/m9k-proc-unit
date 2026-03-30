"""
On-Chain / Global Market Data Feed.

Uses CoinGecko free API (no key required, 30 req/min limit):
- BTC Dominance (BTC.D) — indicates altcoin market health
- Global market cap change — overall sentiment
- Stablecoin dominance — buying power indicator

Confidence impact:
- High BTC.D (>62%) + BUY altcoin  → caution, reduce boost
- Low  BTC.D (<40%) + BUY altcoin  → strong momentum, boost
- Market cap falling fast (-5%/day) → SELL signals boosted
"""
import logging
import time
from typing import Optional
import requests

logger = logging.getLogger("matrix_trader.data.onchain")

_COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
_CACHE: dict = {}
_CACHE_TTL = 900  # 15 minutes


def fetch_global_data() -> Optional[dict]:
    """
    Fetch global crypto market data from CoinGecko.

    Returns dict with:
        btc_dominance: float (e.g. 52.4)
        eth_dominance: float
        stablecoin_dominance: float (USDT + USDC + BUSD combined %)
        market_cap_change_24h: float (% change in total market cap)
        total_market_cap_usd: float
        active_cryptocurrencies: int
    """
    now = time.time()
    cached = _CACHE.get("global")
    if cached and now - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    try:
        r = requests.get(_COINGECKO_GLOBAL, timeout=10)
        if r.status_code != 200:
            logger.warning(f"CoinGecko global API error: {r.status_code}")
            return None

        raw = r.json().get("data", {})
        dominance = raw.get("market_cap_percentage", {})

        btc_dom = float(dominance.get("btc", 0))
        eth_dom = float(dominance.get("eth", 0))
        # Stablecoin dominance = USDT + USDC + DAI + BUSD
        stable_dom = sum(float(dominance.get(k, 0)) for k in ("usdt", "usdc", "dai", "busd"))

        result = {
            "btc_dominance": round(btc_dom, 2),
            "eth_dominance": round(eth_dom, 2),
            "stablecoin_dominance": round(stable_dom, 2),
            "market_cap_change_24h": round(
                float(raw.get("market_cap_change_percentage_24h_usd", 0)), 2
            ),
            "total_market_cap_usd": float(
                raw.get("total_market_cap", {}).get("usd", 0)
            ),
            "active_cryptocurrencies": int(raw.get("active_cryptocurrencies", 0)),
        }

        _CACHE["global"] = {"ts": now, "data": result}
        logger.debug(
            f"OnChain: BTC.D={result['btc_dominance']}%, "
            f"market_chg={result['market_cap_change_24h']}%"
        )
        return result

    except Exception as e:
        logger.warning(f"OnChain fetch error: {e}")
        return None


def get_onchain_confidence_boost(
    global_data: Optional[dict],
    symbol: str,
    direction: str,
    is_crypto: bool = True,
) -> dict:
    """
    Calculate confidence boost/penalty from on-chain market data.

    Args:
        global_data: from fetch_global_data()
        symbol: e.g. "BTC/USDT" or "ETH/USDT"
        direction: "BUY" or "SELL"
        is_crypto: False = BIST, skip on-chain for BIST

    Returns:
        {"boost": int, "reason": str}
    """
    if not is_crypto or not global_data:
        return {"boost": 0, "reason": "N/A"}

    from src.config import BTC_DOMINANCE_HIGH_THRESHOLD, BTC_DOMINANCE_LOW_THRESHOLD

    btc_dom = global_data.get("btc_dominance", 50)
    market_chg = global_data.get("market_cap_change_24h", 0)
    stable_dom = global_data.get("stablecoin_dominance", 10)

    boost = 0
    reasons = []

    is_btc = "BTC" in symbol.upper()
    is_eth = "ETH" in symbol.upper()

    # ── BTC Dominance impact ────────────────────────────────
    if not is_btc:
        if btc_dom > BTC_DOMINANCE_HIGH_THRESHOLD:
            # BTC.D high = capital flowing from alts to BTC = bad for alts
            if direction in ("BUY", "LONG", "AL"):
                boost -= 5
                reasons.append(f"high_btc_dom({btc_dom:.1f}%)")
            else:
                boost += 3
                reasons.append(f"high_btc_dom_sell_alt({btc_dom:.1f}%)")
        elif btc_dom < BTC_DOMINANCE_LOW_THRESHOLD:
            # BTC.D low = alt season
            if direction in ("BUY", "LONG", "AL"):
                boost += 5
                reasons.append(f"alt_season({btc_dom:.1f}%)")

    # ── Market Cap Change ───────────────────────────────────
    if market_chg < -5:
        # Market down >5% in 24h = strong bear
        if direction in ("BUY", "LONG", "AL"):
            boost -= 8
            reasons.append(f"market_crash({market_chg:.1f}%)")
        else:
            boost += 5
            reasons.append(f"bear_market_sell_boost({market_chg:.1f}%)")
    elif market_chg < -2:
        if direction in ("BUY", "LONG", "AL"):
            boost -= 4
            reasons.append(f"market_down({market_chg:.1f}%)")
        else:
            boost += 2
    elif market_chg > 5:
        # Market up >5% in 24h = strong bull
        if direction in ("BUY", "LONG", "AL"):
            boost += 5
            reasons.append(f"bull_market({market_chg:.1f}%)")
        else:
            boost -= 4
            reasons.append(f"bull_market_vs_sell({market_chg:.1f}%)")
    elif market_chg > 2:
        if direction in ("BUY", "LONG", "AL"):
            boost += 2
        else:
            boost -= 2

    # ── Stablecoin Dominance (high stable_dom = dry powder, bullish) ──
    if stable_dom > 12 and direction in ("BUY", "LONG", "AL"):
        boost += 2
        reasons.append(f"high_stable_dom({stable_dom:.1f}%)")

    # Clamp
    boost = max(-10, min(10, boost))

    return {
        "boost": boost,
        "reason": ", ".join(reasons) if reasons else "normal_market",
        "btc_dominance": btc_dom,
        "market_change_24h": market_chg,
    }
