"""
Crypto data feed using ccxt with multi-exchange fallback.
Tries exchanges in order until one works (handles US geo-blocking on GitHub Actions).
Supported: Gate.io → KuCoin → MEXC → OKX → Binance → Bybit
"""
import asyncio
import logging
from typing import Optional
import ccxt.async_support as ccxt
import pandas as pd
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.data.crypto")

# Exchanges to try in order — prioritize those without US geo-block
EXCHANGE_CANDIDATES = [
    ("gate", {"enableRateLimit": True}),
    ("kucoin", {"enableRateLimit": True}),
    ("mexc", {"enableRateLimit": True}),
    ("okx", {"enableRateLimit": True}),
    ("binance", {"enableRateLimit": True}),
    ("bybit", {"enableRateLimit": True}),
]


class CryptoFeed:
    """Async crypto data feed with automatic exchange failover."""

    def __init__(self):
        self.exchange = None
        self._exchange_name = None
        self._initialized = False

    async def _ensure_exchange(self):
        """Lazy-init: try exchanges until one responds."""
        if self._initialized:
            return
        for name, opts in EXCHANGE_CANDIDATES:
            try:
                ex_class = getattr(ccxt, name)
                ex = ex_class({**opts, "options": {"defaultType": "spot"}})
                # Quick connectivity test
                await ex.fetch_ticker("BTC/USDT")
                self.exchange = ex
                self._exchange_name = name
                self._initialized = True
                logger.info(f"✅ Connected to {name} exchange")
                return
            except Exception as e:
                logger.warning(f"Exchange {name} unavailable: {type(e).__name__}: {str(e)[:80]}")
                try:
                    await ex.close()
                except Exception:
                    pass
        raise RuntimeError("All crypto exchanges unavailable — check network/geo-block")

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles for a symbol."""
        try:
            await self._ensure_exchange()
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                logger.warning(f"Insufficient data for {symbol} ({timeframe}): {len(ohlcv) if ohlcv else 0} candles")
                return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error fetching {symbol} ({timeframe}): {e}")
            return None

    async def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetch current ticker info."""
        try:
            await self._ensure_exchange()
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "price": safe_float(ticker.get("last", 0)),
                "volume_24h": safe_float(ticker.get("quoteVolume", 0)),
                "change_24h": safe_float(ticker.get("percentage", 0)),
                "high_24h": safe_float(ticker.get("high", 0)),
                "low_24h": safe_float(ticker.get("low", 0)),
            }
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error fetching ticker {symbol}: {e}")
            return None

    async def fetch_multi_timeframe(self, symbol: str, timeframes: list[str], limit: int = 200) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple timeframes concurrently."""
        await self._ensure_exchange()
        results = {}
        for tf in timeframes:
            df = await self.fetch_ohlcv(symbol, tf, limit)
            if df is not None:
                results[tf] = df
            await asyncio.sleep(0.1)  # Rate limit respect
        return results

    async def fetch_batch_tickers(self, symbols: list[str]) -> list[dict]:
        """Fetch tickers for a batch of symbols."""
        await self._ensure_exchange()
        tickers = []
        try:
            all_tickers = await self.exchange.fetch_tickers(symbols)
            for symbol in symbols:
                if symbol in all_tickers:
                    t = all_tickers[symbol]
                    tickers.append({
                        "symbol": symbol,
                        "price": safe_float(t.get("last", 0)),
                        "volume_24h": safe_float(t.get("quoteVolume", 0)),
                        "change_24h": safe_float(t.get("percentage", 0)),
                        "high_24h": safe_float(t.get("high", 0)),
                        "low_24h": safe_float(t.get("low", 0)),
                    })
        except Exception as e:
            logger.error(f"Error fetching batch tickers: {e}")
        return tickers

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[dict]:
        """Fetch order book for whale detection."""
        try:
            await self._ensure_exchange()
            ob = await self.exchange.fetch_order_book(symbol, limit)
            total_bid_volume = sum(b[1] for b in ob["bids"])
            total_ask_volume = sum(a[1] for a in ob["asks"])
            return {
                "bid_volume": total_bid_volume,
                "ask_volume": total_ask_volume,
                "bid_ask_ratio": total_bid_volume / max(total_ask_volume, 0.001),
                "spread_pct": ((ob["asks"][0][0] - ob["bids"][0][0]) / ob["bids"][0][0] * 100) if ob["bids"] and ob["asks"] else 0,
            }
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error fetching order book {symbol}: {e}")
            return None

    async def _get_swap_exchange(self):
        """Get a swap/futures exchange instance for funding rates."""
        await self._ensure_exchange()
        ex_class = getattr(ccxt, self._exchange_name)
        return ex_class({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    async def fetch_funding_rate(self, symbol: str) -> Optional[dict]:
        """Fetch current funding rate for perpetual futures.
        Positive = longs pay shorts (bearish signal when extreme)
        Negative = shorts pay longs (bullish signal when extreme)
        """
        try:
            futures_exchange = await self._get_swap_exchange()
            try:
                funding = await futures_exchange.fetch_funding_rate(symbol)
                rate = safe_float(funding.get("fundingRate", 0))
                timestamp = funding.get("fundingTimestamp")

                return {
                    "symbol": symbol,
                    "funding_rate": rate,
                    "funding_rate_pct": round(rate * 100, 4),
                    "annualized_pct": round(rate * 3 * 365 * 100, 2),  # 8h intervals
                    "timestamp": timestamp,
                    "bias": "BEARISH" if rate > 0.01 else "BULLISH" if rate < -0.01 else "NEUTRAL",
                    "extreme": abs(rate) > 0.05,  # >5% = extreme
                }
            finally:
                await futures_exchange.close()
        except Exception as e:
            logger.debug(f"Funding rate not available for {symbol}: {e}")
            return None

    async def fetch_price_aggregated(self, symbol: str) -> Optional[dict]:
        """
        Fetch price from multiple exchanges and return median for accuracy.
        Falls back to primary exchange if aggregation fails.

        Returns: {"price": float, "median": float, "sources": int, ...}
        """
        from src.config import MULTI_EXCHANGE_AGGREGATION
        if not MULTI_EXCHANGE_AGGREGATION:
            return await self.fetch_ticker(symbol)

        # Prioritize fast exchanges without geo-block for aggregation
        agg_candidates = [
            ("gate", {"enableRateLimit": True}),
            ("kucoin", {"enableRateLimit": True}),
            ("mexc", {"enableRateLimit": True}),
        ]
        prices = []

        for ex_name, opts in agg_candidates:
            try:
                ex_class = getattr(ccxt, ex_name)
                ex = ex_class({**opts, "options": {"defaultType": "spot"}})
                ticker = await asyncio.wait_for(ex.fetch_ticker(symbol), timeout=8)
                price = safe_float(ticker.get("last", 0))
                if price > 0:
                    prices.append(price)
                await ex.close()
            except Exception:
                try:
                    await ex.close()
                except Exception:
                    pass

        if not prices:
            return await self.fetch_ticker(symbol)

        import statistics
        median_price = statistics.median(prices)

        return {
            "symbol": symbol,
            "price": median_price,
            "median": median_price,
            "mean": round(sum(prices) / len(prices), 8),
            "sources": len(prices),
            "all_prices": prices,
            "volume_24h": 0,
            "change_24h": 0,
            "high_24h": max(prices),
            "low_24h": min(prices),
        }

    async def fetch_batch_funding_rates(self, symbols: list[str]) -> dict:
        """Fetch funding rates for multiple symbols."""
        results = {}
        try:
            futures_exchange = await self._get_swap_exchange()
        except Exception:
            return results
        try:
            for symbol in symbols:
                try:
                    funding = await futures_exchange.fetch_funding_rate(symbol)
                    rate = safe_float(funding.get("fundingRate", 0))
                    results[symbol] = {
                        "funding_rate": rate,
                        "funding_rate_pct": round(rate * 100, 4),
                        "bias": "BEARISH" if rate > 0.01 else "BULLISH" if rate < -0.01 else "NEUTRAL",
                    }
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
        finally:
            await futures_exchange.close()
        return results
