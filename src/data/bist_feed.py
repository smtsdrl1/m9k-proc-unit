"""
BIST data feed using yfinance.
Handles ISE tickers with .IS suffix.
"""
import logging
from typing import Optional
import pandas as pd
import yfinance as yf
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.data.bist")


class BistFeed:
    """yfinance-based BIST data feed (15min delayed)."""

    @staticmethod
    def _ticker(symbol: str) -> str:
        """Ensure .IS suffix for BIST symbols."""
        return symbol if symbol.endswith(".IS") else f"{symbol}.IS"

    def fetch_ohlcv(self, symbol: str, period: str = "3mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles for a BIST symbol."""
        try:
            ticker = yf.Ticker(self._ticker(symbol))
            df = ticker.history(period=period, interval=interval)
            if df is None or len(df) < 20:
                logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0}")
                return None

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df.index.name = "timestamp"
            return df
        except Exception as e:
            logger.error(f"Error fetching BIST {symbol}: {e}")
            return None

    def fetch_multi_timeframe(self, symbol: str, timeframes: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch multiple timeframes for BIST.
        Maps: 1h -> period=1mo/interval=1h, 1d -> 6mo/1d, 1wk -> 2y/1wk
        """
        tf_map = {
            "15m": ("5d", "15m"),
            "1h":  ("1mo", "1h"),
            "1d":  ("6mo", "1d"),
            "1wk": ("2y", "1wk"),
        }
        results = {}
        for tf in timeframes:
            if tf in tf_map:
                period, interval = tf_map[tf]
                df = self.fetch_ohlcv(symbol, period=period, interval=interval)
                if df is not None:
                    results[tf] = df
        return results

    def fetch_fundamental(self, symbol: str) -> Optional[dict]:
        """Fetch fundamental data for a BIST stock."""
        try:
            ticker = yf.Ticker(self._ticker(symbol))
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("shortName", symbol),
                "sector": info.get("sector", "N/A"),
                "pe_ratio": safe_float(info.get("trailingPE")),
                "pb_ratio": safe_float(info.get("priceToBook")),
                "market_cap": safe_float(info.get("marketCap")),
                "dividend_yield": safe_float(info.get("dividendYield")) * 100,
                "roe": safe_float(info.get("returnOnEquity")) * 100,
                "debt_to_equity": safe_float(info.get("debtToEquity")),
                "revenue_growth": safe_float(info.get("revenueGrowth")) * 100,
                "profit_margin": safe_float(info.get("profitMargins")) * 100,
                "current_price": safe_float(info.get("currentPrice", info.get("regularMarketPrice"))),
                "target_price": safe_float(info.get("targetMeanPrice")),
                "52w_high": safe_float(info.get("fiftyTwoWeekHigh")),
                "52w_low": safe_float(info.get("fiftyTwoWeekLow")),
            }
        except Exception as e:
            logger.error(f"Error fetching fundamental {symbol}: {e}")
            return None

    def fetch_news(self, symbol: str) -> list[dict]:
        """Fetch recent news for a BIST stock."""
        try:
            ticker = yf.Ticker(self._ticker(symbol))
            news = ticker.news or []
            return [
                {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": item.get("providerPublishTime", 0),
                }
                for item in news[:10]
            ]
        except Exception as e:
            logger.error(f"Error fetching news {symbol}: {e}")
            return []

    def fetch_batch_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices for multiple BIST symbols."""
        prices = {}
        tickers = [self._ticker(s) for s in symbols]
        try:
            data = yf.download(tickers, period="1d", progress=False, threads=True)
            if "Close" in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else "Close" in data.columns:
                for symbol in symbols:
                    ts = self._ticker(symbol)
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            price = data["Close"][ts].iloc[-1]
                        else:
                            price = data["Close"].iloc[-1]
                        prices[symbol] = safe_float(price)
                    except (KeyError, IndexError):
                        pass
        except Exception as e:
            logger.error(f"Error fetching batch BIST prices: {e}")
        return prices
