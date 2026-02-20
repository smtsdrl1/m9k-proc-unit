"""
Macro data feed — DXY, USDTRY, Gold, VIX, S&P500, US10Y.
Used for macro correlation filters.
"""
import logging
from typing import Optional
import pandas as pd
import yfinance as yf
from src.config import MACRO_SYMBOLS
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.data.macro")


class MacroFeed:
    """Fetch macro indicator data via yfinance."""

    def fetch_indicator(self, name: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a macro indicator."""
        symbol = MACRO_SYMBOLS.get(name)
        if not symbol:
            logger.warning(f"Unknown macro indicator: {name}")
            return None
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df is None or len(df) < 5:
                return None
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df.index.name = "timestamp"
            return df
        except Exception as e:
            logger.error(f"Error fetching macro {name}: {e}")
            return None

    def fetch_all_current(self) -> dict[str, dict]:
        """Fetch current values for all macro indicators."""
        results = {}
        for name, symbol in MACRO_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if hist is not None and len(hist) >= 2:
                    current = safe_float(hist["Close"].iloc[-1])
                    prev = safe_float(hist["Close"].iloc[-2])
                    change_pct = ((current - prev) / prev * 100) if prev > 0 else 0.0
                    results[name] = {
                        "value": current,
                        "change_pct": round(change_pct, 2),
                        "prev": prev,
                    }
            except Exception as e:
                logger.error(f"Error fetching macro {name}: {e}")
        return results

    async def fetch_fear_greed(self) -> Optional[dict]:
        """Fetch Crypto Fear & Greed Index via alternative.me API."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        item = data.get("data", [{}])[0]
                        return {
                            "value": int(item.get("value", 50)),
                            "classification": item.get("value_classification", "Neutral"),
                        }
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
        return None
