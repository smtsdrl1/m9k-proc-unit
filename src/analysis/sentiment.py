"""
Sentiment Analysis Engine — News + AI-powered sentiment scoring.
Fetches news headlines and uses Groq to score market sentiment.
"""
import logging
from typing import Optional
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger("matrix_trader.analysis.sentiment")


async def fetch_crypto_news(symbol: str, limit: int = 10) -> list[str]:
    """Fetch recent crypto news headlines from CryptoPanic API (free tier)."""
    coin = symbol.split("/")[0] if "/" in symbol else symbol
    url = f"https://cryptopanic.com/api/free/v1/posts/?auth_token=free&currencies={coin}&public=true"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    return [r.get("title", "") for r in results[:limit] if r.get("title")]
    except Exception as e:
        logger.debug(f"CryptoPanic unavailable for {coin}: {e}")

    # Fallback: Google News RSS
    return await _fetch_google_news(coin, limit)


async def _fetch_google_news(query: str, limit: int = 10) -> list[str]:
    """Fetch news from Google News RSS feed."""
    url = f"https://news.google.com/rss/search?q={query}+crypto&hl=en&gl=US&ceid=US:en"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    items = soup.find_all("item")
                    return [item.find("title").text for item in items[:limit] if item.find("title")]
    except Exception as e:
        logger.debug(f"Google News unavailable for {query}: {e}")
    return []


async def fetch_bist_news(symbol: str, limit: int = 10) -> list[str]:
    """Fetch BIST news from Google News (Turkish)."""
    url = f"https://news.google.com/rss/search?q={symbol}+hisse+borsa&hl=tr&gl=TR&ceid=TR:tr"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    items = soup.find_all("item")
                    return [item.find("title").text for item in items[:limit] if item.find("title")]
    except Exception as e:
        logger.debug(f"Google News TR unavailable for {symbol}: {e}")
    return []


def build_sentiment_prompt(symbol: str, headlines: list[str], is_crypto: bool = True) -> str:
    """Build a prompt for Groq to analyze sentiment from headlines."""
    market = "kripto" if is_crypto else "BIST hisse"
    headline_text = "\n".join(f"- {h}" for h in headlines[:10]) if headlines else "- Haber bulunamadı"

    return f"""Aşağıdaki {market} haberleri {symbol} için piyasa duygu analizi yap.

HABERLER:
{headline_text}

GÖREV:
1. Genel duygu skorunu -100 (aşırı negatif) ile +100 (aşırı pozitif) arasında ver.
2. Kısa bir özet yaz (max 2 cümle).
3. Bu haberlerin fiyat üzerindeki muhtemel etkisini değerlendir.

SADECE JSON formatında yanıt ver:
{{"score": <-100..100>, "summary": "<özet>", "impact": "POSITIVE|NEGATIVE|NEUTRAL"}}"""


class SentimentResult:
    """Structured sentiment analysis result."""

    def __init__(self, score: int = 0, summary: str = "", impact: str = "NEUTRAL",
                 headlines: list[str] = None, fear_greed: int = 50):
        self.score = score          # -100 to +100
        self.summary = summary
        self.impact = impact        # POSITIVE, NEGATIVE, NEUTRAL
        self.headlines = headlines or []
        self.fear_greed = fear_greed  # 0-100 (crypto only)

    @property
    def normalized_score(self) -> float:
        """Normalize sentiment to 0-1 range for scoring."""
        return (self.score + 100) / 200.0

    @property
    def is_bullish(self) -> bool:
        return self.score > 20

    @property
    def is_bearish(self) -> bool:
        return self.score < -20

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "summary": self.summary,
            "impact": self.impact,
            "fear_greed": self.fear_greed,
            "headline_count": len(self.headlines),
        }


# Keyword-based sentiment scoring — no Groq needed
_POSITIVE_KEYWORDS = [
    "surge", "soar", "rally", "bull", "moon", "pump", "gain", "profit",
    "rise", "up", "high", "record", "ath", "breakout", "buy", "growth",
    "yüksel", "artış", "rekor", "kazanç", "alım", "büyüme", "güçlen",
    "pozitif", "boğa", "patlama", "ralli", "destek", "ivme",
    "adoption", "partnership", "launch", "approve", "bullish",
    "onay", "ortaklık", "lansman", "benimse",
]

_NEGATIVE_KEYWORDS = [
    "crash", "dump", "plunge", "bear", "fall", "drop", "loss", "sell",
    "down", "low", "fear", "panic", "hack", "scam", "fraud", "ban",
    "düşüş", "kayıp", "satış", "panik", "çöküş", "yasakla", "hack",
    "negatif", "ayı", "risk", "endişe", "kriz", "daralma",
    "regulation", "sec", "lawsuit", "investigation", "warning",
    "uyarı", "soruşturma", "dava", "ceza",
]


def keyword_sentiment_score(headlines: list[str]) -> dict:
    """
    Fast keyword-based sentiment scoring. No AI needed.
    Returns dict compatible with Groq sentiment output format.
    """
    if not headlines:
        return {"score": 0, "summary": "Haber bulunamadı", "impact": "NEUTRAL"}

    pos_count = 0
    neg_count = 0
    total = len(headlines)

    for headline in headlines:
        lower = headline.lower()
        for kw in _POSITIVE_KEYWORDS:
            if kw in lower:
                pos_count += 1
                break
        for kw in _NEGATIVE_KEYWORDS:
            if kw in lower:
                neg_count += 1
                break

    # Calculate score: -100 to +100
    if total > 0:
        net = pos_count - neg_count
        score = int((net / total) * 100)
        score = max(-100, min(100, score))
    else:
        score = 0

    if score > 20:
        impact = "POSITIVE"
        summary = f"{total} haberden {pos_count} pozitif, {neg_count} negatif — genel olumlu"
    elif score < -20:
        impact = "NEGATIVE"
        summary = f"{total} haberden {neg_count} negatif, {pos_count} pozitif — genel olumsuz"
    else:
        impact = "NEUTRAL"
        summary = f"{total} haberden {pos_count} pozitif, {neg_count} negatif — dengeli"

    return {
        "score": score,
        "summary": summary,
        "impact": impact,
        "key_events": headlines[:3],
    }
