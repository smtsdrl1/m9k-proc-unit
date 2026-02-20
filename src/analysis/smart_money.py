"""
Smart Money Detection — Volume anomalies, whale activity, unusual patterns.
Simulates institutional flow detection without paid data.
"""
import logging
import numpy as np
import pandas as pd
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.analysis.smart_money")


def detect_volume_anomaly(df: pd.DataFrame, z_threshold: float = 2.0) -> dict:
    """
    Detect abnormal volume spikes using Z-score.
    Z > 2.0 = unusual volume = potential institutional activity.
    """
    if df is None or len(df) < 30:
        return {"anomaly": False, "z_score": 0.0, "volume_ratio": 1.0, "interpretation": "Yetersiz veri"}

    volume = df["volume"].astype(float)
    mean_vol = volume.rolling(window=20).mean().iloc[-1]
    std_vol = volume.rolling(window=20).std().iloc[-1]
    current_vol = volume.iloc[-1]

    if std_vol == 0 or np.isnan(std_vol):
        return {"anomaly": False, "z_score": 0.0, "volume_ratio": 1.0, "interpretation": "Hacim verisi yok"}

    z_score = (current_vol - mean_vol) / std_vol
    vol_ratio = current_vol / max(mean_vol, 1)

    # Determine if price moved with volume
    close = df["close"].astype(float)
    price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) >= 2 else 0

    anomaly = abs(z_score) >= z_threshold

    if anomaly:
        if price_change > 0.5 and z_score > 0:
            interpretation = "🐋 TOPLAMA: Yüksek hacimle fiyat artışı — Kurumsal alım olabilir"
        elif price_change < -0.5 and z_score > 0:
            interpretation = "🐋 DAĞITIM: Yüksek hacimle fiyat düşüşü — Kurumsal satış olabilir"
        elif abs(price_change) < 0.3 and z_score > 0:
            interpretation = "🔍 BİRİKİM: Yüksek hacim, düşük volatilite — Pozisyon oluşturuluyor"
        else:
            interpretation = "⚠️ ANORMAL HACİM: Dikkatli takip et"
    else:
        interpretation = "Normal hacim"

    return {
        "anomaly": anomaly,
        "z_score": round(float(z_score), 2),
        "volume_ratio": round(float(vol_ratio), 2),
        "price_change_pct": round(float(price_change), 2),
        "interpretation": interpretation,
    }


def detect_large_candles(df: pd.DataFrame, atr: float, threshold: float = 2.5) -> list[dict]:
    """
    Detect unusually large candles that may indicate institutional orders.
    Candle body > 2.5x ATR = potential whale move.
    """
    if df is None or len(df) < 5 or atr <= 0:
        return []

    results = []
    recent = df.tail(5)

    for i, (idx, row) in enumerate(recent.iterrows()):
        body = abs(row["close"] - row["open"])
        wick_upper = row["high"] - max(row["close"], row["open"])
        wick_lower = min(row["close"], row["open"]) - row["low"]

        if body >= atr * threshold:
            direction = "BUY" if row["close"] > row["open"] else "SELL"
            results.append({
                "index": i,
                "timestamp": str(idx),
                "direction": direction,
                "body_atr_ratio": round(body / atr, 2),
                "type": "ENGULFING" if body > atr * 3 else "LARGE_CANDLE",
            })

    return results


def detect_accumulation_distribution(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Analyze Accumulation/Distribution pattern.
    Rising A/D with flat price = accumulation (smart money buying).
    Falling A/D with flat price = distribution (smart money selling).
    """
    if df is None or len(df) < lookback:
        return {"pattern": "NONE", "strength": 0, "description": "Yetersiz veri"}

    close = df["close"].tail(lookback)
    high = df["high"].tail(lookback)
    low = df["low"].tail(lookback)
    volume = df["volume"].tail(lookback)

    # Calculate Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    ad_line = mfv.cumsum()

    # Price trend
    price_slope = np.polyfit(range(len(close)), close.values, 1)[0]
    ad_slope = np.polyfit(range(len(ad_line)), ad_line.values, 1)[0]

    # Normalize slopes
    price_pct = price_slope / close.mean() * 100
    ad_pct = ad_slope / (abs(ad_line.mean()) + 1e-10) * 100

    if ad_pct > 1 and abs(price_pct) < 0.5:
        return {
            "pattern": "ACCUMULATION",
            "strength": min(round(abs(ad_pct) * 10), 100),
            "description": "A/D hattı yükseliyor, fiyat yatay — Kurumsal birikim",
        }
    elif ad_pct < -1 and abs(price_pct) < 0.5:
        return {
            "pattern": "DISTRIBUTION",
            "strength": min(round(abs(ad_pct) * 10), 100),
            "description": "A/D hattı düşüyor, fiyat yatay — Kurumsal dağıtım",
        }
    elif ad_pct > 1 and price_pct > 0.5:
        return {
            "pattern": "HEALTHY_UPTREND",
            "strength": min(round(abs(ad_pct) * 5), 100),
            "description": "Fiyat ve A/D birlikte yükseliyor — Sağlıklı trend",
        }
    elif ad_pct < -1 and price_pct < -0.5:
        return {
            "pattern": "HEALTHY_DOWNTREND",
            "strength": min(round(abs(ad_pct) * 5), 100),
            "description": "Fiyat ve A/D birlikte düşüyor — Sağlıklı düşüş",
        }
    else:
        return {"pattern": "NONE", "strength": 0, "description": "Belirgin bir akıllı para hareketi yok"}


def smart_money_analysis(df: pd.DataFrame, atr: float, order_book: dict = None) -> dict:
    """Full smart money analysis combining all detectors."""
    volume = detect_volume_anomaly(df)
    large_candles = detect_large_candles(df, atr)
    ad_pattern = detect_accumulation_distribution(df)

    # Combine into overall signal
    signals = []
    score = 0  # -100 (sell pressure) to +100 (buy pressure)

    if volume["anomaly"]:
        if volume["price_change_pct"] > 0:
            score += 20
            signals.append("VOLUME_SPIKE_BULLISH")
        elif volume["price_change_pct"] < 0:
            score -= 20
            signals.append("VOLUME_SPIKE_BEARISH")

    for candle in large_candles:
        if candle["direction"] == "BUY":
            score += 15
            signals.append("LARGE_BUY_CANDLE")
        else:
            score -= 15
            signals.append("LARGE_SELL_CANDLE")

    if ad_pattern["pattern"] == "ACCUMULATION":
        score += 25
        signals.append("ACCUMULATION")
    elif ad_pattern["pattern"] == "DISTRIBUTION":
        score -= 25
        signals.append("DISTRIBUTION")

    # Order book imbalance (if available)
    if order_book:
        ratio = order_book.get("bid_ask_ratio", 1.0)
        if ratio > 1.5:
            score += 10
            signals.append("BID_HEAVY")
        elif ratio < 0.67:
            score -= 10
            signals.append("ASK_HEAVY")

    direction = "BUY" if score > 15 else "SELL" if score < -15 else "NEUTRAL"

    return {
        "direction": direction,
        "score": max(-100, min(100, score)),
        "signals": signals,
        "volume_anomaly": volume,
        "large_candles": large_candles,
        "ad_pattern": ad_pattern,
    }
