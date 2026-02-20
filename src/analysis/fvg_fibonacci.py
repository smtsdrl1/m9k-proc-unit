"""
FVG (Fair Value Gap) + Fibonacci Confluence Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kaynak: Alper INCE @alper3968 — XU100 & Kripto analizi
        https://x.com/alper3968/status/1862990567153557955
        Hashtag: #xu100 #fibonacci #fvg

Bu modül, ICT (Inner Circle Trader) konseptinden gelen Fair Value Gap (FVG) ile
klasik Fibonacci retracement seviyelerini birleştirerek yüksek kaliteli giriş
noktaları tespit eder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Fair Value Gap (FVG) Nedir?
  Fiyat çok hızlı ve tek yönlü hareket ettiğinde, 1. ve 3. mumun arasında
  piyasa "doldurulmamış" bir boşluk bırakır. Bu bölgeler piyasada "imbalance"
  oluşturur ve fiyat bu bölgeleri doldurmaya (fill etmeye) meyillidir.

  Bullish FVG:
    ┌ mum3.low ─────────────────  ← FVG üst sınırı
    │   [BOŞLUK = FVG bölgesi]
    └ mum1.high ─────────────────  ← FVG alt sınırı
    (mum2 = hareket mumu)

  Bearish FVG:
    ┌ mum1.low ──────────────────  ← FVG üst sınırı
    │   [BOŞLUK = FVG bölgesi]
    └ mum3.high ─────────────────  ← FVG alt sınırı

📌 Fibonacci + FVG Confluence:
  Fibonacci retracement seviyelerinin (özellikle 0.618 = altın oran) bir FVG
  bölgesiyle örtüşmesi → en yüksek olasılıklı geri dönüş noktası = SNIPER GİRİŞ
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from src.utils.helpers import safe_float

logger = logging.getLogger("matrix_trader.analysis.fvg_fibonacci")


# ─── Sabitler ────────────────────────────────────────────────────────────────

# Fibonacci seviyeleri ve göreli güçleri
FIB_CONFIG = {
    "0.236": {"weight": 0.40, "label": "Düşük Destek"},
    "0.382": {"weight": 0.70, "label": "Orta Destek"},
    "0.500": {"weight": 0.80, "label": "Psikolojik Seviye"},
    "0.618": {"weight": 1.00, "label": "Altın Oran 🏆"},
    "0.786": {"weight": 0.65, "label": "Derin Retracement"},
}

CONFLUENCE_TOLERANCE = 0.015  # %1.5 tolerans (FVG–Fib örtüşmesi için)
FVG_LOOKBACK         = 60     # Kaç mum geriye bakılacak
FIB_LOOKBACK         = 100    # Fibonacci için kaç mum geriye bakılacak


# ─── 1. FVG Tespit ───────────────────────────────────────────────────────────

def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = FVG_LOOKBACK) -> list[dict]:
    """
    OHLCV DataFrame üzerinde Fair Value Gap (FVG) tespit eder.

    Args:
        df:       OHLCV DataFrame (kolonlar: open, high, low, close, volume — küçük harf)
        lookback: Geriye bakılacak mum sayısı

    Returns:
        Aktif (doldurulmamış) FVG listesi — en yeni önce sıralı
        Her FVG: {type, top, bottom, midpoint, size_pct, idx, filled}
    """
    if df is None or len(df) < 5:
        return []

    # Kolon isimleri normalize et
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ("high", "h"):       col_map["high"]  = col
        elif col_lower in ("low", "l"):      col_map["low"]   = col
        elif col_lower in ("close", "c"):    col_map["close"] = col
        elif col_lower in ("open", "o"):     col_map["open"]  = col

    if not all(k in col_map for k in ("high", "low", "close")):
        # Positional fallback — varsay: [Open, High, Low, Close, Volume]
        cols = list(df.columns)
        if len(cols) >= 4:
            col_map = {"open": cols[0], "high": cols[1], "low": cols[2], "close": cols[3]}
        else:
            return []

    high_col  = col_map["high"]
    low_col   = col_map["low"]
    close_col = col_map["close"]

    start = max(0, len(df) - lookback - 2)
    fvgs: list[dict] = []

    for i in range(start, len(df) - 2):
        c0_high = safe_float(df[high_col].iloc[i])
        c0_low  = safe_float(df[low_col].iloc[i])
        c2_high = safe_float(df[high_col].iloc[i + 2])
        c2_low  = safe_float(df[low_col].iloc[i + 2])

        if c0_high <= 0 or c0_low <= 0 or c2_high <= 0 or c2_low <= 0:
            continue

        # ── Bullish FVG: c[i].high < c[i+2].low ─────────────
        if c2_low > c0_high:
            gap_size = c2_low - c0_high
            mid = (c0_high + c2_low) / 2
            fvgs.append({
                "type":     "bullish",
                "top":      c2_low,
                "bottom":   c0_high,
                "midpoint": mid,
                "size":     gap_size,
                "size_pct": gap_size / mid * 100 if mid > 0 else 0,
                "idx":      i + 1,
                "filled":   False,
            })

        # ── Bearish FVG: c[i].low > c[i+2].high ─────────────
        elif c0_low > c2_high:
            gap_size = c0_low - c2_high
            mid = (c2_high + c0_low) / 2
            fvgs.append({
                "type":     "bearish",
                "top":      c0_low,
                "bottom":   c2_high,
                "midpoint": mid,
                "size":     gap_size,
                "size_pct": gap_size / mid * 100 if mid > 0 else 0,
                "idx":      i + 1,
                "filled":   False,
            })

    # ── Dolumu kontrol et ────────────────────────────────────
    current_close = safe_float(df[close_col].iloc[-1])

    for fvg in fvgs:
        if fvg["type"] == "bullish":
            # Fiyat FVG'nin altına tamamen geçtiyse kapandı
            if current_close < fvg["bottom"] * 0.998:
                fvg["filled"] = True
            # Ne kadar dolduruldu?
            fill_pct = max(0.0, min(1.0,
                (current_close - fvg["bottom"]) / max(fvg["size"], 1e-10)
            ))
            fvg["fill_pct"] = round(fill_pct * 100, 1)
        else:
            if current_close > fvg["top"] * 1.002:
                fvg["filled"] = True
            fill_pct = max(0.0, min(1.0,
                (fvg["top"] - current_close) / max(fvg["size"], 1e-10)
            ))
            fvg["fill_pct"] = round(fill_pct * 100, 1)

    # Aktif FVG'ler — en yeni önce
    active = [f for f in fvgs if not f["filled"]]
    active.sort(key=lambda x: x["idx"], reverse=True)
    return active


# ─── 2. Fibonacci Seviyeleri ─────────────────────────────────────────────────

def calculate_fibonacci_levels(
    df: pd.DataFrame,
    lookback: int = FIB_LOOKBACK,
) -> dict:
    """
    Swing high/low üzerinden Fibonacci retracement ve extension seviyeleri hesaplar.

    Args:
        df:       OHLCV DataFrame
        lookback: Swing tespit için geriye bakılacak mum sayısı

    Returns: {
        swing_high, swing_low, direction,
        "0.236", "0.382", "0.500", "0.618", "0.786", "1.000",
        "ext_1.272", "ext_1.618"  (extension)
    }
    """
    if df is None or len(df) < 10:
        return {}

    # Kolon isimleri bul
    high_col = low_col = close_col = None
    for col in df.columns:
        cl = col.lower()
        if cl in ("high", "h"):       high_col  = col
        elif cl in ("low", "l"):      low_col   = col
        elif cl in ("close", "c"):    close_col = col

    if high_col is None or low_col is None or close_col is None:
        cols = list(df.columns)
        if len(cols) >= 4:
            high_col, low_col, close_col = cols[1], cols[2], cols[3]
        else:
            return {}

    recent     = df.tail(min(lookback, len(df)))
    swing_high = safe_float(recent[high_col].max())
    swing_low  = safe_float(recent[low_col].min())
    diff       = swing_high - swing_low

    if diff <= 0 or swing_high <= 0:
        return {}

    # Trend yönü belirleme: son 20 mumun slope'u
    tail20     = df[close_col].tail(min(20, len(df)))
    first_c    = safe_float(tail20.iloc[0])
    last_c     = safe_float(tail20.iloc[-1])
    direction  = "uptrend" if last_c > first_c else "downtrend"

    # Swing High'dan aşağı retracement (uptrend sonrası)
    # veya Swing Low'dan yukarı extension (downtrend sonrası)
    levels = {
        "swing_high": swing_high,
        "swing_low":  swing_low,
        "direction":  direction,
        "diff":       diff,
        "0.000":      swing_high,
        "0.236":      swing_high - diff * 0.236,
        "0.382":      swing_high - diff * 0.382,
        "0.500":      swing_high - diff * 0.500,
        "0.618":      swing_high - diff * 0.618,   # Altın oran
        "0.786":      swing_high - diff * 0.786,
        "1.000":      swing_low,
        "ext_1.272":  swing_low  - diff * 0.272,   # Extension
        "ext_1.618":  swing_low  - diff * 0.618,   # Golden extension
    }
    return levels


# ─── 3. Confluence Tespit ────────────────────────────────────────────────────

def find_fvg_fibonacci_confluence(
    current_price: float,
    fvgs: list[dict],
    fib_levels: dict,
    tolerance: float = CONFLUENCE_TOLERANCE,
) -> Optional[dict]:
    """
    FVG bölgesi ile Fibonacci seviyesi arasındaki confluence'ı tespit eder.

    Alper INCE metodolojisi gereği:
    • FVG zone içinde veya yakınında kritik Fib seviyesi varsa → confluence
    • 0.618 (altın oran) → en güçlü sinyal  
    • Birden fazla confluence varsa en güçlüsü döndürülür

    Args:
        current_price: Şu anki kapanış fiyatı
        fvgs:          detect_fair_value_gaps() çıktısı
        fib_levels:    calculate_fibonacci_levels() çıktısı
        tolerance:     Örtüşme toleransı (default %1.5)

    Returns: confluence dict veya None
    """
    if not fvgs or not fib_levels:
        return None

    best_confluence = None
    best_strength   = -1.0

    for fvg in fvgs:
        for fib_key, fib_cfg in FIB_CONFIG.items():
            fib_price = fib_levels.get(fib_key)
            if fib_price is None:
                continue

            # FVG bölgesinde mi?
            in_zone = (
                fvg["bottom"] * (1 - tolerance) <= fib_price <= fvg["top"] * (1 + tolerance)
            )

            # Yakın mı? (toleransın 2 katı içinde)
            near_zone_dist = abs(fib_price - fvg["midpoint"]) / max(fvg["midpoint"], 1e-10)
            near_zone = near_zone_dist < tolerance * 2

            if not (in_zone or near_zone):
                continue

            # Fiyat bu confluence bölgesine yakın mı?
            dist_to_price = abs(current_price - fvg["midpoint"]) / max(fvg["midpoint"], 1e-10)
            if dist_to_price > tolerance * 4:
                continue

            # ── Confluence gücü hesapla ──────────────────────
            proximity_factor  = max(0, 1.0 - dist_to_price / (tolerance * 4))
            in_zone_bonus     = 0.20 if in_zone else 0.0
            golden_ratio_mult = 1.30 if fib_key == "0.618" else 1.0

            strength = (
                fib_cfg["weight"]
                * proximity_factor
                * golden_ratio_mult
                + in_zone_bonus
            )

            if strength > best_strength:
                best_strength = strength
                best_confluence = {
                    # FVG bilgisi
                    "fvg_type":      fvg["type"],
                    "fvg_top":       round(fvg["top"], 8),
                    "fvg_bottom":    round(fvg["bottom"], 8),
                    "fvg_midpoint":  round(fvg["midpoint"], 8),
                    "fvg_size_pct":  round(fvg.get("size_pct", 0), 3),
                    "fvg_fill_pct":  fvg.get("fill_pct", 0),
                    # Fibonacci bilgisi
                    "fib_level":     fib_key,
                    "fib_label":     fib_cfg["label"],
                    "fib_price":     round(fib_price, 8),
                    "is_golden":     fib_key == "0.618",
                    # Konum bilgisi
                    "in_zone":       in_zone,
                    "distance_pct":  round(dist_to_price * 100, 3),
                    "proximity":     round(proximity_factor, 3),
                    # Sinyal gücü
                    "strength":      round(min(strength, 1.5), 3),
                    "score_boost":   int(round(strength * 15)),  # +0 to +22 puan (100 üzerinden)
                    # Yön
                    "signal":        "BUY"  if fvg["type"] == "bullish" else "SELL",
                }

    return best_confluence


# ─── 4. Ana Analiz Fonksiyonu ─────────────────────────────────────────────────

def analyze_fvg_fibonacci(
    df: pd.DataFrame,
    fvg_lookback: int  = FVG_LOOKBACK,
    fib_lookback: int  = FIB_LOOKBACK,
    tolerance:    float = CONFLUENCE_TOLERANCE,
) -> dict:
    """
    FVG + Fibonacci tam analizi — tek çağrıyla tüm bilgileri döndürür.

    Kullanım:
        result = analyze_fvg_fibonacci(ohlcv_df)

        result["has_confluence"]   → True/False
        result["confluence"]       → En güçlü confluence detayı
        result["signal"]           → "BUY" / "SELL" / "NEUTRAL"
        result["score_boost"]      → Mevcut puana eklenecek bonus (0-22)
        result["active_fvgs"]      → Aktif FVG listesi
        result["fib_levels"]       → Fibonacci seviyeleri dict
        result["summary"]          → İnsan okuyabilir özet string

    Args:
        df:           OHLCV DataFrame
        fvg_lookback: FVG tarama penceresi
        fib_lookback: Fibonacci penceresi
        tolerance:    Confluence toleransı

    Returns: analiz sonucu dict
    """
    result = {
        "has_confluence":  False,
        "confluence":      None,
        "signal":          "NEUTRAL",
        "score_boost":     0,
        "active_fvgs":     [],
        "fib_levels":      {},
        "bullish_fvg_count": 0,
        "bearish_fvg_count": 0,
        "summary":         "Analiz yapılamadı.",
    }

    if df is None or len(df) < 50:
        result["summary"] = "Yetersiz veri (min 50 mum gerekli)"
        return result

    # Güncel fiyat
    close_col = [c for c in df.columns if c.lower() in ("close", "c")]
    close_col = close_col[0] if close_col else (list(df.columns)[3] if len(df.columns) >= 4 else None)
    if close_col is None:
        result["summary"] = "Fiyat kolonu bulunamadı"
        return result

    current_price = safe_float(df[close_col].iloc[-1])

    # 1. FVG'leri bul
    fvgs = detect_fair_value_gaps(df, lookback=fvg_lookback)
    bullish_fvgs = [f for f in fvgs if f["type"] == "bullish"]
    bearish_fvgs = [f for f in fvgs if f["type"] == "bearish"]

    # 2. Fibonacci seviyeleri hesapla
    fib = calculate_fibonacci_levels(df, lookback=fib_lookback)

    # 3. Confluence kontrolü
    confluence = find_fvg_fibonacci_confluence(current_price, fvgs, fib, tolerance)

    result.update({
        "active_fvgs":       fvgs,
        "bullish_fvg_count": len(bullish_fvgs),
        "bearish_fvg_count": len(bearish_fvgs),
        "fib_levels":        {k: v for k, v in fib.items() if isinstance(v, (int, float))},
    })

    if not confluence:
        summary_parts = [
            f"Aktif FVG: {len(fvgs)} ({len(bullish_fvgs)} bullish, {len(bearish_fvgs)} bearish)"
        ]
        if fib:
            summary_parts.append(
                f"Fib 0.618: {fib.get('0.618', 0):.4f} | "
                f"0.382: {fib.get('0.382', 0):.4f}"
            )
        summary_parts.append("FVG+Fib confluence yok — bekleme")
        result["summary"] = " | ".join(summary_parts)
        return result

    # Confluence bulundu!
    result.update({
        "has_confluence": True,
        "confluence":     confluence,
        "signal":         confluence["signal"],
        "score_boost":    confluence["score_boost"],
    })

    golden_tag = " [🏆 ALTIN ORAN]" if confluence["is_golden"] else ""
    in_zone_tag = " [BÖLGE İÇİ ✅]" if confluence["in_zone"] else ""

    result["summary"] = (
        f"🎯 FVG+Fib CONFLUENCE {confluence['signal']}{golden_tag}{in_zone_tag}\n"
        f"   FVG ({confluence['fvg_type'].upper()}): "
        f"{confluence['fvg_bottom']:.4f} – {confluence['fvg_top']:.4f} "
        f"(fill: %{confluence['fvg_fill_pct']:.0f})\n"
        f"   Fibonacci {confluence['fib_level']} ({confluence['fib_label']}): "
        f"{confluence['fib_price']:.4f}\n"
        f"   Mesafe: %{confluence['distance_pct']:.2f} | "
        f"Güç: {confluence['strength']:.2f} | "
        f"+{confluence['score_boost']} puan"
    )

    logger.info(result["summary"])
    return result


# ─── 5. Telegram Mesajı Formatı ──────────────────────────────────────────────

def format_fvg_fib_telegram(analysis: dict, symbol: str) -> str:
    """
    FVG+Fib analizini Telegram mesaj formatına çevirir.

    Args:
        analysis: analyze_fvg_fibonacci() çıktısı
        symbol:   Sembol adı

    Returns: Telegram HTML formatında string
    """
    if not analysis.get("has_confluence"):
        fvg_cnt = len(analysis.get("active_fvgs", []))
        return (
            f"📐 <b>FVG+Fibonacci</b> ({symbol})\n"
            f"   Aktif FVG: {fvg_cnt} | Confluence yok\n"
        )

    c = analysis["confluence"]
    golden = " 🏆" if c["is_golden"] else ""
    in_zone = " ✅" if c["in_zone"] else ""
    arrow = "📈" if c["signal"] == "BUY" else "📉"

    return (
        f"📐 <b>FVG+Fibonacci Confluence</b> ({symbol}){golden}\n"
        f"{arrow} <b>{c['signal']}</b> — "
        f"Fib <b>{c['fib_level']}</b>{' (Altın Oran)' if c['is_golden'] else ''}{in_zone}\n"
        f"   FVG Bölgesi: {c['fvg_bottom']:.4f} – {c['fvg_top']:.4f}\n"
        f"   Fib Fiyatı : {c['fib_price']:.4f}\n"
        f"   Mesafe     : %{c['distance_pct']:.2f} | Güç: {c['strength']:.2f}\n"
        f"   +{c['score_boost']} puan bonus\n"
    )
