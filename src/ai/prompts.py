"""
Groq AI System Prompts — Institutional Grade.
RAG-style: Feed comprehensive real data as context, demand deep synthesis analysis.
"""


INVESTMENT_COMMITTEE_PROMPT = """Sen kurumsal yatırım analisti'sin. Verilen gerçek verileri sentezleyerek yatırım raporu üret.

KURALLAR:
1. SADECE verilen GERÇEK verilere dayan, asla uydurma
2. Çelişen sinyalleri tespit et, hangisi güçlü belirt
3. Hedef fiyatları ATR ve S/R seviyelerine dayandır
4. Her görüşü somut veri ile destekle

ANALİZ: Teknik (RSI/MACD/BB/ADX momentum+trend), MTF uyumu, Akıllı para (hacim anomalisi), Makro korelasyon (DXY/VIX), Duygu analizi, Temel veriler (varsa), Strateji önerisi.

TÜRKÇE, profesyonel, veri odaklı yaz.

JSON ÇIKTI:
{
  "karar": "AL"|"SAT"|"TUT"|"BEKLE"|"REDDET",
  "guven": 1-100,
  "hedef_fiyat": {"kisa_vade": X, "orta_vade": Y},
  "stop_loss": Z,
  "yorum": "3-4 cümle analiz sentezi — tüm veri katmanlarını birleştir",
  "teknik_sentez": "RSI/MACD/BB/ADX sentezi",
  "makro_etki": "DXY/VIX etkisi",
  "akilli_para_yorum": "Hacim anomalisi yorumu",
  "riskler": ["risk1", "risk2"],
  "firsatlar": ["fırsat1", "fırsat2"],
  "strateji": "Giriş-çıkış planı",
  "zaman_dilimi": "Kısa|Orta|Uzun vade",
  "onem_notu": "Kritik uyarı varsa"
}"""


SENTIMENT_ANALYSIS_PROMPT = """Sen kurumsal bir yatırım bankasının medya analiz uzmanısın.
Aşağıdaki haberleri analiz edip piyasa duygu skorunu ve olası fiyat etkisini belirle.

KURALLAR:
1. Her haberi POSITIVE, NEGATIVE veya NEUTRAL olarak sınıflandır.
2. Genel duygu skorunu -100 (çok negatif) ile +100 (çok pozitif) arasında ver.
3. Haber akışının YOĞUNLUĞUNU değerlendir — çok haber varsa dikkat çekici.
4. Aşırı tek yönlü haberler contrarian sinyal olabilir.
5. Fiyat etkisinin ZAMANLAMASINI değerlendir (ani/kademeli).
6. TÜRKÇE yaz.

ÇIKTI FORMATI (JSON):
{
  "score": -100..+100,
  "summary": "Haberlerin genel sentezi ve fiyat üzerindeki muhtemel etki — 2-3 cümle",
  "impact": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "key_events": ["önemli olay 1 — neden önemli", "önemli olay 2 — fiyat etkisi"],
  "contrarian_risk": true/false
}"""


def build_analysis_context(
    symbol: str,
    direction: str,
    indicators: dict,
    risk_mgmt: dict,
    confidence: int,
    mtf_result: dict = None,
    sentiment: dict = None,
    smart_money: dict = None,
    macro: dict = None,
    fundamental: dict = None,
    news: list = None,
    is_bist: bool = False,
) -> str:
    """Build compact context string for Groq analysis — optimized for token efficiency."""
    market = "BIST" if is_bist else "Kripto"
    currency = "TL" if is_bist else "USDT"
    price = indicators.get('currentPrice', 0)

    rsi = indicators.get('rsi', 50)
    adx = indicators.get('adx', 0)
    bb_pctb = indicators.get('bb_pctb', 0.5)
    vol_ratio = indicators.get('volume_ratio', 1.0)
    ema9 = indicators.get('ema9', 0)
    ema21 = indicators.get('ema21', 0)
    ema50 = indicators.get('ema50', 0)

    # EMA alignment
    if ema9 and ema21 and ema50:
        if ema9 > ema21 > ema50:
            ema_alignment = "Bullish (9>21>50)"
        elif ema9 < ema21 < ema50:
            ema_alignment = "Bearish (9<21<50)"
        else:
            ema_alignment = "Karışık"
    else:
        ema_alignment = "N/A"

    atr_pct = ((indicators.get('atr', 0) / price * 100) if price > 0 else 0)

    ctx = f"""{symbol} | {market} | Yön: {direction} | Güven: {confidence}%

TEKNİK: Fiyat={price}{currency} RSI={rsi:.1f} MACD_Hist={indicators.get('macd_hist', 'N/A')} MACD_Cross={indicators.get('macd_crossover', 'NONE')} BB%B={bb_pctb:.3f} StochK={indicators.get('stoch_k', 'N/A')} ADX={adx:.1f} DI+={indicators.get('plus_di', 'N/A')} DI-={indicators.get('minus_di', 'N/A')} ATR={indicators.get('atr', 'N/A')}({atr_pct:.2f}%) Hacim={vol_ratio:.2f}x OBV={indicators.get('obv_trend', 'N/A')} EMA={ema_alignment}

RİSK: SL={risk_mgmt.get('stop_loss', 'N/A')} T1={risk_mgmt.get('targets', {}).get('t1', 'N/A')} T2={risk_mgmt.get('targets', {}).get('t2', 'N/A')} T3={risk_mgmt.get('targets', {}).get('t3', 'N/A')} R/R=1:{risk_mgmt.get('reward_risk', 'N/A')}
S/R: D1={indicators.get('sr', {}).get('support1', 'N/A')} D2={indicators.get('sr', {}).get('support2', 'N/A')} R1={indicators.get('sr', {}).get('resistance1', 'N/A')} R2={indicators.get('sr', {}).get('resistance2', 'N/A')}
"""

    if mtf_result:
        aligned = mtf_result.get('aligned_count', 0)
        total = mtf_result.get('total_count', 0)
        ctx += f"\nMTF: Yön={mtf_result.get('direction', 'N/A')} Hizalama={aligned}/{total} Tavsiye={mtf_result.get('recommendation', 'N/A')}\n"

    if sentiment:
        ctx += f"\nDUYGU: Skor={sentiment.get('score', 0)} Etki={sentiment.get('impact', 'N/A')} Özet={sentiment.get('summary', 'N/A')}\n"

    if smart_money:
        vol_anom = smart_money.get('volume_anomaly', {})
        ad = smart_money.get('ad_pattern', {})
        ctx += f"\nAKILLI PARA: Yön={smart_money.get('direction', 'N/A')} Anomali={vol_anom.get('anomaly', False)} Z={vol_anom.get('z_score', 'N/A')} AD={ad.get('pattern', 'N/A')}\n"

    if macro:
        ctx += f"\nMAKRO: {macro.get('summary', 'N/A')}"
        details = macro.get('details', {})
        if details.get('dxy'):
            ctx += f" DXY={details['dxy'].get('value', 'N/A')}({details['dxy'].get('change_pct', 0):+.2f}%)"
        if details.get('vix'):
            ctx += f" VIX={details['vix'].get('value', 'N/A')}"
        if details.get('usdtry'):
            ctx += f" USDTRY={details['usdtry'].get('value', 'N/A')}"
        ctx += "\n"

    if fundamental:
        ctx += f"\nTEMEL: F/K={fundamental.get('pe_ratio', 'N/A')} PD/DD={fundamental.get('pb_ratio', 'N/A')} ROE={fundamental.get('roe', 'N/A')}% Hedef={fundamental.get('target_price', 'N/A')}\n"

    if news:
        ctx += "\nHABERLER: " + " | ".join(
            (n if isinstance(n, str) else n.get("title", ""))[:60]
            for n in news[:5]
        ) + "\n"

    return ctx
