"""
Telegram Message Formatter — Beautiful, emoji-rich signal messages.
Smart price formatting to prevent 0.00 display for micro-cap coins.
Includes: partial TP, trailing SL, funding rate, circuit breaker status.
"""
from src.utils.helpers import format_price, format_pct, format_number, calculate_change_pct
from src.config import PARTIAL_TP_ENABLED, PARTIAL_TP_RATIOS, TRAILING_STOP_ENABLED


def format_signal_message(
    symbol: str,
    direction: str,
    tier_name: str,
    confidence: int,
    grade: str,
    indicators: dict,
    risk_mgmt: dict,
    is_bist: bool,
    ai_analysis: dict = None,
    mtf_result: dict = None,
    sentiment: dict = None,
    smart_money: dict = None,
    macro: dict = None,
    reasons: list = None,
    funding_rate: dict = None,
    time_estimates: dict = None,
) -> str:
    """Format a complete signal message for Telegram."""
    is_buy = direction == "BUY"
    currency = "TL" if is_bist else "USDT"
    price = indicators.get("currentPrice", 0)

    # Header
    if is_bist and not is_buy:
        header = "<b>🔴 ÇIKIŞ TAVSİYESİ</b>"
    else:
        icon = "🟢" if is_buy else "🔴"
        action = "AL SİNYALİ" if is_buy else "SAT SİNYALİ (SHORT)"
        header = f"{icon} <b>{action}</b>"

    msg = f"{header}\n"
    msg += f"📊 <b>{symbol}</b> | {tier_name}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # Entry
    msg += f"💰 <b>GİRİŞ:</b> {format_price(price, is_bist)} {currency}\n"
    msg += f"🎯 <b>GÜVEN:</b> {confidence}% (Grade: {grade})\n\n"

    # Stop Loss
    sl = risk_mgmt.get("stop_loss", 0)
    sl_pct = calculate_change_pct(sl, price)
    msg += f"🛡 <b>STOP LOSS:</b> {format_price(sl, is_bist)} {currency} ({format_pct(sl_pct)})\n"
    msg += f"   Risk: {format_price(risk_mgmt.get('risk_amount', 0), is_bist)} {currency}\n\n"

    # Targets with kademeli kar alma + time estimates
    msg += "🎯 <b>HEDEFLER:</b>\n"
    targets = risk_mgmt.get("targets", {})
    partial_tp = risk_mgmt.get("partial_tp", {})
    for tname, tval in targets.items():
        t_pct = calculate_change_pct(tval, price)
        close_info = ""
        if PARTIAL_TP_ENABLED and partial_tp:
            t_key = f"{tname}_close_pct"
            close_pct = partial_tp.get(t_key, 0)
            if close_pct > 0:
                close_info = f" → %{close_pct:.0f} kapat"
        # Time estimate
        time_info = ""
        if time_estimates and tname in time_estimates:
            te = time_estimates[tname]
            label = te.get("label", "")
            if label and label != "—":
                time_info = f" ⏱{label}"
        msg += f"   {tname.upper()}: {format_price(tval, is_bist)} {currency} ({format_pct(t_pct)}){close_info}{time_info}\n"
    msg += f"   R/R: 1:{risk_mgmt.get('reward_risk', 0)}\n"

    # Trailing SL info
    if TRAILING_STOP_ENABLED:
        trailing_sl = risk_mgmt.get("trailing_sl")
        if trailing_sl:
            msg += f"   🔒 Trailing SL: T1 sonrası aktif (ATR bazlı)\n"

    msg += "\n"

    # Position Size
    pos_size = min(risk_mgmt.get("position_size", 0), 100000)
    msg += f"💼 <b>POZİSYON:</b> {pos_size:.1f} adet ({risk_mgmt.get('position_value', 0):.0f} {currency})\n\n"

    # Technical Summary
    rsi = indicators.get("rsi", 0)
    macd_status = "Pozitif 📈" if indicators.get("macd_hist", 0) > 0 else "Negatif 📉"
    msg += f"📐 <b>TEKNİK:</b>\n"
    msg += f"   RSI: {rsi:.1f} | MACD: {macd_status}\n"
    msg += f"   ADX: {indicators.get('adx', 0):.0f} | Hacim: {indicators.get('volume_ratio', 1):.1f}x\n"

    if indicators.get("cross") and indicators["cross"] != "NONE":
        cross_name = "Golden Cross 🌟" if indicators["cross"] == "GOLDEN_CROSS" else "Death Cross 💀"
        msg += f"   ⚡ {cross_name}\n"

    # MTF Confluence
    if mtf_result and mtf_result.get("direction") != "NEUTRAL":
        msg += f"\n🕐 <b>ÇOKLU ZAMAN DİLİMİ:</b>\n"
        msg += f"   {mtf_result.get('recommendation', '')}\n"

    # Smart Money
    if smart_money and smart_money.get("direction") != "NEUTRAL":
        msg += f"\n🐋 <b>AKILLI PARA:</b>\n"
        vol_info = smart_money.get("volume_anomaly", {})
        if vol_info.get("anomaly"):
            msg += f"   {vol_info.get('interpretation', '')}\n"
        ad_info = smart_money.get("ad_pattern", {})
        if ad_info.get("pattern") != "NONE":
            msg += f"   {ad_info.get('description', '')}\n"

    # Sentiment
    if sentiment and sentiment.get("summary"):
        msg += f"\n📰 <b>DUYGU:</b> {sentiment.get('summary', '')}\n"

    # Macro
    if macro and macro.get("summary") and macro["summary"] != "Makro ortam normal":
        msg += f"\n🌍 <b>MAKRO:</b> {macro.get('summary', '')}\n"

    # Funding Rate (crypto only)
    if funding_rate and not is_bist:
        fr_pct = funding_rate.get("funding_rate_pct", 0)
        fr_bias = funding_rate.get("bias", "NEUTRAL")
        extreme = funding_rate.get("extreme", False)
        ann_pct = funding_rate.get("annualized_pct", 0)
        fr_icon = "⚠️" if extreme else "📊"
        msg += f"\n{fr_icon} <b>FUNDING:</b> {fr_pct:+.4f}% ({ann_pct:+.1f}%/yıl) — {fr_bias}\n"
        if extreme:
            msg += f"   ⚡ Aşırı funding oranı!\n"

    # Signal Reasons
    if reasons:
        msg += "\n📋 <b>SEBEPLER:</b>\n"
        for r in reasons[:5]:
            msg += f"   • {r}\n"

    # AI Analysis — Full Institutional Report
    if ai_analysis:
        ai_decision = ai_analysis.get("karar", "")
        ai_comment = ai_analysis.get("yorum", "")
        ai_confidence = ai_analysis.get("guven", "")
        is_fallback = ai_analysis.get("_fallback", False)

        ai_label = "KURAL BAZLI ANALİZ" if is_fallback else "AI KURUMSAL ANALİZ"
        msg += f"\n🤖 <b>{ai_label}:</b>\n"
        msg += f"   Karar: <b>{ai_decision}</b>"
        if ai_confidence:
            msg += f" ({ai_confidence}% güven)"
        msg += "\n"

        if ai_comment:
            msg += f"   {ai_comment[:600]}\n"

        # Technical synthesis
        teknik = ai_analysis.get("teknik_sentez", "")
        if teknik:
            msg += f"\n   📐 <b>Teknik Sentez:</b> {teknik[:250]}\n"

        # Macro impact
        makro = ai_analysis.get("makro_etki", "")
        if makro:
            msg += f"   🌍 <b>Makro Etki:</b> {makro[:200]}\n"

        # Smart money comment
        akilli_para = ai_analysis.get("akilli_para_yorum", "")
        if akilli_para:
            msg += f"   🐋 <b>Akıllı Para:</b> {akilli_para[:200]}\n"

        # Strategy
        strateji = ai_analysis.get("strateji", "")
        if strateji:
            msg += f"   📋 <b>Strateji:</b> {strateji[:250]}\n"

        # AI targets
        ai_targets = ai_analysis.get("hedef_fiyat", {})
        if ai_targets and isinstance(ai_targets, dict):
            parts = []
            for k, v in ai_targets.items():
                if v:
                    parts.append(f"{k}: {v}")
            if parts:
                msg += f"   🎯 <b>AI Hedefler:</b> {' | '.join(parts)}\n"

        # Opportunities
        firsatlar = ai_analysis.get("firsatlar", [])
        if firsatlar:
            msg += "   ✅ " + " | ".join(f[:80] for f in firsatlar[:3]) + "\n"

        # Risks
        risks = ai_analysis.get("riskler", [])
        if risks:
            msg += "   ⚠️ " + " | ".join(r[:80] for r in risks[:3]) + "\n"

        # Important note
        onem = ai_analysis.get("onem_notu", "")
        if onem:
            msg += f"   🔔 <b>{onem[:200]}</b>\n"

        # Time horizon
        zaman = ai_analysis.get("zaman_dilimi", "")
        if zaman:
            msg += f"   ⏰ {zaman}\n"

    msg += f"\n<i>Matrix Trader AI v3.0 Institutional | {confidence}% güven</i>"
    return msg


def format_accuracy_report(stats: dict) -> str:
    """Format periodic accuracy report for Telegram."""
    total = stats.get("total", 0)
    if total == 0:
        return "📊 <b>DOĞRULUK RAPORU</b>\n\nHenüz yeterli veri yok."

    wins = stats.get("wins", 0)
    win_rate = stats.get("win_rate", 0)
    t1 = stats.get("t1_hits", 0)
    t2 = stats.get("t2_hits", 0)
    t3 = stats.get("t3_hits", 0)
    sl = stats.get("sl_hits", 0)
    avg_pnl = stats.get("avg_pnl", 0)

    msg = "📊 <b>DOĞRULUK RAPORU</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    pct = lambda n: f"{n/total*100:.1f}%" if total else "0%"

    msg += f"📈 Toplam Sinyal: <b>{total}</b>\n"
    msg += f"✅ Doğru: <b>{wins}</b> ({win_rate:.1f}%)\n"
    msg += f"❌ Yanlış: <b>{sl}</b> ({pct(sl)})\n\n"

    msg += f"🎯 <b>HEDEF İSABETLERİ:</b>\n"
    msg += f"   T1: {t1} ({pct(t1)})\n"
    msg += f"   T2: {t2} ({pct(t2)})\n"
    msg += f"   T3: {t3} ({pct(t3)})\n"
    msg += f"   SL: {sl} ({pct(sl)})\n\n"

    msg += f"💰 Ort. PnL: <b>{avg_pnl:+.2f}%</b>\n"

    # Duration averages
    for label, key in [("T1", "avg_t1_duration_min"), ("T2", "avg_t2_duration_min"), ("T3", "avg_t3_duration_min")]:
        dur = stats.get(key)
        if dur:
            if dur < 60:
                msg += f"   ⏱ {label} ort: {dur:.0f}dk\n"
            else:
                msg += f"   ⏱ {label} ort: {dur/60:.1f}sa\n"

    # Tier breakdown
    by_tier = stats.get("by_tier", {})
    if by_tier:
        msg += "\n📋 <b>TIER BAZLI:</b>\n"
        for tier, ts in sorted(by_tier.items()):
            t_total = ts.get("total", 0)
            t_wins = ts.get("wins", 0)
            t_wr = f"{t_wins/t_total*100:.0f}%" if t_total else "N/A"
            msg += f"   {tier}: {t_wins}/{t_total} ({t_wr})\n"

    msg += "\n━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "<i>Matrix Trader AI v3.0</i>"
    return msg


def format_analysis_message(
    symbol: str,
    indicators: dict,
    risk_mgmt: dict,
    is_bist: bool,
    signal: dict,
    mtf_result: dict = None,
    fundamental: dict = None,
    ai_analysis: dict = None,
) -> str:
    """Format /analiz command response — comprehensive analysis report."""
    currency = "TL" if is_bist else "USDT"
    price = indicators.get("currentPrice", 0)
    direction = signal.get("direction", "NEUTRAL")

    if direction == "BUY":
        header = "🟢 YÜKSELİŞ EĞİLİMİ"
    elif direction == "SELL":
        header = "🔴 DÜŞÜŞ EĞİLİMİ"
    else:
        header = "⚪ KARARSIZ"

    msg = f"📊 <b>{symbol} ANALİZ RAPORU</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"💰 Fiyat: <b>{format_price(price, is_bist)} {currency}</b>\n"
    msg += f"📈 Trend: {header} | {signal.get('tier_name', 'N/A')}\n\n"

    # Indicators
    msg += "📐 <b>İNDİKATÖRLER:</b>\n"
    rsi = indicators.get("rsi", 50)
    rsi_status = "Aşırı Alım ⚠️" if rsi > 70 else "Aşırı Satım ⚠️" if rsi < 30 else "Normal"
    msg += f"   RSI(14): {rsi:.1f} — {rsi_status}\n"

    macd_hist = indicators.get("macd_hist", 0)
    macd_status = "Yükseliş 📈" if macd_hist > 0 else "Düşüş 📉"
    msg += f"   MACD: {macd_status}"
    if indicators.get("macd_crossover") != "NONE":
        msg += f" ({indicators['macd_crossover']})"
    msg += "\n"

    bb_pctb = indicators.get("bb_pctb", 0.5)
    bb_status = "Üst Banda Yakın" if bb_pctb > 0.8 else "Alt Banda Yakın" if bb_pctb < 0.2 else "Ortada"
    msg += f"   Bollinger: %B={bb_pctb:.2f} — {bb_status}\n"

    msg += f"   StochRSI K: {indicators.get('stoch_k', 50):.0f}\n"
    msg += f"   ADX: {indicators.get('adx', 20):.0f} (DI+: {indicators.get('plus_di', 20):.0f}, DI-: {indicators.get('minus_di', 20):.0f})\n"
    msg += f"   ATR: {format_price(indicators.get('atr', 0), is_bist)}\n"
    msg += f"   Hacim Oranı: {indicators.get('volume_ratio', 1):.1f}x\n"

    # Support/Resistance
    sr = indicators.get("sr", {})
    msg += f"\n📏 <b>DESTEK/DİRENÇ:</b>\n"
    msg += f"   D1: {format_price(sr.get('support1', 0), is_bist)} | D2: {format_price(sr.get('support2', 0), is_bist)}\n"
    msg += f"   R1: {format_price(sr.get('resistance1', 0), is_bist)} | R2: {format_price(sr.get('resistance2', 0), is_bist)}\n"

    # EMAs
    msg += f"\n📊 <b>HAREKETLI ORTALAMALAR:</b>\n"
    msg += f"   EMA9: {format_price(indicators.get('ema9', 0), is_bist)} | EMA21: {format_price(indicators.get('ema21', 0), is_bist)}\n"
    msg += f"   EMA50: {format_price(indicators.get('ema50', 0), is_bist)}\n"

    # MTF
    if mtf_result and mtf_result.get("direction") != "NEUTRAL":
        msg += f"\n🕐 <b>ÇOKLU ZAMAN DİLİMİ:</b>\n"
        msg += f"   {mtf_result.get('recommendation', 'N/A')}\n"
        for tf, analysis in mtf_result.get("timeframes", {}).items():
            tf_icon = "🟢" if analysis["direction"] == "BUY" else "🔴" if analysis["direction"] == "SELL" else "⚪"
            msg += f"   {tf_icon} {tf}: {analysis['direction']} ({analysis['strength']})\n"

    # Fundamental (BIST)
    if fundamental:
        msg += f"\n📋 <b>TEMEL VERİLER:</b>\n"
        msg += f"   F/K: {fundamental.get('pe_ratio', 'N/A')} | PD/DD: {fundamental.get('pb_ratio', 'N/A')}\n"
        msg += f"   ROE: {fundamental.get('roe', 'N/A')}%\n"

    # Risk Management
    if direction != "NEUTRAL":
        msg += f"\n🎯 <b>RİSK YÖNETİMİ:</b>\n"
        msg += f"   SL: {format_price(risk_mgmt.get('stop_loss', 0), is_bist)} {currency}\n"
        targets = risk_mgmt.get("targets", {})
        for tname, tval in targets.items():
            msg += f"   {tname.upper()}: {format_price(tval, is_bist)} {currency}\n"
        msg += f"   R/R: 1:{risk_mgmt.get('reward_risk', 0)}\n"

    # AI
    if ai_analysis and ai_analysis.get("yorum"):
        msg += f"\n🤖 <b>AI KURUMSAL ANALİZ:</b>\n"
        msg += f"   Karar: <b>{ai_analysis.get('karar', 'N/A')}</b>\n"
        msg += f"   {ai_analysis['yorum'][:600]}\n"
        teknik = ai_analysis.get("teknik_sentez", "")
        if teknik:
            msg += f"   📐 {teknik[:250]}\n"
        strateji = ai_analysis.get("strateji", "")
        if strateji:
            msg += f"   📋 Strateji: {strateji[:250]}\n"
        risks = ai_analysis.get("riskler", [])
        if risks:
            msg += "   ⚠️ " + " | ".join(r[:80] for r in risks[:3]) + "\n"

    msg += f"\n<i>Matrix Trader AI v3.0 Institutional</i>"
    return msg


def format_alarm_message(symbol: str, target_price: float, current_price: float,
                         direction: str, is_bist: bool) -> str:
    """Format alarm triggered notification."""
    currency = "TL" if is_bist else "USDT"
    icon = "⬆️" if direction == "above" else "⬇️"
    return (
        f"🔔 <b>ALARM TETİKLENDİ!</b>\n\n"
        f"{icon} <b>{symbol}</b>\n"
        f"Hedef: {format_price(target_price, is_bist)} {currency}\n"
        f"Mevcut: {format_price(current_price, is_bist)} {currency}\n\n"
        f"<i>Matrix Trader AI</i>"
    )


def format_watchlist_message(watchlist: list[dict]) -> str:
    """Format watchlist display."""
    if not watchlist:
        return "📋 <b>Takip Listeniz Boş</b>\n\n/ekle <sembol> ile ekleyebilirsiniz."

    msg = "📋 <b>TAKİP LİSTESİ</b>\n━━━━━━━━━━━━━━━━━━━━━━\n\n"
    for i, item in enumerate(watchlist, 1):
        market = "BIST" if item.get("is_bist") else "Kripto"
        msg += f"{i}. {item['symbol']} ({market})\n"
    msg += f"\n<i>Toplam: {len(watchlist)} sembol</i>"
    return msg
