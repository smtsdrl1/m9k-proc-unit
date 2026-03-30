"""
Post-Trade AI Journal.

After every paper trade is closed, uses Groq LLM to generate a structured
post-trade analysis:
  - What went right/wrong
  - Which indicators were accurate
  - What the system should learn
  - Pattern recognition over time

Stored in the `paper_trade_journal` DB table.
Reports sent to Telegram with weekly digest.
"""
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("matrix_trader.paper_trading.journal")

_POST_TRADE_PROMPT = """Sen bir profesyonel kripto/borsa trader'ı ve sistem analistisisin.
Aşağıdaki tamamlanmış paper trade işlemini analiz et:

Sembol: {symbol}
Yön: {direction}
Tier: {tier}
Confidence: {confidence}%
Sinyal Fiyatı: {signal_price}
Giriş Fiyatı: {entry_price}
Stop-Loss: {stop_loss}
Hedef 1/2/3: {t1}/{t2}/{t3}
Sonuç: {status}
Çıkış Fiyatı: {exit_price}
PnL: {pnl_pct:.2f}%
Süre: {duration} dakika
Max Favorable Move: {mfe:.2f}%
Max Adverse Move: {mae:.2f}%

3 maddede Türkçe yanıtla:
1. Bu işlem neden kazandı/kaybetti? (max 2 cümle)
2. Hangi risk faktörü önemliydi? (max 1 cümle)
3. Sistem için öğrenim: Sonraki benzer kurulumda ne değişmeli? (max 1 cümle)

JSON formatında yanıtla:
{{"neden": "...", "risk": "...", "ogrenim": "...", "skor": 1-10}}
skor: bu işlemin kalitesini 1-10 arası puanla (10=mükemmel kurulum, 1=kötü sinyal)
"""


def generate_journal_entry(trade: dict, groq_engine) -> Optional[dict]:
    """
    Generate AI-powered post-trade journal entry.

    Args:
        trade: paper_trade dict from DB
        groq_engine: GroqEngine instance

    Returns:
        journal dict with AI analysis, or None if failed
    """
    from src.config import JOURNAL_ENABLED
    if not JOURNAL_ENABLED:
        return None

    if not groq_engine or not groq_engine.available:
        # Rule-based fallback analysis
        return _rule_based_analysis(trade)

    try:
        prompt = _POST_TRADE_PROMPT.format(
            symbol=trade.get("symbol", ""),
            direction=trade.get("direction", ""),
            tier=trade.get("signal_tier", ""),
            confidence=trade.get("signal_confidence", 0),
            signal_price=trade.get("signal_entry_price", 0),
            entry_price=trade.get("actual_entry_price", 0),
            stop_loss=trade.get("stop_loss", 0),
            t1=trade.get("target1", 0),
            t2=trade.get("target2", 0),
            t3=trade.get("target3", 0),
            status=trade.get("status", ""),
            exit_price=trade.get("exit_price", 0),
            pnl_pct=trade.get("pnl_pct", 0),
            duration=trade.get("duration_minutes", 0),
            mfe=trade.get("max_favorable_pct", 0),
            mae=trade.get("max_adverse_pct", 0),
        )

        raw = groq_engine._call_groq(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        if not raw:
            return _rule_based_analysis(trade)

        # Parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = {"neden": raw[:200], "risk": "", "ogrenim": "", "skor": 5}

        return {
            "trade_id": trade.get("id"),
            "symbol": trade.get("symbol"),
            "status": trade.get("status"),
            "pnl_pct": trade.get("pnl_pct", 0),
            "analysis": analysis,
            "source": "groq",
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.warning(f"Journal AI analysis failed for trade {trade.get('id')}: {e}")
        return _rule_based_analysis(trade)


def _rule_based_analysis(trade: dict) -> dict:
    """
    Rule-based post-trade analysis when Groq is unavailable.
    Uses trade outcome + MAE/MFE to derive insights.
    """
    status = trade.get("status", "")
    pnl_pct = trade.get("pnl_pct", 0)
    mfe = trade.get("max_favorable_pct", 0)
    mae = trade.get("max_adverse_pct", 0)
    conf = trade.get("signal_confidence", 0)
    tier = trade.get("signal_tier", "")
    duration = trade.get("duration_minutes", 0)

    is_win = pnl_pct > 0
    t1_hit = bool(trade.get("t1_hit_at"))
    t2_hit = bool(trade.get("t2_hit_at"))
    t3_hit = bool(trade.get("t3_hit_at"))

    # Determine 'neden' (why won/lost)
    if "SL_HIT" in status or "TRAILING" in status:
        if mae > mfe * 2:
            neden = f"Fiyat hemen ters yönde hareket etti (MAE={mae:.1f}%). Giriş zamanlaması erken olabilir."
        else:
            neden = f"Stop-loss tetiklendi. MFE={mfe:.1f}% seviyesine ulaşıldı ama geri döndü."
    elif "T3_HIT" in status:
        neden = f"Tüm hedeflere ulaşıldı. Güçlü momentum. Duration: {duration}dk."
    elif "T2_HIT" in status:
        neden = f"T1 ve T2 hedeflerine ulaşıldı. T3 olmadan kapandı."
    elif "T1_HIT" in status:
        neden = f"Sadece T1 hedefine ulaşıldı. Momentum yeterli değildi."
    else:
        neden = f"İşlem tamamlandı. PnL: {pnl_pct:.2f}%."

    # Determine 'risk'
    if mae > 2:
        risk = f"Yüksek adverse excursion ({mae:.1f}%) — stop daha geniş olabilirdi."
    elif conf < 60:
        risk = f"Düşük confidence ({conf}%) — sinyal kalitesi iyileştirilebilir."
    else:
        risk = "Risk yönetimi uygun görünüyor."

    # Determine 'ogrenim'
    if not is_win and conf > 70:
        ogrenim = "Yüksek confidence'lı sinyaller de kaybedebilir — stop mesafesi optimize edilmeli."
    elif is_win and duration < 60:
        ogrenim = "Hızlı hedeflere ulaşıldı — bu tier/saat kombinasyonu verimli."
    elif not t1_hit:
        ogrenim = "Hiçbir hedefe ulaşılmadı — sinyal kalitesi veya giriş zamanlaması gözden geçirilmeli."
    else:
        ogrenim = "Kısmi kar alma stratejisi çalışıyor."

    # Quality score
    skor = 5
    if is_win:
        skor = 7
        if t3_hit:
            skor = 10
        elif t2_hit:
            skor = 8
    else:
        if mae > mfe * 3:
            skor = 2
        elif mfe > abs(pnl_pct) * 2:
            skor = 4  # Potential was there but failed

    return {
        "trade_id": trade.get("id"),
        "symbol": trade.get("symbol"),
        "status": status,
        "pnl_pct": pnl_pct,
        "analysis": {
            "neden": neden,
            "risk": risk,
            "ogrenim": ogrenim,
            "skor": skor,
        },
        "source": "rule_based",
        "created_at": datetime.utcnow().isoformat(),
    }


def format_journal_message(journal: dict) -> str:
    """Format journal entry as Telegram message."""
    if not journal:
        return ""

    status = journal.get("status", "")
    symbol = journal.get("symbol", "")
    pnl = journal.get("pnl_pct", 0)
    analysis = journal.get("analysis", {})
    source = journal.get("source", "rule_based")

    pnl_icon = "✅" if pnl >= 0 else "❌"
    source_icon = "🤖" if source == "groq" else "📐"

    return (
        f"📓 <b>TRADE JOURNAL</b> {source_icon}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{pnl_icon} {symbol} — {status} ({pnl:+.2f}%)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💡 <b>Neden?</b>\n{analysis.get('neden', '')}\n\n"
        f"⚠️ <b>Risk:</b>\n{analysis.get('risk', '')}\n\n"
        f"📚 <b>Öğrenim:</b>\n{analysis.get('ogrenim', '')}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⭐ Kalite Skoru: {analysis.get('skor', 5)}/10"
    )
