"""
Daily Report — End-of-day summary with accuracy metrics.
Sends comprehensive daily digest with signal performance, accuracy stats, and ML model info.
"""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.groq_engine import GroqEngine
from src.data.macro_feed import MacroFeed
from src.telegram.sender import TelegramSender
from src.database.db import Database
from src.ml.model import SignalPredictor
from src.utils.helpers import setup_logging, format_pct, get_istanbul_time

logger = logging.getLogger("matrix_trader.daily_report")


def _fmt_dur(minutes):
    """Format minutes to readable duration."""
    if not minutes:
        return "N/A"
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes}dk"
    h = minutes // 60
    m = minutes % 60
    if h < 24:
        return f"{h}sa {m}dk"
    d = h // 24
    return f"{d}gün {h % 24}sa"


async def main():
    setup_logging()
    logger.info("📊 Daily Report generating...")

    sender = TelegramSender()
    db = Database()
    groq = GroqEngine()

    try:
        today = get_istanbul_time().strftime("%Y-%m-%d")

        # Get today's signals
        signals = db.get_recent_signals(50)
        today_signals = [s for s in signals if s.get("sent_at", "").startswith(today)]

        # Get accuracy stats (30 days and 7 days)
        stats_30d = db.get_accuracy_stats(days=30)
        stats_7d = db.get_accuracy_stats(days=7)

        # Build report
        msg = f"📊 <b>GÜNLÜK RAPOR — {today}</b>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # Today's signal summary
        if today_signals:
            buy_count = sum(1 for s in today_signals if s["direction"] in ("BUY", "LONG", "AL"))
            sell_count = sum(1 for s in today_signals if s["direction"] in ("SELL", "SHORT", "SAT"))
            avg_confidence = sum(s["confidence"] for s in today_signals) / len(today_signals)

            msg += f"📡 <b>BUGÜNKÜ SİNYALLER:</b>\n"
            msg += f"   Toplam: {len(today_signals)} ({buy_count} AL / {sell_count} SAT)\n"
            msg += f"   Ort. Güven: {avg_confidence:.0f}%\n\n"

            # Top signals
            msg += "🏆 <b>EN İYİ SİNYALLER:</b>\n"
            top_signals = sorted(today_signals, key=lambda x: x["confidence"], reverse=True)[:5]
            for s in top_signals:
                icon = "🟢" if s["direction"] in ("BUY", "LONG", "AL") else "🔴"
                outcome = s.get("outcome", "PENDING")
                outcome_icon = {"PENDING": "⏳", "T1_HIT": "🎯", "T2_HIT": "🎯🎯",
                                "T3_HIT": "🎯🎯🎯", "SL_HIT": "❌", "EXPIRED": "⌛"}.get(outcome, "⏳")
                msg += f"   {icon} {s['symbol']} — {s['direction']} ({s['confidence']}%) {outcome_icon}\n"
            msg += "\n"
        else:
            msg += "📡 Bugün sinyal üretilmedi.\n\n"

        # 7-Day Accuracy
        if stats_7d.get("total", 0) > 0:
            msg += "<b>📈 7 GÜNLÜK PERFORMANS:</b>\n"
            msg += f"   Sinyal: {stats_7d['total']} | Win Rate: {stats_7d['win_rate']}%\n"
            msg += f"   T1: {stats_7d['t1_rate']}% | T2: {stats_7d['t2_rate']}% | T3: {stats_7d['t3_rate']}%\n"
            msg += f"   Ort. PnL: {stats_7d.get('avg_pnl', 0):+.2f}%\n\n"

        # 30-Day Accuracy
        if stats_30d.get("total", 0) > 0:
            msg += "<b>📊 30 GÜNLÜK PERFORMANS:</b>\n"
            msg += f"   Sinyal: {stats_30d['total']} | Win Rate: {stats_30d['win_rate']}%\n"
            msg += f"   T1: {stats_30d['t1_rate']}% | T2: {stats_30d['t2_rate']}% | T3: {stats_30d['t3_rate']}%\n"
            msg += f"   Ort. PnL: {stats_30d.get('avg_pnl', 0):+.2f}%\n"

            # Avg target durations
            if stats_30d.get("avg_t1_duration_min"):
                msg += f"\n   ⏱ Ort. Hedef Süresi:\n"
                if stats_30d.get("avg_t1_duration_min"):
                    msg += f"      T1: {_fmt_dur(stats_30d['avg_t1_duration_min'])}\n"
                if stats_30d.get("avg_t2_duration_min"):
                    msg += f"      T2: {_fmt_dur(stats_30d['avg_t2_duration_min'])}\n"
                if stats_30d.get("avg_t3_duration_min"):
                    msg += f"      T3: {_fmt_dur(stats_30d['avg_t3_duration_min'])}\n"
            msg += "\n"

        # ML Model Status
        try:
            predictor = SignalPredictor(db)
            ml_info = predictor.get_model_info()
            if ml_info.get("status") == "ACTIVE":
                msg += f"🤖 <b>ML MODEL:</b>\n"
                msg += f"   Durum: Aktif ✅\n"
                msg += f"   Doğruluk: {ml_info['accuracy']:.1f}%\n"
                msg += f"   Eğitim Verisi: {ml_info['total_samples']} sinyal\n"
                metrics = ml_info.get("metrics", {})
                if metrics.get("top_features"):
                    top3 = metrics["top_features"][:3]
                    msg += f"   En Önemli: {', '.join(f[0] for f in top3)}\n"
                msg += "\n"

            # Auto-retrain if needed
            if predictor.should_retrain():
                logger.info("Auto-retraining ML model...")
                train_result = predictor.train()
                if train_result:
                    msg += f"🔄 <b>ML YENİDEN EĞİTİLDİ:</b>\n"
                    msg += f"   Yeni doğruluk: {train_result.get('cv_accuracy', train_result.get('train_accuracy', 0)):.1f}%\n"
                    msg += f"   Örnek sayısı: {train_result['total_samples']}\n\n"
        except Exception as e:
            logger.warning(f"ML info error: {e}")

        # Macro overview
        try:
            macro_feed = MacroFeed()
            macro_data = macro_feed.fetch_all_current()
            fear_greed = await macro_feed.fetch_fear_greed()

            if macro_data:
                msg += "🌍 <b>MAKRO ÖZET:</b>\n"
                for name, data in macro_data.items():
                    change = data.get("change_pct", 0)
                    icon = "🔺" if change > 0.3 else "🔻" if change < -0.3 else "➖"
                    msg += f"   {name}: {icon} {format_pct(change)}\n"
                msg += "\n"

            if fear_greed:
                msg += f"😱 Fear & Greed: {fear_greed.get('value', 'N/A')} ({fear_greed.get('classification', 'N/A')})\n\n"

        except Exception as e:
            logger.warning(f"Macro data error: {e}")

        # AI daily summary
        if groq.available and today_signals:
            try:
                ai_summary = groq.get_summary_report(today_signals)
                if ai_summary:
                    msg += f"🤖 <b>AI GÜNLÜK YORUM:</b>\n{ai_summary[:500]}\n\n"
            except Exception as e:
                logger.warning(f"AI summary error: {e}")

        # Save daily stats
        db.save_daily_stats(today, len(today_signals),
                            sum(1 for s in today_signals if s.get("is_crypto")),
                            sum(1 for s in today_signals if not s.get("is_crypto")))

        msg += "<i>Matrix Trader AI v2.0 — ML Destekli Günlük Rapor</i>"

        await sender.send_message(msg)
        logger.info("Daily report sent successfully.")

    except Exception as e:
        logger.error(f"Daily report error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
