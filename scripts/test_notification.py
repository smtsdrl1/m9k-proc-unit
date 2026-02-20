"""Test notification — sends sample signal templates to Telegram."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telegram.sender import TelegramSender
from src.telegram.formatter import format_signal_message


async def main():
    sender = TelegramSender()

    if not sender.available:
        print("❌ TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID ayarlanmamış!")
        return

    # ─── 1. Kripto LONG Sinyal Örneği ────────────────────
    crypto_msg = format_signal_message(
        symbol="BTC/USDT",
        direction="BUY",
        tier_name="🔥 EXTREME",
        confidence=87,
        grade="A",
        indicators={
            "currentPrice": 98450.25,
            "rsi": 42.3,
            "macd_hist": 125.5,
            "adx": 38,
            "volume_ratio": 2.4,
            "cross": "GOLDEN_CROSS",
        },
        risk_mgmt={
            "stop_loss": 95200.00,
            "risk_amount": 3250.25,
            "targets": {"t1": 101500.00, "t2": 105800.00, "t3": 112000.00},
            "reward_risk": 4.2,
            "position_size": 0.102,
            "position_value": 10041.65,
        },
        is_bist=False,
        ai_analysis={
            "karar": "GÜÇLÜ AL",
            "guven": 85,
            "yorum": "Bitcoin güçlü destek seviyesinden sıçrama yaptı. Golden Cross oluşumu ve artan hacim yükseliş trendinin devamına işaret ediyor. Makro ortam olumlu, DXY zayıflıyor.",
            "riskler": ["98K direnç kırılamazsa geri çekilme", "Fed toplantısı yaklaşıyor"],
        },
        mtf_result={
            "direction": "BUY",
            "recommendation": "✅ 4/4 zaman dilimi YÜKSELİŞ yönünde — Güçlü uyum",
        },
        sentiment={"summary": "Olumlu — Kurumsal alım haberleri hakim"},
        smart_money={
            "direction": "BUY",
            "volume_anomaly": {"anomaly": True, "interpretation": "Hacim 2.4x ortalamanın üstünde — Balina aktivitesi"},
            "ad_pattern": {"pattern": "ACCUMULATION", "description": "Fiyat düşerken A/D yükseldi — Birikim"},
        },
        macro={"summary": "DXY düşüşte, VIX normal — Risk-on ortam"},
        reasons=[
            "RSI 42 → Aşırı satım bölgesinden çıkış",
            "MACD pozitif crossover",
            "Golden Cross (EMA50 > EMA200)",
            "Hacim 2.4x ortalama üstü",
            "4/4 zaman dilimi uyumlu",
        ],
    )

    print("📤 Kripto sinyal gönderiliyor...")
    await sender.send_message(crypto_msg)
    await asyncio.sleep(1)

    # ─── 2. BIST AL Sinyal Örneği ────────────────────────
    bist_msg = format_signal_message(
        symbol="THYAO",
        direction="BUY",
        tier_name="💪 STRONG",
        confidence=78,
        grade="B",
        indicators={
            "currentPrice": 342.50,
            "rsi": 35.2,
            "macd_hist": 1.8,
            "adx": 42,
            "volume_ratio": 1.9,
            "cross": "NONE",
        },
        risk_mgmt={
            "stop_loss": 330.00,
            "risk_amount": 12.50,
            "targets": {"t1": 355.00, "t2": 370.00, "t3": 390.00},
            "reward_risk": 3.8,
            "position_size": 16,
            "position_value": 5480.00,
        },
        is_bist=True,
        ai_analysis={
            "karar": "AL",
            "guven": 75,
            "yorum": "THYAO güçlü teknik görünüm sergilemekte. Yolcu taşıma verileri olumlu, dolar bazlı gelirler TL zayıflığından faydalanıyor. 355 TL ilk direnç seviyesi.",
            "riskler": ["Jet yakıt fiyatlarında artış riski", "Küresel resesyon endişesi"],
        },
        mtf_result={
            "direction": "BUY",
            "recommendation": "✅ 3/3 zaman dilimi YÜKSELİŞ — BIST uyumu güçlü",
        },
        sentiment={"summary": "Nötr-Olumlu — Yaz sezonu beklentisi"},
        smart_money=None,
        macro={"summary": "USDTRY yükselişte — İhracatçı avantajlı"},
        reasons=[
            "RSI 35 → Aşırı satım",
            "MACD pozitif dönüş",
            "ADX 42 → Güçlü trend",
            "Hacim 1.9x ortalama üstü",
        ],
    )

    print("📤 BIST sinyal gönderiliyor...")
    await sender.send_message(bist_msg)
    await asyncio.sleep(1)

    # ─── 3. Kripto SHORT Sinyal Örneği ───────────────────
    short_msg = format_signal_message(
        symbol="SOL/USDT",
        direction="SELL",
        tier_name="📊 MODERATE",
        confidence=65,
        grade="C",
        indicators={
            "currentPrice": 187.42,
            "rsi": 78.5,
            "macd_hist": -2.3,
            "adx": 28,
            "volume_ratio": 1.3,
            "cross": "NONE",
        },
        risk_mgmt={
            "stop_loss": 195.80,
            "risk_amount": 8.38,
            "targets": {"t1": 178.00, "t2": 170.00, "t3": 158.00},
            "reward_risk": 3.5,
            "position_size": 23.9,
            "position_value": 4479.50,
        },
        is_bist=False,
        ai_analysis={
            "karar": "SAT",
            "guven": 62,
            "yorum": "SOL aşırı alım bölgesinde. MACD negatif crossover ve zayıflayan hacim düzeltme sinyali veriyor. 178 USDT ilk destek.",
            "riskler": ["Ani ecosystem haberleri yukarı kırabilir"],
        },
        mtf_result={
            "direction": "SELL",
            "recommendation": "⚠️ 3/4 zaman dilimi DÜŞÜŞ — Kısa vadeli düzeltme beklentisi",
        },
        sentiment=None,
        smart_money=None,
        macro=None,
        reasons=[
            "RSI 78.5 → Aşırı alım",
            "MACD negatif crossover",
            "Bollinger üst bandından dönüş",
        ],
    )

    print("📤 SHORT sinyal gönderiliyor...")
    await sender.send_message(short_msg)
    await asyncio.sleep(1)

    # ─── 4. Sistem Bildirim Mesajı ───────────────────────
    system_msg = (
        "🤖 <b>Matrix Trader AI v1.0 — Sistem Testi</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✅ Telegram bağlantısı başarılı\n"
        "✅ Mesaj formatı çalışıyor\n"
        "✅ HTML parse modu aktif\n\n"
        "📡 <b>Aktif Modüller:</b>\n"
        "   🔬 Teknik Analiz (RSI, MACD, BB, ADX, ATR)\n"
        "   🕐 Çoklu Zaman Dilimi (15d, 1S, 4S, 1G)\n"
        "   🐋 Akıllı Para Tespiti\n"
        "   📰 Duygu Analizi + Fear/Greed\n"
        "   🌍 Makro Filtre (DXY, USDTRY, VIX)\n"
        "   🤖 Groq AI Yatırım Komitesi\n"
        "   📊 Backtest Motoru\n"
        "   📈 Grafik Çıktısı (mplfinance)\n\n"
        "📊 <b>Tarama Kapsamı:</b>\n"
        "   ₿ 100 Kripto (Binance)\n"
        "   🏛 88 BIST Hisse\n\n"
        "⏰ <b>Otomatik Tarama:</b>\n"
        "   Kripto: Her 15 dakika\n"
        "   BIST: Her 30 dakika (10:00-18:00)\n"
        "   Makro: Her saat\n"
        "   Günlük Rapor: 18:30\n\n"
        "<i>Bot hazır. /start ile komutları görün.</i>"
    )

    print("📤 Sistem testi gönderiliyor...")
    await sender.send_message(system_msg)

    print("✅ 4 test mesajı gönderildi!")


if __name__ == "__main__":
    asyncio.run(main())
