"""
Interactive Telegram Bot — Long-running bot with commands.
/start, /analiz, /alarm, /backtest, /watchlist, /ekle, /sil, /rapor, /help
"""
import asyncio
import logging
from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)
from src.config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, CRYPTO_SYMBOLS, BIST_100,
    CRYPTO_TIMEFRAMES, BIST_TIMEFRAMES, CAPITAL, RISK_PERCENT,
)
from src.data.crypto_feed import CryptoFeed
from src.data.bist_feed import BistFeed
from src.data.macro_feed import MacroFeed
from src.analysis.technical import calculate_indicators
from src.analysis.multi_timeframe import multi_timeframe_confluence
from src.analysis.smart_money import smart_money_analysis
from src.analysis.sentiment import fetch_crypto_news, fetch_bist_news, SentimentResult
from src.analysis.macro_filter import analyze_macro
from src.signals.detector import detect_signal
from src.signals.risk_manager import calculate_risk
from src.signals.scorer import calculate_confidence
from src.signals.validator import validate_signal
from src.ai.groq_engine import GroqEngine
from src.backtest.engine import BacktestEngine
from src.backtest.reporter import format_backtest_report
from src.visualization.charts import generate_analysis_chart, generate_backtest_chart
from src.telegram.formatter import (
    format_analysis_message, format_alarm_message, format_watchlist_message,
)
from src.database.db import Database
from src.utils.helpers import format_price, setup_logging

logger = logging.getLogger("matrix_trader.telegram.bot")


class MatrixTraderBot:
    """Interactive Telegram bot with trading commands."""

    def __init__(self):
        self.db = Database()
        self.groq = GroqEngine()
        self.crypto_feed = CryptoFeed()
        self.bist_feed = BistFeed()
        self.macro_feed = MacroFeed()

    def _is_crypto(self, symbol: str) -> bool:
        """Determine if symbol is crypto based on format."""
        return "/" in symbol or symbol.upper() + "/USDT" in CRYPTO_SYMBOLS

    def _normalize_symbol(self, symbol: str, is_crypto: bool) -> str:
        """Normalize symbol format."""
        symbol = symbol.upper()
        if is_crypto and "/" not in symbol:
            symbol = f"{symbol}/USDT"
        return symbol

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        msg = """🤖 <b>Matrix Trader AI v1.0</b>
━━━━━━━━━━━━━━━━━━━━━━

Profesyonel AI destekli finansal analiz botu.
BIST 100 + Kripto piyasalarını takip ediyorum.

<b>KOMUTLAR:</b>
/analiz &lt;sembol&gt; — Detaylı teknik analiz
/alarm &lt;sembol&gt; &lt;fiyat&gt; — Fiyat alarmı kur
/backtest &lt;sembol&gt; — Geriye dönük test
/watchlist — Takip listeni gör
/ekle &lt;sembol&gt; — Takip listesine ekle
/sil &lt;sembol&gt; — Takip listesinden çıkar
/rapor — Günlük performans raporu
/help — Yardım

<b>ÖRNEKLER:</b>
/analiz THYAO
/analiz BTC
/alarm THYAO 350
/backtest ETH
/ekle SOL AVAX THYAO

<i>Otomatik sinyal bildirimleri aktif 📡</i>"""
        await update.message.reply_text(msg, parse_mode="HTML")

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        msg = """📖 <b>Matrix Trader AI — Yardım</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>/analiz &lt;sembol&gt;</b>
Detaylı teknik analiz raporu.
Multi-timeframe, RSI, MACD, BB, ADX, S/R seviyeler.
+ AI yorumu + Grafik çıktısı.

Örnek: /analiz THYAO, /analiz BTC, /analiz SOL

<b>/alarm &lt;sembol&gt; &lt;fiyat&gt;</b>
Fiyat belirtilen seviyeye ulaştığında bildirim alırsın.

Örnek: /alarm BTC 100000, /alarm THYAO 350

<b>/backtest &lt;sembol&gt;</b>
Son 1 yılda stratejimiz bu sembole uygulansaydı sonuçlar ne olurdu?

Örnek: /backtest THYAO, /backtest ETH

<b>/ekle &lt;sembol1&gt; &lt;sembol2&gt; ...</b>
Takip listesine sembol ekle (birden fazla olabilir).

<b>/sil &lt;sembol&gt;</b>
Takip listesinden çıkar.

<b>Modüller:</b>
🔬 Teknik Analiz (RSI, MACD, BB, Stoch, ADX, ATR)
🕐 Çoklu Zaman Dilimi (15d, 1S, 4S, 1G)
🐋 Akıllı Para Tespiti (Hacim anomalisi, A/D)
📰 Duygu Analizi (Haberler + Fear/Greed)
🌍 Makro Filtre (DXY, USDTRY, VIX)
🤖 Groq AI Yorum (llama3-70b)
📊 Backtest Motoru
📈 Grafik Çıktısı"""
        await update.message.reply_text(msg, parse_mode="HTML")

    async def analiz(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analiz <symbol> command — Full analysis."""
        if not context.args:
            await update.message.reply_text("❌ Kullanım: /analiz <sembol>\nÖrnek: /analiz THYAO")
            return

        symbol_input = context.args[0].upper()
        is_crypto = self._is_crypto(symbol_input)
        symbol = self._normalize_symbol(symbol_input, is_crypto)

        await update.message.reply_text(f"🔍 {symbol} analiz ediliyor...")

        try:
            # Fetch data
            if is_crypto:
                tf_data_raw = await self.crypto_feed.fetch_multi_timeframe(symbol, CRYPTO_TIMEFRAMES)
            else:
                tf_data_raw = self.bist_feed.fetch_multi_timeframe(symbol_input, BIST_TIMEFRAMES)

            if not tf_data_raw:
                await update.message.reply_text(f"❌ {symbol} için veri bulunamadı.")
                return

            # Use the longest timeframe for primary analysis
            primary_tf = list(tf_data_raw.keys())[-1]
            primary_df = tf_data_raw[primary_tf]
            indicators = calculate_indicators(primary_df)

            if not indicators:
                await update.message.reply_text(f"❌ {symbol} için göstergeler hesaplanamadı.")
                return

            # Multi-timeframe analysis
            tf_indicators = {}
            for tf, df in tf_data_raw.items():
                ind = calculate_indicators(df)
                if ind:
                    tf_indicators[tf] = ind
            mtf_result = multi_timeframe_confluence(tf_indicators)

            # Smart money
            sm_result = smart_money_analysis(primary_df, indicators["atr"])

            # Signal detection
            signal = detect_signal(indicators, mtf_result, sm_result)

            # Risk management
            risk_mgmt = calculate_risk(
                indicators["currentPrice"], indicators["atr"],
                indicators["sr"], signal["direction"],
                is_bist, CAPITAL, RISK_PERCENT,
            )

            # Fundamental (BIST only)
            fundamental = None
            if not is_crypto:
                fundamental = self.bist_feed.fetch_fundamental(symbol_input)

            # AI Analysis (for strong signals)
            ai_analysis = None
            if signal["tier"] <= 3 and self.groq.available:
                ai_analysis = self.groq.get_investment_analysis(
                    symbol, signal["direction"], indicators, risk_mgmt,
                    70, mtf_result, None, sm_result, None, fundamental,
                    is_bist=not is_crypto,
                )

            # Format message
            msg = format_analysis_message(
                symbol, indicators, risk_mgmt, not is_crypto,
                signal, mtf_result, fundamental, ai_analysis,
            )
            await update.message.reply_text(msg, parse_mode="HTML")

            # Generate & send chart
            sr = indicators.get("sr", {})
            chart_path = generate_analysis_chart(
                primary_df, symbol,
                indicators=indicators,
                signal_direction=signal["direction"] if signal["direction"] != "NEUTRAL" else None,
                support_levels=[sr.get("support1", 0), sr.get("support2", 0)],
                resistance_levels=[sr.get("resistance1", 0), sr.get("resistance2", 0)],
            )
            if chart_path:
                with open(chart_path, "rb") as photo:
                    await update.message.reply_photo(photo=photo, caption=f"📊 {symbol} Teknik Grafik")

        except Exception as e:
            logger.error(f"Analiz error for {symbol}: {e}")
            await update.message.reply_text(f"❌ Analiz sırasında hata: {str(e)[:200]}")
        finally:
            if is_crypto:
                await self.crypto_feed.close()
                self.crypto_feed = CryptoFeed()

    async def alarm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alarm <symbol> <price> command."""
        if len(context.args) < 2:
            await update.message.reply_text("❌ Kullanım: /alarm <sembol> <fiyat>\nÖrnek: /alarm BTC 100000")
            return

        symbol_input = context.args[0].upper()
        try:
            target_price = float(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Geçerli bir fiyat girin.")
            return

        is_crypto = self._is_crypto(symbol_input)
        user_id = str(update.effective_user.id)

        # Determine direction (above/below current price)
        direction = "above"  # Default
        alarm_id = self.db.add_alarm(user_id, symbol_input, target_price, direction, not is_crypto)

        currency = "TL" if not is_crypto else "USDT"
        await update.message.reply_text(
            f"✅ <b>Alarm Kuruldu!</b>\n\n"
            f"📊 {symbol_input}\n"
            f"🎯 Hedef: {target_price} {currency}\n"
            f"🆔 Alarm ID: {alarm_id}\n\n"
            f"Fiyat hedefe ulaştığında bildirim alacaksınız.",
            parse_mode="HTML",
        )

    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backtest <symbol> command."""
        if not context.args:
            await update.message.reply_text("❌ Kullanım: /backtest <sembol>\nÖrnek: /backtest THYAO")
            return

        symbol_input = context.args[0].upper()
        is_crypto = self._is_crypto(symbol_input)
        symbol = self._normalize_symbol(symbol_input, is_crypto)

        await update.message.reply_text(f"⏳ {symbol} backtest çalıştırılıyor (1 yıl)...")

        try:
            # Fetch 1 year of daily data
            if is_crypto:
                df = await self.crypto_feed.fetch_ohlcv(symbol, "1d", 365)
            else:
                df = self.bist_feed.fetch_ohlcv(symbol_input, period="1y", interval="1d")

            if df is None or len(df) < 100:
                await update.message.reply_text(f"❌ {symbol} için yeterli veri yok (min 100 gün).")
                return

            # Run backtest
            engine = BacktestEngine(CAPITAL, RISK_PERCENT)
            result = engine.run(df, symbol, not is_crypto, min_confidence=50)

            # Format report
            report = format_backtest_report(result)
            await update.message.reply_text(report, parse_mode="HTML")

            # Generate equity curve chart
            if result.equity_curve and len(result.equity_curve) > 10:
                chart_path = generate_backtest_chart(result.equity_curve, result.trades, symbol)
                if chart_path:
                    with open(chart_path, "rb") as photo:
                        await update.message.reply_photo(photo=photo, caption=f"📈 {symbol} Equity Curve")

        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            await update.message.reply_text(f"❌ Backtest hatası: {str(e)[:200]}")
        finally:
            if is_crypto:
                await self.crypto_feed.close()
                self.crypto_feed = CryptoFeed()

    async def watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /watchlist command."""
        user_id = str(update.effective_user.id)
        items = self.db.get_watchlist(user_id)
        msg = format_watchlist_message(items)
        await update.message.reply_text(msg, parse_mode="HTML")

    async def ekle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ekle <symbol1> <symbol2> ... command."""
        if not context.args:
            await update.message.reply_text("❌ Kullanım: /ekle <sembol1> <sembol2> ...\nÖrnek: /ekle THYAO SOL BTC")
            return

        user_id = str(update.effective_user.id)
        added = []
        for sym in context.args:
            sym = sym.upper()
            is_crypto = self._is_crypto(sym)
            self.db.add_to_watchlist(user_id, sym, not is_crypto)
            added.append(sym)

        await update.message.reply_text(f"✅ Eklendi: {', '.join(added)}")

    async def sil(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sil <symbol> command."""
        if not context.args:
            await update.message.reply_text("❌ Kullanım: /sil <sembol>")
            return

        user_id = str(update.effective_user.id)
        symbol = context.args[0].upper()
        removed = self.db.remove_from_watchlist(user_id, symbol)

        if removed:
            await update.message.reply_text(f"✅ {symbol} takip listesinden çıkarıldı.")
        else:
            await update.message.reply_text(f"❌ {symbol} takip listesinde bulunamadı.")

    async def rapor(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rapor command — Full performance & accuracy report."""
        # Get accuracy stats
        stats = self.db.get_accuracy_stats(days=30)

        msg = "📊 <b>PERFORMANS RAPORU (Son 30 Gün)</b>\n━━━━━━━━━━━━━━━━━━━━━━\n\n"

        if stats.get("total", 0) > 0:
            msg += f"📡 <b>TOPLAM:</b> {stats['total']} sinyal\n"
            msg += f"✅ <b>Kazanç:</b> {stats.get('wins', 0)} | "
            msg += f"❌ <b>Kayıp:</b> {stats.get('sl_hits', 0)}\n"
            msg += f"📈 <b>Win Rate:</b> {stats['win_rate']}%\n"
            msg += f"💰 <b>Ort. PnL:</b> {stats.get('avg_pnl', 0):+.2f}%\n\n"

            msg += "<b>🎯 HEDEF BAŞARI ORANLARI:</b>\n"
            msg += f"   T1: {stats['t1_rate']}% ({stats['t1_hits']} hit)\n"
            msg += f"   T2: {stats['t2_rate']}% ({stats['t2_hits']} hit)\n"
            msg += f"   T3: {stats['t3_rate']}% ({stats['t3_hits']} hit)\n"
            msg += f"   SL: {stats['sl_rate']}% ({stats['sl_hits']} hit)\n\n"

            # Avg durations
            if stats.get("avg_t1_duration_min"):
                msg += "<b>⏱ ORT. SÜRE (Hedefe Ulaşma):</b>\n"
                if stats.get("avg_t1_duration_min"):
                    msg += f"   T1: {self._format_dur(stats['avg_t1_duration_min'])}\n"
                if stats.get("avg_t2_duration_min"):
                    msg += f"   T2: {self._format_dur(stats['avg_t2_duration_min'])}\n"
                if stats.get("avg_t3_duration_min"):
                    msg += f"   T3: {self._format_dur(stats['avg_t3_duration_min'])}\n"
                msg += "\n"

            # By tier
            by_tier = stats.get("by_tier", {})
            if by_tier:
                msg += "<b>📊 TIER BAZINDA:</b>\n"
                for tier, ts in sorted(by_tier.items()):
                    wr = round(ts["wins"] / ts["total"] * 100) if ts["total"] > 0 else 0
                    msg += f"   {tier}: {ts['total']} sinyal, {wr}% win\n"
                msg += "\n"

            # ML model info
            try:
                from src.ml.model import SignalPredictor
                predictor = SignalPredictor(self.db)
                ml_info = predictor.get_model_info()
                if ml_info.get("status") == "ACTIVE":
                    msg += f"🤖 <b>ML MODEL:</b> Aktif\n"
                    msg += f"   Doğruluk: {ml_info['accuracy']:.1f}%\n"
                    msg += f"   Eğitim verisi: {ml_info['total_samples']} sinyal\n"
                    msg += f"   Son eğitim: {ml_info['trained_at'][:16]}\n\n"
                else:
                    msg += "🤖 <b>ML MODEL:</b> Henüz eğitilmedi\n\n"
            except Exception:
                pass

        else:
            msg += "📡 Henüz tamamlanmış sinyal verisi yok.\n\n"

        # Recent signals
        signals = self.db.get_recent_signals(5)
        if signals:
            msg += "<b>📋 SON SİNYALLER:</b>\n"
            for s in signals:
                icon = "🟢" if s["direction"] in ("BUY", "LONG", "AL") else "🔴"
                outcome_icon = {
                    "PENDING": "⏳", "T1_HIT": "🎯", "T2_HIT": "🎯🎯",
                    "T3_HIT": "🎯🎯🎯", "SL_HIT": "❌", "EXPIRED": "⌛",
                }.get(s.get("outcome", "PENDING"), "⏳")
                msg += (
                    f"   {icon} {s['symbol']} {s['direction']} "
                    f"({s['confidence']}%) {outcome_icon} {s.get('outcome', 'PENDING')}\n"
                )

        msg += "\n<i>Matrix Trader AI v2.0 — ML Destekli</i>"
        await update.message.reply_text(msg, parse_mode="HTML")

    @staticmethod
    def _format_dur(minutes):
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

    def run(self):
        """Start the bot with polling."""
        if not TELEGRAM_TOKEN:
            logger.error("TELEGRAM_TOKEN not set!")
            return

        setup_logging()
        app = Application.builder().token(TELEGRAM_TOKEN).build()

        # Register commands
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help_cmd))
        app.add_handler(CommandHandler("analiz", self.analiz))
        app.add_handler(CommandHandler("alarm", self.alarm))
        app.add_handler(CommandHandler("backtest", self.backtest))
        app.add_handler(CommandHandler("watchlist", self.watchlist))
        app.add_handler(CommandHandler("ekle", self.ekle))
        app.add_handler(CommandHandler("sil", self.sil))
        app.add_handler(CommandHandler("rapor", self.rapor))

        logger.info("Matrix Trader Bot starting...")
        print("🤖 Matrix Trader AI Bot başlatıldı. Ctrl+C ile durdurun.")
        app.run_polling(drop_pending_updates=True)
