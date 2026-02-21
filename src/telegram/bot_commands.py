"""
Telegram Bot Commands Handler.

Processes user commands received via Telegram messages.
Uses offset-based polling (getUpdates) — no persistent connection needed.
Stores last_update_id in DB to avoid replaying old messages.

Supported commands:
  /paper       — Portfolio summary (balance, PnL, win rate)
  /trades      — Currently open paper trades
  /history     — Last 10 closed paper trades
  /performance — Full performance report (30 days)
  /reset       — Reset paper trading balance (requires confirmation)
  /pause       — Toggle paper trading on/off
  /threshold   — Show adaptive threshold status
  /mode        — Show drawdown guard mode
  /help        — Show command list
"""
import logging
import time
from typing import Optional
import requests

logger = logging.getLogger("matrix_trader.telegram.bot_commands")

_TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramBotHandler:
    """Polls Telegram for new commands and processes them."""

    def __init__(self, token: str, chat_id: int, db):
        self.token = token
        self.chat_id = int(chat_id)
        self.db = db
        self.base_url = _TELEGRAM_API.format(token=token)

    def get_updates(self, offset: int = 0, timeout: int = 5) -> list[dict]:
        """Fetch new updates from Telegram API."""
        try:
            r = requests.get(
                f"{self.base_url}/getUpdates",
                params={"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
                timeout=timeout + 5,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("ok"):
                    return data.get("result", [])
            logger.warning(f"getUpdates error: {r.status_code} {r.text[:100]}")
        except Exception as e:
            logger.warning(f"getUpdates exception: {e}")
        return []

    def send_reply(self, chat_id: int, text: str) -> bool:
        """Send a reply message."""
        try:
            import re
            safe_text = re.sub(r'<(?!/?(b|i|u|s|a|code|pre)\b)', '&lt;', text)
            r = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": chat_id, "text": safe_text, "parse_mode": "HTML"},
                timeout=15,
            )
            if r.status_code == 200:
                return True
            # Retry without HTML on parse errors
            if "can't parse" in r.text.lower():
                r2 = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=15,
                )
                return r2.status_code == 200
            logger.warning(f"sendMessage error: {r.status_code}")
        except Exception as e:
            logger.warning(f"sendMessage exception: {e}")
        return False

    def process_updates(self) -> int:
        """
        Fetch and process all new commands since last run.
        Returns number of commands processed.
        """
        last_update_id = self.db.get_bot_state("last_update_id", default=0)
        offset = last_update_id + 1 if last_update_id else 0

        updates = self.get_updates(offset=offset)
        processed = 0

        for update in updates:
            update_id = update.get("update_id", 0)
            message = update.get("message", {})

            if not message:
                last_update_id = max(last_update_id, update_id)
                continue

            # Only process messages from our configured chat
            msg_chat_id = message.get("chat", {}).get("id", 0)
            text = message.get("text", "").strip()

            if not text.startswith("/"):
                last_update_id = max(last_update_id, update_id)
                continue

            # Extract command (handle /command@botname format)
            command = text.split()[0].split("@")[0].lower()
            args = text.split()[1:] if len(text.split()) > 1 else []

            logger.info(f"Telegram command: {command} from chat_id={msg_chat_id}")

            # Process command
            reply = self._handle_command(command, args, msg_chat_id)
            if reply:
                self.send_reply(msg_chat_id, reply)
                processed += 1

            last_update_id = max(last_update_id, update_id)

        if updates:
            self.db.set_bot_state("last_update_id", last_update_id)

        return processed

    def _handle_command(self, command: str, args: list, chat_id: int) -> Optional[str]:
        """Route command to appropriate handler."""
        # Security: only allow commands from configured chat
        if chat_id != self.chat_id and self.chat_id != 0:
            logger.warning(f"Unauthorized command attempt from chat_id={chat_id}")
            return None

        handlers = {
            "/help":        self._cmd_help,
            "/paper":       self._cmd_paper,
            "/trades":      self._cmd_trades,
            "/history":     self._cmd_history,
            "/performance": self._cmd_performance,
            "/reset":       self._cmd_reset,
            "/pause":       self._cmd_pause,
            "/threshold":   self._cmd_threshold,
            "/mode":        self._cmd_mode,
            "/start":       self._cmd_help,
        }

        handler = handlers.get(command)
        if handler:
            try:
                return handler(args)
            except Exception as e:
                logger.error(f"Command {command} handler error: {e}")
                return f"❌ Komut işlenirken hata: {e}"
        return f"❓ Bilinmeyen komut: {command}\n/help ile komutları görebilirsin."

    # ─── Command Handlers ────────────────────────────────────────

    def _cmd_help(self, args: list) -> str:
        return (
            "🤖 <b>MATRIX TRADER BOT</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📋 <b>Paper Trading</b>\n"
            "  /paper       — Portfolio özeti\n"
            "  /trades      — Açık işlemler\n"
            "  /history     — Son 10 işlem\n"
            "  /performance — 30 günlük rapor\n"
            "  /reset       — Bakiyeyi sıfırla\n"
            "  /pause       — Paper trading durdur/devam\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "⚙️ <b>Sistem</b>\n"
            "  /threshold   — Adaptif eşik durumu\n"
            "  /mode        — Drawdown guard modu\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📡 Sorular: @MatrixTraderBot"
        )

    def _cmd_paper(self, args: list) -> str:
        """Portfolio summary."""
        from src.paper_trading.executor import PaperTradeExecutor
        from src.config import PAPER_TRADING_CAPITAL

        stats = self.db.get_paper_trade_stats(days=365)
        if not stats or stats.get("total_trades", 0) == 0:
            return "📋 Henüz paper trade yok. Sinyal geldiğinde otomatik açılır."

        return PaperTradeExecutor.format_portfolio_stats_message(stats)

    def _cmd_trades(self, args: list) -> str:
        """Show open paper trades."""
        trades = self.db.get_open_paper_trades()
        if not trades:
            return "📊 Şu an açık paper trade yok."

        lines = [f"📊 <b>AÇIK PAPER TRADES ({len(trades)})</b>\n━━━━━━━━━━━━━━━━━━"]
        for t in trades[:8]:
            symbol = t["symbol"]
            direction = t["direction"]
            entry = t["actual_entry_price"]
            is_crypto = t.get("is_crypto", True)
            currency = "$" if is_crypto else "₺"
            conf = t.get("signal_confidence", 0)
            tier = t.get("signal_tier", "")
            sl = t.get("stop_loss", 0)
            t1 = t.get("target1", 0)

            # Duration
            from datetime import datetime
            try:
                entry_dt = datetime.fromisoformat(t["entry_timestamp"])
                dur_min = int((datetime.utcnow() - entry_dt).total_seconds() / 60)
                if dur_min < 60:
                    dur_str = f"{dur_min}dk"
                else:
                    dur_str = f"{dur_min // 60}sa {dur_min % 60}dk"
            except Exception:
                dur_str = "?"

            dir_icon = "📈" if direction in ("BUY", "LONG", "AL") else "📉"
            lines.append(
                f"{dir_icon} <b>{symbol}</b> — {direction}\n"
                f"  Tier: {tier} | Conf: {conf}%\n"
                f"  Giriş: {currency}{entry:.4f}\n"
                f"  SL: {currency}{sl:.4f} → T1: {currency}{t1:.4f}\n"
                f"  Süre: {dur_str}"
            )

        if len(trades) > 8:
            lines.append(f"\n...ve {len(trades) - 8} adet daha")

        return "\n".join(lines)

    def _cmd_history(self, args: list) -> str:
        """Show last 10 closed paper trades."""
        limit = int(args[0]) if args and args[0].isdigit() else 10
        limit = min(limit, 20)

        trades = self.db.get_closed_paper_trades(limit=limit)
        if not trades:
            return "📊 Henüz kapanan paper trade yok."

        lines = [f"📋 <b>SON {len(trades)} PAPER TRADE</b>\n━━━━━━━━━━━━━━━━━━"]
        for t in trades:
            symbol = t["symbol"]
            direction = t["direction"]
            status = t.get("status", "")
            pnl_pct = t.get("pnl_pct", 0)
            pnl_amount = t.get("pnl_amount", 0)
            is_crypto = t.get("is_crypto", True)
            currency = "$" if is_crypto else "₺"

            icon = "✅" if pnl_pct >= 0 else "❌"
            status_short = status.replace("_HIT", "").replace("TRAILING_", "T")
            lines.append(
                f"{icon} {symbol} ({direction}) — {status_short}\n"
                f"   PnL: {pnl_pct:+.2f}% | {pnl_amount:+.2f}{currency}"
            )

        return "\n".join(lines)

    def _cmd_performance(self, args: list) -> str:
        """Full 30-day performance report."""
        stats = self.db.get_paper_trade_stats(days=30)
        if not stats or stats.get("total_trades", 0) == 0:
            return "📊 30 günde paper trade verisi yok."

        from src.paper_trading.executor import PaperTradeExecutor
        return PaperTradeExecutor.format_portfolio_stats_message(stats)

    def _cmd_reset(self, args: list) -> str:
        """Reset paper trading balance — requires 'confirm' argument."""
        if not args or args[0].lower() != "confirm":
            return (
                "⚠️ <b>PAPER TRADING SIFIRLA</b>\n\n"
                "Bu işlem demo bakiyeni sıfırlayacak!\n\n"
                "Onaylamak için: <code>/reset confirm</code>"
            )
        try:
            self.db.reset_paper_trading()
            self.db.set_bot_state("paper_paused", "0")
            return (
                "✅ <b>Paper trading sıfırlandı!</b>\n"
                f"💼 Yeni bakiye: $10,000\n"
                "Açık işlemler kapatıldı. Sistem hazır."
            )
        except Exception as e:
            return f"❌ Sıfırlama hatası: {e}"

    def _cmd_pause(self, args: list) -> str:
        """Toggle paper trading pause."""
        current = self.db.get_bot_state("paper_paused", default="0")
        if current == "1":
            self.db.set_bot_state("paper_paused", "0")
            return "▶️ <b>Paper trading devam ediyor.</b>\nYeni sinyaller takip edilecek."
        else:
            self.db.set_bot_state("paper_paused", "1")
            return (
                "⏸ <b>Paper trading durduruldu.</b>\n"
                "Yeni sinyaller açılmayacak.\n"
                "Devam etmek için: /pause"
            )

    def _cmd_threshold(self, args: list) -> str:
        """Show adaptive threshold status."""
        try:
            from src.signals.adaptive_threshold import get_threshold_status
            ts = get_threshold_status(self.db)

            mode_icons = {"RELAX": "🟢", "TIGHTEN": "🔴", "NORMAL": "⚪"}
            icon = mode_icons.get(ts["mode"], "⚪")

            adj_str = f"{ts['adjustment']:+d}" if ts["adjustment"] != 0 else "0"

            return (
                f"⚙️ <b>ADAPTİF EŞİK DURUMU</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"{icon} Mod: {ts['mode']}\n"
                f"📊 7-gün Win Rate: {ts['win_rate_7d']:.1f}% ({ts['signals_7d']} sinyal)\n"
                f"🔧 Ayarlama: {adj_str} puan\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"🔵 Crypto Eşik: {ts['crypto_threshold']}\n"
                f"🟡 BIST Eşik:   {ts['bist_threshold']}"
            )
        except Exception as e:
            return f"❌ Eşik verisi alınamadı: {e}"

    def _cmd_mode(self, args: list) -> str:
        """Show drawdown guard mode."""
        try:
            from src.paper_trading.drawdown_guard import DrawdownGuard
            guard = DrawdownGuard(self.db)
            info = guard.get_mode()

            emoji = info.get("emoji", "")
            mode = info["mode"]
            dd = info["drawdown_pct"]
            balance = info["balance"]
            mult = info["position_mult"]
            min_tier = info["min_tier"]

            return (
                f"{emoji} <b>DRAWDOWN GUARD: {mode}</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📉 Drawdown: {dd:.1f}%\n"
                f"💼 Bakiye: ${balance:,.2f}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📦 Pozisyon Mult: x{mult:.2f}\n"
                f"🏷 Min Tier: {min_tier}\n"
                f"{'🔴 HALT: Yeni trade açılmıyor!' if mode == 'HALT' else '✅ İşlemler devam ediyor'}"
            )
        except Exception as e:
            return f"❌ Guard verisi alınamadı: {e}"
