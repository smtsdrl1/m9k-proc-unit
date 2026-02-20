"""
Telegram Sender — Push notifications (used by GitHub Actions scanner scripts).
Sends text messages to a Telegram chat using the HTTP API via requests.
"""
import re
import logging
import time
from typing import Optional
import requests as _requests
from src.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("matrix_trader.telegram.sender")

_API = "https://api.telegram.org/bot{token}"


class TelegramSender:
    """Send messages to Telegram via HTTP API (sync, no event-loop issues)."""

    def __init__(self, token: str = None, chat_id=None):
        self.token = (token or TELEGRAM_TOKEN or "").strip()
        self.chat_id = chat_id or TELEGRAM_CHAT_ID or 0
        if isinstance(self.chat_id, str):
            self.chat_id = int(self.chat_id) if self.chat_id.strip() else 0
        self.base_url = _API.format(token=self.token)

    @property
    def available(self) -> bool:
        return bool(self.token and self.chat_id)

    # Keep both sync and async interfaces so callers using `await` still work
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        return self._send_sync(text, parse_mode)

    def send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        return self._send_sync(text, parse_mode)

    def _send_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.available:
            logger.warning("Telegram not configured, skipping message")
            print(f"[TELEGRAM-SKIP] {text[:100]}...")
            return False

        try:
            # Sanitise stray < > that break HTML (e.g. "9<21<50")
            safe_text = re.sub(r'<(?!/?(b|i|u|s|a|code|pre)\b)', '&lt;', text)

            chunks = self._split_message(safe_text, 4000)
            for chunk in chunks:
                ok = self._send_chunk(chunk, parse_mode)
                if not ok:
                    return False
                if len(chunks) > 1:
                    time.sleep(0.4)
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _send_chunk(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send one chunk via HTTP POST."""
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
        try:
            r = _requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            # If HTML parse error, retry without parse_mode
            if "can't parse" in r.text.lower() or "bad request" in r.text.lower():
                logger.warning("HTML parse error, retrying as plain text")
                payload.pop("parse_mode", None)
                r2 = _requests.post(url, json=payload, timeout=15)
                return r2.status_code == 200
            logger.error(f"Telegram API error {r.status_code}: {r.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Telegram request error: {e}")
            return False

    async def send_photo(self, photo_path: str, caption: str = None) -> bool:
        """Send a photo (.png chart) to Telegram."""
        if not self.available:
            return False
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, "rb") as photo:
                files = {"photo": photo}
                data = {"chat_id": self.chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                    data["parse_mode"] = "HTML"
                r = _requests.post(url, data=data, files=files, timeout=30)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")
            return False

    async def send_signal_with_chart(self, text: str, chart_path: str = None) -> bool:
        """Send signal message + chart image."""
        text_sent = await self.send_message(text)
        if chart_path:
            time.sleep(0.5)
            await self.send_photo(chart_path)
        return text_sent

    @staticmethod
    def _split_message(text: str, max_len: int = 4000) -> list[str]:
        """Split long messages at line boundaries."""
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            split_pos = text.rfind("\n", 0, max_len)
            if split_pos == -1:
                split_pos = max_len
            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")
        return chunks
