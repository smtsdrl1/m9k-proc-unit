"""
Telegram Bot Script — Processes user commands.

Runs on-demand (via GitHub Actions cron every 5 minutes) or manually.
Uses getUpdates with stored offset to process commands since last run.
No persistent connection required — fully compatible with GitHub Actions.

Supported commands: /help /paper /trades /history /performance /reset /pause /threshold /mode
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db import Database
from src.telegram.bot_commands import TelegramBotHandler
from src.utils.helpers import setup_logging
from src.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("matrix_trader.telegram_bot")


def main():
    setup_logging()
    logger.info("🤖 Telegram Bot starting...")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set — skipping")
        return

    db = Database()
    handler = TelegramBotHandler(
        token=TELEGRAM_TOKEN,
        chat_id=int(TELEGRAM_CHAT_ID),
        db=db,
    )

    try:
        count = handler.process_updates()
        if count > 0:
            logger.info(f"✅ Processed {count} command(s)")
        else:
            logger.info("No new commands")
    except Exception as e:
        logger.error(f"Bot error: {e}")


if __name__ == "__main__":
    main()
