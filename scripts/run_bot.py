"""
Bot Runner — Entry point for long-running Telegram bot.
Run via: python -m scripts.run_bot
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telegram.bot import MatrixTraderBot


def main():
    print("=" * 50)
    print("  🤖 Matrix Trader AI — Telegram Bot")
    print("  v1.0 — Interactive Commands + Notifications")
    print("=" * 50)
    print()

    bot = MatrixTraderBot()
    bot.run()


if __name__ == "__main__":
    main()
