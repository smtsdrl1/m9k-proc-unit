"""
Macro Monitor — Hourly check of DXY, USDTRY, VIX, Gold.
Sends alert if significant moves detected.
"""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.macro_feed import MacroFeed
from src.analysis.macro_filter import analyze_macro
from src.telegram.sender import TelegramSender
from src.utils.helpers import setup_logging, format_pct, format_price

logger = logging.getLogger("matrix_trader.macro_monitor")


async def main():
    setup_logging()
    logger.info("🌍 Macro Monitor starting...")

    sender = TelegramSender()
    feed = MacroFeed()

    try:
        # Fetch all macro data
        macro_data = feed.fetch_all_current()
        fear_greed = await feed.fetch_fear_greed()

        if not macro_data:
            logger.warning("No macro data available.")
            return

        # Analyze for both crypto and BIST perspectives
        crypto_analysis = analyze_macro(macro_data, fear_greed, is_bist=False)
        bist_analysis = analyze_macro(macro_data, fear_greed, is_bist=True)

        # Build report
        alerts = []
        msg = "🌍 <b>MAKRO GÖSTERGE RAPORU</b>\n━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for name, data in macro_data.items():
            price = data.get("value", 0)
            change = data.get("change_pct", 0)

            if name == "DXY":
                icon = "💵"
                display = f"{price:.2f}"
            elif name == "USDTRY":
                icon = "🇹🇷"
                display = f"{price:.4f}"
            elif name == "VIX":
                icon = "📊"
                display = f"{price:.2f}"
            elif name == "GOLD":
                icon = "🪙"
                display = f"${price:.0f}"
            elif name == "US10Y":
                icon = "📈"
                display = f"{price:.2f}%"
            elif name == "SP500":
                icon = "🇺🇸"
                display = f"{price:.0f}"
            else:
                icon = "📌"
                display = f"{price:.2f}"

            change_icon = "🔺" if change > 0 else "🔻" if change < 0 else "➖"
            msg += f"{icon} <b>{name}:</b> {display} {change_icon} {format_pct(change)}\n"

            # Alert thresholds
            if abs(change) > 1.0:
                alerts.append(f"{name} {format_pct(change)}")

        if fear_greed:
            fg_value = fear_greed.get("value", 50)
            fg_text = fear_greed.get("classification", "Neutral")
            msg += f"\n😱 <b>Fear & Greed:</b> {fg_value} ({fg_text})\n"

        # Crypto implications
        if crypto_analysis.get("alerts"):
            msg += f"\n₿ <b>Kripto Etkisi:</b>\n"
            for alert in crypto_analysis["alerts"]:
                msg += f"   • {alert}\n"

        # BIST implications
        if bist_analysis.get("alerts"):
            msg += f"\n🏛 <b>BIST Etkisi:</b>\n"
            for alert in bist_analysis["alerts"]:
                msg += f"   • {alert}\n"

        msg += f"\n<i>Matrix Trader AI — Makro Monitor</i>"

        # Only send if there are significant moves or it's a scheduled check
        if alerts or fear_greed:
            await sender.send_message(msg)
            logger.info(f"Macro report sent. Alerts: {alerts}")
        else:
            logger.info("No significant macro moves. Skipping notification.")

    except Exception as e:
        logger.error(f"Macro monitor error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
