"""
Signal Tracker Script — Checks pending signals against live prices.
Runs every 15 minutes via GitHub Actions.
Records T1/T2/T3 hits, SL hits, and sends Telegram notifications.
Also auto-retrains ML model when enough new data accumulates.
Sends periodic accuracy reports when enough signals are resolved.
"""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.tracker import SignalTracker
from src.telegram.sender import TelegramSender
from src.telegram.formatter import format_accuracy_report
from src.database.db import Database
from src.ml.model import SignalPredictor
from src.utils.helpers import setup_logging
from src.config import PAPER_TRADING_ENABLED

logger = logging.getLogger("matrix_trader.track_signals")


async def main():
    setup_logging()
    logger.info("🔍 Signal Tracker starting...")

    db = Database()
    tracker = SignalTracker(db=db)
    sender = TelegramSender()

    try:
        # 0. Expire old signals (>72h)
        db.expire_old_signals(max_age_hours=72)

        # 1. Track all pending signals against live prices
        events = tracker.track_all_pending()

        if events:
            logger.info(f"📊 {len(events)} event(s) detected")

            # Send notifications for each event
            for event in events:
                try:
                    msg = tracker.format_event_message(event)
                    await sender.send_message(msg)
                    logger.info(f"📨 Notification sent: {event['type']} {event['symbol']}")
                except Exception as e:
                    logger.error(f"Failed to send notification: {e}")
        else:
            logger.info("No events — all signals still pending or no open signals")

        # 2. Daily accuracy report — throttled to once per day (15:30-15:45 UTC window)
        # Without throttling this would spam every 15 minutes when resolved signals exist
        try:
            from datetime import datetime as _dt
            _now = _dt.utcnow()
            _in_report_window = (
                _now.weekday() < 5           # Mon-Fri only
                and _now.hour == 15          # 15:xx UTC = ~18:xx Istanbul
                and 30 <= _now.minute < 45   # 15-minute window
            )
            if _in_report_window:
                stats = db.get_accuracy_stats(30)
                total_resolved = stats.get("total", 0)
                if total_resolved >= 5:
                    report_msg = format_accuracy_report(stats)
                    await sender.send_message(report_msg)
                    logger.info(f"📊 Accuracy report sent: {total_resolved} signals, {stats.get('win_rate', 0)}% win rate")
        except Exception as e:
            logger.warning(f"Accuracy report error: {e}")

        # 3. Auto-retrain ML model if enough new data
        try:
            predictor = SignalPredictor(db)
            if predictor.should_retrain():
                logger.info("🤖 ML model retraining triggered...")
                metrics = predictor.train()
                if metrics:
                    accuracy = metrics.get("cv_accuracy") or metrics.get("train_accuracy", 0)
                    msg = (
                        f"🤖 <b>ML MODEL GÜNCELLENDİ</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"📈 Doğruluk: {accuracy:.1f}%\n"
                        f"📊 Örnek: {metrics['total_samples']} "
                        f"({metrics['win_samples']}W/{metrics['loss_samples']}L)\n"
                        f"🏆 En Önemli: {', '.join(f[0] for f in metrics.get('top_features', [])[:3])}\n"
                        f"━━━━━━━━━━━━━━━━━━"
                    )
                    await sender.send_message(msg)
                    logger.info(f"ML model retrained — accuracy: {accuracy:.1f}%")
        except Exception as e:
            logger.warning(f"ML retrain check error: {e}")

        # 4. Paper trading — check open paper trades against live prices
        if PAPER_TRADING_ENABLED:
            try:
                from src.paper_trading.executor import PaperTradeExecutor
                executor = PaperTradeExecutor(db)

                paper_events = executor.check_open_trades()
                for event in paper_events:
                    try:
                        msg = executor.format_trade_event_message(event)
                        await sender.send_message(msg)
                        logger.info(f"📋 Paper trade event: {event['type']} {event['symbol']}")
                    except Exception as e:
                        logger.error(f"Paper trade notification error: {e}")

                # Daily paper trading stats (15:30-15:45 UTC window)
                try:
                    from datetime import datetime as _dt
                    _now = _dt.utcnow()
                    _in_report_window = (
                        _now.hour == 15
                        and 30 <= _now.minute < 45
                    )
                    if _in_report_window:
                        paper_stats = db.get_paper_trade_stats(30)
                        if paper_stats.get("total_trades", 0) >= 1:
                            stats_msg = executor.format_portfolio_stats_message(paper_stats)
                            await sender.send_message(stats_msg)
                            logger.info(
                                f"📊 Paper trading report sent: "
                                f"{paper_stats['total_trades']} trades, "
                                f"win_rate={paper_stats['win_rate']}%"
                            )
                except Exception as e:
                    logger.warning(f"Paper trading report error: {e}")

            except Exception as e:
                logger.error(f"Paper trading check error: {e}")

        # 5. Summary log
        pending = db.get_pending_signals()
        closed = db.get_closed_signals(100)
        stats = db.get_accuracy_stats(30)
        logger.info(
            f"📊 Status: {len(pending)} pending, {len(closed)} closed, "
            f"win_rate={stats.get('win_rate', 0)}%"
        )

    except Exception as e:
        logger.error(f"Signal tracker error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
