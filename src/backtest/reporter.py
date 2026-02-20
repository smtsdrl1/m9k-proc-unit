"""
Backtest Result Reporter — Format results for Telegram.
"""
from src.backtest.engine import BacktestResult
from src.utils.helpers import format_pct


def format_backtest_report(result: BacktestResult) -> str:
    """Format backtest results for Telegram message."""
    if result.total_trades == 0:
        return f"📊 <b>{result.symbol} Backtest</b>\n\n⚠️ Bu dönemde sinyal üretilmedi."

    # Emoji based on performance
    if result.total_return_pct > 20:
        perf_emoji = "🚀"
    elif result.total_return_pct > 5:
        perf_emoji = "📈"
    elif result.total_return_pct > 0:
        perf_emoji = "➡️"
    elif result.total_return_pct > -10:
        perf_emoji = "📉"
    else:
        perf_emoji = "💥"

    # Win rate emoji
    if result.win_rate >= 60:
        wr_emoji = "✅"
    elif result.win_rate >= 45:
        wr_emoji = "⚠️"
    else:
        wr_emoji = "❌"

    msg = f"""{perf_emoji} <b>{result.symbol} BACKTEST RAPORU</b>
━━━━━━━━━━━━━━━━━━━━━━

📊 <b>GENEL PERFORMANS:</b>
• Toplam Getiri: <b>{format_pct(result.total_return_pct)}</b>
• İşlem Sayısı: {result.total_trades}
• {wr_emoji} Başarı Oranı: {result.win_rate:.1f}%
• Kazanan: {result.winning_trades} | Kaybeden: {result.losing_trades}

💰 <b>KÂRLAMA DETAYI:</b>
• Ort. Kazanç: {format_pct(result.avg_profit_pct)}
• Ort. Kayıp: {format_pct(result.avg_loss_pct)}
• En İyi: {format_pct(result.best_trade_pct)}
• En Kötü: {format_pct(result.worst_trade_pct)}

📐 <b>RİSK METRİKLERİ:</b>
• Max Drawdown: {format_pct(result.max_drawdown_pct)}
• Sharpe Ratio: {result.sharpe_ratio:.2f}
• Profit Factor: {result.profit_factor:.2f}
• Ort. Pozisyon Süresi: {result.avg_holding_bars:.0f} bar

"""

    # Verdict
    if result.total_return_pct > 10 and result.win_rate > 50 and result.sharpe_ratio > 0.5:
        msg += "✅ <b>KARAR:</b> Strateji bu dönemde başarılı"
    elif result.total_return_pct > 0 and result.win_rate > 40:
        msg += "⚠️ <b>KARAR:</b> Strateji marjinal — iyileştirme gerekebilir"
    else:
        msg += "❌ <b>KARAR:</b> Strateji bu dönemde başarısız"

    msg += f"\n\n<i>Dönem: {result.period}</i>"
    return msg
