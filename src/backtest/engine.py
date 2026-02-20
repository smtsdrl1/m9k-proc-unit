"""
Backtesting Engine — Replay historical data through our signal strategy.
"/test THYAO" → "Bu strateji geçen yıl %45 kazandırırdı."
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from src.analysis.technical import calculate_indicators
from src.signals.detector import detect_signal
from src.signals.risk_manager import calculate_risk
from src.utils.helpers import safe_float, safe_positive

logger = logging.getLogger("matrix_trader.backtest.engine")


@dataclass
class Trade:
    """A single backtest trade."""
    entry_idx: int
    entry_price: float
    direction: str
    stop_loss: float
    target1: float
    target2: float
    target3: float
    confidence: int
    tier: int
    exit_idx: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    is_win: bool = False


@dataclass
class BacktestResult:
    """Complete backtest results."""
    symbol: str
    period: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    avg_profit_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_holding_bars: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "period": self.period,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 1),
            "total_return_pct": round(self.total_return_pct, 2),
            "avg_profit_pct": round(self.avg_profit_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "best_trade_pct": round(self.best_trade_pct, 2),
            "worst_trade_pct": round(self.worst_trade_pct, 2),
            "avg_holding_bars": round(self.avg_holding_bars, 1),
        }


class BacktestEngine:
    """Run strategy backtests on historical OHLCV data."""

    def __init__(self, initial_capital: float = 10000, risk_pct: float = 2.0):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        is_bist: bool = False,
        min_confidence: int = 55,
        warmup: int = 60,
    ) -> BacktestResult:
        """
        Backtest our signal strategy on historical data.

        Args:
            df: OHLCV DataFrame (daily recommended)
            symbol: Symbol name
            is_bist: True for BIST, False for crypto
            min_confidence: Minimum confidence to take trade
            warmup: Skip first N bars for indicator stabilization
        """
        result = BacktestResult(symbol=symbol, period=f"{len(df)} bars")

        if df is None or len(df) < warmup + 30:
            logger.warning(f"Insufficient data for backtest: {len(df) if df is not None else 0}")
            return result

        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        current_trade: Optional[Trade] = None
        max_equity = capital

        for i in range(warmup, len(df)):
            # Extract slice for indicator calculation (need lookback data)
            lookback_start = max(0, i - 200)
            slice_df = df.iloc[lookback_start:i + 1].copy()

            current_bar = df.iloc[i]
            close = safe_float(current_bar["close"])
            high = safe_float(current_bar["high"])
            low = safe_float(current_bar["low"])

            # ─── Check existing trade ────────────────────────
            if current_trade:
                exit_price = None
                exit_reason = ""

                if current_trade.direction == "BUY":
                    if low <= current_trade.stop_loss:
                        exit_price = current_trade.stop_loss
                        exit_reason = "STOP_LOSS"
                    elif high >= current_trade.target3:
                        exit_price = current_trade.target3
                        exit_reason = "TARGET_3"
                    elif high >= current_trade.target2:
                        exit_price = current_trade.target2
                        exit_reason = "TARGET_2"
                    elif high >= current_trade.target1:
                        exit_price = current_trade.target1
                        exit_reason = "TARGET_1"
                else:  # SELL
                    if high >= current_trade.stop_loss:
                        exit_price = current_trade.stop_loss
                        exit_reason = "STOP_LOSS"
                    elif low <= current_trade.target3:
                        exit_price = current_trade.target3
                        exit_reason = "TARGET_3"
                    elif low <= current_trade.target2:
                        exit_price = current_trade.target2
                        exit_reason = "TARGET_2"
                    elif low <= current_trade.target1:
                        exit_price = current_trade.target1
                        exit_reason = "TARGET_1"

                if exit_price:
                    if current_trade.direction == "BUY":
                        pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price * 100
                    else:
                        pnl_pct = (current_trade.entry_price - exit_price) / current_trade.entry_price * 100

                    current_trade.exit_idx = i
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = exit_reason
                    current_trade.pnl_pct = round(pnl_pct, 2)
                    current_trade.is_win = pnl_pct > 0

                    # Update capital
                    capital *= (1 + pnl_pct / 100)
                    trades.append(current_trade)
                    current_trade = None

            # ─── Check for new signal ────────────────────────
            elif current_trade is None:
                indicators = calculate_indicators(slice_df)
                if not indicators:
                    equity_curve.append(capital)
                    continue

                signal = detect_signal(indicators)
                if signal["direction"] == "NEUTRAL" or signal["tier"] > 4:
                    equity_curve.append(capital)
                    continue

                # Simple confidence estimation (no MTF/sentiment in backtest for speed)
                confidence = _quick_confidence(indicators, signal["direction"])
                if confidence < min_confidence:
                    equity_curve.append(capital)
                    continue

                # Calculate risk management
                risk_mgmt = calculate_risk(
                    close, indicators["atr"], indicators["sr"],
                    signal["direction"], is_bist, capital, self.risk_pct
                )

                current_trade = Trade(
                    entry_idx=i,
                    entry_price=close,
                    direction=signal["direction"],
                    stop_loss=risk_mgmt["stop_loss"],
                    target1=risk_mgmt["targets"]["t1"],
                    target2=risk_mgmt["targets"]["t2"],
                    target3=risk_mgmt["targets"]["t3"],
                    confidence=confidence,
                    tier=signal["tier"],
                )

            equity_curve.append(capital)

            # Track drawdown
            max_equity = max(max_equity, capital)

        # ─── Close any open trade at last price ──────────
        if current_trade:
            last_close = safe_float(df.iloc[-1]["close"])
            if current_trade.direction == "BUY":
                pnl_pct = (last_close - current_trade.entry_price) / current_trade.entry_price * 100
            else:
                pnl_pct = (current_trade.entry_price - last_close) / current_trade.entry_price * 100

            current_trade.exit_idx = len(df) - 1
            current_trade.exit_price = last_close
            current_trade.exit_reason = "END_OF_DATA"
            current_trade.pnl_pct = round(pnl_pct, 2)
            current_trade.is_win = pnl_pct > 0
            trades.append(current_trade)
            capital *= (1 + pnl_pct / 100)

        # ─── Compute Statistics ──────────────────────────
        result.total_trades = len(trades)
        result.trades = trades
        result.equity_curve = equity_curve

        if trades:
            wins = [t for t in trades if t.is_win]
            losses = [t for t in trades if not t.is_win]
            pnls = [t.pnl_pct for t in trades]

            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / len(trades) * 100
            result.total_return_pct = (capital - self.initial_capital) / self.initial_capital * 100
            result.avg_profit_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
            result.avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
            result.best_trade_pct = max(pnls)
            result.worst_trade_pct = min(pnls)
            result.avg_holding_bars = np.mean([t.exit_idx - t.entry_idx for t in trades])

            # Max Drawdown
            eq = np.array(equity_curve)
            peak = np.maximum.accumulate(eq)
            drawdown = (eq - peak) / peak * 100
            result.max_drawdown_pct = float(drawdown.min())

            # Sharpe Ratio (assuming daily returns)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                result.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

            # Profit Factor
            gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 1
            result.profit_factor = gross_profit / max(gross_loss, 0.01)

        return result


def _quick_confidence(indicators: dict, direction: str) -> int:
    """Quick confidence score for backtesting (simplified, no external data)."""
    score = 50
    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd_hist", 0)
    adx = indicators.get("adx", 20)
    volume_ratio = indicators.get("volume_ratio", 1.0)

    if direction == "BUY":
        if rsi < 35:
            score += 12
        if macd_hist > 0:
            score += 8
    else:
        if rsi > 65:
            score += 12
        if macd_hist < 0:
            score += 8

    if adx > 25:
        score += 8
    if volume_ratio > 1.3:
        score += 5

    return min(100, max(0, score))
