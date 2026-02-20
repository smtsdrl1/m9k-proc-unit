"""
Chart Visualization — mplfinance candlestick charts with overlays.
Generates .png charts with Bollinger Bands, S/R lines, buy/sell markers, RSI/MACD subplots.
"""
import os
import logging
from typing import Optional
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import pandas_ta as ta

logger = logging.getLogger("matrix_trader.visualization.charts")

# Chart output directory
CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Custom dark style
MATRIX_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#00ff88", down="#ff4444",
        edge={"up": "#00ff88", "down": "#ff4444"},
        wick={"up": "#00ff88", "down": "#ff4444"},
        volume={"up": "#00ff8844", "down": "#ff444444"},
    ),
    facecolor="#0a0a1a",
    figcolor="#0a0a1a",
    gridcolor="#1a1a3a",
    rc={
        "font.size": 9,
        "axes.labelcolor": "#cccccc",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
    },
)


def generate_analysis_chart(
    df: pd.DataFrame,
    symbol: str,
    indicators: dict = None,
    signal_direction: str = None,
    support_levels: list[float] = None,
    resistance_levels: list[float] = None,
    show_bb: bool = True,
    show_ema: bool = True,
    last_n_bars: int = 100,
) -> Optional[str]:
    """
    Generate a professional analysis chart with overlays.

    Returns: File path to saved .png chart, or None on error.
    """
    try:
        # Take last N bars
        chart_df = df.tail(last_n_bars).copy()
        if len(chart_df) < 20:
            logger.warning(f"Insufficient data for chart: {len(chart_df)} bars")
            return None

        # Ensure proper column names
        chart_df = chart_df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        addplots = []

        # ─── Bollinger Bands ─────────────────────────────
        if show_bb:
            bb = ta.bbands(chart_df["Close"], length=20, std=2)
            if bb is not None and len(bb.columns) >= 3:
                chart_df["BB_Upper"] = bb.iloc[:, 0].values
                chart_df["BB_Middle"] = bb.iloc[:, 1].values
                chart_df["BB_Lower"] = bb.iloc[:, 2].values

                addplots.append(mpf.make_addplot(chart_df["BB_Upper"], color="#4488ff", width=0.7, linestyle="--"))
                addplots.append(mpf.make_addplot(chart_df["BB_Middle"], color="#4488ff44", width=0.5))
                addplots.append(mpf.make_addplot(chart_df["BB_Lower"], color="#4488ff", width=0.7, linestyle="--"))

        # ─── EMAs ────────────────────────────────────────
        if show_ema:
            if len(chart_df) >= 21:
                chart_df["EMA9"] = ta.ema(chart_df["Close"], length=9)
                chart_df["EMA21"] = ta.ema(chart_df["Close"], length=21)
                addplots.append(mpf.make_addplot(chart_df["EMA9"], color="#ffaa00", width=0.8))
                addplots.append(mpf.make_addplot(chart_df["EMA21"], color="#ff6600", width=0.8))

        # ─── RSI Subplot ─────────────────────────────────
        rsi_series = ta.rsi(chart_df["Close"], length=14)
        if rsi_series is not None:
            chart_df["RSI"] = rsi_series.values
            # Overbought/oversold lines
            chart_df["RSI_70"] = 70
            chart_df["RSI_30"] = 30
            addplots.append(mpf.make_addplot(chart_df["RSI"], panel=2, color="#ff88ff", ylabel="RSI"))
            addplots.append(mpf.make_addplot(chart_df["RSI_70"], panel=2, color="#ff444488", width=0.5, linestyle="--"))
            addplots.append(mpf.make_addplot(chart_df["RSI_30"], panel=2, color="#00ff8888", width=0.5, linestyle="--"))

        # ─── MACD Subplot ────────────────────────────────
        macd_result = ta.macd(chart_df["Close"], fast=12, slow=26, signal=9)
        if macd_result is not None and len(macd_result.columns) >= 3:
            chart_df["MACD"] = macd_result.iloc[:, 0].values
            chart_df["MACD_Signal"] = macd_result.iloc[:, 1].values
            chart_df["MACD_Hist"] = macd_result.iloc[:, 2].values

            # Color histogram bars
            hist_colors = ["#00ff88" if v >= 0 else "#ff4444" for v in chart_df["MACD_Hist"].fillna(0)]

            addplots.append(mpf.make_addplot(chart_df["MACD"], panel=3, color="#00aaff", ylabel="MACD"))
            addplots.append(mpf.make_addplot(chart_df["MACD_Signal"], panel=3, color="#ffaa00", width=0.7))
            addplots.append(mpf.make_addplot(chart_df["MACD_Hist"], panel=3, type="bar", color=hist_colors))

        # ─── Support/Resistance Horizontal Lines ─────────
        hline_kwargs = {}
        all_hlines = []
        all_colors = []
        if support_levels:
            valid_sup = [s for s in support_levels if s and s > 0]
            all_hlines.extend(valid_sup)
            all_colors.extend(["#00ff88"] * len(valid_sup))
        if resistance_levels:
            valid_res = [r for r in resistance_levels if r and r > 0]
            all_hlines.extend(valid_res)
            all_colors.extend(["#ff4444"] * len(valid_res))

        if all_hlines:
            hline_kwargs = {
                "hlines": dict(
                    hlines=all_hlines,
                    colors=all_colors,
                    linestyle="-.",
                    linewidths=0.7,
                ),
            }

        # ─── Buy/Sell Marker ─────────────────────────────
        if signal_direction:
            marker_data = [np.nan] * len(chart_df)
            if signal_direction == "BUY":
                marker_data[-1] = chart_df["Low"].iloc[-1] * 0.995
                addplots.append(mpf.make_addplot(
                    marker_data, type="scatter", markersize=200,
                    marker="^", color="#00ff88",
                ))
            elif signal_direction == "SELL":
                marker_data[-1] = chart_df["High"].iloc[-1] * 1.005
                addplots.append(mpf.make_addplot(
                    marker_data, type="scatter", markersize=200,
                    marker="v", color="#ff4444",
                ))

        # ─── Generate Chart ──────────────────────────────
        title = f"{symbol} Technical Analysis"
        filename = f"{symbol.replace('/', '_')}_{signal_direction or 'analysis'}.png"
        filepath = os.path.join(CHART_DIR, filename)

        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=MATRIX_STYLE,
            title=title,
            volume=True,
            addplot=addplots if addplots else None,
            figsize=(14, 10),
            panel_ratios=(4, 1, 1.2, 1.2),
            returnfig=True,
            tight_layout=True,
            **hline_kwargs,
        )

        # Add watermark
        fig.text(0.5, 0.01, "Matrix Trader AI v1.0", ha="center",
                 fontsize=8, color="#444444", style="italic")

        fig.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)

        logger.info(f"Chart saved: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Chart generation failed for {symbol}: {e}")
        return None


def generate_backtest_chart(
    equity_curve: list[float],
    trades: list,
    symbol: str,
) -> Optional[str]:
    """Generate backtest equity curve chart."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.set_facecolor("#0a0a1a")

        # Equity curve
        ax1.set_facecolor("#0a0a1a")
        ax1.plot(equity_curve, color="#00ff88", linewidth=1.5)
        ax1.fill_between(range(len(equity_curve)), equity_curve,
                         equity_curve[0], alpha=0.1, color="#00ff88")
        ax1.axhline(y=equity_curve[0], color="#ffffff33", linestyle="--", linewidth=0.5)
        ax1.set_title(f"{symbol} Backtest — Equity Curve", color="white", fontsize=14)
        ax1.set_ylabel("Capital", color="#cccccc")
        ax1.tick_params(colors="#888888")

        # Drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak * 100

        ax2.set_facecolor("#0a0a1a")
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color="#ff4444")
        ax2.set_title("Drawdown %", color="white", fontsize=11)
        ax2.set_ylabel("Drawdown %", color="#cccccc")
        ax2.tick_params(colors="#888888")

        fig.text(0.5, 0.01, "Matrix Trader AI v1.0 — Backtest", ha="center",
                 fontsize=8, color="#444444", style="italic")

        filename = f"{symbol.replace('/', '_')}_backtest.png"
        filepath = os.path.join(CHART_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a", edgecolor="none")
        plt.close(fig)

        logger.info(f"Backtest chart saved: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Backtest chart generation failed: {e}")
        return None
