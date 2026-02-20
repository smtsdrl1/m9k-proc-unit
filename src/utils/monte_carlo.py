"""
Monte Carlo Risk Simulation for Matrix Trader AI.
Bootstrap resampling from historical trade returns.
"""
import logging
import random
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("matrix_trader.utils.monte_carlo")


@dataclass
class SimulationResult:
    median_max_dd: float
    p95_max_dd: float
    p99_max_dd: float
    ruin_risk_pct: float
    kelly_fraction: float
    cagr_median: float
    verdict: str
    n_simulations: int


def run_monte_carlo(
    trade_returns: List[float],
    initial_capital: float = 10000,
    n_simulations: int = 3000,
    n_trades: int = 100,
) -> SimulationResult:
    """
    Bootstrap Monte Carlo simulation.

    Args:
        trade_returns: List of per-trade returns as fractions (0.04 = 4%)
        initial_capital: Starting capital
        n_simulations: Number of simulation paths
        n_trades: Trades per simulation path

    Returns:
        SimulationResult with risk metrics
    """
    if not trade_returns or len(trade_returns) < 5:
        return SimulationResult(
            median_max_dd=0, p95_max_dd=0, p99_max_dd=0,
            ruin_risk_pct=0, kelly_fraction=0.02, cagr_median=0,
            verdict="INSUFFICIENT_DATA", n_simulations=0
        )

    max_drawdowns, final_capitals = [], []
    ruin_count = 0
    ruin_threshold = initial_capital * 0.5

    for _ in range(n_simulations):
        capital = initial_capital
        peak = initial_capital
        max_dd = 0.0
        sampled = random.choices(trade_returns, k=n_trades)

        for r in sampled:
            capital *= (1 + r)
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd

        max_drawdowns.append(max_dd)
        final_capitals.append(capital)
        if capital < ruin_threshold:
            ruin_count += 1

    max_drawdowns.sort()
    final_capitals.sort()

    n = len(max_drawdowns)
    median_dd   = max_drawdowns[n // 2]
    p95_dd      = max_drawdowns[int(n * 0.95)]
    p99_dd      = max_drawdowns[int(n * 0.99)]
    ruin_risk   = ruin_count / n_simulations * 100
    median_cap  = final_capitals[n // 2]
    cagr_median = ((median_cap / initial_capital) - 1) * 100

    # Kelly Criterion from sample
    wins   = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    if wins and losses:
        win_rate = len(wins) / len(trade_returns)
        avg_win  = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        kelly    = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else 0
        kelly_fraction = max(0.0, min(kelly * 0.5, 0.25))
    else:
        kelly_fraction = 0.02

    if ruin_risk < 2 and p95_dd < 0.25:
        verdict = "SAFE"
    elif ruin_risk < 5 and p95_dd < 0.40:
        verdict = "ACCEPTABLE"
    elif ruin_risk < 10:
        verdict = "RISKY"
    else:
        verdict = "DANGEROUS"

    return SimulationResult(
        median_max_dd=round(median_dd * 100, 2),
        p95_max_dd=round(p95_dd * 100, 2),
        p99_max_dd=round(p99_dd * 100, 2),
        ruin_risk_pct=round(ruin_risk, 2),
        kelly_fraction=round(kelly_fraction, 4),
        cagr_median=round(cagr_median, 2),
        verdict=verdict,
        n_simulations=n_simulations,
    )
