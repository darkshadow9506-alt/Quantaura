"""Event-driven backtester for single-asset strategies.

For every bar where the strategy produces a TradePlan and we are flat,
we open a position at that bar's close (a realistic end-of-day fill) and
then walk forward bar-by-bar, exiting on:
  * stop hit       -> -1.0 R
  * target hit     -> +rr  R   (rr = reward/risk of the plan)
  * max-hold reached -> mark-to-close R

If a single bar's range spans BOTH stop and target, we pessimistically
assume the stop filled first. There is no look-ahead: exits use future
bars only after entry, and entry uses only data up to the signal bar.

Results are summarized as per-trade R statistics, the unit in which we
gate signals (see signal_gate in config.yaml).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import BacktestStats, Side
from .strategies import TradePlan


# ---------------------------------------------------------------------
def stats_from_R(returns_R: list[float]) -> BacktestStats:
    """Canonical per-trade R -> BacktestStats. Shared by all backtests."""
    if not returns_R:
        return BacktestStats()
    arr = np.asarray(returns_R, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    win_rate = len(wins) / len(arr)
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = 999.0 if gross_win > 0 else 0.0

    std = arr.std(ddof=0)
    # significance-weighted quality score (t-stat style): mean/std*sqrt(N).
    # Used purely as a relative gate, labelled "sharpe" for familiarity.
    sharpe = float(arr.mean() / std * math.sqrt(len(arr))) if std > 0 else 0.0

    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    max_dd = float((peak - eq).max()) if len(eq) else 0.0

    return BacktestStats(
        trades=int(len(arr)),
        win_rate=round(float(win_rate), 4),
        profit_factor=round(float(min(profit_factor, 999.0)), 4),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        avg_R=round(float(arr.mean()), 4),
        expectancy_R=round(float(arr.mean()), 4),
        returns_R=[round(float(x), 6) for x in arr],
    )


def out_of_sample(stats: BacktestStats, split: float = 0.7) -> BacktestStats:
    """Walk-forward holdout: stats on the most recent (1-split) of trades.

    `returns_R` is chronological, so slicing the tail gives a genuine
    out-of-sample window the strategy parameters never 'saw'.
    """
    r = stats.returns_R
    if len(r) < 4:
        return BacktestStats()
    cut = int(len(r) * split)
    return stats_from_R(r[cut:])


# ---------------------------------------------------------------------
@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    side: str
    entry: float
    exit: float
    R: float
    outcome: str


def _simulate_trailing(df, i, plan, max_hold, trail_mult):
    """Chandelier-exit simulation: trail the stop at (extreme - mult·ATR).

    The fixed target is dropped so winners can run; exit is the trailed
    stop or max_hold. The trail uses only bars strictly before the current
    one (no look-ahead). Returns (R, exit_idx, exit_price, outcome).
    """
    entry, risk = plan.entry, plan.risk_per_unit
    atr = plan.atr
    n = len(df)
    end = min(i + max_hold, n - 1)
    long = plan.side is Side.LONG
    ext = float(df["high"].iloc[i]) if long else float(df["low"].iloc[i])
    if long:
        stop = max(plan.stop, ext - trail_mult * atr)
    else:
        stop = min(plan.stop, ext + trail_mult * atr)

    for j in range(i + 1, end + 1):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])
        if long and low <= stop:
            return (stop - entry) / risk, j, stop, "trail"
        if (not long) and high >= stop:
            return (entry - stop) / risk, j, stop, "trail"
        # ratchet the trail using this bar's extreme (for the NEXT bar)
        if long:
            ext = max(ext, high)
            stop = max(stop, ext - trail_mult * atr)
        else:
            ext = min(ext, low)
            stop = min(stop, ext + trail_mult * atr)

    exit_price = float(df["close"].iloc[end])
    R = (exit_price - entry) / risk if long else (entry - exit_price) / risk
    return R, end, exit_price, "time"


def _simulate_trade(df: pd.DataFrame, i: int, plan: TradePlan, max_hold: int,
                    trail_atr_mult: float = 0.0):
    """Walk forward from entry bar i; return (R, exit_idx, exit_price, outcome)."""
    if trail_atr_mult > 0 and plan.atr > 0:
        return _simulate_trailing(df, i, plan, max_hold, trail_atr_mult)

    entry = plan.entry
    stop = plan.stop
    target = plan.target
    risk = plan.risk_per_unit
    rr = plan.rr_ratio
    n = len(df)

    end = min(i + max_hold, n - 1)
    for j in range(i + 1, end + 1):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])
        if plan.side is Side.LONG:
            hit_stop = low <= stop
            hit_target = high >= target
            if hit_stop and hit_target:      # pessimistic: stop first
                return -1.0, j, stop, "stop"
            if hit_stop:
                return -1.0, j, stop, "stop"
            if hit_target:
                return rr, j, target, "target"
        else:  # SHORT
            hit_stop = high >= stop
            hit_target = low <= target
            if hit_stop and hit_target:
                return -1.0, j, stop, "stop"
            if hit_stop:
                return -1.0, j, stop, "stop"
            if hit_target:
                return rr, j, target, "target"

    # time exit at close of `end`
    exit_price = float(df["close"].iloc[end])
    if plan.side is Side.LONG:
        R = (exit_price - entry) / risk if risk > 0 else 0.0
    else:
        R = (entry - exit_price) / risk if risk > 0 else 0.0
    return R, end, exit_price, "time"


def backtest_strategy(strategy, df: pd.DataFrame, max_hold: int = 60,
                      trail_atr_mult: float = 0.0):
    """Backtest one strategy over a prepared OHLCV frame.

    `trail_atr_mult > 0` enables a Chandelier trailing-stop exit instead of
    the fixed target. Returns (BacktestStats, list[Trade]).
    """
    prepared = strategy.prepare(df)
    n = len(prepared)
    trades: list[Trade] = []
    returns_R: list[float] = []

    i = 0
    # warmup: skip until indicators are populated (first non-NaN atr)
    while i < n:
        plan = None
        try:
            plan = strategy.evaluate(prepared, i)
        except Exception:
            plan = None
        if plan is not None and plan.valid():
            R, exit_idx, exit_price, outcome = _simulate_trade(
                prepared, i, plan, max_hold, trail_atr_mult)
            trades.append(
                Trade(i, exit_idx, plan.side.value, plan.entry, exit_price, R, outcome)
            )
            returns_R.append(R)
            i = exit_idx + 1   # no overlapping positions
        else:
            i += 1

    return stats_from_R(returns_R), trades
