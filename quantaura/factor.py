"""Cross-sectional momentum — the academic 'momentum factor'.

Reference: Jegadeesh & Titman (1993); used at scale by AQR, Two Sigma and
most systematic equity shops. The rule: within a universe, rank assets by
their trailing return (here ~6 months, skipping the most recent month to
avoid short-term reversal), then go LONG the strongest names and SHORT the
weakest. The edge is relative, not absolute.

Unlike the single-asset strategies, this is validated by a **panel
backtest**: at each monthly rebalance we form the long-short basket from
information available at that time and measure its forward return. The
basket's historical win rate / profit factor gate the live signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import stats_from_R
from .indicators import atr as atr_ind
from .models import BacktestStats, Side


@dataclass
class FactorLeg:
    symbol: str
    side: Side
    score: float           # trailing momentum return
    rank: int              # 1 = strongest (longs) / 1 = weakest (shorts)
    entry: float
    stop: float
    target: float
    atr: float

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    def valid(self) -> bool:
        if not all(math.isfinite(x) for x in (self.entry, self.stop, self.target)):
            return False
        if self.risk_per_unit <= 0:
            return False
        if self.side is Side.LONG:
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop


def _momentum_score(close: pd.Series, lookback: int, skip: int) -> float:
    """Trailing return from (skip+lookback) bars ago to skip bars ago."""
    if len(close) < lookback + skip + 1:
        return float("nan")
    end = close.iloc[-1 - skip]
    start = close.iloc[-1 - skip - lookback]
    if start <= 0 or not math.isfinite(start) or not math.isfinite(end):
        return float("nan")
    return float(end / start - 1.0)


# ---------------------------------------------------------------------
def backtest_cross_sectional(panel: pd.DataFrame, cfg: dict) -> BacktestStats:
    """Long-short basket backtest, rebalanced every `rebalance_days`.

    `panel` columns are symbols, rows are aligned closes. No look-ahead:
    formation uses data up to the rebalance bar; the position's return is
    measured strictly forward.
    """
    lookback = int(cfg.get("lookback_days", 126))
    skip = int(cfg.get("skip_days", 21))
    hold = int(cfg.get("rebalance_days", 21))
    top_n = int(cfg.get("top_n", 2))
    bottom_n = int(cfg.get("bottom_n", 2))
    allow_short = bool(cfg.get("allow_short", True))

    panel = panel.dropna(how="any")
    n = len(panel)
    start = lookback + skip
    if n < start + hold + 1 or panel.shape[1] < max(top_n, 1) + (bottom_n if allow_short else 0):
        return BacktestStats()

    basket_returns: list[float] = []
    t = start
    while t + hold < n:
        # formation: trailing return ending `skip` bars before t
        form_end = t - skip
        form_start = t - skip - lookback
        if form_start < 0:
            t += hold
            continue
        scores = (panel.iloc[form_end] / panel.iloc[form_start] - 1.0).dropna()
        if len(scores) < top_n + (bottom_n if allow_short else 0):
            t += hold
            continue
        ranked = scores.sort_values(ascending=False)
        longs = ranked.index[:top_n]
        fwd = panel.iloc[t + hold] / panel.iloc[t] - 1.0
        long_ret = float(fwd[longs].mean())
        if allow_short and bottom_n > 0:
            shorts = ranked.index[-bottom_n:]
            short_ret = float(fwd[shorts].mean())
            basket_returns.append(long_ret - short_ret)
        else:
            basket_returns.append(long_ret)
        t += hold

    return stats_from_R(basket_returns)


# ---------------------------------------------------------------------
def rank_live(
    symbol_to_df: dict[str, pd.DataFrame], cfg: dict
) -> list[FactorLeg]:
    """Build live long/short legs from the latest data in each frame."""
    lookback = int(cfg.get("lookback_days", 126))
    skip = int(cfg.get("skip_days", 21))
    top_n = int(cfg.get("top_n", 2))
    bottom_n = int(cfg.get("bottom_n", 2))
    allow_short = bool(cfg.get("allow_short", True))
    atr_period = int(cfg.get("atr_period", 14))
    atr_mult = float(cfg.get("atr_stop_mult", 3.0))
    target_R = float(cfg.get("min_target_R", 2.0))

    scores: dict[str, float] = {}
    for sym, df in symbol_to_df.items():
        if df is None or len(df) < lookback + skip + 2:
            continue
        s = _momentum_score(df["close"], lookback, skip)
        if math.isfinite(s):
            scores[sym] = s
    if len(scores) < top_n + (bottom_n if allow_short else 0):
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    legs: list[FactorLeg] = []

    def _make(sym: str, side: Side, score: float, rank: int) -> Optional[FactorLeg]:
        df = symbol_to_df[sym]
        entry = float(df["close"].iloc[-1])
        a = float(atr_ind(df, atr_period).iloc[-1])
        if not math.isfinite(a) or a <= 0:
            return None
        risk = atr_mult * a
        if side is Side.LONG:
            leg = FactorLeg(sym, side, score, rank, entry, entry - risk,
                            entry + target_R * risk, a)
        else:
            leg = FactorLeg(sym, side, score, rank, entry, entry + risk,
                            entry - target_R * risk, a)
        return leg if leg.valid() else None

    for rank, (sym, score) in enumerate(ranked[:top_n], start=1):
        leg = _make(sym, Side.LONG, score, rank)
        if leg:
            legs.append(leg)
    if allow_short and bottom_n > 0:
        for rank, (sym, score) in enumerate(reversed(ranked[-bottom_n:]), start=1):
            leg = _make(sym, Side.SHORT, score, rank)
            if leg:
                legs.append(leg)
    return legs


def passes_factor_gate(stats: BacktestStats, cfg: dict) -> bool:
    return (
        stats.trades >= int(cfg.get("min_basket_rebalances", 24))
        and stats.win_rate >= float(cfg.get("min_basket_winrate", 0.45))
        and stats.profit_factor >= float(cfg.get("min_basket_profit_factor", 1.15))
    )
