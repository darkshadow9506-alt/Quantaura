"""Walk-forward parameter optimisation.

Grid-searches a strategy's key parameters and scores each combination by
its **out-of-sample** expectancy — NOT its in-sample fit. Scoring on the
held-out tail is exactly the discipline the methodology doc prescribes to
avoid curve-fitting: a combo that only looks good in-sample is ignored.

Usage:  python -m quantaura optimize AAPL --strategy trend
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .backtest import backtest_strategy, out_of_sample
from .models import BacktestStats
from .strategies import (
    DualThrust,
    MacdTrend,
    MeanReversion,
    SqueezeBreakout,
    TrendBreakout,
)

# strategy -> (class, config-section, parameter grid)
_GRIDS = {
    "trend": (TrendBreakout, "trend", {
        "donchian_entry": [10, 20, 40],
        "atr_stop_mult": [2.0, 2.5, 3.0],
        "min_target_R": [1.5, 2.0, 3.0],
    }),
    "mean_reversion": (MeanReversion, "mean_reversion", {
        "zscore_entry": [1.5, 2.0, 2.5],
        "atr_stop_mult": [2.5, 3.0, 3.5],
    }),
    "macd": (MacdTrend, "macd", {
        "atr_stop_mult": [2.0, 2.5, 3.0],
        "min_target_R": [1.5, 2.0, 3.0],
    }),
    "dual_thrust": (DualThrust, "dual_thrust", {
        "range_bars": [3, 4, 5],
        "k1": [0.4, 0.5, 0.7],
        "atr_stop_mult": [1.5, 2.0, 2.5],
    }),
    "squeeze": (SqueezeBreakout, "squeeze", {
        "atr_stop_mult": [1.0, 1.5, 2.0],
        "min_target_R": [2.0, 3.0, 4.0],
    }),
}


@dataclass
class OptResult:
    params: dict
    full: BacktestStats
    oos: BacktestStats
    score: float


def _grid_combos(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def optimize_on_df(
    df: pd.DataFrame,
    strategy_name: str,
    base_cfg: dict,
    min_trades: int = 10,
    oos_min_trades: int = 3,
    oos_split: float = 0.7,
) -> list[OptResult]:
    if strategy_name not in _GRIDS:
        raise ValueError(f"unknown strategy '{strategy_name}'")
    cls, _section, grid = _GRIDS[strategy_name]
    # for dual_thrust keep k2 == k1 unless overridden
    results: list[OptResult] = []
    for combo in _grid_combos(grid):
        cfg = dict(base_cfg)
        cfg.update(combo)
        if strategy_name == "dual_thrust" and "k1" in combo:
            cfg.setdefault("k2", combo["k1"])
            cfg["k2"] = combo["k1"]
        strat = cls(cfg)
        full, _ = backtest_strategy(strat, df)
        oos = out_of_sample(full, oos_split)
        if full.trades >= min_trades and oos.trades >= oos_min_trades:
            score = oos.expectancy_R          # score on held-out data
        else:
            score = float("-inf")
        results.append(OptResult(combo, full, oos, score))
    results.sort(key=lambda r: (r.score, r.full.expectancy_R), reverse=True)
    return results


def optimize_symbol(symbol: str, strategy_name: str, settings) -> list[OptResult]:
    from . import data as data_mod
    from .data import asset_class_of

    ac = asset_class_of(symbol, settings.universe)
    d = settings.data
    df = data_mod.get_ohlcv(
        symbol, ac, timeframe=d.get("timeframe", "1d"),
        lookback=int(d.get("lookback_bars", 1250)),
        cache_minutes=int(d.get("cache_minutes", 30)),
        ccxt_exchange=settings.ccxt_exchange,
    )
    base = settings.section(_GRIDS[strategy_name][1])
    return optimize_on_df(df, strategy_name, base)


def format_report(results: list[OptResult], top: int = 8) -> str:
    if not results:
        return "No optimisation results."
    lines = ["Walk-forward optimisation (scored by out-of-sample expectancy):",
             f"{'params':<46}{'OOS exp':>9}{'OOS n':>7}{'full exp':>10}{'win%':>7}"]
    shown = 0
    for r in results[:top]:
        if r.score == float("-inf") and shown > 0:
            continue
        p = ", ".join(f"{k}={v}" for k, v in r.params.items())
        score = "n/a" if r.score == float("-inf") else f"{r.oos.expectancy_R:+.2f}R"
        full_exp = f"{r.full.expectancy_R:+.2f}R"
        win = f"{r.full.win_rate*100:.0f}%"
        lines.append(f"{p:<46}{score:>9}{r.oos.trades:>7}{full_exp:>10}{win:>7}")
        shown += 1
    best = results[0]
    if best.score != float("-inf"):
        lines.append("")
        lines.append(f"➡ Best out-of-sample: {best.params} "
                     f"(OOS exp {best.oos.expectancy_R:+.2f}R over {best.oos.trades} trades)")
    return "\n".join(lines)
