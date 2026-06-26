"""Single-asset strategies + regime detection.

Each strategy is a small class that:
  * `prepare(df)`  -> returns a copy of the OHLCV frame with the
                      indicator columns it needs, computed once.
  * `evaluate(df, i)` -> returns a `TradePlan` IF the strategy would
                      open a position at bar `i`, using ONLY data up to
                      and including bar `i` (no look-ahead), else None.

The same `evaluate` is used both live (i = last bar) and inside the
backtester (i swept across history). Entry/stop/target are concrete
prices derived from ATR and market structure — never guesses.

Strategy rationale (grounded in the methodology in README):
  * TrendBreakout   — time-series momentum / trend following. Donchian
                      breakout, filtered by the 200-period MA so we only
                      trade breakouts in the direction of the major trend.
  * MeanReversion   — short-horizon overreaction reversal. Bollinger /
                      z-score stretch + Connors RSI-2, taken only WITH
                      the long-term trend (buy dips up, fade rips down).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .models import Side


# ---------------------------------------------------------------------
@dataclass
class TradePlan:
    """A strategy's concrete trade idea at one bar (pre-sizing)."""

    side: Side
    entry: float
    stop: float
    target: float
    atr: float
    rationale: str
    regime: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.target - self.entry)

    @property
    def rr_ratio(self) -> float:
        r = self.risk_per_unit
        return self.reward_per_unit / r if r > 0 else 0.0

    def valid(self) -> bool:
        if not all(math.isfinite(x) for x in (self.entry, self.stop, self.target, self.atr)):
            return False
        if self.risk_per_unit <= 0 or self.reward_per_unit <= 0:
            return False
        if self.side is Side.LONG:
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop


def _finite(*vals: float) -> bool:
    return all(v is not None and math.isfinite(v) for v in vals)


# ---------------------------------------------------------------------
# Regime detection (ADX): trending vs ranging vs neutral
# ---------------------------------------------------------------------
def detect_regime(df: pd.DataFrame, cfg: dict) -> str:
    period = int(cfg.get("adx_period", 14))
    trend_th = float(cfg.get("adx_trend_threshold", 25))
    range_th = float(cfg.get("adx_range_threshold", 20))
    adx_series = ind.adx(df, period)
    if adx_series.dropna().empty:
        return "neutral"
    val = float(adx_series.iloc[-1])
    if not math.isfinite(val):
        return "neutral"
    if val >= trend_th:
        return "trending"
    if val < range_th:
        return "ranging"
    return "neutral"


# ---------------------------------------------------------------------
class TrendBreakout:
    name = "trend_breakout"
    preferred_regime = "trending"

    def __init__(self, cfg: dict):
        self.donchian_entry = int(cfg.get("donchian_entry", 20))
        self.donchian_exit = int(cfg.get("donchian_exit", 10))
        self.ma_fast = int(cfg.get("ma_fast", 50))
        self.ma_slow = int(cfg.get("ma_slow", 200))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 2.5))
        self.min_target_R = float(cfg.get("min_target_R", 2.0))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_slow"] = ind.sma(d["close"], self.ma_slow)
        d["dc_up"], d["dc_low"] = ind.donchian(d, self.donchian_entry)
        d["dc_exit_up"], d["dc_exit_low"] = ind.donchian(d, self.donchian_exit)
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        row = df.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        atr_v = float(row["atr"])
        ma_slow = float(row["ma_slow"])
        dc_up, dc_low = float(row["dc_up"]), float(row["dc_low"])
        if not _finite(close, high, low, atr_v, ma_slow, dc_up, dc_low) or atr_v <= 0:
            return None

        risk = self.atr_stop_mult * atr_v

        # Long: break above prior N-bar high, in an uptrend
        if high > dc_up and close > ma_slow:
            entry = close
            stop = entry - risk
            target = entry + self.min_target_R * risk
            return TradePlan(
                Side.LONG, entry, stop, target, atr_v,
                rationale=(f"{self.donchian_entry}-bar breakout above {dc_up:.4f} "
                           f"with price over {self.ma_slow}MA (uptrend)."),
                meta={"dc_level": dc_up},
            )

        # Short: break below prior N-bar low, in a downtrend
        if low < dc_low and close < ma_slow:
            entry = close
            stop = entry + risk
            target = entry - self.min_target_R * risk
            return TradePlan(
                Side.SHORT, entry, stop, target, atr_v,
                rationale=(f"{self.donchian_entry}-bar breakdown below {dc_low:.4f} "
                           f"with price under {self.ma_slow}MA (downtrend)."),
                meta={"dc_level": dc_low},
            )
        return None


# ---------------------------------------------------------------------
class MeanReversion:
    name = "mean_reversion"
    preferred_regime = "ranging"

    def __init__(self, cfg: dict):
        self.bb_period = int(cfg.get("bb_period", 20))
        self.bb_std = float(cfg.get("bb_std", 2.0))
        self.zscore_entry = float(cfg.get("zscore_entry", 2.0))
        self.rsi_period = int(cfg.get("rsi_period", 2))
        self.rsi_buy_below = float(cfg.get("rsi_buy_below", 10))
        self.rsi_sell_above = float(cfg.get("rsi_sell_above", 90))
        self.ma_trend = int(cfg.get("ma_trend", 200))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", 3.0))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["atr"] = ind.atr(d, self.atr_period)
        d["ma_trend"] = ind.sma(d["close"], self.ma_trend)
        mid, up, low = ind.bollinger(d["close"], self.bb_period, self.bb_std)
        d["bb_mid"], d["bb_up"], d["bb_low"] = mid, up, low
        d["zscore"] = ind.rolling_zscore(d["close"], self.bb_period)
        d["rsi"] = ind.rsi(d["close"], self.rsi_period)
        return d

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[TradePlan]:
        row = df.iloc[i]
        close = float(row["close"])
        atr_v = float(row["atr"])
        ma_trend = float(row["ma_trend"])
        bb_mid = float(row["bb_mid"])
        bb_up, bb_low = float(row["bb_up"]), float(row["bb_low"])
        z = float(row["zscore"])
        rsi_v = float(row["rsi"])
        if not _finite(close, atr_v, ma_trend, bb_mid, bb_up, bb_low, z, rsi_v) or atr_v <= 0:
            return None

        risk = self.atr_stop_mult * atr_v

        # Long: oversold dip below lower band, but major trend still up
        if (close < bb_low and z <= -self.zscore_entry and rsi_v < self.rsi_buy_below
                and close > ma_trend):
            entry = close
            stop = entry - risk
            target = bb_mid  # revert to the mean
            plan = TradePlan(
                Side.LONG, entry, stop, target, atr_v,
                rationale=(f"Oversold: z={z:.2f}, RSI{self.rsi_period}={rsi_v:.1f} below "
                           f"lower band, price above {self.ma_trend}MA. Target = mean."),
                meta={"z": z, "rsi": rsi_v},
            )
            return plan if plan.valid() else None

        # Short: overbought spike above upper band, but major trend still down
        if (close > bb_up and z >= self.zscore_entry and rsi_v > self.rsi_sell_above
                and close < ma_trend):
            entry = close
            stop = entry + risk
            target = bb_mid
            plan = TradePlan(
                Side.SHORT, entry, stop, target, atr_v,
                rationale=(f"Overbought: z={z:.2f}, RSI{self.rsi_period}={rsi_v:.1f} above "
                           f"upper band, price below {self.ma_trend}MA. Target = mean."),
                meta={"z": z, "rsi": rsi_v},
            )
            return plan if plan.valid() else None
        return None


# ---------------------------------------------------------------------
def build_strategies(settings) -> list:
    """Instantiate enabled single-asset strategies from settings."""
    out = []
    if settings.trend.get("enabled", True):
        out.append(TrendBreakout(settings.trend))
    if settings.mean_reversion.get("enabled", True):
        out.append(MeanReversion(settings.mean_reversion))
    return out
