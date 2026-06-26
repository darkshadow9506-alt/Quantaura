"""Vectorized technical indicators.

All functions take/return pandas Series or DataFrames and never look
ahead: value at index t uses only data up to and including t. This is
enforced by using only rolling / ewm / shift operations.

Indicator definitions follow the standard references (Wilder 1978 for
ATR/RSI/ADX, Bollinger for bands). They are unit-tested in tests/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------
# Volatility: True Range / ATR (Wilder smoothing)
# ---------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA)."""
    tr = true_range(df)
    # Wilder RMA == EMA with alpha = 1/period
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------
# RSI (Wilder)
# ---------------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 -> RSI = 100; when avg_gain == 0 -> RSI = 0
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss == 0, 0.0))
    return out


# ---------------------------------------------------------------------
# Bollinger Bands & rolling z-score
# ---------------------------------------------------------------------
def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def rolling_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


# ---------------------------------------------------------------------
# Donchian channel (breakout levels)
# ---------------------------------------------------------------------
def donchian(df: pd.DataFrame, period: int = 20):
    """Return (upper, lower) Donchian channel built from PRIOR bars.

    Shifted by 1 so the level at bar t is the highest high / lowest low
    of the *preceding* `period` bars — i.e. a level you could actually
    have traded a breakout of at bar t. No look-ahead.
    """
    upper = df["high"].rolling(window=period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(window=period, min_periods=period).min().shift(1)
    return upper, lower


# ---------------------------------------------------------------------
# ADX / DMI (Wilder) — regime detection
# ---------------------------------------------------------------------
def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength, 0..100)."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_.replace(0.0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_.replace(0.0, np.nan)
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_


# ---------------------------------------------------------------------
# MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------------------
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------------------------------------------------------------------
# Keltner channels (EMA + ATR bands) — used by the TTM squeeze
# ---------------------------------------------------------------------
def keltner(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, mult: float = 1.5):
    mid = ema(df["close"], ema_period)
    a = atr(df, atr_period)
    upper = mid + mult * a
    lower = mid - mult * a
    return mid, upper, lower


# ---------------------------------------------------------------------
# Dual Thrust range (Michael Chalek): max(HH-LC, HC-LL) over prior N bars
# ---------------------------------------------------------------------
def dual_thrust_range(df: pd.DataFrame, n: int = 4) -> pd.Series:
    """Breakout range built from the PRIOR n bars (shifted, no look-ahead)."""
    hh = df["high"].rolling(n, min_periods=n).max().shift(1)
    ll = df["low"].rolling(n, min_periods=n).min().shift(1)
    hc = df["close"].rolling(n, min_periods=n).max().shift(1)
    lc = df["close"].rolling(n, min_periods=n).min().shift(1)
    return pd.concat([(hh - lc), (hc - ll)], axis=1).max(axis=1)


# ---------------------------------------------------------------------
# Annualized realized volatility (for reporting)
# ---------------------------------------------------------------------
def realized_vol(close: pd.Series, period: int = 20, ann: int = 252) -> pd.Series:
    rets = np.log(close / close.shift(1))
    return rets.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(ann)
