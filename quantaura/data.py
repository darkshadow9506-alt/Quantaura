"""Market data providers.

A single function -- `get_ohlcv(symbol, asset_class, ...)` -- returns a
normalized DataFrame with columns [open, high, low, close, volume] and a
tz-aware DatetimeIndex, regardless of the underlying source.

Sources:
  * stocks / forex / indices  -> yfinance
  * crypto                    -> ccxt (default exchange: binance)

Network libraries are imported lazily so the analytics/backtest code and
the unit tests run without them installed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import AssetClass

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

_COLUMNS = ["open", "high", "low", "close", "volume"]

# ccxt timeframe + yfinance interval mapping
_CCXT_TF = {"1d": "1d", "1h": "1h", "4h": "4h", "15m": "15m", "1w": "1w"}
_YF_INTERVAL = {"1d": "1d", "1h": "1h", "1wk": "1wk", "1w": "1wk"}


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("=", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _read_cache(key: str, max_age_minutes: int) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age_min = (time.time() - path.stat().st_mtime) / 60.0
    if age_min > max_age_minutes:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(key))
    except Exception:
        # parquet engine may be missing; caching is best-effort only
        pass


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, keep OHLCV, drop incomplete rows, sort."""
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    keep = [c for c in _COLUMNS if c in df.columns]
    df = df[keep].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[c for c in ("open", "high", "low", "close") if c in df.columns])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ---------------------------------------------------------------------
# yfinance (stocks / forex)
# ---------------------------------------------------------------------
def _fetch_yfinance(symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    import yfinance as yf  # lazy

    interval = _YF_INTERVAL.get(timeframe, "1d")
    # request generous period; yfinance caps intraday history
    if interval.endswith(("m", "h")):
        period = "60d" if interval == "1h" else "7d"
    else:
        # ~ lookback trading days plus buffer, capped at "max"
        days = min(int(lookback * 1.6) + 30, 4000)
        period = f"{days}d"

    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned no data for {symbol}")
    # yfinance can return a MultiIndex column frame for single tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = _normalize(raw)
    return df.tail(lookback)


# ---------------------------------------------------------------------
# ccxt (crypto)
# ---------------------------------------------------------------------
def _fetch_ccxt(symbol: str, timeframe: str, lookback: int, exchange_id: str) -> pd.DataFrame:
    import ccxt  # lazy

    tf = _CCXT_TF.get(timeframe, "1d")
    ex_class = getattr(ccxt, exchange_id)
    exchange = ex_class({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=lookback)
    if not raw:
        raise DataError(f"ccxt returned no data for {symbol} on {exchange_id}")
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return _normalize(df).tail(lookback)


class DataError(RuntimeError):
    """Raised when a provider cannot return usable data."""


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_ohlcv(
    symbol: str,
    asset_class: AssetClass,
    timeframe: str = "1d",
    lookback: int = 400,
    cache_minutes: int = 30,
    ccxt_exchange: str = "binance",
) -> pd.DataFrame:
    """Return normalized OHLCV for one symbol. Retries with backoff."""
    key = f"{asset_class.value}_{symbol}_{timeframe}"
    cached = _read_cache(key, cache_minutes)
    if cached is not None and not cached.empty:
        return cached

    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            if asset_class is AssetClass.CRYPTO:
                df = _fetch_ccxt(symbol, timeframe, lookback, ccxt_exchange)
            else:
                df = _fetch_yfinance(symbol, timeframe, lookback)
            if df.empty:
                raise DataError(f"empty frame for {symbol}")
            _write_cache(key, df)
            return df
        except Exception as exc:  # network / parsing
            last_err = exc
            time.sleep(2 ** attempt)  # 1,2,4,8s backoff
    raise DataError(f"failed to fetch {symbol}: {last_err}")


def asset_class_of(symbol: str, universe: dict[str, list[str]]) -> AssetClass:
    """Resolve which asset class a symbol belongs to from the configured universe."""
    if symbol in universe.get("crypto", []) or "/" in symbol:
        return AssetClass.CRYPTO
    if symbol in universe.get("forex", []) or symbol.endswith("=X"):
        return AssetClass.FOREX
    return AssetClass.STOCK
