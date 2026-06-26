"""Live signal tracking — resolve open signals against real price action.

Pure resolvers (`resolve_bar`, `resolve_path`) decide whether a signal
hit its take-profit or stop given subsequent bars, using the same
pessimistic same-bar rule as the backtester (if a bar spans both, the
stop is assumed first). `update_open_signals` ties this to live data and
the SQLite store, so the bot can report a real, honest track record.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .models import AssetClass


def resolve_bar(side: str, entry: float, stop: float, target: float,
                high: float, low: float):
    """Outcome of one bar. Returns (status, result_R) or (None, None)."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 0.0
    if side == "LONG":
        hit_stop = low <= stop
        hit_tp = high >= target
    else:
        hit_stop = high >= stop
        hit_tp = low <= target
    if hit_stop:                 # pessimistic: stop wins a tie
        return "sl", -1.0
    if hit_tp:
        return "tp", rr
    return None, None


def resolve_path(side: str, entry: float, stop: float, target: float,
                 bars):  # bars: iterable of (high, low) in chronological order
    for high, low in bars:
        status, r = resolve_bar(side, entry, stop, target, high, low)
        if status:
            return status, r
    return None, None


def _mark_to_market_R(side: str, entry: float, stop: float, last_close: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    move = (last_close - entry) if side == "LONG" else (entry - last_close)
    return round(move / risk, 4)


def update_open_signals(store, settings, max_hold_days: int = 30) -> dict:
    """Resolve every open signal against bars printed after it was issued.

    Returns a summary dict {checked, closed_tp, closed_sl, expired}.
    """
    from . import data as data_mod

    uni = settings.universe
    d = settings.data
    summary = {"checked": 0, "tp": 0, "sl": 0, "expired": 0}

    for row in store.open_signals():
        summary["checked"] += 1
        try:
            ac = AssetClass(row["asset_class"])
            created = datetime.fromisoformat(row["created_at"])
        except (ValueError, KeyError):
            continue
        try:
            df = data_mod.get_ohlcv(
                row["symbol"], ac, timeframe=d.get("timeframe", "1d"),
                lookback=int(d.get("lookback_bars", 1250)),
                cache_minutes=int(d.get("cache_minutes", 30)),
                ccxt_exchange=settings.ccxt_exchange,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue

        # bars strictly after the signal's timestamp (handle tz-aware crypto
        # indices and tz-naive equity/FX indices uniformly)
        import pandas as pd

        created_pd = pd.Timestamp(created)
        if df.index.tz is not None:
            cmp = (created_pd.tz_convert(df.index.tz) if created_pd.tz is not None
                   else created_pd.tz_localize(df.index.tz))
        else:
            cmp = created_pd.tz_localize(None) if created_pd.tz is not None else created_pd
        after = df[df.index > cmp]
        if after.empty:
            continue

        bars = list(zip(after["high"].tolist(), after["low"].tolist()))
        status, r = resolve_path(row["side"], row["entry"], row["stop"],
                                 row["target"], bars)
        if status:
            store.close_signal(row["id"], status, r)
            summary[status] += 1
            continue

        # expire stale, still-open signals (mark to market)
        age_days = (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).days
        if age_days >= max_hold_days:
            r = _mark_to_market_R(row["side"], row["entry"], row["stop"],
                                  float(after["close"].iloc[-1]))
            store.close_signal(row["id"], "expired", r)
            summary["expired"] += 1

    return summary


def format_performance(perf: dict) -> str:
    if perf["total"] == 0:
        return ("📒 *Track record*\nNo signals recorded yet. Run a scan and the "
                "bot will start journaling every published signal.")
    return (
        "📒 *Track record (live)*\n"
        f"Total signals: {perf['total']}  |  open: {perf['open']}  |  "
        f"closed: {perf['closed']}\n"
        f"Hit target: {perf['tp']}  |  stopped: {perf['sl']}\n"
        f"Win rate: {perf['win_rate']*100:.0f}%  |  avg {perf['avg_R']:+.2f}R  |  "
        f"total {perf['total_R']:+.1f}R"
    )
