"""Smart-Money-Concepts structure detection (quantified, look-ahead-free).

These are deliberately *mechanical* approximations of discretionary SMC /
ICT ideas. They will not match a chartist's hand-drawn reading exactly —
that part is subjective — but they capture the quantifiable core and feed
the same support/resistance machinery used for stop and target placement.

  * Fair Value Gap (FVG): a 3-bar imbalance.
      bullish (support) at bar i when high[i-2] < low[i]
      bearish (resistance) at bar i when low[i-2]  > high[i]
    Confirmed at bar i (uses bars i-2..i only) -> no look-ahead.

  * Order Block (OB): the last opposite candle before a displacement.
      bullish (support):  bar i closes above bar i-1's high AND bar i-1
                          was a down candle -> i-1's low is the OB support.
      bearish (resistance): bar i closes below bar i-1's low AND bar i-1
                          was an up candle -> i-1's high is the OB resistance.

  * Swing pivots double as the liquidity pools (stops cluster just beyond
    prior swing highs/lows); placing stops beyond them avoids the sweep.

All detectors return Series carrying the level price at the bar where the
structure is established (NaN elsewhere).
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind


def fair_value_gaps(df: pd.DataFrame):
    """Return (fvg_support, fvg_resistance) level Series."""
    high, low = df["high"], df["low"]
    h2, l2 = high.shift(2), low.shift(2)
    bullish = h2 < low          # gap up -> support around high[i-2]
    bearish = l2 > high         # gap down -> resistance around low[i-2]
    return h2.where(bullish), l2.where(bearish)


def order_blocks(df: pd.DataFrame):
    """Return (ob_support, ob_resistance) level Series."""
    open_, close = df["open"], df["close"]
    high, low = df["high"], df["low"]
    prev_high, prev_low = high.shift(1), low.shift(1)
    prev_down = close.shift(1) < open_.shift(1)
    prev_up = close.shift(1) > open_.shift(1)
    up_disp = close > prev_high      # displacement up
    down_disp = close < prev_low     # displacement down
    ob_sup = prev_low.where(up_disp & prev_down)
    ob_res = prev_high.where(down_disp & prev_up)
    return ob_sup, ob_res


def add_levels(df: pd.DataFrame, swing_width: int = 3) -> pd.DataFrame:
    """Attach all structural support/resistance level columns to a copy."""
    d = df
    piv_low, piv_high = ind.swing_pivots(d, swing_width)
    fvg_sup, fvg_res = fair_value_gaps(d)
    ob_sup, ob_res = order_blocks(d)
    d["piv_low"], d["piv_high"] = piv_low, piv_high
    d["fvg_sup"], d["fvg_res"] = fvg_sup, fvg_res
    d["ob_sup"], d["ob_res"] = ob_sup, ob_res
    return d


# columns that act as resistance (above price) and support (below price)
RES_COLS = ("piv_high", "fvg_res", "ob_res")
SUP_COLS = ("piv_low", "fvg_sup", "ob_sup")


def collect_levels(df: pd.DataFrame, lo: int, hi: int, cols) -> list[float]:
    """All non-NaN level prices in df[cols] over the bar window [lo, hi]."""
    out: list[float] = []
    for c in cols:
        if c in df.columns:
            out.extend(df[c].iloc[lo:hi + 1].dropna().tolist())
    return out
