import numpy as np
import pandas as pd

from quantaura import smc
from quantaura.models import Side
from quantaura.strategies import _refine_stop, _stop_and_target


def test_bullish_fvg():
    # bar 2 bullish FVG: high[0] < low[2] -> support at high[0]
    df = pd.DataFrame({
        "open": [10, 11, 13, 13], "high": [10.5, 12, 14, 14],
        "low": [9.5, 11, 11.5, 12], "close": [10, 11.5, 13.5, 13]})
    sup, res = smc.fair_value_gaps(df)
    assert abs(sup.iloc[2] - 10.5) < 1e-9
    assert np.isnan(res.iloc[2])


def test_bearish_fvg():
    # bar 2 bearish FVG: low[0] > high[2] -> resistance at low[0]
    df = pd.DataFrame({
        "open": [20, 19, 17, 17], "high": [20.5, 19.5, 16.5, 17],
        "low": [19.5, 18, 16, 16.5], "close": [20, 18.5, 16.2, 17]})
    sup, res = smc.fair_value_gaps(df)
    assert abs(res.iloc[2] - 19.5) < 1e-9


def test_order_blocks():
    # bar3 closes above prev high, prev was down candle -> bullish OB = prev low
    df = pd.DataFrame({
        "open": [10, 12, 11.0, 11], "high": [10.5, 12.5, 11.2, 13],
        "low": [9.5, 11.5, 10.5, 11], "close": [10, 12, 10.7, 13]})
    obs, obr = smc.order_blocks(df)
    assert abs(obs.iloc[3] - 10.5) < 1e-9


def test_add_levels_columns():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 60))
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                       "close": close, "volume": np.full(60, 1e6)})
    smc.add_levels(df, 3)
    for c in ("piv_low", "piv_high", "fvg_sup", "fvg_res", "ob_sup", "ob_res"):
        assert c in df.columns


def _levels_df(marks, n=130):
    """marks: list of (column, bar_index, price)."""
    cols = ("piv_low", "piv_high", "fvg_sup", "fvg_res", "ob_sup", "ob_res")
    d = pd.DataFrame({c: [np.nan] * n for c in cols})
    for col, bar, price in marks:
        d.loc[bar, col] = price
    return d


SCFG = {"enabled": True, "structural_stop": True, "swing_width": 3,
        "buffer_atr": 0.25, "lookback": 120, "stop_min_atr": 0.8, "stop_max_atr": 4.0,
        "min_rr": 0.8}


def test_structural_stop_short_beyond_resistance():
    d = _levels_df([("piv_high", 50, 210.0)])
    stop = _refine_stop(d, 100, Side.SHORT, 200.0, 4.0, base_stop=210.0, scfg=SCFG)
    assert abs(stop - 211.0) < 1e-9            # 210 + 0.25*4


def test_structural_stop_long_below_support_via_fvg():
    d = _levels_df([("fvg_sup", 50, 190.0)])
    stop = _refine_stop(d, 100, Side.LONG, 200.0, 4.0, base_stop=190.0, scfg=SCFG)
    assert abs(stop - 189.0) < 1e-9            # 190 - 0.25*4


def test_structural_stop_too_far_keeps_blind():
    d = _levels_df([("piv_high", 60, 230.0)])
    stop = _refine_stop(d, 100, Side.SHORT, 200.0, 4.0, base_stop=210.0, scfg=SCFG)
    assert stop == 210.0                       # 30 > 4*ATR -> keep blind


def test_structural_stop_too_close_keeps_blind():
    d = _levels_df([("piv_high", 60, 201.0)])
    stop = _refine_stop(d, 100, Side.SHORT, 200.0, 4.0, base_stop=210.0, scfg=SCFG)
    assert stop == 210.0                       # 1 < 0.8*ATR(=3.2) -> keep blind


def test_stop_and_target_geometry():
    # support below for a long, resistance above for stop
    d = _levels_df([("piv_low", 50, 188.0)])
    stop, target = _stop_and_target(d, 100, Side.SHORT, 200.0, 4.0,
                                    atr_stop_mult=2.5, min_target_R=2.0, scfg=SCFG)
    assert target < 200.0 < stop               # valid SHORT geometry
    assert target > 0


def test_empty_structure_is_backcompat():
    d = _levels_df([("piv_high", 50, 210.0)])
    assert _refine_stop(d, 100, Side.SHORT, 200.0, 4.0, base_stop=210.0, scfg={}) == 210.0
