import numpy as np
import pandas as pd

from quantaura.backtest import backtest_strategy
from quantaura.models import Side
from quantaura.strategies import MeanReversion, TrendBreakout, detect_regime


def _ohlc(close, seed=0, noise=0.001):
    rng = np.random.default_rng(seed)
    close = np.asarray(close, float)
    idx = pd.date_range("2021-01-01", periods=len(close), freq="B")
    high = close * (1 + rng.uniform(0, noise, len(close)))
    low = close * (1 - rng.uniform(0, noise, len(close)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([open_, high, close]),
         "low": np.minimum.reduce([open_, low, close]), "close": close,
         "volume": np.full(len(close), 1e6)}, index=idx)


def test_trend_long_geometry(trending_df):
    s = TrendBreakout({"donchian_entry": 20, "donchian_exit": 10, "ma_fast": 50,
                       "ma_slow": 200, "atr_period": 14, "atr_stop_mult": 2.5,
                       "min_target_R": 2.0})
    prepared = s.prepare(trending_df)
    # any plan produced over history must have valid geometry
    plans = [s.evaluate(prepared, i) for i in range(len(prepared))]
    plans = [p for p in plans if p is not None]
    assert all(p.valid() for p in plans)
    for p in plans:
        if p.side is Side.LONG:
            assert p.stop < p.entry < p.target
        else:
            assert p.target < p.entry < p.stop


def test_trend_target_is_2R(trending_df):
    s = TrendBreakout({"donchian_entry": 20, "ma_slow": 200, "atr_period": 14,
                       "atr_stop_mult": 2.5, "min_target_R": 2.0})
    prepared = s.prepare(trending_df)
    for i in range(len(prepared)):
        p = s.evaluate(prepared, i)
        if p:
            assert abs(p.rr_ratio - 2.0) < 1e-6


def test_backtest_produces_trades(trending_df):
    s = TrendBreakout({"donchian_entry": 20, "ma_slow": 200, "atr_period": 14,
                       "atr_stop_mult": 2.5, "min_target_R": 2.0})
    stats, trades = backtest_strategy(s, trending_df)
    assert stats.trades == len(trades)
    assert all(np.isfinite(t.R) for t in trades)
    # trades never overlap
    for a, b in zip(trades, trades[1:]):
        assert b.entry_idx > a.exit_idx


def test_mean_reversion_long_fires():
    # steep uptrend then shallow dip -> oversold-in-uptrend long
    close = np.concatenate([np.linspace(100, 150, 240), [147, 144]])
    df = _ohlc(close, seed=9)
    s = MeanReversion({"bb_period": 20, "bb_std": 2.0, "zscore_entry": 2.0,
                       "rsi_period": 2, "rsi_buy_below": 10, "rsi_sell_above": 90,
                       "ma_trend": 100, "atr_period": 14, "atr_stop_mult": 3.0})
    prepared = s.prepare(df)
    plan = s.evaluate(prepared, len(prepared) - 1)
    assert plan is not None and plan.side is Side.LONG
    assert plan.valid()
    assert plan.target == prepared["bb_mid"].iloc[-1]  # target is the mean


def test_regime_label(trending_df):
    r = detect_regime(trending_df, {"adx_period": 14, "adx_trend_threshold": 25,
                                    "adx_range_threshold": 20})
    assert r in ("trending", "ranging", "neutral")


def test_macd_and_dualthrust_geometry(trending_df):
    from quantaura.strategies import MacdTrend, DualThrust
    for s in (MacdTrend({"ma_slow": 200, "atr_period": 14, "atr_stop_mult": 2.5,
                         "min_target_R": 2.0}),
              DualThrust({"range_bars": 4, "k1": 0.5, "k2": 0.5, "ma_slow": 200,
                          "atr_period": 14, "atr_stop_mult": 2.0, "min_target_R": 2.0})):
        prepared = s.prepare(trending_df)
        plans = [s.evaluate(prepared, i) for i in range(len(prepared))]
        plans = [p for p in plans if p is not None]
        assert all(p.valid() for p in plans)


def test_squeeze_fires_on_release():
    from quantaura.strategies import SqueezeBreakout
    rng = np.random.default_rng(0)
    rise = np.linspace(80, 100, 205)
    flat = 100 + rng.normal(0, 0.05, 18)
    close = np.concatenate([rise, flat, np.array([104.0, 107.0])])
    n = len(close)
    hi = close + 0.4; lo = close - 0.4
    hi[205:223] = close[205:223] + 0.4; lo[205:223] = close[205:223] - 0.4
    op = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": op, "high": np.maximum.reduce([op, hi, close]),
                       "low": np.minimum.reduce([op, lo, close]), "close": close,
                       "volume": np.full(n, 1e6)}, index=idx)
    s = SqueezeBreakout({"bb_period": 20, "bb_std": 2.0, "kc_ema": 20, "kc_atr": 10,
                         "kc_mult": 1.5, "ma_slow": 200, "atr_period": 14,
                         "atr_stop_mult": 1.5, "min_target_R": 3.0})
    p = s.prepare(df)
    fired = next((s.evaluate(p, k) for k in range(220, n)
                  if s.evaluate(p, k) is not None), None)
    assert fired is not None and fired.side is Side.LONG and fired.valid()
    assert abs(fired.rr_ratio - 3.0) < 1e-6   # squeeze uses a 3R target


def test_stop_and_target_emits_structure_note():
    import numpy as np
    import pandas as pd
    from quantaura.strategies import _stop_and_target
    n = 130
    d = pd.DataFrame({c: [np.nan] * n for c in
                      ("piv_low", "piv_high", "fvg_sup", "fvg_res", "ob_sup", "ob_res")})
    d.loc[50, "piv_high"] = 108.0      # resistance ahead of a long target
    scfg = {"enabled": True, "structural_stop": True, "swing_width": 3,
            "buffer_atr": 0.25, "lookback": 120, "stop_min_atr": 0.8,
            "stop_max_atr": 4.0, "min_rr": 0.8}
    stop, target, note = _stop_and_target(d, 100, Side.LONG, 100.0, 2.0, 2.5, 2.0, scfg)
    assert "Structure" in note and "resistance" in note and "108" in note
    # no structure -> empty note (back-compat)
    d2 = pd.DataFrame({c: [np.nan] * n for c in
                       ("piv_low", "piv_high", "fvg_sup", "fvg_res", "ob_sup", "ob_res")})
    _, _, note2 = _stop_and_target(d2, 100, Side.LONG, 100.0, 2.0, 2.5, 2.0, scfg)
    assert note2 == ""


def test_structure_target_refinement():
    import numpy as np
    import pandas as pd
    from quantaura.strategies import _refine_target
    n = 130
    df = pd.DataFrame({"piv_low": [np.nan] * n, "piv_high": [np.nan] * n})
    scfg = {"enabled": True, "swing_width": 3, "buffer_atr": 0.25,
            "lookback": 120, "min_rr": 0.8}

    # SHORT: a support at 185 sits between target(180) and entry(200) ->
    # target pulled up to 185 + 0.25*ATR
    df.loc[50, "piv_low"] = 185.0
    t, lvl = _refine_target(df, 100, Side.SHORT, 200.0, 180.0, 4.0, 10.0, scfg)
    assert abs(t - 186.0) < 1e-9 and lvl == 185.0

    # LONG: a resistance at 215 between entry(200) and target(220) ->
    # target pulled down to 215 - 0.25*ATR
    df2 = pd.DataFrame({"piv_low": [np.nan] * n, "piv_high": [np.nan] * n})
    df2.loc[50, "piv_high"] = 215.0
    t2, lvl2 = _refine_target(df2, 100, Side.LONG, 200.0, 220.0, 4.0, 10.0, scfg)
    assert abs(t2 - 214.0) < 1e-9 and lvl2 == 215.0

    # no qualifying level -> mechanical target unchanged
    assert _refine_target(df, 100, Side.SHORT, 200.0, 190.0, 4.0, 10.0, scfg) == (190.0, None)

    # empty/disabled structure -> unchanged (back-compat)
    assert _refine_target(df, 100, Side.SHORT, 200.0, 180.0, 4.0, 10.0, {}) == (180.0, None)
    assert _refine_target(df, 100, Side.SHORT, 200.0, 180.0, 4.0, 10.0,
                          {"enabled": False}) == (180.0, None)

    # refined reward:risk below min_rr -> keep mechanical
    df.loc[60, "piv_low"] = 199.0     # support 1 point below entry -> tiny reward
    df3 = pd.DataFrame({"piv_low": [np.nan] * n, "piv_high": [np.nan] * n})
    df3.loc[60, "piv_low"] = 199.0
    assert _refine_target(df3, 100, Side.SHORT, 200.0, 180.0, 4.0, 10.0, scfg) == (180.0, None)


def test_structure_geometry_stays_valid(trending_df):
    # with structure enabled, any produced plan must still be valid geometry
    from quantaura.strategies import TrendBreakout
    s = TrendBreakout(
        {"donchian_entry": 20, "ma_slow": 200, "atr_period": 14,
         "atr_stop_mult": 2.5, "min_target_R": 2.0},
        structure={"enabled": True, "swing_width": 3, "buffer_atr": 0.25,
                   "lookback": 120, "min_rr": 0.8})
    prepared = s.prepare(trending_df)
    for i in range(len(prepared)):
        p = s.evaluate(prepared, i)
        if p:
            assert p.valid()
            # refined target never exceeds the original 2R distance
            assert p.rr_ratio <= 2.0 + 1e-9


def test_build_strategies_count():
    from quantaura.config import Settings
    from quantaura.strategies import build_strategies
    strats = build_strategies(Settings.load())
    names = {s.name for s in strats}
    assert {"trend_breakout", "mean_reversion", "macd_trend",
            "dual_thrust", "squeeze_breakout"} <= names
