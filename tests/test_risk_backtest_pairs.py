import numpy as np
import pandas as pd

from quantaura import pairs as pairs_mod
from quantaura import risk as risk_mod


def test_fixed_fractional_risk_cap():
    units = risk_mod.fixed_fractional_units(equity=10000, risk_pct=1.0, risk_per_unit=5.0)
    # losing the stop must cost exactly 1% = $100 -> 20 units
    assert abs(units - 20.0) < 1e-9
    assert abs(units * 5.0 - 100.0) < 1e-9


def test_kelly_zero_on_no_edge():
    # 50% win rate with 1:1 payoff -> zero edge -> f=0
    assert risk_mod.kelly_fraction(0.5, 1.0, 1.0) == 0.0
    # negative-expectancy -> clipped to 0
    assert risk_mod.kelly_fraction(0.3, 1.0, 1.0) == 0.0


def test_kelly_positive_on_edge():
    f = risk_mod.kelly_fraction(0.6, 2.0, 1.0)  # b=2 -> f = .6 - .4/2 = .4
    assert abs(f - 0.4) < 1e-9


def test_size_position_takes_min_and_caps():
    res = risk_mod.size_position(
        equity=10000, entry=100, risk_per_unit=2.0, win_rate=0.6,
        avg_win_R=2.0, avg_loss_R=1.0, risk_per_trade_pct=1.0,
        kelly_fraction_mult=0.5, max_kelly_pct=5.0)
    assert res.units >= 0
    assert res.risk_amount <= 100.0 + 1e-6      # never exceeds fixed-fractional cap
    assert res.notional <= 10000 * 0.05 + 1e-6  # never exceeds max-Kelly notional


def test_confidence_monotone():
    low = risk_mod.confidence_from_backtest(0.40, 1.1, 0.2)
    high = risk_mod.confidence_from_backtest(0.70, 2.0, 1.5)
    assert 0.0 <= low <= high <= 1.0


def _coint_pair(n=400, seed=3):
    rng = np.random.default_rng(seed)
    b = 50 + np.cumsum(rng.normal(0, 0.5, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.6 * spread[t - 1] + rng.normal(0, 1.0)
    a = 1.8 * b + 10 + spread
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


def test_pairs_backtest_and_eval():
    a, b = _coint_pair()
    cfg = {"lookback": 252, "coint_pvalue_max": 0.10, "zscore_entry": 2.0,
           "zscore_exit": 0.5, "zscore_stop": 3.5}
    stats = pairs_mod.backtest_pair(a, b, cfg)
    assert stats.trades >= 0
    assert np.isfinite(stats.win_rate)
    plan = pairs_mod.evaluate_pair("A", "B", a, b, cfg, atr_a=1.0)
    if plan is not None:
        assert plan.valid()
        assert plan.coint_pvalue <= 0.10


def test_non_cointegrated_pair_rejected():
    rng = np.random.default_rng(7)
    a = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))
    b = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))  # independent walk
    cfg = {"lookback": 252, "coint_pvalue_max": 0.01, "zscore_entry": 2.0,
           "zscore_exit": 0.5, "zscore_stop": 3.5}
    plan = pairs_mod.evaluate_pair("A", "B", a, b, cfg)
    # independent random walks are essentially never cointegrated at p<=0.01
    assert plan is None


def test_coint_fallback_without_statsmodels(monkeypatch):
    """A missing statsmodels must not break pairs: numpy ADF takes over."""
    import builtins
    real_import = builtins.__import__

    def _no_statsmodels(name, *a, **k):
        if name.startswith("statsmodels"):
            raise ImportError("statsmodels hidden for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_statsmodels)
    a, b = _coint_pair()
    rng = np.random.default_rng(7)
    a2 = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))
    b2 = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))  # independent

    p_coint = pairs_mod._coint_pvalue(a, b)
    p_indep = pairs_mod._coint_pvalue(a2, b2)
    assert 0.0 <= p_coint <= 1.0 and 0.0 <= p_indep <= 1.0
    # the fallback must still discriminate: cointegrated << independent
    assert p_coint < p_indep
    # and the backtest must run end-to-end with no statsmodels present
    stats = pairs_mod.backtest_pair(a, b, {"lookback": 200})
    assert np.isfinite(stats.win_rate) and stats.trades >= 0
