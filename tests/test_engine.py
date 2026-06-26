"""End-to-end engine tests with injected (synthetic) data — no network.

These prove the full orchestration: data -> regime -> strategy ->
backtest gate -> sizing -> Signal, including failure robustness.
"""
import numpy as np
import pandas as pd
import pytest

from quantaura import engine
from quantaura.config import Settings
from quantaura.models import AssetClass, Side


def _rising(n=320, seed=5):
    rng = np.random.default_rng(seed)
    # steady uptrend with mild noise: makes fresh highs -> trend breakouts,
    # but with pullbacks so the backtest has a realistic mix of outcomes.
    steps = rng.normal(0.0012, 0.01, n)
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([open_, high, close]),
         "low": np.minimum.reduce([open_, low, close]), "close": close,
         "volume": rng.uniform(1e6, 5e6, n)}, index=idx)


@pytest.fixture
def settings():
    return Settings.load()


def test_scan_symbol_orchestration(monkeypatch, settings):
    df = _rising()
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: df)

    sigs = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=False)
    assert isinstance(sigs, list)
    for s in sigs:
        assert s.is_valid_geometry()
        assert 0.0 <= s.confidence <= 1.0
        assert s.position_units >= 0
        assert s.risk_per_unit > 0
        if s.side is Side.LONG:
            assert s.stop < s.entry < s.target
        # backtest stats attached
        assert s.backtest.trades >= 0
        # Monte Carlo + out-of-sample layers attached and sane
        assert 0.0 <= s.montecarlo.win_prob <= 1.0
        assert 0.0 <= s.montecarlo.prob_profitable <= 1.0
        assert 0.0 <= s.montecarlo.risk_of_ruin <= 1.0
        assert s.confluence >= 1


def test_scan_symbol_robust_to_bad_data(monkeypatch, settings):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", boom)
    # scan_universe must swallow per-symbol errors and never crash
    out = engine.scan_universe(settings, classes=["stocks"], include_pairs=False)
    assert out == []


def test_scan_symbol_short_history_returns_empty(monkeypatch, settings):
    short = _rising(n=30)
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: short)
    out = engine.scan_symbol("FAKE", AssetClass.STOCK, settings, publish_only=False)
    assert out == []


def test_scan_factor_builds_signals(monkeypatch, settings):
    # distinct trend per symbol so ranking is non-degenerate
    def fake(symbol, asset_class, **k):
        h = (abs(hash(symbol)) % 7) - 3
        return _rising(n=400, seed=100 + h)
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", fake)
    sigs = engine.scan_factor(settings, publish_only=False)
    assert any(s.strategy == "factor_momentum" for s in sigs)
    for s in sigs:
        assert s.is_valid_geometry()
        assert s.risk_per_unit > 0
        assert 0.0 <= s.confidence <= 1.0
        assert s.regime == "cross-sectional"
