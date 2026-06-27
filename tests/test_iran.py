import numpy as np
import pandas as pd

from quantaura import data as data_mod
from quantaura.config import Settings
from quantaura.data import _parse_tgju, asset_class_of
from quantaura.models import AssetClass, Side


SAMPLE = {"total": 4, "data": [
    ["739,500", "738,000", "741,200", "740,800", "800", "0.11",
     "<span class=\"x\">2024/01/17</span>", "1402/10/27"],
    ["738,700", "737,000", "740,000", "739,500", "500", "0.07", "2024/01/16", "1402/10/26"],
    ["737,000", "735,500", "739,000", "738,700", "1200", "0.16", "2024/01/15", "1402/10/25"],
    ["bad", "row", "ignored", "x", "", "", "", ""],     # unparseable -> skipped
]}


def test_parse_tgju_basic():
    df = _parse_tgju(SAMPLE, 400)
    assert len(df) == 3                        # bad row skipped
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing    # sorted ascending
    assert df.index[0].year == 2024            # gregorian, not jalali (1402)
    assert (df["high"] >= df["low"]).all()
    assert df["open"].iloc[-1] == 739500.0


def test_parse_tgju_empty_raises():
    import pytest
    with pytest.raises(data_mod.DataError):
        _parse_tgju({"data": []}, 100)


def test_iran_routing():
    uni = {"iran": ["price_dollar_rl", "geram18"]}
    assert asset_class_of("price_dollar_rl", uni) is AssetClass.IRAN
    assert asset_class_of("geram18", {}) is AssetClass.IRAN        # via IRAN_NAMES
    assert asset_class_of("AAPL", uni) is AssetClass.STOCK


def _iran_df(n=320, seed=1):
    # a steadily devaluing rial -> strong uptrend in USD/gold (rial terms)
    rng = np.random.default_rng(seed)
    close = 500000 * np.exp(np.cumsum(rng.normal(0.0015, 0.01, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    hi = close * (1 + rng.uniform(0, 0.004, n))
    lo = close * (1 - rng.uniform(0, 0.004, n))
    op = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": op, "high": np.maximum.reduce([op, hi, close]),
         "low": np.minimum.reduce([op, lo, close]), "close": close,
         "volume": np.zeros(n)}, index=idx)


def test_iran_scan_is_long_only(monkeypatch):
    from quantaura import engine
    settings = Settings.load()
    monkeypatch.setattr(engine.data_mod, "get_ohlcv", lambda *a, **k: _iran_df())
    sigs = engine.scan_symbol("price_dollar_rl", AssetClass.IRAN, settings, publish_only=False)
    # long-only filter (config iran.long_only) must drop any SHORT signal
    assert all(s.side is Side.LONG for s in sigs)
