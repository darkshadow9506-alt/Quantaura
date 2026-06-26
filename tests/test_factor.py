import numpy as np
import pandas as pd

from quantaura import factor as factor_mod
from quantaura.models import Side


def _panel(n=400, k=6, seed=20):
    frames = {}
    for j in range(k):
        rng = np.random.default_rng(seed + j)
        drift = 0.0003 * (j - 2)        # different trend strengths
        steps = rng.normal(drift, 0.012, n)
        close = 100 * np.exp(np.cumsum(steps))
        idx = pd.date_range("2021-01-01", periods=n, freq="B")
        high = close * (1 + rng.uniform(0, 0.003, n))
        low = close * (1 - rng.uniform(0, 0.003, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        frames[f"SYM{j}"] = pd.DataFrame(
            {"open": open_, "high": np.maximum.reduce([open_, high, close]),
             "low": np.minimum.reduce([open_, low, close]), "close": close,
             "volume": np.full(n, 1e6)}, index=idx)
    return frames


CFG = {"lookback_days": 126, "skip_days": 21, "rebalance_days": 21,
       "top_n": 2, "bottom_n": 2, "allow_short": True, "atr_period": 14,
       "atr_stop_mult": 3.0, "min_target_R": 2.0,
       "min_basket_rebalances": 5, "min_basket_winrate": 0.0,
       "min_basket_profit_factor": 0.0}


def test_panel_backtest_runs():
    frames = _panel()
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    stats = factor_mod.backtest_cross_sectional(panel, CFG)
    assert stats.trades > 0
    assert 0.0 <= stats.win_rate <= 1.0


def test_rank_live_long_short_legs():
    frames = _panel()
    legs = factor_mod.rank_live(frames, CFG)
    assert len(legs) == 4                       # 2 long + 2 short
    assert {l.side for l in legs} == {Side.LONG, Side.SHORT}
    assert all(l.valid() for l in legs)
    # the strongest long must out-score the weakest short
    longs = [l for l in legs if l.side is Side.LONG]
    shorts = [l for l in legs if l.side is Side.SHORT]
    assert max(l.score for l in longs) > min(s.score for s in shorts)


def test_factor_long_only_mode():
    cfg = dict(CFG, allow_short=False, bottom_n=0)
    frames = _panel()
    legs = factor_mod.rank_live(frames, cfg)
    assert all(l.side is Side.LONG for l in legs)


def test_gate_logic():
    good = factor_mod.passes_factor_gate(
        type("S", (), {"trades": 30, "win_rate": 0.6, "profit_factor": 1.5})(),
        {"min_basket_rebalances": 24, "min_basket_winrate": 0.45,
         "min_basket_profit_factor": 1.15})
    bad = factor_mod.passes_factor_gate(
        type("S", (), {"trades": 10, "win_rate": 0.6, "profit_factor": 1.5})(),
        {"min_basket_rebalances": 24, "min_basket_winrate": 0.45,
         "min_basket_profit_factor": 1.15})
    assert good and not bad
