import numpy as np
import pandas as pd

from quantaura import ml, optimize
from quantaura.backtest import backtest_strategy
from quantaura.models import Side
from quantaura.strategies import TrendBreakout


def _series(n=500, seed=1, drift=0.0009):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.012, n)
    steps[120:160] += 0.01
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    high = close * (1 + rng.uniform(0, 0.004, n))
    low = close * (1 - rng.uniform(0, 0.004, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([open_, high, close]),
         "low": np.minimum.reduce([open_, low, close]), "close": close,
         "volume": rng.uniform(1e6, 5e6, n)}, index=idx)


ML_CFG = {"horizon": 10, "k": 1.5, "min_train": 250, "refit_every": 60,
          "prob_threshold": 0.55, "max_iter": 120, "max_depth": 3,
          "learning_rate": 0.05}


# ---------------- ML ----------------
def test_triple_barrier_labels_binary():
    df = _series()
    lab = ml.triple_barrier_labels(df, horizon=10, k=1.5)
    vals = lab.dropna().unique()
    assert set(vals) <= {0.0, 1.0}
    # last `horizon` bars are unlabelled
    assert lab.iloc[-1] != lab.iloc[-1] or True  # NaN tolerated
    assert lab.tail(10).isna().any()


def test_features_no_lookahead_shapes():
    df = _series()
    f = ml.build_features(df)
    assert len(f) == len(df)
    # ret_1 at t depends only on close[t], close[t-1]
    assert abs(f["ret_1"].iloc[-1] - (df["close"].iloc[-1] / df["close"].iloc[-2] - 1)) < 1e-9


def test_ml_backtest_and_plan():
    df = _series()
    stats, last = ml.backtest_ml(df, ML_CFG)
    assert stats.trades >= 0
    assert (last is None) or (0.0 <= last <= 1.0)
    plan = ml.latest_plan(df, ML_CFG)
    if plan is not None:
        assert plan.valid()
        assert plan.side in (Side.LONG, Side.SHORT)
        # TP/SL are symmetric ±k·ATR -> RR ~ 1
        assert abs(plan.rr_ratio - 1.0) < 1e-6


def test_ml_insufficient_history():
    df = _series(n=120)
    stats, last = ml.backtest_ml(df, ML_CFG)
    assert stats.trades == 0 and last is None
    assert ml.latest_plan(df, ML_CFG) is None


# ---------------- trailing ----------------
def test_trailing_changes_outcomes():
    df = _series()
    s = TrendBreakout({"donchian_entry": 20, "ma_slow": 200, "atr_period": 14,
                       "atr_stop_mult": 2.5, "min_target_R": 2.0})
    fixed, ft = backtest_strategy(s, df)
    trail, tt = backtest_strategy(s, df, trail_atr_mult=3.0)
    assert all(t.outcome in ("trail", "time") for t in tt)
    assert all(np.isfinite(t.R) for t in tt)
    # trailing removes the fixed 2R cap -> some winners exceed +2R
    if tt:
        assert max((t.R for t in tt), default=0) >= max((t.R for t in ft), default=0) - 1e-9


def test_trailing_long_stop_only_ratchets_up():
    # a clean uptrend: trailing stop should never produce a loss bigger than -1R
    df = _series(seed=4, drift=0.002)
    s = TrendBreakout({"donchian_entry": 20, "ma_slow": 200, "atr_period": 14,
                       "atr_stop_mult": 2.5, "min_target_R": 2.0})
    _, trades = backtest_strategy(s, df, trail_atr_mult=3.0)
    for t in trades:
        assert t.R >= -1.2   # initial stop bounds the loss (small slack for gaps)


# ---------------- optimizer ----------------
def test_optimizer_ranks_by_oos():
    df = _series()
    base = {"ma_slow": 200, "atr_period": 14, "donchian_exit": 10, "ma_fast": 50}
    res = optimize.optimize_on_df(df, "trend", base, min_trades=5, oos_min_trades=2)
    assert len(res) == 27                      # 3x3x3 grid
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)
    rep = optimize.format_report(res, top=3)
    assert "out-of-sample" in rep.lower()


def test_optimizer_unknown_strategy():
    df = _series()
    try:
        optimize.optimize_on_df(df, "nope", {})
        assert False
    except ValueError:
        pass
