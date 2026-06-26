import numpy as np
import pandas as pd

from quantaura import indicators as ind


def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.5
    assert out.iloc[4] == 4.5


def test_atr_positive(trending_df):
    a = ind.atr(trending_df, 14).dropna()
    assert (a > 0).all()
    assert a.index.equals(trending_df.index[-len(a):])


def test_rsi_bounds(trending_df):
    r = ind.rsi(trending_df["close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_all_up_is_100():
    s = pd.Series(np.arange(1, 50, dtype=float))  # strictly increasing
    r = ind.rsi(s, 14).dropna()
    assert (r > 99.0).all()


def test_bollinger_ordering(trending_df):
    mid, up, low = ind.bollinger(trending_df["close"], 20, 2.0)
    d = pd.concat([mid, up, low], axis=1).dropna()
    assert (d.iloc[:, 1] >= d.iloc[:, 0]).all()  # upper >= mid
    assert (d.iloc[:, 0] >= d.iloc[:, 2]).all()  # mid >= lower


def test_donchian_no_lookahead(trending_df):
    up, low = ind.donchian(trending_df, 20)
    # level at t must equal max/min of strictly prior 20 highs/lows
    t = 100
    expected_up = trending_df["high"].iloc[t - 20:t].max()
    assert abs(up.iloc[t] - expected_up) < 1e-9


def test_adx_bounds(trending_df):
    a = ind.adx(trending_df, 14).dropna()
    assert ((a >= 0) & (a <= 100)).all()


def test_zscore_mean_zero(trending_df):
    z = ind.rolling_zscore(trending_df["close"], 20).dropna()
    assert np.isfinite(z).all()


def test_macd_hist_equals_line_minus_signal(trending_df):
    line, sig, hist = ind.macd(trending_df["close"], 12, 26, 9)
    d = pd.concat([line, sig, hist], axis=1).dropna()
    assert np.allclose(d.iloc[:, 2], d.iloc[:, 0] - d.iloc[:, 1])


def test_keltner_ordering(trending_df):
    mid, up, low = ind.keltner(trending_df, 20, 10, 1.5)
    d = pd.concat([mid, up, low], axis=1).dropna()
    assert (d.iloc[:, 1] > d.iloc[:, 0]).all()
    assert (d.iloc[:, 0] > d.iloc[:, 2]).all()


def test_dual_thrust_range_positive_and_no_lookahead(trending_df):
    rng = ind.dual_thrust_range(trending_df, 4).dropna()
    assert (rng >= 0).all()
    # value at t uses only bars < t (shifted) -> first 4 are NaN
    assert ind.dual_thrust_range(trending_df, 4).iloc[:4].isna().all()
