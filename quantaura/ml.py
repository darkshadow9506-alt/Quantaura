"""Machine-learning alpha — gradient boosting with triple-barrier labels.

This is the "supervised learning" strategy from the methodology doc:
gradient boosting (the workhorse of tabular prediction) on engineered
price/volatility features, trained to predict the **triple-barrier**
label of López de Prado — i.e. *will price reach +k·ATR before −k·ATR
within a horizon?*

Honesty by construction:
  * Features are all look-ahead-free (only past data).
  * Labels use a forward window, so during BACKTEST we train with
    **purging**: the model that predicts bar t is fit only on bars whose
    forward label window closed at or before t (no leakage).
  * The trade taken matches the label exactly (TP = +k·ATR, SL = −k·ATR),
    so the model's job is literally to estimate P(win).
  * The resulting rule is then put through the same backtest gate +
    out-of-sample + Monte Carlo checks as every other strategy.

scikit-learn is imported lazily so the rest of the package runs without it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import stats_from_R
from .models import BacktestStats, Side
from .strategies import TradePlan


# ---------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered, look-ahead-free features (one row per bar)."""
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    for w in (1, 3, 5, 10, 20):
        f[f"ret_{w}"] = c.pct_change(w)
    f["rsi_14"] = ind.rsi(c, 14)
    f["rsi_2"] = ind.rsi(c, 2)
    _, _, hist = ind.macd(c)
    f["macd_hist"] = hist / c
    atr = ind.atr(df, 14)
    f["atr_pct"] = atr / c
    f["zscore_20"] = ind.rolling_zscore(c, 20)
    f["adx_14"] = ind.adx(df, 14)
    mid, up, low = ind.bollinger(c, 20, 2.0)
    width = (up - low).replace(0.0, np.nan)
    f["bb_pctb"] = (c - low) / width
    f["dist_ma50"] = c / ind.sma(c, 50) - 1.0
    f["dist_ma200"] = c / ind.sma(c, 200) - 1.0
    f["rvol_20"] = ind.realized_vol(c, 20)
    f["mom_63"] = c.pct_change(63)
    f["mom_126"] = c.pct_change(126)
    if "volume" in df.columns:
        f["vol_z"] = ind.rolling_zscore(df["volume"], 20)
    return f


def triple_barrier_labels(
    df: pd.DataFrame, horizon: int = 10, k: float = 1.5, atr_period: int = 14
) -> pd.Series:
    """1 if +k·ATR is hit before −k·ATR within `horizon` bars, else 0.

    If neither barrier is touched, label by the sign of the horizon return.
    The last `horizon` bars get NaN (their window is incomplete).
    """
    atr = ind.atr(df, atr_period).values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    labels = np.full(n, np.nan)
    for i in range(n):
        a = atr[i]
        if not math.isfinite(a) or a <= 0:
            continue
        if i + 1 > n - 1:
            break
        up = close[i] + k * a
        dn = close[i] - k * a
        end = min(i + horizon, n - 1)
        lab = np.nan
        for j in range(i + 1, end + 1):
            if high[j] >= up:
                lab = 1.0
                break
            if low[j] <= dn:
                lab = 0.0
                break
        if math.isnan(lab):
            if end > i:
                lab = 1.0 if close[end] > close[i] else 0.0
            else:
                continue
        labels[i] = lab
    return pd.Series(labels, index=df.index)


def _make_model(cfg: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=int(cfg.get("max_depth", 3)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_iter=int(cfg.get("max_iter", 200)),
        l2_regularization=float(cfg.get("l2", 1.0)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 20)),
        random_state=42,
    )


@dataclass
class MLConfig:
    horizon: int = 10
    k: float = 1.5
    min_train: int = 250
    refit_every: int = 42
    prob_threshold: float = 0.58
    atr_period: int = 14

    @classmethod
    def from_dict(cls, d: dict) -> "MLConfig":
        return cls(
            horizon=int(d.get("horizon", 10)),
            k=float(d.get("k", 1.5)),
            min_train=int(d.get("min_train", 250)),
            refit_every=int(d.get("refit_every", 42)),
            prob_threshold=float(d.get("prob_threshold", 0.58)),
            atr_period=int(d.get("atr_period", 14)),
        )


def _fit_if_ready(model_cfg, X, y, tr_end, min_rows=80):
    """Fit a model on rows [0:tr_end] with finite labels and both classes."""
    idx = np.where(np.isfinite(y[:tr_end]))[0]
    if len(idx) < min_rows:
        return None
    yv = y[idx]
    if len(np.unique(yv)) < 2:
        return None
    model = _make_model(model_cfg)
    model.fit(X[idx], yv)
    return model


# ---------------------------------------------------------------------
def backtest_ml(df: pd.DataFrame, cfg: dict):
    """Purged walk-forward backtest of the gradient-boosting rule.

    Returns (BacktestStats, last_prob_up | None).
    """
    mc = MLConfig.from_dict(cfg)
    feats = build_features(df)
    labels = triple_barrier_labels(df, mc.horizon, mc.k, mc.atr_period)
    atr = ind.atr(df, mc.atr_period).values
    X = feats.values
    y = labels.values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    if n < mc.min_train + mc.horizon + 20:
        return BacktestStats(), None

    returns: list[float] = []
    model = None
    last_fit = -10 ** 9
    i = mc.min_train
    while i < n - 1:
        if model is None or (i - last_fit) >= mc.refit_every:
            m = _fit_if_ready(cfg, X, y, tr_end=i - mc.horizon)
            if m is not None:
                model, last_fit = m, i
        if model is None or not np.isfinite(X[i]).any() or atr[i] <= 0:
            i += 1
            continue
        prob_up = float(model.predict_proba(X[i:i + 1])[0, 1])

        side = None
        if prob_up >= mc.prob_threshold:
            side = Side.LONG
        elif prob_up <= 1.0 - mc.prob_threshold:
            side = Side.SHORT
        if side is None:
            i += 1
            continue

        # simulate the matching triple-barrier trade forward from i
        a = atr[i]
        up = close[i] + mc.k * a
        dn = close[i] - mc.k * a
        end = min(i + mc.horizon, n - 1)
        R = None
        for j in range(i + 1, end + 1):
            if side is Side.LONG:
                if low[j] <= dn:
                    R = -1.0; break
                if high[j] >= up:
                    R = 1.0; break
            else:
                if high[j] >= up:
                    R = -1.0; break
                if low[j] <= dn:
                    R = 1.0; break
        if R is None:  # mark to close
            move = close[end] - close[i]
            R = (move if side is Side.LONG else -move) / (mc.k * a)
        returns.append(float(R))
        i = end + 1  # no overlapping trades

    # probability for the latest fully-featured bar
    last_prob = None
    if model is not None and np.isfinite(X[n - 1]).any():
        last_prob = float(model.predict_proba(X[n - 1:n])[0, 1])
    return stats_from_R(returns), last_prob


def latest_plan(df: pd.DataFrame, cfg: dict) -> Optional[TradePlan]:
    """Train on all completed-label history; emit a plan for the last bar."""
    mc = MLConfig.from_dict(cfg)
    feats = build_features(df)
    labels = triple_barrier_labels(df, mc.horizon, mc.k, mc.atr_period)
    atr = ind.atr(df, mc.atr_period)
    X = feats.values
    y = labels.values
    n = len(df)
    if n < mc.min_train + mc.horizon + 1:
        return None
    model = _fit_if_ready(cfg, X, y, tr_end=n - mc.horizon)
    if model is None or not np.isfinite(X[n - 1]).any():
        return None
    a = float(atr.iloc[-1])
    if not math.isfinite(a) or a <= 0:
        return None
    prob_up = float(model.predict_proba(X[n - 1:n])[0, 1])
    entry = float(df["close"].iloc[-1])
    risk = mc.k * a

    if prob_up >= mc.prob_threshold:
        plan = TradePlan(Side.LONG, entry, entry - risk, entry + risk, a,
                         rationale=(f"Gradient-boosting model P(up)={prob_up:.0%} "
                                    f"(triple-barrier ±{mc.k}·ATR, {mc.horizon}-bar horizon)."),
                         regime="ml", meta={"prob_up": prob_up})
    elif prob_up <= 1.0 - mc.prob_threshold:
        plan = TradePlan(Side.SHORT, entry, entry + risk, entry - risk, a,
                         rationale=(f"Gradient-boosting model P(down)={1-prob_up:.0%} "
                                    f"(triple-barrier ±{mc.k}·ATR, {mc.horizon}-bar horizon)."),
                         regime="ml", meta={"prob_up": prob_up})
    else:
        return None
    return plan if plan.valid() else None
