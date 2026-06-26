"""Statistical arbitrage — cointegration pairs trading.

Method (Engle-Granger, the textbook stat-arb pipeline):
  1. Align two price series A, B over a lookback window.
  2. OLS hedge ratio beta from  A = alpha + beta*B + e.
  3. Test the residual spread for cointegration (statsmodels.coint).
     Only trade pairs with p-value <= threshold (statistically stable).
  4. Standardize the spread to a rolling z-score (Ornstein-Uhlenbeck
     style mean reversion).
  5. Enter when |z| >= entry, exit toward the mean, stop on structural
     break (|z| >= stop). Long the cheap leg, short the rich leg.

We express the entry/stop/target as concrete prices on leg A (holding B
at its current price), so the published signal is precise and tradable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .models import BacktestStats, Side


@dataclass
class PairPlan:
    symbol_a: str
    symbol_b: str
    side_a: Side          # side for leg A; leg B is the opposite
    entry_a: float
    stop_a: float
    target_a: float
    hedge_ratio: float
    spread_z: float
    coint_pvalue: float
    atr_a: float
    rationale: str

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_a - self.stop_a)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.target_a - self.entry_a)

    @property
    def rr_ratio(self) -> float:
        r = self.risk_per_unit
        return self.reward_per_unit / r if r > 0 else 0.0

    def valid(self) -> bool:
        vals = (self.entry_a, self.stop_a, self.target_a, self.hedge_ratio)
        if not all(math.isfinite(v) for v in vals):
            return False
        if self.risk_per_unit <= 0 or self.reward_per_unit <= 0:
            return False
        if self.side_a is Side.LONG:
            return self.stop_a < self.entry_a < self.target_a
        return self.target_a < self.entry_a < self.stop_a


def _hedge_ratio_and_spread(a: pd.Series, b: pd.Series):
    """OLS hedge ratio (beta) and spread = a - beta*b - alpha."""
    import statsmodels.api as sm

    x = sm.add_constant(b.values)
    model = sm.OLS(a.values, x).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    spread = a - (beta * b + alpha)
    return alpha, beta, spread


def _coint_pvalue(a: pd.Series, b: pd.Series) -> float:
    from statsmodels.tsa.stattools import coint

    try:
        _, pvalue, _ = coint(a.values, b.values)
        return float(pvalue)
    except Exception:
        return 1.0


def evaluate_pair(
    symbol_a: str,
    symbol_b: str,
    close_a: pd.Series,
    close_b: pd.Series,
    cfg: dict,
    atr_a: float = 0.0,
) -> Optional[PairPlan]:
    """Return a PairPlan if the pair is cointegrated and stretched now."""
    lookback = int(cfg.get("lookback", 252))
    p_max = float(cfg.get("coint_pvalue_max", 0.05))
    z_entry = float(cfg.get("zscore_entry", 2.0))
    z_exit = float(cfg.get("zscore_exit", 0.5))
    z_stop = float(cfg.get("zscore_stop", 3.5))

    df = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).dropna()
    if len(df) < lookback:
        return None
    df = df.tail(lookback)
    a, b = df["a"], df["b"]

    pvalue = _coint_pvalue(a, b)
    if pvalue > p_max:
        return None

    alpha, beta, spread = _hedge_ratio_and_spread(a, b)
    if not math.isfinite(beta) or beta == 0:
        return None
    mu = float(spread.mean())
    sigma = float(spread.std(ddof=0))
    if sigma <= 0:
        return None

    z_now = (float(spread.iloc[-1]) - mu) / sigma
    if abs(z_now) < z_entry:
        return None

    b_now = float(b.iloc[-1])
    a_now = float(a.iloc[-1])

    def a_price_at_z(z: float) -> float:
        # spread = mu + z*sigma ; spread = a - (beta*b + alpha)
        return mu + z * sigma + beta * b_now + alpha

    if z_now <= -z_entry:
        # spread too low -> A cheap vs B -> LONG A, SHORT B; expect z up
        side_a = Side.LONG
        entry_a = a_now
        target_a = a_price_at_z(-z_exit)   # higher than entry
        stop_a = a_price_at_z(-z_stop)     # lower than entry
        rationale = (f"Spread z={z_now:.2f} (cointegrated p={pvalue:.3f}). "
                     f"{symbol_a} cheap vs {symbol_b}: long {symbol_a}, short {symbol_b}.")
    else:
        # spread too high -> A rich vs B -> SHORT A, LONG B; expect z down
        side_a = Side.SHORT
        entry_a = a_now
        target_a = a_price_at_z(z_exit)    # lower than entry
        stop_a = a_price_at_z(z_stop)      # higher than entry
        rationale = (f"Spread z={z_now:.2f} (cointegrated p={pvalue:.3f}). "
                     f"{symbol_a} rich vs {symbol_b}: short {symbol_a}, long {symbol_b}.")

    plan = PairPlan(
        symbol_a=symbol_a, symbol_b=symbol_b, side_a=side_a,
        entry_a=entry_a, stop_a=stop_a, target_a=target_a,
        hedge_ratio=beta, spread_z=z_now, coint_pvalue=pvalue,
        atr_a=atr_a, rationale=rationale,
    )
    return plan if plan.valid() else None


# ---------------------------------------------------------------------
def backtest_pair(close_a: pd.Series, close_b: pd.Series, cfg: dict) -> BacktestStats:
    """Walk-forward-ish backtest of the z-score spread on the whole history.

    Uses a rolling estimation window so the hedge ratio / mean / std at
    each bar are computed only from PRIOR data (no look-ahead). Returns
    per-trade R statistics.
    """
    import statsmodels.api as sm

    lookback = int(cfg.get("lookback", 252))
    z_entry = float(cfg.get("zscore_entry", 2.0))
    z_exit = float(cfg.get("zscore_exit", 0.5))
    z_stop = float(cfg.get("zscore_stop", 3.5))

    df = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).dropna()
    if len(df) < lookback + 30:
        return BacktestStats()

    a_all, b_all = df["a"].values, df["b"].values
    n = len(df)

    returns_R: list[float] = []
    position = 0          # +1 long spread, -1 short spread, 0 flat
    entry_z = 0.0

    for t in range(lookback, n):
        # estimate on the trailing window ending at t-1 (prior data only)
        a_w = a_all[t - lookback:t]
        b_w = b_all[t - lookback:t]
        x = sm.add_constant(b_w)
        try:
            beta = np.linalg.lstsq(x, a_w, rcond=None)[0]
            alpha_, beta_ = beta[0], beta[1]
        except Exception:
            continue
        spread_w = a_w - (beta_ * b_w + alpha_)
        mu, sigma = spread_w.mean(), spread_w.std()
        if sigma <= 0:
            continue
        spread_now = a_all[t] - (beta_ * b_all[t] + alpha_)
        z = (spread_now - mu) / sigma

        if position == 0:
            if z <= -z_entry:
                position, entry_z = 1, z
            elif z >= z_entry:
                position, entry_z = -1, z
        elif position == 1:        # long spread, want z -> up
            if z >= -z_exit:       # reverted toward mean (win)
                returns_R.append((z - entry_z) / (z_stop - z_entry))
                position = 0
            elif z <= -z_stop:     # structural break (loss)
                returns_R.append(-1.0)
                position = 0
        elif position == -1:       # short spread, want z -> down
            if z <= z_exit:
                returns_R.append((entry_z - z) / (z_stop - z_entry))
                position = 0
            elif z >= z_stop:
                returns_R.append(-1.0)
                position = 0

    from .backtest import stats_from_R  # local import avoids any import cycle

    return stats_from_R(returns_R)
