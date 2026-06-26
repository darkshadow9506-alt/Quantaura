"""Monte Carlo robustness & probability tools.

Two independent probabilistic checks, both straight out of the standard
quant toolkit (the doc you supplied lists "Monte Carlo simulation of
extreme scenarios" and "out-of-sample testing" under risk management):

1. `bootstrap_paths` — resample the backtested per-trade R outcomes to
   estimate the probability the edge is actually profitable going
   forward, the bad-case (5th-percentile) result, and the risk of ruin.
   This guards against a backtest that looks good only because of trade
   ordering / a few lucky trades.

2. `prob_target_before_stop` — simulate the price as a random walk with
   the symbol's *current* drift and volatility (ATR) to estimate the
   probability the take-profit is hit before the stop. Reported next to
   the driftless structural baseline 1/(1+RR), so you can see whether the
   measured drift actually tilts the odds in your favour.
"""
from __future__ import annotations

import numpy as np

from .models import MonteCarloStats, Side


def bootstrap_paths(
    returns_R: list[float],
    n_sims: int = 5000,
    horizon: int | None = None,
    ruin_R: float = 10.0,
    seed: int = 42,
):
    """Resample trade outcomes -> (prob_profitable, median, p05, risk_of_ruin)."""
    arr = np.asarray(returns_R, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 1.0
    horizon = horizon or arr.size
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_sims, horizon), replace=True)
    cum = np.cumsum(draws, axis=1)
    totals = cum[:, -1]
    mins = cum.min(axis=1)
    prob_profitable = float((totals > 0).mean())
    median_total = float(np.median(totals))
    p05_total = float(np.percentile(totals, 5))
    risk_of_ruin = float((mins <= -ruin_R).mean())
    return prob_profitable, median_total, p05_total, risk_of_ruin


def _analytic_baseline(side: Side, entry: float, stop: float, target: float) -> float:
    """Driftless gambler's-ruin P(target before stop) = risk/(risk+reward)."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    denom = risk + reward
    return float(risk / denom) if denom > 0 else 0.0


def prob_target_before_stop(
    side: Side,
    entry: float,
    stop: float,
    target: float,
    sigma: float,
    drift: float = 0.0,
    max_bars: int = 60,
    n_sims: int = 4000,
    seed: int = 7,
):
    """MC P(TP before SL) with per-bar Normal(drift, sigma) steps.

    Returns (mc_win_prob, baseline_win_prob). Undecided paths (neither
    barrier touched within max_bars) count as non-wins (conservative).
    """
    baseline = _analytic_baseline(side, entry, stop, target)
    if sigma <= 0 or max_bars < 1:
        return baseline, baseline
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, sigma, size=(n_sims, max_bars))
    paths = entry + np.cumsum(steps, axis=1)

    if side is Side.LONG:
        hit_t = paths >= target
        hit_s = paths <= stop
    else:
        hit_t = paths <= target
        hit_s = paths >= stop

    big = max_bars + 1
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), big)
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), big)
    win = (first_t < first_s).mean()
    return float(win), float(baseline)


def prob_spread_reversion(
    z_now: float,
    z_exit: float,
    z_stop: float,
    phi: float,
    sigma_eps: float,
    max_bars: int = 60,
    n_sims: int = 4000,
    seed: int = 11,
):
    """P(a pairs spread reverts to |z|<=z_exit before diverging to |z|>=z_stop).

    The z-score is modelled as a discrete Ornstein-Uhlenbeck / AR(1)
    process  z_t = phi·z_{t-1} + eps  (phi<1 -> mean reverting). This is
    the correct model for a stat-arb trade — its edge is mean reversion of
    the spread, not directional drift in either leg.

    Returns (mc_win_prob, baseline). The baseline is the driftless
    random-walk gambler's-ruin probability in the |z| interval.
    """
    span = z_stop - z_exit
    baseline = (z_stop - abs(z_now)) / span if span > 0 else 0.0
    baseline = float(min(1.0, max(0.0, baseline)))
    if not (0.0 < phi < 1.0) or sigma_eps <= 0 or max_bars < 1:
        return baseline, baseline

    rng = np.random.default_rng(seed)
    z = np.full(n_sims, float(z_now))
    eps = rng.normal(0.0, sigma_eps, size=(n_sims, max_bars))
    won = np.zeros(n_sims, dtype=bool)
    done = np.zeros(n_sims, dtype=bool)
    for t in range(max_bars):
        z = phi * z + eps[:, t]
        az = np.abs(z)
        hit_exit = (~done) & (az <= z_exit)
        hit_stop = (~done) & (az >= z_stop)
        won |= hit_exit
        done |= hit_exit | hit_stop
    return float(won.mean()), baseline


def assess_pairs(
    *,
    returns_R: list[float],
    z_now: float,
    z_exit: float,
    z_stop: float,
    phi: float,
    sigma_eps: float,
    ruin_R: float = 10.0,
    max_bars: int = 60,
) -> MonteCarloStats:
    """Monte Carlo bundle for a pairs trade (spread-reversion win prob)."""
    prob_profitable, median_total, p05_total, risk_of_ruin = bootstrap_paths(
        returns_R, ruin_R=ruin_R
    )
    win_prob, baseline = prob_spread_reversion(
        z_now, z_exit, z_stop, phi, sigma_eps, max_bars=max_bars
    )
    return MonteCarloStats(
        prob_profitable=round(prob_profitable, 4),
        median_total_R=round(median_total, 4),
        p05_total_R=round(p05_total, 4),
        risk_of_ruin=round(risk_of_ruin, 4),
        win_prob=round(win_prob, 4),
        baseline_win_prob=round(baseline, 4),
    )


def assess(
    *,
    side: Side,
    entry: float,
    stop: float,
    target: float,
    returns_R: list[float],
    atr: float,
    drift: float = 0.0,
    ruin_R: float = 10.0,
    max_bars: int = 60,
) -> MonteCarloStats:
    """Bundle both checks into a MonteCarloStats."""
    prob_profitable, median_total, p05_total, risk_of_ruin = bootstrap_paths(
        returns_R, ruin_R=ruin_R
    )
    win_prob, baseline = prob_target_before_stop(
        side, entry, stop, target, sigma=atr, drift=drift, max_bars=max_bars
    )
    return MonteCarloStats(
        prob_profitable=round(prob_profitable, 4),
        median_total_R=round(median_total, 4),
        p05_total_R=round(p05_total, 4),
        risk_of_ruin=round(risk_of_ruin, 4),
        win_prob=round(win_prob, 4),
        baseline_win_prob=round(baseline, 4),
    )
