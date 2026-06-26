import numpy as np

from quantaura import montecarlo as mc
from quantaura.backtest import out_of_sample, stats_from_R
from quantaura.models import Side
from quantaura.risk import combined_confidence


def test_bootstrap_profitable_edge():
    # +0.5R average with 2:1 winners -> clearly profitable
    returns = [2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, 2.0, -1.0, 2.0]
    prob, median, p05, ruin = mc.bootstrap_paths(returns, n_sims=4000, seed=1)
    assert 0.0 <= prob <= 1.0
    assert prob > 0.7              # a real edge is usually profitable forward
    assert median > 0


def test_bootstrap_losing_edge():
    returns = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0]
    prob, *_ = mc.bootstrap_paths(returns, n_sims=4000, seed=1)
    assert prob < 0.5


def test_bootstrap_empty():
    prob, median, p05, ruin = mc.bootstrap_paths([], n_sims=10)
    assert prob == 0.0 and ruin == 1.0


def test_barrier_baseline_matches_rr():
    # RR = 2 -> driftless baseline = risk/(risk+reward) = 1/3
    win, base = mc.prob_target_before_stop(
        Side.LONG, entry=100, stop=98, target=104, sigma=0.0)
    assert abs(base - (2.0 / 6.0)) < 1e-9
    assert abs(win - base) < 1e-9   # sigma=0 -> falls back to baseline


def test_barrier_positive_drift_raises_winprob():
    # with upward drift, a long should beat its driftless baseline
    win, base = mc.prob_target_before_stop(
        Side.LONG, entry=100, stop=98, target=104, sigma=1.0, drift=0.4,
        max_bars=80, n_sims=6000, seed=3)
    assert win > base


def test_barrier_symmetry_short():
    win_l, base_l = mc.prob_target_before_stop(
        Side.LONG, 100, 98, 104, sigma=1.0, drift=0.0, n_sims=6000, seed=5)
    win_s, base_s = mc.prob_target_before_stop(
        Side.SHORT, 100, 102, 96, sigma=1.0, drift=0.0, n_sims=6000, seed=5)
    assert abs(base_l - base_s) < 1e-9          # same geometry -> same baseline
    assert abs(win_l - win_s) < 0.06            # roughly symmetric under zero drift


def test_out_of_sample_split():
    returns = [1.0, -1.0] * 20          # 40 trades, chronological
    stats = stats_from_R(returns)
    oos = out_of_sample(stats, split=0.7)
    assert oos.trades == 12              # last 30% of 40
    assert len(stats.returns_R) == 40


def test_combined_confidence_monotone_and_bounded():
    low = combined_confidence(backtest_conf=0.2, prob_profitable=0.5, mc_win_prob=0.3,
                              baseline_win_prob=0.33, oos_expectancy_R=-0.1, confluence=1)
    high = combined_confidence(backtest_conf=0.9, prob_profitable=0.95, mc_win_prob=0.6,
                               baseline_win_prob=0.33, oos_expectancy_R=0.3, confluence=3)
    assert 0.0 <= low < high <= 1.0


def test_assess_bundle():
    returns = [2.0, -1.0, 2.0, -1.0, 2.0, 2.0, -1.0, 2.0]
    m = mc.assess(side=Side.LONG, entry=100, stop=98, target=104,
                  returns_R=returns, atr=1.0, drift=0.2)
    assert 0.0 <= m.prob_profitable <= 1.0
    assert 0.0 <= m.win_prob <= 1.0
    assert 0.0 <= m.risk_of_ruin <= 1.0
