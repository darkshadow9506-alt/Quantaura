import numpy as np

from quantaura import montecarlo as mc
from quantaura import portfolio as pf
from quantaura.models import AssetClass, Side, Signal


# ---------------- spread-reversion win probability ----------------
def test_spread_reversion_baseline():
    # driftless gambler's ruin: (z_stop-|z|)/(z_stop-z_exit)
    win, base = mc.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.0, sigma_eps=0.0)
    assert abs(base - (1.0 / 3.0)) < 1e-9
    assert win == base                      # no AR(1) dynamics -> baseline


def test_strong_mean_reversion_beats_baseline():
    # phi=0.5 -> fast reversion; a z=2.5 spread should very likely revert
    win, base = mc.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.5, sigma_eps=1.0,
                                         n_sims=8000)
    assert win > base
    assert win > 0.8                        # strong reversion -> high win prob


def test_no_reversion_when_phi_near_one():
    # phi≈1 is a random walk -> close to the driftless baseline
    win, base = mc.prob_spread_reversion(2.5, 0.5, 3.5, phi=0.999, sigma_eps=1.0,
                                         n_sims=8000)
    assert abs(win - base) < 0.15


def test_assess_pairs_bundle():
    m = mc.assess_pairs(returns_R=[1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                        z_now=2.5, z_exit=0.5, z_stop=3.5, phi=0.5, sigma_eps=1.0)
    assert 0.0 <= m.win_prob <= 1.0
    assert 0.0 <= m.prob_profitable <= 1.0


# ---------------- portfolio summary ----------------
def _sig(symbol, side, ac, risk, notional):
    return Signal(symbol=symbol, asset_class=ac, strategy="x", side=side,
                  entry=100, stop=98, target=104, risk_per_unit=2,
                  reward_per_unit=4, rr_ratio=2.0,
                  risk_amount=risk, position_notional=notional)


def test_portfolio_totals_and_net():
    sigs = [_sig("A", Side.LONG, AssetClass.STOCK, 100, 5000),
            _sig("B", Side.LONG, AssetClass.STOCK, 150, 6000),
            _sig("C", Side.SHORT, AssetClass.CRYPTO, 120, 4000)]
    s = pf.summarize(sigs, equity=10000, max_risk_pct=6.0, max_per_class=5)
    assert s.n == 3 and s.longs == 2 and s.shorts == 1
    assert abs(s.total_risk - 370) < 1e-9
    assert abs(s.total_risk_pct - 3.7) < 1e-9
    assert abs(s.gross_exposure - 15000) < 1e-9
    assert abs(s.net_exposure - 7000) < 1e-9   # 11000 long - 4000 short
    assert s.by_class_risk["stock"] == 250


def test_portfolio_over_budget_warning():
    sigs = [_sig(f"S{i}", Side.LONG, AssetClass.STOCK, 200, 5000) for i in range(5)]
    s = pf.summarize(sigs, equity=10000, max_risk_pct=6.0, max_per_class=10)
    assert s.total_risk_pct == 10.0
    assert any("budget" in w for w in s.warnings)
    assert any("one-directional" in w for w in s.warnings)


def test_portfolio_concentration_warning():
    sigs = [_sig(f"S{i}", Side.LONG if i % 2 else Side.SHORT, AssetClass.STOCK, 50, 1000)
            for i in range(7)]
    s = pf.summarize(sigs, equity=100000, max_risk_pct=50.0, max_per_class=5)
    assert any("concentrated" in w for w in s.warnings)


def test_portfolio_format_and_empty():
    assert pf.format_summary(pf.summarize([], 10000), 10000) == ""
    sigs = [_sig("A", Side.LONG, AssetClass.STOCK, 100, 5000)]
    text = pf.format_summary(pf.summarize(sigs, 10000), 10000)
    assert "Portfolio risk" in text and "Risk at stop" in text
