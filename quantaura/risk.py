"""Position sizing & risk geometry.

Two ideas, both standard in the literature (see README):

1. Fixed-fractional risk: never risk more than `risk_per_trade_pct` of
   equity on a single trade. Units = (equity * risk%) / (per-unit risk).

2. Fractional Kelly: scale exposure by the *measured* edge from the
   backtest. We use HALF-Kelly capped at `max_kelly_pct` because full
   Kelly is famously too aggressive on noisy, non-stationary estimates.

The final suggested size is the MINIMUM of the two — i.e. we let Kelly
shrink the bet when the edge is weak, but never let it exceed the hard
fixed-fractional risk cap.
"""
from __future__ import annotations

from dataclasses import dataclass


def fixed_fractional_units(equity: float, risk_pct: float, risk_per_unit: float) -> float:
    """Units such that a stop-out loses exactly risk_pct of equity."""
    if risk_per_unit <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    return risk_amount / risk_per_unit


def kelly_fraction(win_rate: float, avg_win_R: float, avg_loss_R: float = 1.0) -> float:
    """Kelly fraction of equity for a bet with payoff b:1.

    b = avg_win / avg_loss (in R units). f* = W - (1-W)/b.
    Returns 0 when the edge is non-positive. Clipped to [0, 1].
    """
    if avg_win_R <= 0 or avg_loss_R <= 0:
        return 0.0
    b = avg_win_R / avg_loss_R
    w = max(0.0, min(1.0, win_rate))
    f = w - (1.0 - w) / b
    return max(0.0, min(1.0, f))


@dataclass
class SizingResult:
    units: float
    notional: float
    risk_amount: float
    kelly_used: float


def size_position(
    *,
    equity: float,
    entry: float,
    risk_per_unit: float,
    win_rate: float,
    avg_win_R: float,
    avg_loss_R: float,
    risk_per_trade_pct: float,
    kelly_fraction_mult: float,
    max_kelly_pct: float,
) -> SizingResult:
    """Combine fixed-fractional and fractional-Kelly into a unit count."""
    if entry <= 0 or risk_per_unit <= 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0)

    # 1) hard cap from fixed-fractional risk
    ff_units = fixed_fractional_units(equity, risk_per_trade_pct, risk_per_unit)

    # 2) Kelly-implied equity fraction -> equivalent notional -> units
    f_kelly = kelly_fraction(win_rate, avg_win_R, avg_loss_R) * kelly_fraction_mult
    f_kelly = min(f_kelly, max_kelly_pct / 100.0)
    kelly_notional = equity * f_kelly
    kelly_units = kelly_notional / entry if entry > 0 else 0.0

    # final = the more conservative of the two
    units = max(0.0, min(ff_units, kelly_units)) if kelly_units > 0 else ff_units * 0.0
    notional = units * entry
    risk_amount = units * risk_per_unit
    return SizingResult(
        units=units,
        notional=notional,
        risk_amount=risk_amount,
        kelly_used=f_kelly,
    )


def confidence_from_backtest(
    win_rate: float, profit_factor: float, sharpe: float
) -> float:
    """Map backtest quality to a 0..1 confidence score (heuristic, monotone).

    Blends three independent quality axes so no single metric dominates.
    """
    wr = max(0.0, min(1.0, (win_rate - 0.35) / 0.40))          # 0.35->0, 0.75->1
    pf = max(0.0, min(1.0, (profit_factor - 1.0) / 1.0))        # 1.0->0, 2.0->1
    sh = max(0.0, min(1.0, sharpe / 2.0))                       # 0->0, 2.0->1
    return round(0.40 * wr + 0.35 * pf + 0.25 * sh, 3)
