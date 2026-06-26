"""Portfolio-level risk aggregation across a batch of signals.

This is the "position management / risk budget" layer from the
methodology: a single trade can be sized correctly yet a *basket* of
simultaneous signals can still over-concentrate risk. After a scan we
summarise total risk-at-stop, gross/net exposure, and flag concentration
so the user sees the whole book, not just one idea.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Side, Signal


@dataclass
class PortfolioSummary:
    n: int = 0
    longs: int = 0
    shorts: int = 0
    total_risk: float = 0.0          # account currency at risk if every stop hits
    total_risk_pct: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    by_class_risk: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def summarize(signals: list[Signal], equity: float,
              max_risk_pct: float = 6.0, max_per_class: int = 5) -> PortfolioSummary:
    s = PortfolioSummary()
    if not signals:
        return s
    s.n = len(signals)
    long_notional = short_notional = 0.0
    class_count: dict[str, int] = {}
    for sig in signals:
        if sig.side is Side.LONG:
            s.longs += 1
            long_notional += sig.position_notional
        else:
            s.shorts += 1
            short_notional += sig.position_notional
        s.total_risk += sig.risk_amount
        cls = sig.asset_class.value
        s.by_class_risk[cls] = s.by_class_risk.get(cls, 0.0) + sig.risk_amount
        class_count[cls] = class_count.get(cls, 0) + 1

    s.gross_exposure = long_notional + short_notional
    s.net_exposure = long_notional - short_notional
    s.total_risk_pct = (s.total_risk / equity * 100.0) if equity > 0 else 0.0

    if s.total_risk_pct > max_risk_pct:
        s.warnings.append(
            f"Total risk-at-stop is {s.total_risk_pct:.1f}% of equity "
            f"(> {max_risk_pct:.0f}% budget) — consider taking fewer / smaller positions.")
    for cls, cnt in class_count.items():
        if cnt > max_per_class:
            s.warnings.append(
                f"{cnt} {cls} positions (> {max_per_class}) — concentrated in one asset class.")
    # directional skew
    if s.n >= 4 and (s.longs == 0 or s.shorts == 0):
        side = "long" if s.shorts == 0 else "short"
        s.warnings.append(f"All {s.n} signals are {side} — the book is one-directional.")
    return s


def format_summary(summary: PortfolioSummary, equity: float) -> str:
    if summary.n == 0:
        return ""
    lines = ["", "📁 *Portfolio risk (if you took all signals)*",
             f"Signals: {summary.n}  ({summary.longs} long / {summary.shorts} short)",
             f"Risk at stop: ${summary.total_risk:,.0f}  ({summary.total_risk_pct:.1f}% of equity)",
             f"Gross exposure: ${summary.gross_exposure:,.0f}  |  "
             f"Net: ${summary.net_exposure:,.0f}"]
    if summary.by_class_risk:
        parts = ", ".join(f"{k} ${v:,.0f}" for k, v in summary.by_class_risk.items())
        lines.append(f"Risk by class: {parts}")
    for w in summary.warnings:
        lines.append(f"⚠️ {w}")
    return "\n".join(lines)
