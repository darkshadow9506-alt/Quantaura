"""Human-readable rendering of Signal objects (Telegram / CLI)."""
from __future__ import annotations

from .models import PairSignal, Side, Signal

_DISCLAIMER = (
    "⚠️ Educational signal from a backtested quant model. Markets are "
    "probabilistic — no signal is guaranteed. Risk only what you can lose."
)


def _fmt_price(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _bar(conf: float, width: int = 10) -> str:
    filled = int(round(conf * width))
    return "█" * filled + "░" * (width - filled)


def format_signal(sig: Signal, md: bool = True) -> str:
    side_icon = "🟢 LONG" if sig.side is Side.LONG else "🔴 SHORT"
    bt = sig.backtest
    lines = []
    title = f"{side_icon}  *{sig.symbol}*  ({sig.asset_class.value})"
    lines.append(title)
    lines.append(f"Strategy: `{sig.strategy}`  |  Regime: {sig.regime}  |  TF: {sig.timeframe}")
    lines.append("")
    lines.append(f"🎯 Entry:  `{_fmt_price(sig.entry)}`")
    lines.append(f"🛑 Stop:   `{_fmt_price(sig.stop)}`")
    lines.append(f"✅ Target: `{_fmt_price(sig.target)}`")
    lines.append(f"⚖️ R:R = {sig.rr_ratio}  |  ATR = {_fmt_price(sig.atr)}")
    lines.append("")

    if isinstance(sig, PairSignal):
        leg_b = sig.leg_b_side or "?"
        lines.append(
            f"🔗 Pair leg: {leg_b} *{sig.pair_symbol}*  "
            f"(hedge β={sig.hedge_ratio}, spread z={sig.spread_z})"
        )
        lines.append("")

    if sig.position_units > 0:
        lines.append(
            f"📐 Size: {_fmt_price(sig.position_units)} units "
            f"(~${sig.position_notional:,.0f}) | risk ${sig.risk_amount:,.0f}"
        )
    lines.append(
        f"📊 Backtest: {bt.trades} trades | win {bt.win_rate*100:.0f}% | "
        f"PF {bt.profit_factor:.2f} | exp {bt.expectancy_R:+.2f}R | "
        f"maxDD {bt.max_drawdown:.1f}R"
    )
    lines.append(f"🔆 Confidence: {_bar(sig.confidence)} {sig.confidence*100:.0f}%")
    if not sig.passed_gate:
        lines.append("❗ Below publish threshold — shown for inspection only.")
    lines.append("")
    lines.append(f"💡 {sig.rationale}")
    lines.append("")
    lines.append(_DISCLAIMER)

    text = "\n".join(lines)
    if not md:
        text = text.replace("*", "").replace("`", "")
    return text


def format_scan_summary(signals: list[Signal]) -> str:
    if not signals:
        return ("No signals passed the backtest gate this scan. "
                "That is normal — the model stays flat unless it has a measured edge.")
    head = f"📡 *{len(signals)} signal(s)* passed the gate (best first):\n"
    rows = []
    for s in signals:
        icon = "🟢" if s.side is Side.LONG else "🔴"
        rows.append(
            f"{icon} *{s.symbol}* `{s.strategy}` "
            f"entry {_fmt_price(s.entry)} → tgt {_fmt_price(s.target)} "
            f"(R:R {s.rr_ratio}, conf {s.confidence*100:.0f}%)"
        )
    return head + "\n".join(rows)
