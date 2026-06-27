"""Active management advice for an OPEN position.

Given a signal and the current market state, recommend what to do *now*:
move the stop to breakeven, take partial profit, trail the stop, or close
early because the thesis is breaking. The advice is phrased as the current
*recommended state* (not one-off events), so re-checking every scan is
idempotent and never spams.

The core `review()` is a pure function of scalars (fully unit-tested); the
engine computes those scalars from live data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Side


@dataclass
class ManagementReview:
    R_now: float = 0.0
    recommended_sl: float = 0.0
    recommended_tp: float = 0.0       # where to take profit given the live state
    tp_reason: str = ""
    at_breakeven: bool = False
    trailed: bool = False
    near_target: bool = False
    danger: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        # worth pushing unprompted: at risk-free, near target, or in danger
        return self.at_breakeven or self.trailed or self.near_target or self.danger


def review(*, side: Side, entry: float, stop: float, target: float, current: float,
           atr: float, ma_trend: float, macd_hist: float,
           hi_since: float, lo_since: float, cfg: dict,
           next_level: float | None = None) -> ManagementReview:
    risk = abs(entry - stop)
    if risk <= 0 or atr <= 0:
        return ManagementReview(recommended_sl=stop, recommended_tp=target)
    be_R = float(cfg.get("breakeven_R", 1.0))
    trail_mult = float(cfg.get("trail_atr_mult", 3.0))
    near_mult = float(cfg.get("near_target_atr", 0.5))
    use_ma = bool(cfg.get("danger_on_ma_break", True))
    use_macd = bool(cfg.get("danger_on_macd_flip", True))

    long = side is Side.LONG
    R_now = (current - entry) / risk if long else (entry - current) / risk

    rec = stop
    at_be = trailed = False
    notes: list[str] = []

    if R_now >= be_R:
        # 1) lock in: stop to at least breakeven
        if long:
            rec = max(rec, entry)
        else:
            rec = min(rec, entry)
        at_be = (rec == entry)
        # 2) trail behind the move
        if long:
            trail_sl = hi_since - trail_mult * atr
            if trail_sl > rec and trail_sl < current:
                rec, trailed = trail_sl, True
        else:
            trail_sl = lo_since + trail_mult * atr
            if trail_sl < rec and trail_sl > current:
                rec, trailed = trail_sl, True
        notes.append(f"In profit ({R_now:+.2f}R) — take partial profit and move stop to "
                     + ("breakeven." if not trailed else f"the trail at {rec:.4f}."))

    # near target?
    near = abs(target - current) <= near_mult * atr
    if near:
        notes.append("Within reach of target — consider taking profit / tightening.")

    # danger: trend or momentum flipped against the position
    danger = False
    if long and ((use_ma and current < ma_trend) or (use_macd and macd_hist < 0)):
        danger = True
        notes.append("⚠️ Price below trend MA / MACD turned down — long thesis weakening; "
                     "consider closing or tightening the stop.")
    if (not long) and ((use_ma and current > ma_trend) or (use_macd and macd_hist > 0)):
        danger = True
        notes.append("⚠️ Price above trend MA / MACD turned up — short thesis weakening; "
                     "consider closing or tightening the stop.")

    # --- live take-profit recommendation (adapts to the current state) ---
    tp_buf = float(cfg.get("tp_buffer_atr", 0.25)) * atr
    rec_tp, tp_reason = target, "let it run toward the target"
    if danger:
        rec_tp, tp_reason = current, "thesis weakening — take profit / close now"
    elif near:
        rec_tp, tp_reason = target, "in the target zone — take profit"
    elif next_level is not None:
        if long and current < next_level < target:
            c = next_level - tp_buf
            if c > current:
                rec_tp, tp_reason = c, f"resistance ~{next_level:.4f} ahead — bank profit before a bounce"
        elif (not long) and target < next_level < current:
            c = next_level + tp_buf
            if c < current:
                rec_tp, tp_reason = c, f"support ~{next_level:.4f} ahead — bank profit before a bounce"
    notes.append(f"Take profit ~{rec_tp:.4f} — {tp_reason}.")

    if not [n for n in notes if not n.startswith("Take profit")]:
        notes.insert(0, f"Holding ({R_now:+.2f}R). Let it work toward target/stop.")

    return ManagementReview(R_now=round(R_now, 2), recommended_sl=round(rec, 6),
                            recommended_tp=round(rec_tp, 6), tp_reason=tp_reason,
                            at_breakeven=at_be, trailed=trailed, near_target=near,
                            danger=danger, notes=notes)
