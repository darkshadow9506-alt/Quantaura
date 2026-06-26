"""Core data models shared across the system."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class AssetClass(str, Enum):
    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class BacktestStats:
    """Performance of the originating strategy on this symbol's history."""

    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_R: float = 0.0
    expectancy_R: float = 0.0
    # chronological per-trade R outcomes (used for OOS split + Monte Carlo).
    # Excluded from as_dict() to keep serialized signals compact.
    returns_R: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("returns_R", None)
        return d


@dataclass
class MonteCarloStats:
    """Probabilistic robustness of the backtested edge."""

    prob_profitable: float = 0.0     # P(positive total over a forward run)
    median_total_R: float = 0.0
    p05_total_R: float = 0.0         # 5th-percentile (bad-case) outcome
    risk_of_ruin: float = 1.0        # P(drawdown breaches the ruin threshold)
    win_prob: float = 0.0            # MC P(hit TP before SL) given drift+vol
    baseline_win_prob: float = 0.0   # driftless structural baseline = 1/(1+RR)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    """A precise, fully-specified trade idea.

    Entry / stop / target are concrete prices computed from real OHLCV
    data — never placeholders. `confidence` and `passed_gate` come from
    the strategy's own backtest on this symbol, so a signal is only
    'published' when it has a measured historical edge.
    """

    symbol: str
    asset_class: AssetClass
    strategy: str
    side: Side

    entry: float
    stop: float
    target: float

    # risk geometry
    risk_per_unit: float          # |entry - stop|
    reward_per_unit: float        # |target - entry|
    rr_ratio: float               # reward / risk

    # position sizing suggestion (units & notional)
    position_units: float = 0.0
    position_notional: float = 0.0
    risk_amount: float = 0.0      # account currency risked if stop hit

    # context
    atr: float = 0.0
    regime: str = ""
    rationale: str = ""
    management: str = ""          # how to manage the trade (e.g. trailing stop)
    backtest: BacktestStats = field(default_factory=BacktestStats)
    oos: BacktestStats = field(default_factory=BacktestStats)   # out-of-sample
    montecarlo: MonteCarloStats = field(default_factory=MonteCarloStats)
    confluence: int = 1           # how many strategies agree (same symbol+side)
    confidence: float = 0.0       # 0..1, blended quality score
    passed_gate: bool = False

    timeframe: str = "1d"
    price_at_signal: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["asset_class"] = self.asset_class.value
        d["side"] = self.side.value
        for key in ("backtest", "oos"):
            if isinstance(d.get(key), dict):
                d[key].pop("returns_R", None)
        return d

    def is_valid_geometry(self) -> bool:
        """Sanity: stop and target on the correct sides of entry."""
        if self.side is Side.LONG:
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop


@dataclass
class PairSignal(Signal):
    """A pairs/stat-arb signal trades two legs simultaneously."""

    pair_symbol: str = ""        # the second leg
    hedge_ratio: float = 1.0     # units of leg B per unit of leg A
    spread_z: float = 0.0
    leg_b_side: Optional[str] = None
