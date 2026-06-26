"""Signal engine — turns market data into gated, sized Signal objects.

For each symbol:
  1. fetch OHLCV
  2. detect regime (ADX)
  3. run every enabled strategy whose preferred regime matches (or
     'neutral' -> run all); take the latest-bar TradePlan if any
  4. backtest THAT strategy on THAT symbol's history
  5. only publish the signal if the backtest clears the signal_gate
  6. size the position (fixed-fractional capped, half-Kelly) and attach
     confidence derived from the backtest

This is the mechanism that keeps signals honest: nothing is published
without a measured historical edge on the very instrument it targets.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from . import data as data_mod
from . import factor as factor_mod
from . import pairs as pairs_mod
from . import risk as risk_mod
from .backtest import backtest_strategy
from .config import Settings
from .indicators import atr as atr_ind
from .models import AssetClass, BacktestStats, PairSignal, Side, Signal
from .strategies import build_strategies, detect_regime


# ---------------------------------------------------------------------
def _passes_gate(stats: BacktestStats, gate: dict) -> bool:
    return (
        stats.trades >= int(gate.get("min_backtest_trades", 15))
        and stats.win_rate >= float(gate.get("min_win_rate", 0.40))
        and stats.profit_factor >= float(gate.get("min_profit_factor", 1.15))
        and stats.sharpe >= float(gate.get("min_sharpe", 0.3))
    )


def _kelly_b(win_rate: float, profit_factor: float) -> float:
    """Recover the payoff ratio b = avg_win/avg_loss from W and PF."""
    if win_rate <= 0 or win_rate >= 1 or not math.isfinite(profit_factor):
        return 0.0
    return profit_factor * (1.0 - win_rate) / win_rate


def _build_signal(
    *,
    symbol: str,
    asset_class: AssetClass,
    strategy_name: str,
    plan,
    stats: BacktestStats,
    regime: str,
    settings: Settings,
    passed_gate: bool,
) -> Signal:
    risk_cfg = settings.risk
    b = _kelly_b(stats.win_rate, stats.profit_factor)
    sizing = risk_mod.size_position(
        equity=settings.account_equity,
        entry=plan.entry,
        risk_per_unit=plan.risk_per_unit,
        win_rate=stats.win_rate,
        avg_win_R=b,
        avg_loss_R=1.0,
        risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 1.0)),
        kelly_fraction_mult=float(risk_cfg.get("kelly_fraction", 0.5)),
        max_kelly_pct=float(risk_cfg.get("max_kelly_pct", 5.0)),
    )
    confidence = risk_mod.confidence_from_backtest(
        stats.win_rate, stats.profit_factor, stats.sharpe
    )
    return Signal(
        symbol=symbol,
        asset_class=asset_class,
        strategy=strategy_name,
        side=plan.side,
        entry=round(plan.entry, 6),
        stop=round(plan.stop, 6),
        target=round(plan.target, 6),
        risk_per_unit=round(plan.risk_per_unit, 6),
        reward_per_unit=round(plan.reward_per_unit, 6),
        rr_ratio=round(plan.rr_ratio, 3),
        position_units=round(sizing.units, 6),
        position_notional=round(sizing.notional, 2),
        risk_amount=round(sizing.risk_amount, 2),
        atr=round(plan.atr, 6),
        regime=regime,
        rationale=plan.rationale,
        backtest=stats,
        confidence=confidence,
        passed_gate=passed_gate,
        timeframe=settings.data.get("timeframe", "1d"),
        price_at_signal=round(plan.entry, 6),
    )


# ---------------------------------------------------------------------
def scan_symbol(
    symbol: str,
    asset_class: AssetClass,
    settings: Settings,
    publish_only: bool = True,
) -> list[Signal]:
    """Return signals for one symbol. If publish_only, only gated ones."""
    d = settings.data
    df = data_mod.get_ohlcv(
        symbol,
        asset_class,
        timeframe=d.get("timeframe", "1d"),
        lookback=int(d.get("lookback_bars", 400)),
        cache_minutes=int(d.get("cache_minutes", 30)),
        ccxt_exchange=settings.ccxt_exchange,
    )
    if df is None or len(df) < 60:
        return []

    regime = detect_regime(df, settings.regime)
    gate = settings.signal_gate
    out: list[Signal] = []

    for strat in build_strategies(settings):
        # regime alignment: run a strategy when the market suits it, or
        # when the regime is neutral (let the backtest gate decide).
        if regime != "neutral" and strat.preferred_regime != regime:
            continue
        prepared = strat.prepare(df)
        plan = strat.evaluate(prepared, len(prepared) - 1)
        if plan is None or not plan.valid():
            continue
        stats, _ = backtest_strategy(strat, df)
        passed = _passes_gate(stats, gate)
        if publish_only and not passed:
            continue
        out.append(
            _build_signal(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=strat.name,
                plan=plan,
                stats=stats,
                regime=regime,
                settings=settings,
                passed_gate=passed,
            )
        )
    return out


# ---------------------------------------------------------------------
def scan_pairs(settings: Settings, publish_only: bool = True) -> list[PairSignal]:
    pcfg = settings.pairs
    if not pcfg.get("enabled", True):
        return []
    gate = settings.signal_gate
    d = settings.data
    out: list[PairSignal] = []

    # pairs candidates are equities here
    cache: dict[str, pd.DataFrame] = {}

    def _load(sym: str) -> Optional[pd.DataFrame]:
        if sym in cache:
            return cache[sym]
        try:
            df = data_mod.get_ohlcv(
                sym, AssetClass.STOCK,
                timeframe=d.get("timeframe", "1d"),
                lookback=int(d.get("lookback_bars", 400)),
                cache_minutes=int(d.get("cache_minutes", 30)),
            )
        except Exception:
            df = None
        cache[sym] = df
        return df

    for cand in pcfg.get("candidates", []):
        if not isinstance(cand, (list, tuple)) or len(cand) != 2:
            continue
        sym_a, sym_b = cand
        da, db = _load(sym_a), _load(sym_b)
        if da is None or db is None or len(da) < 60 or len(db) < 60:
            continue

        atr_a = float(atr_ind(da, 14).iloc[-1]) if len(da) > 20 else 0.0
        plan = pairs_mod.evaluate_pair(
            sym_a, sym_b, da["close"], db["close"], pcfg, atr_a=atr_a
        )
        if plan is None:
            continue
        stats = pairs_mod.backtest_pair(da["close"], db["close"], pcfg)
        passed = _passes_gate(stats, gate)
        if publish_only and not passed:
            continue

        # size on leg A
        risk_cfg = settings.risk
        b = _kelly_b(stats.win_rate, stats.profit_factor)
        sizing = risk_mod.size_position(
            equity=settings.account_equity,
            entry=plan.entry_a,
            risk_per_unit=plan.risk_per_unit,
            win_rate=stats.win_rate,
            avg_win_R=b,
            avg_loss_R=1.0,
            risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 1.0)),
            kelly_fraction_mult=float(risk_cfg.get("kelly_fraction", 0.5)),
            max_kelly_pct=float(risk_cfg.get("max_kelly_pct", 5.0)),
        )
        confidence = risk_mod.confidence_from_backtest(
            stats.win_rate, stats.profit_factor, stats.sharpe
        )
        leg_b_side = Side.SHORT if plan.side_a is Side.LONG else Side.LONG
        out.append(
            PairSignal(
                symbol=plan.symbol_a,
                asset_class=AssetClass.STOCK,
                strategy="pairs_statarb",
                side=plan.side_a,
                entry=round(plan.entry_a, 6),
                stop=round(plan.stop_a, 6),
                target=round(plan.target_a, 6),
                risk_per_unit=round(plan.risk_per_unit, 6),
                reward_per_unit=round(plan.reward_per_unit, 6),
                rr_ratio=round(plan.rr_ratio, 3),
                position_units=round(sizing.units, 6),
                position_notional=round(sizing.notional, 2),
                risk_amount=round(sizing.risk_amount, 2),
                atr=round(plan.atr_a, 6),
                regime="stat-arb",
                rationale=plan.rationale,
                backtest=stats,
                confidence=confidence,
                passed_gate=passed,
                timeframe=d.get("timeframe", "1d"),
                price_at_signal=round(plan.entry_a, 6),
                pair_symbol=plan.symbol_b,
                hedge_ratio=round(plan.hedge_ratio, 6),
                spread_z=round(plan.spread_z, 3),
                leg_b_side=leg_b_side.value,
            )
        )
    return out


# ---------------------------------------------------------------------
def scan_factor(settings: Settings, publish_only: bool = True) -> list[Signal]:
    """Cross-sectional momentum factor, scanned per asset class."""
    fcfg = settings.section("factor_momentum")
    if not fcfg.get("enabled", True):
        return []
    d = settings.data
    class_map = {
        "stocks": AssetClass.STOCK,
        "forex": AssetClass.FOREX,
        "crypto": AssetClass.CRYPTO,
    }
    out: list[Signal] = []

    for cls, ac in class_map.items():
        symbols = settings.universe.get(cls, [])
        if len(symbols) < int(fcfg.get("top_n", 2)) + int(fcfg.get("bottom_n", 2)):
            continue
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                frames[sym] = data_mod.get_ohlcv(
                    sym, ac, timeframe=d.get("timeframe", "1d"),
                    lookback=int(d.get("lookback_bars", 1250)),
                    cache_minutes=int(d.get("cache_minutes", 30)),
                    ccxt_exchange=settings.ccxt_exchange,
                )
            except Exception:
                continue
        frames = {k: v for k, v in frames.items() if v is not None and len(v) > 60}
        if len(frames) < int(fcfg.get("top_n", 2)) + int(fcfg.get("bottom_n", 2)):
            continue

        panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
        stats = factor_mod.backtest_cross_sectional(panel, fcfg)
        passed = factor_mod.passes_factor_gate(stats, fcfg)
        if publish_only and not passed:
            continue

        legs = factor_mod.rank_live(frames, fcfg)
        confidence = risk_mod.confidence_from_backtest(
            stats.win_rate, stats.profit_factor, stats.sharpe
        )
        risk_cfg = settings.risk
        b = _kelly_b(stats.win_rate, stats.profit_factor)
        for leg in legs:
            sizing = risk_mod.size_position(
                equity=settings.account_equity, entry=leg.entry,
                risk_per_unit=leg.risk_per_unit, win_rate=stats.win_rate,
                avg_win_R=b, avg_loss_R=1.0,
                risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 1.0)),
                kelly_fraction_mult=float(risk_cfg.get("kelly_fraction", 0.5)),
                max_kelly_pct=float(risk_cfg.get("max_kelly_pct", 5.0)),
            )
            rr = (abs(leg.target - leg.entry) / leg.risk_per_unit
                  if leg.risk_per_unit > 0 else 0.0)
            out.append(Signal(
                symbol=leg.symbol, asset_class=ac, strategy="factor_momentum",
                side=leg.side, entry=round(leg.entry, 6), stop=round(leg.stop, 6),
                target=round(leg.target, 6), risk_per_unit=round(leg.risk_per_unit, 6),
                reward_per_unit=round(abs(leg.target - leg.entry), 6), rr_ratio=round(rr, 3),
                position_units=round(sizing.units, 6),
                position_notional=round(sizing.notional, 2),
                risk_amount=round(sizing.risk_amount, 2), atr=round(leg.atr, 6),
                regime="cross-sectional",
                rationale=(f"Ranked #{leg.rank} by 6-1 momentum "
                           f"({leg.score*100:+.1f}%) among {cls} — "
                           f"{'strongest→long' if leg.side is Side.LONG else 'weakest→short'}."),
                backtest=stats, confidence=confidence, passed_gate=passed,
                timeframe=d.get("timeframe", "1d"), price_at_signal=round(leg.entry, 6),
            ))
    return out


# ---------------------------------------------------------------------
def scan_universe(
    settings: Settings,
    classes: Optional[list[str]] = None,
    include_pairs: bool = True,
) -> list[Signal]:
    """Scan the whole configured universe. Returns gated signals only."""
    uni = settings.universe
    classes = classes or ["stocks", "forex", "crypto"]
    class_map = {
        "stocks": AssetClass.STOCK,
        "forex": AssetClass.FOREX,
        "crypto": AssetClass.CRYPTO,
    }
    signals: list[Signal] = []
    for cls in classes:
        ac = class_map.get(cls)
        if ac is None:
            continue
        for sym in uni.get(cls, []):
            try:
                signals.extend(scan_symbol(sym, ac, settings, publish_only=True))
            except Exception:
                # one bad symbol must never break a full scan
                continue
    if include_pairs:
        try:
            signals.extend(scan_pairs(settings, publish_only=True))
        except Exception:
            pass
    if include_pairs:  # also run the cross-sectional factor on a full scan
        try:
            signals.extend(scan_factor(settings, publish_only=True))
        except Exception:
            pass

    # best signals first
    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals
