"""Command-line interface.

Usage:
  python -m quantaura bot                     # run the Telegram bot
  python -m quantaura scan [--class stocks]   # scan & print signals
  python -m quantaura signal AAPL             # analyse one symbol
  python -m quantaura pairs                    # scan cointegration pairs
  python -m quantaura selftest                # offline self-check (no network)
"""
from __future__ import annotations

import argparse
import sys

from .config import Settings
from .data import asset_class_of
from .formatting import format_scan_summary, format_signal


def _cmd_scan(args, settings: Settings) -> int:
    from . import engine

    classes = None if args.cls in (None, "all") else [args.cls]
    signals = engine.scan_universe(settings, classes, include_pairs=True)
    print(format_scan_summary(signals).replace("*", "").replace("`", ""))
    print()
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_signal(args, settings: Settings) -> int:
    from . import engine

    sym = args.symbol.strip().upper()
    ac = asset_class_of(sym, settings.universe)
    try:
        signals = engine.scan_symbol(sym, ac, settings, publish_only=False)
    except Exception as exc:
        print(f"Could not analyse {sym}: {exc}")
        return 1
    if not signals:
        print(f"No active setup on {sym} right now.")
        return 0
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_pairs(args, settings: Settings) -> int:
    from . import engine

    signals = engine.scan_pairs(settings, publish_only=False)
    if not signals:
        print("No pair setups right now.")
        return 0
    for s in signals:
        print(format_signal(s, md=False))
        print("-" * 60)
    return 0


def _cmd_bot(args, settings: Settings) -> int:
    from .bot import run

    run(settings)
    return 0


def _cmd_selftest(args, settings: Settings) -> int:
    from .selftest import run_selftest

    ok = run_selftest()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantaura", description="QuantAura signal bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan the universe")
    p_scan.add_argument("--class", dest="cls", default="all",
                        choices=["all", "stocks", "forex", "crypto"])

    p_sig = sub.add_parser("signal", help="analyse one symbol")
    p_sig.add_argument("symbol")

    sub.add_parser("pairs", help="scan cointegration pairs")
    sub.add_parser("bot", help="run the Telegram bot")
    sub.add_parser("selftest", help="offline self-check (no network)")

    args = parser.parse_args(argv)
    settings = Settings.load()

    dispatch = {
        "scan": _cmd_scan,
        "signal": _cmd_signal,
        "pairs": _cmd_pairs,
        "bot": _cmd_bot,
        "selftest": _cmd_selftest,
    }
    return dispatch[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
