"""QuantAura — a backtest-validated quant trading signal bot.

Pipeline:  data -> indicators -> strategies (gated by regime) -> risk
           -> backtest validation -> Signal -> Telegram.

See README.md for the methodology behind every strategy.
"""

__version__ = "1.0.0"
