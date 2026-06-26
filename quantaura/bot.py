"""Telegram bot front-end for QuantAura.

Commands:
  /start, /help        — usage
  /scan [stocks|forex|crypto|all]  — scan the universe, push gated signals
  /signal SYMBOL       — analyse one symbol now (e.g. /signal AAPL,
                         /signal EURUSD=X, /signal BTC/USDT)
  /pairs               — scan the cointegration pairs
  /factor              — scan the cross-sectional momentum factor
  /ml [SYMBOL]         — gradient-boosting model signal(s)
  /status              — show the active configuration

Heavy work (network + backtests) runs in a worker thread via
asyncio.to_thread so the event loop never blocks. If a broadcast chat id
and the job-queue extra are configured, the bot also pushes a scheduled
scan automatically.

All Markdown sends fall back to plain text if Telegram rejects the
entities, so a stray character in a rationale can never break a command.
"""
from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from . import engine
from . import portfolio as portfolio_mod
from .config import Settings
from .data import asset_class_of
from .formatting import format_scan_summary, format_signal

log = logging.getLogger("quantaura.bot")

_MAX_DETAIL_SIGNALS = 8  # cap how many full cards we push per scan


# ---------------------------------------------------------------------
def _plain(text: str) -> str:
    return text.replace("*", "").replace("`", "")


async def _reply(message, text: str, md: bool = True) -> None:
    """Reply, preferring Markdown but falling back to plain text on error."""
    if md:
        try:
            await message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            return
        except BadRequest:
            log.warning("Markdown reply rejected; sending plain text.")
    await message.reply_text(_plain(text))


async def _broadcast(bot, chat_id: str, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except BadRequest:
        await bot.send_message(chat_id=chat_id, text=_plain(text))


def _authorized(settings: Settings, update: Update) -> bool:
    allowed = settings.telegram_allowed_users
    if not allowed:
        return True
    user = update.effective_user
    return bool(user and user.id in allowed)


async def _guard(settings: Settings, update: Update) -> bool:
    if not _authorized(settings, update):
        await update.message.reply_text("⛔ You are not authorized to use this bot.")
        return False
    return True


# ---------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    text = (
        "*QuantAura* — backtest-validated quant signals.\n\n"
        "Every signal carries a precise *entry / stop / target* and is only "
        "published if the strategy has a measured edge on that instrument's "
        "own history.\n\n"
        "*Commands*\n"
        "• `/scan [stocks|forex|crypto|all]` — scan & push gated signals\n"
        "• `/signal SYMBOL` — analyse one symbol now\n"
        "• `/pairs` — scan cointegration pairs\n"
        "• `/factor` — scan the cross-sectional momentum factor\n"
        "• `/ml [SYMBOL]` — gradient-boosting model signal(s)\n"
        "• `/status` — show configuration\n\n"
        "Examples: `/signal AAPL` · `/signal EURUSD=X` · `/signal BTC/USDT`"
    )
    await _reply(update.message, text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    uni = settings.universe
    g = settings.signal_gate
    text = (
        "*QuantAura status*\n"
        f"Timeframe: `{settings.data.get('timeframe')}`\n"
        f"Universe: {len(uni.get('stocks', []))} stocks, "
        f"{len(uni.get('forex', []))} FX, {len(uni.get('crypto', []))} crypto\n"
        "Strategies: `trend, macd, dual_thrust, squeeze, mean_reversion, "
        "pairs, factor_momentum, ml_gboost`\n"
        f"Trailing stop: {settings.risk.get('use_trailing_stop')} "
        f"({settings.risk.get('trail_atr_mult')}×ATR)\n"
        f"Gate: ≥{g.get('min_backtest_trades')} trades, "
        f"win≥{float(g.get('min_win_rate',0))*100:.0f}%, "
        f"PF≥{g.get('min_profit_factor')}, "
        f"P(profit)≥{float(g.get('min_prob_profitable',0))*100:.0f}%\n"
        f"Risk: {settings.risk.get('risk_per_trade_pct')}%/trade, "
        f"{settings.risk.get('kelly_fraction')}×Kelly\n"
        f"Account equity: ${settings.account_equity:,.0f}"
    )
    await _reply(update.message, text)


async def _send_signals(update_or_chat, context, signals, header: str | None = None,
                        portfolio: bool = False):
    if header:
        await _reply(update_or_chat,
                     format_scan_summary(signals) if signals else header)
    for sig in signals[:_MAX_DETAIL_SIGNALS]:
        await _reply(update_or_chat, format_signal(sig, md=True))
    if portfolio and signals:
        settings: Settings = context.application.bot_data["settings"]
        summary = portfolio_mod.summarize(
            signals, settings.account_equity,
            max_risk_pct=float(settings.risk.get("portfolio_max_risk_pct", 6.0)),
            max_per_class=int(settings.risk.get("max_open_per_class", 5)))
        text = portfolio_mod.format_summary(summary, settings.account_equity)
        if text:
            await _reply(update_or_chat, text)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    arg = (context.args[0].lower() if context.args else "all")
    classes = None if arg in ("all", "") else [arg]
    if classes and arg not in ("stocks", "forex", "crypto"):
        await update.message.reply_text(
            "Usage: /scan [stocks|forex|crypto|all]"
        )
        return

    await update.message.reply_text(
        f"🔎 Scanning {'all markets' if not classes else arg}… this can take a minute."
    )
    signals = await asyncio.to_thread(
        engine.scan_universe, settings, classes, True
    )
    await _reply(update.message, format_scan_summary(signals))
    await _send_signals(update.message, context, signals, portfolio=True)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /signal SYMBOL  (e.g. /signal AAPL)")
        return
    symbol = context.args[0].strip().upper()
    ac = asset_class_of(symbol, settings.universe)
    await update.message.reply_text(f"🔎 Analysing {symbol} ({ac.value})…")
    try:
        # publish_only=False so the user sees the analysis even if it
        # doesn't clear the gate (clearly flagged as such).
        signals = await asyncio.to_thread(
            engine.scan_symbol, symbol, ac, settings, False
        )
    except Exception as exc:
        await update.message.reply_text(f"⚠️ Could not analyse {symbol}: {exc}")
        return
    if not signals:
        await update.message.reply_text(
            f"No active setup on {symbol} right now. No trigger from any strategy "
            f"on the latest bar — the model stays flat."
        )
        return
    await _send_signals(update.message, context, signals, portfolio=True)


async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    await update.message.reply_text("🔎 Scanning cointegration pairs…")
    signals = await asyncio.to_thread(engine.scan_pairs, settings, True)
    await _reply(update.message, format_scan_summary(signals))
    await _send_signals(update.message, context, signals, portfolio=True)


async def cmd_factor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    await update.message.reply_text("🔎 Scanning the cross-sectional momentum factor…")
    signals = await asyncio.to_thread(engine.scan_factor, settings, True)
    await _reply(update.message, format_scan_summary(signals))
    await _send_signals(update.message, context, signals, portfolio=True)


async def cmd_ml(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not await _guard(settings, update):
        return
    if context.args:
        symbol = context.args[0].strip().upper()
        ac = asset_class_of(symbol, settings.universe)
        await update.message.reply_text(f"🤖 Training the ML model on {symbol}…")
        signals = await asyncio.to_thread(engine.scan_ml_symbol, symbol, ac, settings, False)
    else:
        await update.message.reply_text("🤖 Running the ML model across the universe… (slow)")
        signals = await asyncio.to_thread(engine.scan_ml, settings, True)
    await _reply(update.message, format_scan_summary(signals))
    await _send_signals(update.message, context, signals, portfolio=True)


async def _scheduled_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    chat_id = settings.telegram_broadcast_chat_id
    if not chat_id:
        return
    signals = await asyncio.to_thread(engine.scan_universe, settings, None, True)
    if not signals:
        return
    await _broadcast(context.bot, chat_id, format_scan_summary(signals))
    for sig in signals[:_MAX_DETAIL_SIGNALS]:
        await _broadcast(context.bot, chat_id, format_signal(sig, md=True))
    summary = portfolio_mod.summarize(
        signals, settings.account_equity,
        max_risk_pct=float(settings.risk.get("portfolio_max_risk_pct", 6.0)),
        max_per_class=int(settings.risk.get("max_open_per_class", 5)))
    text = portfolio_mod.format_summary(summary, settings.account_equity)
    if text:
        await _broadcast(context.bot, chat_id, text)


# ---------------------------------------------------------------------
def build_application(settings: Settings) -> Application:
    if not settings.telegram_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    app = Application.builder().token(settings.telegram_token).build()
    app.bot_data["settings"] = settings

    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("pairs", cmd_pairs))
    app.add_handler(CommandHandler("factor", cmd_factor))
    app.add_handler(CommandHandler("ml", cmd_ml))

    # optional scheduled broadcast (needs the job-queue extra + a chat id)
    if settings.telegram_broadcast_chat_id and app.job_queue is not None:
        app.job_queue.run_repeating(_scheduled_scan, interval=6 * 3600, first=30)
        log.info("Scheduled scan enabled (every 6h).")

    return app


def run(settings: Settings | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = settings or Settings.load()
    app = build_application(settings)
    log.info("QuantAura bot starting (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
