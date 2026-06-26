# QuantAura — a backtest-validated quant trading signal bot

QuantAura turns real market data into **precise, fully-specified trade
signals** (exact entry, stop, target and position size) and delivers them
over **Telegram**. It is built on *published, mainstream* quantitative
strategies — trend / time-series momentum, short-horizon mean reversion,
and cointegration pairs (statistical arbitrage) — with strict risk
management on top.

> **Read this first.** QuantAura is an educational tool. Every signal is a
> *backtested probability*, not a promise. Real quant funds win on large
> samples of small edges, with leverage, infrastructure and risk controls.
> No honest system can give you "100% safe, guaranteed" entries — anyone
> claiming that is lying. What QuantAura *does* guarantee is that every
> number it shows you is computed from real data by a transparent, tested
> rule, and that it stays silent unless a setup has a **measured edge on
> that instrument's own history**.

---

## Why these strategies (and not "magic")

The strategy set is taken directly from how quantitative trading actually
works (see *What Is Quantitative Trading?* and the strategy taxonomy that
shipped with this project). The robust, well-documented retail-accessible
edges are:

| Strategy | Family | Edge source | Horizon |
|---|---|---|---|
| **Trend breakout** | Time-series momentum / trend following | Persistence of price moves (the most robust cross-asset anomaly; the core of CTAs like Man AHL) | weeks–months |
| **Mean reversion** | Short-horizon reversal | Temporary supply/demand imbalances revert to fair value | days |
| **Pairs / stat-arb** | Statistical arbitrage | Cointegrated assets converge after diverging | days–weeks |

Everything QuantAura does maps onto the standard quant pipeline:
**data → signal research → backtest → execution → risk management.** We
deliberately *do not* implement HFT or market making — those need
co-located servers, microsecond latency and FPGAs, and are impossible to
run honestly from a Telegram bot. Pretending otherwise would be the
"made-up" behaviour you explicitly asked to avoid.

### The three engines

**1. Trend breakout (momentum).** Donchian-channel breakout filtered by the
200-period moving average. Go long only when price breaks above the prior
20-bar high *and* trades above its 200-MA (and the mirror image for
shorts). Initial stop = `2.5 × ATR`; target = `2R`. This is textbook
trend-following.

**2. Mean reversion.** Bollinger / z-score stretch + Connors **RSI-2**,
taken *only with* the long-term trend (buy oversold dips inside an uptrend,
fade overbought spikes inside a downtrend). Stop = `3 × ATR`; target = the
moving-average mean (reversion to fair value).

**3. Pairs / statistical arbitrage.** Engle–Granger cointegration test +
OLS hedge ratio + z-score of the spread (Ornstein–Uhlenbeck-style
reversion). Only trades pairs that are statistically cointegrated
(p ≤ 0.05). Long the cheap leg, short the rich leg; exit toward the mean,
stop on a structural break (z ≥ 3.5).

A **regime filter (ADX)** decides which engine is even allowed to fire on a
symbol: trending markets → momentum, ranging markets → mean reversion. This
stops the two engines from fighting each other.

### How signals stay honest — the backtest gate

A raw strategy trigger is **not published** unless the *same strategy*,
backtested on the *same instrument's* history, clears all of:

- ≥ 15 trades (a meaningful sample, guards against curve-fits)
- win rate ≥ 40%
- profit factor ≥ 1.15
- significance score ("Sharpe") ≥ 0.3

The backtester is **event-driven and look-ahead-free**: entries use only
data up to the signal bar; exits walk forward bar-by-bar; if a bar's range
spans both stop and target we pessimistically assume the stop filled first.
Thresholds live in `config.yaml → signal_gate` and are fully tunable.

### Risk management

- **Fixed-fractional sizing:** never risk more than `risk_per_trade_pct`
  (default 1%) of equity per trade.
- **Fractional Kelly:** size is *also* scaled by the measured edge using
  **half-Kelly**, capped at `max_kelly_pct`. The final size is the more
  conservative of the two. Full Kelly is intentionally avoided — it is far
  too aggressive on noisy, non-stationary estimates.

---

## Install

```bash
git clone <your-fork-url> QuantAura
cd QuantAura
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Python 3.10+ required.

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

1. **Create a Telegram bot:** open Telegram → talk to
   [@BotFather](https://t.me/BotFather) → `/newbot` → follow the prompts →
   copy the token (`123456789:ABC...`) into `TELEGRAM_BOT_TOKEN`.
2. *(optional)* Put your numeric Telegram id in `TELEGRAM_ALLOWED_USERS`
   (get it from [@userinfobot](https://t.me/userinfobot)) to lock the bot
   to just you.
3. *(optional)* Set `TELEGRAM_BROADCAST_CHAT_ID` to auto-push a scan every
   6 hours.
4. Set `ACCOUNT_EQUITY` so position-size suggestions match your account.

Tune the **universe, strategy parameters, risk and gate** in
`config.yaml` — every knob is documented inline.

## Run

```bash
# 1) Offline self-check (no network, proves the whole pipeline works)
python -m quantaura selftest

# 2) Terminal scans (need market-data access)
python -m quantaura scan --class stocks
python -m quantaura signal AAPL
python -m quantaura signal EURUSD=X
python -m quantaura signal BTC/USDT
python -m quantaura pairs

# 3) Start the Telegram bot
python -m quantaura bot
```

### Telegram commands

| Command | Action |
|---|---|
| `/start`, `/help` | usage |
| `/scan [stocks\|forex\|crypto\|all]` | scan the universe, push gated signals |
| `/signal SYMBOL` | analyse one symbol now (e.g. `/signal AAPL`) |
| `/pairs` | scan the cointegration pairs |
| `/status` | show the active configuration |

A signal card shows side, entry/stop/target, R:R, ATR, suggested size and
dollar risk, the originating strategy's backtest record, and a confidence
bar derived from that backtest.

## Data sources

- **Stocks / ETFs / FX / indices →** [yfinance](https://github.com/ranaroussi/yfinance) (free, end-of-day + intraday).
- **Crypto →** [ccxt](https://github.com/ccxt/ccxt) (default exchange Binance; change via `CCXT_EXCHANGE`).

OHLCV is cached locally for `cache_minutes` to respect rate limits. The
default timeframe is **daily** — the horizon where retail quant edges are
real and execution speed doesn't dominate.

> If you run in a restricted/proxied network and data fetches return 403,
> the destination (e.g. Yahoo/Binance) is blocked by that network's egress
> policy, not by QuantAura. Run on an open network or a VPS.

## Tests

```bash
pytest -q            # 23 unit/integration tests (synthetic data, no network)
python -m quantaura selftest
```

## Project layout

```
quantaura/
  config.py        settings (YAML + .env)
  data.py          unified OHLCV providers (yfinance / ccxt) + cache
  indicators.py    ATR, RSI, Bollinger, z-score, ADX, Donchian (look-ahead-free)
  strategies.py    TrendBreakout, MeanReversion, ADX regime filter
  pairs.py         Engle-Granger cointegration pairs (stat-arb)
  risk.py          fixed-fractional + fractional-Kelly sizing
  backtest.py      event-driven, look-ahead-free backtester + R-metrics
  engine.py        orchestration: data -> signal, with the backtest gate
  formatting.py    Telegram/CLI signal rendering
  bot.py           Telegram front-end
  cli.py           command-line interface
  selftest.py      offline end-to-end validation
tests/             pytest suite
config.yaml        all strategy / risk / gate parameters
```

## Honest limitations

- Daily EOD signals; not for scalping or HFT.
- yfinance/ccxt are free data feeds — fine for research, not institutional
  quality.
- Backtests are realistic but still optimistic vs. live trading (slippage,
  fees, borrow, gaps and partial fills are only partially modelled).
- A measured historical edge can decay; markets are non-stationary.
- **Not financial advice.** You are responsible for every trade you take.

## License

MIT — see `LICENSE`.
