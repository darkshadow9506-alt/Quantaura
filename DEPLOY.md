# Deploying QuantAura 24/7 (always-on)

The bot is a normal long-running Python process. To keep it running around
the clock you need a machine that's always on with open internet access to
**api.telegram.org** and your data sources (Yahoo Finance, Toobit).

> **You probably already have a server.** A "v2ray server" *is* a Linux
> VPS hosted abroad that you already pay for. You can run QuantAura on that
> same server for **no extra cost** — and because it's outside a censored
> region, Telegram and the market-data APIs are reachable directly. Run the
> bot on the **server** (the foreign exit node), not on your home machine.

Pick one of the options below. **Option A (systemd) is the recommended,
easiest "set and forget".**

---

## Option A — systemd on your existing VPS (recommended)

One command. Auto-restarts on crash and on server reboot.

```bash
# on the server (ssh in first)
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip
git clone <your-repo-url> quantaura && cd quantaura
cp .env.example .env
nano .env                 # paste your TELEGRAM_BOT_TOKEN, save (Ctrl+O, Enter, Ctrl+X)
sudo bash deploy/install.sh
```

That's it. Useful commands afterwards:

```bash
journalctl -u quantaura -f          # live logs
sudo systemctl status quantaura     # is it running?
sudo systemctl restart quantaura    # restart
sudo systemctl stop quantaura       # stop
```

To update later:

```bash
cd quantaura && git pull && sudo bash deploy/install.sh
```

---

## Option B — Docker on your VPS

If you prefer containers:

```bash
sudo apt install -y docker.io docker-compose-plugin
git clone <your-repo-url> quantaura && cd quantaura
cp .env.example .env && nano .env   # add your token
docker compose up -d --build
```

`restart: always` keeps it alive across crashes and reboots. The journal
database and OHLCV cache persist in named volumes.

```bash
docker compose logs -f              # live logs
docker compose restart
docker compose down                 # stop
```

---

## Option C — dead-simple (tmux), no root needed

Least robust (won't auto-start after a reboot) but fine for a quick run:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
tmux new -s quantaura               # start a persistent session
python -m quantaura bot
# detach with: Ctrl-b then d   (the bot keeps running)
# reattach later: tmux attach -t quantaura
```

---

## Genuinely-free alternatives (if you don't want to use your VPS)

- **Oracle Cloud Free Tier** — an *always-free* small VM (ARM Ampere or
  AMD micro). Enough to run this bot 24/7 at no cost. Create a VM, then
  follow Option A.
- **Google Cloud free `e2-micro`**, **Fly.io**, **Railway**, **Render** —
  small free allowances; follow Option A or B. Check current limits.
- **Your home PC / a Raspberry Pi** that stays on also works (Option A/C).

> Avoid trying to run a persistent bot on GitHub Actions / serverless
> function platforms — they kill long-running processes.

---

## Notes & troubleshooting

- **Token:** get it from [@BotFather](https://t.me/BotFather) (`/newbot`),
  put it in `.env` as `TELEGRAM_BOT_TOKEN=...`.
- **Restrict access:** set `TELEGRAM_ALLOWED_USERS` to your numeric id
  (from [@userinfobot](https://t.me/userinfobot)) so only you can use it.
- **Scheduled signals:** message the bot `/subscribe`, or set
  `TELEGRAM_BROADCAST_CHAT_ID`. The scheduler pushes a scan every 6h.
- **`Conflict: terminated by other getUpdates`** — the bot is running in
  two places at once. Stop the duplicate (only one instance per token).
- **No data / 403 from Yahoo or Toobit** — your server's network blocks
  that host. Use a server with open egress (most VPSes are fine).
- **Persistence:** the journal/track-record lives in `quantaura_state.db`
  (or the Docker `quantaura_state` volume). Back it up to keep history.
