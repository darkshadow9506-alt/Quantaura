# Deploying QuantAura 24/7 (always-on)

The bot is a normal long-running Python process. To keep it running around
the clock you need a machine that's always on with open internet access to
**api.telegram.org** and your data sources (Yahoo Finance, Toobit).

# Which situation are you in?

There are two very different things people call a "server":

1. **A real VPS** — you have an SSH login and root on a Linux box abroad
   (DigitalOcean, Hetzner, Oracle Cloud, etc.). You *can* run software on
   it → use **Option A / B** below. This is the best setup.

2. **Just a v2ray "config" / subscription** — you bought proxy access from
   a seller. This is **only a tunnel; you do NOT control that server and
   cannot run the bot on it.** Instead, run the bot on **your own
   computer** (or a cheap always-on device) and route its traffic through
   your v2ray config so it can reach Telegram from a censored network →
   use **Option D (proxy)**. For truly always-on without leaving your PC
   on, get a **free** real VPS (Oracle Cloud Always-Free) and use Option A.

Pick the matching option below.

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

## Option D — run on your own PC through a v2ray config (no VPS)

Use this if all you have is a v2ray **config** (proxy access), not a server
you control. The bot runs on your computer; its internet traffic goes out
through your v2ray client, so it can reach Telegram from a censored network.

1. **Run your v2ray client** (v2rayN, Nekoray, Hiddify, v2rayNG…) and import
   your config so it's connected. The client exposes a **local proxy** —
   note its SOCKS port (commonly `10808`, sometimes `2080`/`12334`). In
   v2rayN/Nekoray it's shown under settings as the "local listening" /
   inbound SOCKS port.

2. **Install and configure the bot:**

   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

   Edit `.env` and set your token **and** the proxy:

   ```
   TELEGRAM_BOT_TOKEN=123456789:ABC...
   PROXY_URL=socks5://127.0.0.1:10808     # <- your v2ray client's local SOCKS port
   ```

3. **Run it:**

   ```bash
   python -m quantaura bot
   ```

   The bot now reaches Telegram + Yahoo/Toobit through your config. Keep the
   v2ray client connected and the bot process running.

To keep it always-on on your PC: on Linux/Mac use **Option C (tmux)**; on
Windows, leave the terminal open or use Task Scheduler. (Your PC must stay
on. For 24/7 without that, use a free VM — see below — and you won't need
the proxy at all.)

---

## ⭐ Oracle Cloud Always-Free — step by step (recommended for 24/7)

A genuinely-free, always-on Linux VM abroad. No proxy needed (Telegram and
the data APIs are reachable directly), and your PC doesn't have to stay on.

1. **Create the account:** sign up at <https://www.oracle.com/cloud/free/>.
   A credit/debit card is required for identity verification only — the
   **Always Free** resources are not charged. Pick a home region close to
   you (any works).

2. **Create the VM:** Console → *Compute → Instances → Create instance*.
   - **Image:** Ubuntu 22.04.
   - **Shape:** click *Change shape* and choose an **Always Free-eligible**
     one — either `VM.Standard.A1.Flex` (ARM Ampere, e.g. 1–2 OCPU / 6–12 GB,
     plenty) or `VM.Standard.E2.1.Micro` (AMD). If ARM says "out of
     capacity", try the AMD micro or another availability domain.
   - **SSH keys:** let it generate a key pair and **download the private
     key** (or paste your own public key).
   - Click **Create**. Note the instance's **public IP**.

3. **Connect (SSH):**
   ```bash
   chmod 600 your-key.key
   ssh -i your-key.key ubuntu@YOUR_PUBLIC_IP
   ```
   (On Windows use PuTTY or `ssh` in PowerShell.)

4. **Install the bot (one block):**
   ```bash
   sudo apt update && sudo apt install -y git python3 python3-venv python3-pip
   git clone <your-repo-url> quantaura && cd quantaura
   cp .env.example .env
   nano .env          # paste TELEGRAM_BOT_TOKEN (leave PROXY_URL empty), save
   sudo bash deploy/install.sh
   ```

5. **Done.** The bot auto-starts on boot and restarts on crash. In Telegram,
   message your bot `/start` then `/subscribe`. Logs: `journalctl -u quantaura -f`.

Notes:
- **No inbound ports needed** — the bot uses outbound long-polling, so you
  don't have to open any firewall/security-list ports.
- ARM (aarch64) is fine: all the Python wheels (numpy, pandas, scikit-learn,
  statsmodels, scipy) have ARM builds.
- Leave `PROXY_URL` empty on Oracle — the proxy is only for running behind a
  censored network on your own PC.

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
