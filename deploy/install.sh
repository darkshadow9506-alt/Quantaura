#!/usr/bin/env bash
# ============================================================
#  QuantAura — one-command always-on install (systemd).
#  Run this ON your server (e.g. your existing v2ray VPS):
#
#     git clone <repo> quantaura && cd quantaura
#     cp .env.example .env && nano .env        # add your bot token
#     sudo bash deploy/install.sh
#
#  Re-run it any time to update after `git pull`.
# ============================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${APP_DIR}/.venv"
SERVICE_NAME="quantaura"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> QuantAura install"
echo "    App dir: ${APP_DIR}"

# --- python venv + deps ---------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "!! python3 not found. Install it first:  apt update && apt install -y python3 python3-venv python3-pip"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "==> Creating virtualenv"
  python3 -m venv "${VENV_DIR}"
fi
echo "==> Installing dependencies (this can take a few minutes)"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"

# --- .env check ------------------------------------------------------
if [ ! -f "${APP_DIR}/.env" ]; then
  echo "==> No .env found — creating one from the template"
  cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
fi
if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${APP_DIR}/.env"; then
  echo "!! TELEGRAM_BOT_TOKEN is empty in ${APP_DIR}/.env"
  echo "   Edit it (nano ${APP_DIR}/.env), then re-run this script."
  exit 1
fi

# --- offline sanity check -------------------------------------------
echo "==> Running offline self-test"
"${VENV_DIR}/bin/python" -m quantaura selftest >/dev/null && echo "    self-test OK"

# --- systemd unit ----------------------------------------------------
echo "==> Installing systemd service: ${UNIT_PATH}"
sed -e "s|__APP_DIR__|${APP_DIR}|g" \
    -e "s|__VENV_PY__|${VENV_DIR}/bin/python|g" \
    "${APP_DIR}/deploy/quantaura.service" | sudo tee "${UNIT_PATH}" >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

sleep 2
echo
echo "==> Done. The bot is running and will auto-start on reboot."
echo "    Status:  sudo systemctl status ${SERVICE_NAME}"
echo "    Logs:    journalctl -u ${SERVICE_NAME} -f"
echo "    Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo
sudo systemctl --no-pager --lines=10 status "${SERVICE_NAME}" || true
