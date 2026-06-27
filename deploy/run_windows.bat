@echo off
REM ============================================================
REM  QuantAura - Windows launcher
REM  Double-click this file to run the bot. It creates a virtual
REM  environment, installs dependencies the first time, then keeps
REM  the bot running (auto-restarts if it crashes).
REM
REM  Before first run:
REM    1) Install Python from python.org (tick "Add Python to PATH").
REM    2) Copy .env.example to .env and fill in TELEGRAM_BOT_TOKEN
REM       and PROXY_URL (your v2rayN local SOCKS, e.g.
REM       socks5://127.0.0.1:10808).
REM    3) Make sure v2rayN is connected.
REM ============================================================
chcp 65001 >nul
set PYTHONUTF8=1

REM go to the project root (this file lives in deploy\)
cd /d "%~dp0\.."

if not exist ".env" (
  echo [!] .env not found. Copy .env.example to .env and add your token first.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo [*] Creating virtual environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo [*] Installing / updating dependencies (first run can take a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

echo [*] Self-test...
python -m quantaura selftest

echo.
echo [*] Starting the bot. Keep this window open. Press Ctrl+C to stop.
echo.
:loop
python -m quantaura bot
echo.
echo [!] Bot stopped. Restarting in 10 seconds... (close this window to quit)
timeout /t 10 /nobreak >nul
goto loop
