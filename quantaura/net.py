"""Outbound proxy configuration.

When PROXY_URL is set (e.g. the local SOCKS proxy your v2ray client
exposes, like socks5://127.0.0.1:10808), route the bot's traffic through
it. Setting the standard proxy environment variables makes the
requests-based libraries (yfinance, ccxt) use it automatically; the
Telegram client is configured explicitly in bot.py.
"""
from __future__ import annotations

import os


def apply_proxy(proxy_url: str) -> str:
    """Export proxy env vars so requests/httpx/curl_cffi pick them up.

    Returns the url applied (or "" if none). Safe to call repeatedly.
    """
    url = (proxy_url or "").strip()
    if not url:
        return ""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = url
    return url
