"""Configuration loading: merges config.yaml with environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # optional, but recommended
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def _get_env_list(name: str) -> list[str]:
    raw = os.getenv(name, "") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    """All runtime settings: strategy params from YAML + secrets from env."""

    cfg: dict[str, Any]

    # --- secrets / env ---
    telegram_token: str = ""
    telegram_allowed_users: list[int] = field(default_factory=list)
    telegram_broadcast_chat_id: str = ""
    account_equity: float = 10_000.0
    ccxt_exchange: str = "toobit"
    proxy_url: str = ""          # e.g. socks5://127.0.0.1:10808 (a v2ray local proxy)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Settings":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        allowed = []
        for uid in _get_env_list("TELEGRAM_ALLOWED_USERS"):
            try:
                allowed.append(int(uid))
            except ValueError:
                continue

        try:
            equity = float(os.getenv("ACCOUNT_EQUITY", "10000"))
        except ValueError:
            equity = 10_000.0

        return cls(
            cfg=cfg,
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_allowed_users=allowed,
            telegram_broadcast_chat_id=os.getenv("TELEGRAM_BROADCAST_CHAT_ID", "").strip(),
            account_equity=equity,
            ccxt_exchange=os.getenv("CCXT_EXCHANGE", "toobit").strip() or "toobit",
            proxy_url=os.getenv("PROXY_URL", "").strip(),
        )

    # --- convenient typed accessors ----------------------------------
    def section(self, name: str) -> dict[str, Any]:
        return self.cfg.get(name, {}) or {}

    @property
    def universe(self) -> dict[str, list[str]]:
        return self.section("universe")

    @property
    def data(self) -> dict[str, Any]:
        return self.section("data")

    @property
    def regime(self) -> dict[str, Any]:
        return self.section("regime")

    @property
    def trend(self) -> dict[str, Any]:
        return self.section("trend")

    @property
    def mean_reversion(self) -> dict[str, Any]:
        return self.section("mean_reversion")

    @property
    def pairs(self) -> dict[str, Any]:
        return self.section("pairs")

    @property
    def risk(self) -> dict[str, Any]:
        return self.section("risk")

    @property
    def signal_gate(self) -> dict[str, Any]:
        return self.section("signal_gate")

    @property
    def journal_db_path(self) -> str:
        """DB path: QUANTAURA_DB env overrides config (handy for Docker volumes)."""
        return (os.getenv("QUANTAURA_DB", "").strip()
                or str(self.section("journal").get("db_path", "") or ""))
