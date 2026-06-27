"""Persistence layer (stdlib sqlite3) — no external dependencies.

Stores every published signal so the bot can:
  * deduplicate (don't re-emit the same symbol+strategy+side within a
    cooldown window),
  * track how signals actually resolved (TP / SL / expired) and report a
    live track record,
  * remember Telegram subscribers for scheduled broadcasts.

All methods are synchronous and fast; the bot calls them from worker
threads. The schema is created on first use.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import Signal

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "quantaura_state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    asset_class  TEXT NOT NULL,
    strategy     TEXT NOT NULL,
    side         TEXT NOT NULL,
    entry        REAL NOT NULL,
    stop         REAL NOT NULL,
    target       REAL NOT NULL,
    rr_ratio     REAL,
    confidence   REAL,
    status       TEXT NOT NULL DEFAULT 'open',   -- open | tp | sl | expired
    closed_at    TEXT,
    result_R     REAL
);
CREATE INDEX IF NOT EXISTS idx_signals_open
    ON signals (symbol, strategy, side, status);
CREATE TABLE IF NOT EXISTS subscribers (
    chat_id   INTEGER PRIMARY KEY,
    added_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- signals -------------------------------------------------------
    def has_recent(self, symbol: str, strategy: str, side: str,
                   cooldown_days: float) -> bool:
        """True if an open signal for this key exists within the cooldown."""
        cutoff = datetime.now(timezone.utc).timestamp() - cooldown_days * 86400
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT created_at FROM signals "
                "WHERE symbol=? AND strategy=? AND side=? AND status='open'",
                (symbol, strategy, side),
            )
            for row in cur.fetchall():
                try:
                    ts = datetime.fromisoformat(row["created_at"]).timestamp()
                except ValueError:
                    continue
                if ts >= cutoff:
                    return True
        return False

    def record_signal(self, sig: Signal, cooldown_days: float = 3.0) -> bool:
        """Insert a signal. Returns False (and inserts nothing) if a
        matching open signal exists within the cooldown window."""
        if self.has_recent(sig.symbol, sig.strategy, sig.side.value, cooldown_days):
            return False
        with self._conn:
            self._conn.execute(
                "INSERT INTO signals (created_at, symbol, asset_class, strategy, "
                "side, entry, stop, target, rr_ratio, confidence, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?, 'open')",
                (_now(), sig.symbol, sig.asset_class.value, sig.strategy,
                 sig.side.value, sig.entry, sig.stop, sig.target,
                 sig.rr_ratio, sig.confidence),
            )
        return True

    def clear_signals(self) -> int:
        """Delete every journaled signal (keeps subscribers). Returns count."""
        with self._conn:
            cur = self._conn.execute("DELETE FROM signals")
        return cur.rowcount

    def open_signals(self) -> list[dict]:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT * FROM signals WHERE status='open' ORDER BY id")
            return [dict(r) for r in cur.fetchall()]

    def close_signal(self, signal_id: int, status: str, result_R: float) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE signals SET status=?, closed_at=?, result_R=? WHERE id=?",
                (status, _now(), result_R, signal_id),
            )

    def performance(self) -> dict:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT status, result_R FROM signals")
            rows = cur.fetchall()
        total = len(rows)
        closed = [r for r in rows if r["status"] in ("tp", "sl", "expired")]
        wins = [r for r in closed if r["status"] == "tp"]
        losses = [r for r in closed if r["status"] == "sl"]
        rs = [r["result_R"] for r in closed if r["result_R"] is not None]
        n_closed = len(closed)
        win_rate = len(wins) / n_closed if n_closed else 0.0
        avg_R = sum(rs) / len(rs) if rs else 0.0
        total_R = sum(rs) if rs else 0.0
        return {
            "total": total,
            "open": total - n_closed,
            "closed": n_closed,
            "tp": len(wins),
            "sl": len(losses),
            "win_rate": round(win_rate, 4),
            "avg_R": round(avg_R, 4),
            "total_R": round(total_R, 4),
        }

    # -- subscribers ---------------------------------------------------
    def add_subscriber(self, chat_id: int) -> bool:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO subscribers (chat_id, added_at) VALUES (?,?)",
                    (int(chat_id), _now()),
                )
            return True
        except sqlite3.Error:
            return False

    def remove_subscriber(self, chat_id: int) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM subscribers WHERE chat_id=?", (int(chat_id),))
        return cur.rowcount > 0

    def subscribers(self) -> list[int]:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT chat_id FROM subscribers ORDER BY chat_id")
            return [int(r["chat_id"]) for r in cur.fetchall()]
