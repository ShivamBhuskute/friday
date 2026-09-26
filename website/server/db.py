"""SQLite persistence for turns.

One row per voice interaction. The row is created as soon as the audio lands
and then mutated in place as the pipeline advances, which is what lets the UI
render a live status instead of waiting for the final answer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id             TEXT PRIMARY KEY,
    seq            INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    state          TEXT NOT NULL,
    audio_path     TEXT,
    duration_s     REAL,
    sample_rate    INTEGER,
    channels       INTEGER,
    source         TEXT,
    transcript     TEXT,
    confidence     REAL,
    answer         TEXT,
    error          TEXT,
    tool_calls     TEXT NOT NULL DEFAULT '[]',
    transcript_ms  INTEGER,
    llm_ms         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_turns_created ON turns (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_turns_seq     ON turns (seq DESC);
"""

# Terminal states: no further updates expected.
TERMINAL_STATES = frozenset({"done", "error", "no_speech", "unclear"})
# The same set as a sorted tuple, for use as SQL placeholders. Sorted so the
# query text (and any statement cache keyed on it) is stable between runs.
TERMINAL_DB_STATES: tuple[str, ...] = tuple(sorted(TERMINAL_STATES))


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Database:
    """Thread-safe SQLite wrapper.

    The pipeline runs on a worker thread while the HTTP layer reads from the
    event loop, so every statement is serialised behind one lock and each
    thread gets its own connection.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ write
    def create_turn(self, **fields: Any) -> dict:
        """Insert a new ``uploaded`` turn and return it."""
        turn_id = fields.get("id") or uuid.uuid4().hex[:16]
        now = utcnow()
        with self._write_lock:
            conn = self._connect()
            seq = (conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM turns").fetchone()[0])
            conn.execute(
                """
                INSERT INTO turns (id, seq, created_at, updated_at, state, audio_path,
                                   duration_s, sample_rate, channels, source)
                VALUES (?, ?, ?, ?, 'uploaded', ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    seq,
                    now,
                    now,
                    fields.get("audio_path"),
                    fields.get("duration_s"),
                    fields.get("sample_rate"),
                    fields.get("channels"),
                    fields.get("source"),
                ),
            )
            conn.commit()
        return self.get_turn(turn_id)  # type: ignore[return-value]

    def update_turn(self, turn_id: str, **fields: Any) -> dict | None:
        """Patch a turn. ``None`` values are ignored so partial updates are safe."""
        allowed = {
            "state",
            "audio_path",
            "duration_s",
            "sample_rate",
            "channels",
            "source",
            "transcript",
            "confidence",
            "answer",
            "error",
            "transcript_ms",
            "llm_ms",
        }
        sets: list[str] = []
        args: list[Any] = []
        for key, value in fields.items():
            if key == "tool_calls":
                sets.append("tool_calls = ?")
                args.append(json.dumps(value))
            elif key in allowed and value is not None:
                sets.append(f"{key} = ?")
                args.append(value)
        if not sets:
            return self.get_turn(turn_id)

        sets.append("updated_at = ?")
        args.append(utcnow())
        args.append(turn_id)
        with self._write_lock:
            conn = self._connect()
            conn.execute(f"UPDATE turns SET {', '.join(sets)} WHERE id = ?", args)
            conn.commit()
        return self.get_turn(turn_id)

    def delete_turn(self, turn_id: str) -> bool:
        with self._write_lock:
            conn = self._connect()
            cur = conn.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------- read
    def get_turn(self, turn_id: str) -> dict | None:
        row = self._connect().execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_turns(self, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM turns ORDER BY seq DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def latest_turn(self) -> dict | None:
        row = self._connect().execute("SELECT * FROM turns ORDER BY seq DESC LIMIT 1").fetchone()
        return _row_to_dict(row) if row else None

    def count(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    def all_audio_paths(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT audio_path FROM turns WHERE audio_path IS NOT NULL"
        ).fetchall()
        return [r[0] for r in rows]

    def recover_interrupted(self) -> list[str]:
        """Fail any turn left mid-pipeline by a previous process.

        The worker thread is the only thing that advances a turn past
        ``uploaded``, so a non-terminal state found at startup belongs to a
        process that died -- the device audio is gone and nothing will ever
        finish it. Left alone, the console shows a spinner that never resolves.
        Returns the ids that were recovered.
        """
        rows = self._connect().execute(
            "SELECT id, state FROM turns WHERE state NOT IN (?, ?, ?, ?)",
            TERMINAL_DB_STATES,
        ).fetchall()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        placeholders = ", ".join("?" for _ in ids)
        with self._write_lock:
            conn = self._connect()
            conn.execute(
                f"UPDATE turns SET state = 'error', error = ?, updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [
                    "interrupted: the server restarted while this turn was in flight",
                    utcnow(),
                    *ids,
                ],
            )
            conn.commit()
        return ids

    # -------------------------------------------------------------- retention
    def prune(self, max_turns: int, delete_audio: bool = True) -> list[str]:
        """Trim to the newest ``max_turns``. Returns audio paths that are now orphaned."""
        if max_turns <= 0:
            return []
        keep = {r[0] for r in self._connect().execute(
            "SELECT audio_path FROM turns ORDER BY seq DESC LIMIT ?", (max_turns,)
        )}
        orphans: list[str] = []
        with self._write_lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT id, audio_path FROM turns ORDER BY seq DESC LIMIT -1 OFFSET ?",
                (max_turns,),
            ).fetchall()
            for row in rows:
                if row["audio_path"]:
                    if delete_audio and row["audio_path"] not in keep:
                        orphans.append(row["audio_path"])
                    elif not delete_audio:
                        orphans.append(row["audio_path"])
                conn.execute("DELETE FROM turns WHERE id = ?", (row["id"],))
            conn.commit()
        return orphans

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["tool_calls"] = json.loads(d.get("tool_calls") or "[]")
    except (TypeError, json.JSONDecodeError):
        d["tool_calls"] = []
    return d


def summarise(turns: Iterable[dict]) -> dict:
    """Aggregate stats for the UI vitals strip."""
    turns = list(turns)
    done = [t for t in turns if t.get("state") == "done"]
    stt_ms = [t["transcript_ms"] for t in done if t.get("transcript_ms")]
    llm_ms = [t["llm_ms"] for t in done if t.get("llm_ms")]
    return {
        "total": len(turns),
        "done": len(done),
        "errors": sum(1 for t in turns if t.get("state") == "error"),
        "avg_transcript_ms": round(sum(stt_ms) / len(stt_ms)) if stt_ms else None,
        "avg_llm_ms": round(sum(llm_ms) / len(llm_ms)) if llm_ms else None,
    }
