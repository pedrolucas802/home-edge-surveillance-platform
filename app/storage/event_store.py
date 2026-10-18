from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    camera_id TEXT NOT NULL,
    camera_host TEXT,
    event_at_utc TEXT NOT NULL,
    event_at_epoch REAL NOT NULL,
    frame_index INTEGER,
    detections_count INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    detections_json TEXT NOT NULL DEFAULT '[]',
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events (camera_id, event_at_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (event_at_epoch DESC);
"""


class EventDatabase:
    """SQLite-backed store for detection events.

    Multiple camera-publisher processes and the Streamlit dashboard all open
    this same database file concurrently, so it runs in WAL mode with a busy
    timeout instead of assuming a single writer.
    """

    def __init__(self, db_path: Path, logger: logging.Logger | None = None) -> None:
        self.db_path = db_path
        self.logger = (logger or logging.getLogger(__name__)).getChild("event_store")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def insert_event(self, record: dict[str, object]) -> None:
        event_at_utc = str(record.get("event_at_utc") or _utc_now_iso())
        try:
            event_at_epoch = _parse_iso_to_epoch(event_at_utc)
        except ValueError:
            event_at_epoch = time.time()

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO events (
                        event_id, camera_id, camera_host, event_at_utc, event_at_epoch,
                        frame_index, detections_count, counts_json, detections_json, snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.get("event_id")),
                        str(record.get("camera_id", "")),
                        record.get("camera_host"),
                        event_at_utc,
                        event_at_epoch,
                        record.get("frame_index"),
                        int(record.get("detections_count", 0)),
                        json.dumps(record.get("counts", {}), ensure_ascii=True),
                        json.dumps(record.get("detections", []), ensure_ascii=True),
                        record.get("snapshot_path"),
                    ),
                )
        except sqlite3.OperationalError as exc:
            # The raw JSONL log already captured this event, so a dropped row
            # here (e.g. transient lock contention across processes) is not
            # fatal for the publisher loop.
            self.logger.warning("Failed to persist event to SQLite: %s", exc)

    def query_events(
        self,
        camera_ids: list[str] | None = None,
        class_name: str | None = None,
        start_epoch: float | None = None,
        end_epoch: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if camera_ids:
            placeholders = ",".join("?" for _ in camera_ids)
            clauses.append(f"camera_id IN ({placeholders})")
            params.extend(camera_ids)
        if start_epoch is not None:
            clauses.append("event_at_epoch >= ?")
            params.append(start_epoch)
        if end_epoch is not None:
            clauses.append("event_at_epoch <= ?")
            params.append(end_epoch)
        if class_name:
            clauses.append("counts_json LIKE ?")
            params.append(f'%"{class_name}"%')

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM events {where} ORDER BY event_at_epoch DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def purge_older_than(self, cutoff_epoch: float) -> list[str]:
        """Delete events older than cutoff_epoch, returning their snapshot paths."""
        rows = self._conn.execute(
            "SELECT snapshot_path FROM events WHERE event_at_epoch < ?",
            (cutoff_epoch,),
        ).fetchall()
        with self._conn:
            self._conn.execute("DELETE FROM events WHERE event_at_epoch < ?", (cutoff_epoch,))
        return [str(row["snapshot_path"]) for row in rows if row["snapshot_path"]]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"]) if row else 0


def _row_to_record(row: sqlite3.Row) -> dict[str, object]:
    return {
        "event_id": row["event_id"],
        "camera_id": row["camera_id"],
        "camera_host": row["camera_host"],
        "event_at_utc": row["event_at_utc"],
        "frame_index": row["frame_index"],
        "detections_count": row["detections_count"],
        "counts": json.loads(row["counts_json"]) if row["counts_json"] else {},
        "detections": json.loads(row["detections_json"]) if row["detections_json"] else [],
        "snapshot_path": row["snapshot_path"],
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_to_epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()
