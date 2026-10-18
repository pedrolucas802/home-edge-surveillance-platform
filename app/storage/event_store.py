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
CREATE TABLE IF NOT EXISTS device_state (
    device_id TEXT PRIMARY KEY,
    device_type TEXT NOT NULL,
    status TEXT,
    state_json TEXT NOT NULL DEFAULT '{}',
    telemetry_json TEXT NOT NULL DEFAULT '{}',
    last_event_type TEXT,
    last_event_at_epoch REAL,
    last_seen_at_epoch REAL,
    registered INTEGER NOT NULL DEFAULT 1
);
"""

# Columns added when the store grew from camera-only to any device type.
# camera_id stays as the legacy name for the device id so existing databases
# and readers keep working; device_id is the column new code should read.
_DEVICE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("device_id", "TEXT"),
    ("device_type", "TEXT NOT NULL DEFAULT 'camera'"),
    ("event_type", "TEXT NOT NULL DEFAULT 'detection'"),
    ("data_json", "TEXT NOT NULL DEFAULT '{}'"),
)


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
        self._migrate_device_columns()
        self._conn.commit()

    def _migrate_device_columns(self) -> None:
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        added = False
        for name, definition in _DEVICE_COLUMNS:
            if name in existing:
                continue
            try:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")
                added = True
            except sqlite3.OperationalError as exc:
                # Another process may have migrated between our check and ALTER.
                if "duplicate column" not in str(exc).lower():
                    raise
        if added:
            self._conn.execute("UPDATE events SET device_id = camera_id WHERE device_id IS NULL")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_device_time "
            "ON events (device_id, event_at_epoch DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_time "
            "ON events (device_type, event_at_epoch DESC)"
        )

    def close(self) -> None:
        self._conn.close()

    def insert_event(self, record: dict[str, object]) -> None:
        event_at_utc = str(record.get("event_at_utc") or _utc_now_iso())
        try:
            event_at_epoch = _parse_iso_to_epoch(event_at_utc)
        except ValueError:
            event_at_epoch = time.time()

        device_id = str(record.get("device_id") or record.get("camera_id") or "")
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO events (
                        event_id, camera_id, camera_host, event_at_utc, event_at_epoch,
                        frame_index, detections_count, counts_json, detections_json, snapshot_path,
                        device_id, device_type, event_type, data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.get("event_id")),
                        device_id,
                        record.get("camera_host"),
                        event_at_utc,
                        event_at_epoch,
                        record.get("frame_index"),
                        int(record.get("detections_count", 0)),
                        json.dumps(record.get("counts", {}), ensure_ascii=True),
                        json.dumps(record.get("detections", []), ensure_ascii=True),
                        record.get("snapshot_path"),
                        device_id,
                        str(record.get("device_type") or "camera"),
                        str(record.get("event_type") or "detection"),
                        json.dumps(record.get("data", {}), ensure_ascii=True),
                    ),
                )
        except sqlite3.OperationalError as exc:
            # The raw JSONL log already captured this event, so a dropped row
            # here (e.g. transient lock contention across processes) is not
            # fatal for the publisher loop.
            self.logger.warning("Failed to persist event to SQLite: %s", exc)

    def upsert_device_state(
        self,
        device_id: str,
        device_type: str,
        *,
        status: str | None = None,
        state_patch: dict[str, object] | None = None,
        telemetry: dict[str, object] | None = None,
        last_event_type: str | None = None,
        last_event_at_epoch: float | None = None,
        seen_at_epoch: float | None = None,
        registered: bool = True,
    ) -> None:
        """Merge new information about a device into its last known state."""
        seen_at = seen_at_epoch if seen_at_epoch is not None else time.time()
        try:
            with self._conn:
                row = self._conn.execute(
                    "SELECT * FROM device_state WHERE device_id = ?", (device_id,)
                ).fetchone()
                state = json.loads(row["state_json"]) if row else {}
                state.update(state_patch or {})
                stored_telemetry = json.loads(row["telemetry_json"]) if row else {}
                stored_telemetry.update(telemetry or {})
                self._conn.execute(
                    """
                    INSERT INTO device_state (
                        device_id, device_type, status, state_json, telemetry_json,
                        last_event_type, last_event_at_epoch, last_seen_at_epoch, registered
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(device_id) DO UPDATE SET
                        device_type = excluded.device_type,
                        status = COALESCE(excluded.status, device_state.status),
                        state_json = excluded.state_json,
                        telemetry_json = excluded.telemetry_json,
                        last_event_type = COALESCE(
                            excluded.last_event_type, device_state.last_event_type
                        ),
                        last_event_at_epoch = COALESCE(
                            excluded.last_event_at_epoch, device_state.last_event_at_epoch
                        ),
                        last_seen_at_epoch = MAX(
                            COALESCE(device_state.last_seen_at_epoch, 0),
                            excluded.last_seen_at_epoch
                        ),
                        registered = excluded.registered
                    """,
                    (
                        device_id,
                        device_type,
                        status,
                        json.dumps(state, ensure_ascii=True),
                        json.dumps(stored_telemetry, ensure_ascii=True),
                        last_event_type,
                        last_event_at_epoch,
                        seen_at,
                        1 if registered else 0,
                    ),
                )
        except sqlite3.OperationalError as exc:
            self.logger.warning("Failed to persist state for %s: %s", device_id, exc)

    def device_states(self) -> dict[str, dict[str, object]]:
        rows = self._conn.execute("SELECT * FROM device_state").fetchall()
        return {row["device_id"]: _state_row_to_record(row) for row in rows}

    def query_events(
        self,
        camera_ids: list[str] | None = None,
        class_name: str | None = None,
        device_ids: list[str] | None = None,
        device_types: list[str] | None = None,
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
        if device_ids is not None:
            if not device_ids:
                return []
            placeholders = ",".join("?" for _ in device_ids)
            clauses.append(f"device_id IN ({placeholders})")
            params.extend(device_ids)
        if device_types is not None:
            if not device_types:
                return []
            placeholders = ",".join("?" for _ in device_types)
            clauses.append(f"device_type IN ({placeholders})")
            params.extend(device_types)
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


def run_retention(
    event_db: EventDatabase,
    base_dir: Path,
    retention_days: float,
    logger: logging.Logger,
) -> int:
    """Delete events older than retention_days and their snapshot files."""
    if retention_days <= 0:
        return 0
    cutoff_epoch = time.time() - (retention_days * 86400.0)
    stale_snapshot_paths = event_db.purge_older_than(cutoff_epoch)
    if not stale_snapshot_paths:
        return 0

    removed = 0
    for relative_path in stale_snapshot_paths:
        snapshot_path = base_dir / relative_path
        try:
            snapshot_path.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            logger.warning("Failed to remove stale snapshot %s: %s", snapshot_path, exc)
    logger.info(
        "Event retention purge removed %s snapshot event(s) older than %s day(s), %s file(s) deleted.",
        len(stale_snapshot_paths),
        retention_days,
        removed,
    )
    return len(stale_snapshot_paths)


def _state_row_to_record(row: sqlite3.Row) -> dict[str, object]:
    return {
        "device_id": row["device_id"],
        "device_type": row["device_type"],
        "status": row["status"],
        "state": json.loads(row["state_json"]) if row["state_json"] else {},
        "telemetry": json.loads(row["telemetry_json"]) if row["telemetry_json"] else {},
        "last_event_type": row["last_event_type"],
        "last_event_at_epoch": row["last_event_at_epoch"],
        "last_seen_at_epoch": row["last_seen_at_epoch"],
        "registered": bool(row["registered"]),
    }


def _row_to_record(row: sqlite3.Row) -> dict[str, object]:
    return {
        "event_id": row["event_id"],
        "device_id": row["device_id"] or row["camera_id"],
        "device_type": row["device_type"] or "camera",
        "event_type": row["event_type"] or "detection",
        "data": json.loads(row["data_json"]) if row["data_json"] else {},
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
