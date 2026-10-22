"""SQLite persistence for C4 (stdlib `sqlite3`, no extra dependencies).

Design:
  * Raw events are the source of truth. Each is stored verbatim (`raw_json`) with a few
    indexed key columns, so nothing is lost and the timeline can be rebuilt at any time.
  * Normalized annotations are NOT stored: delta merging needs the whole slide history, so
    `timeline()` rebuilds them from the raw rows on demand.
  * Groups and analyses are caches derived from events; new events for a slide invalidate them.
  * One short-lived connection per operation; WAL mode; schema version in `PRAGMA user_version`.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edusync_c4.normalizer import TimeUnit, normalize_event
from edusync_c4.schemas import (
    AnnotationGroup,
    AnnotationTimeline,
    EventImportResponse,
    GroupAnalysis,
    GroupingConfig,
    LectureSummary,
    NormalizedAnnotation,
    ShapeType,
    SlideInfo,
    SlideUpsert,
    SlideWord,
    SpeechInfo,
    SpeechSegment,
)
from edusync_c4.timeline import build_timeline

SCHEMA_VERSION = 4

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    lecture_id TEXT NOT NULL,
    slide_id TEXT NOT NULL,
    shape_id TEXT NOT NULL,
    timestamp_seconds REAL,
    timestamp_valid INTEGER NOT NULL,
    is_deleted INTEGER NOT NULL,
    is_clear INTEGER NOT NULL,
    time_unit TEXT NOT NULL,
    lecture_start_ms REAL,
    raw_json TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_slide
    ON events (lecture_id, slide_id, timestamp_seconds);
CREATE TABLE IF NOT EXISTS groups (
    group_id TEXT PRIMARY KEY,
    lecture_id TEXT NOT NULL,
    slide_id TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    config_json TEXT NOT NULL,
    group_json TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_groups_slide ON groups (lecture_id, slide_id);
CREATE TABLE IF NOT EXISTS analyses (
    group_id TEXT PRIMARY KEY REFERENCES groups (group_id) ON DELETE CASCADE,
    lecture_id TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS slides (
    lecture_id TEXT NOT NULL,
    slide_id TEXT NOT NULL,
    number INTEGER,
    width REAL,
    height REAL,
    scaled_width REAL,
    scaled_height REAL,
    text TEXT,
    svg TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (lecture_id, slide_id)
);
CREATE TABLE IF NOT EXISTS slide_words (
    lecture_id TEXT NOT NULL,
    slide_id TEXT NOT NULL,
    words_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (lecture_id, slide_id)
);
CREATE TABLE IF NOT EXISTS speech_segments (
    lecture_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    segment_id TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (lecture_id, position)
);
CREATE TABLE IF NOT EXISTS speech_meta (
    lecture_id TEXT PRIMARY KEY,
    offset_seconds REAL NOT NULL,
    method TEXT NOT NULL,
    source TEXT,
    sync_shape_id TEXT,
    updated_at TEXT NOT NULL
);
"""


def _shape_of(item: Any) -> str | None:
    """Shape id of a raw event dict (as stored) or a NormalizedAnnotation."""
    if isinstance(item, NormalizedAnnotation):
        return item.shape_id
    return normalize_event(item).shape_id


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AnnotationRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema v{version} is newer than supported v{SCHEMA_VERSION}"
                )
            conn.executescript(_DDL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:  # one transaction: commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ events

    def import_events(
        self,
        events: list[dict[str, Any]],
        default_lecture_id: str,
        default_slide_id: str,
        time_unit: TimeUnit = "s",
        lecture_start_ms: float | None = None,
    ) -> EventImportResponse:
        """Store events idempotently. Nothing is dropped; problems are counted."""
        resp = EventImportResponse(imported_count=0)
        touched: set[tuple[str, str]] = set()
        with self._connect() as conn:
            for raw in events:
                norm = normalize_event(
                    raw,
                    default_lecture_id=default_lecture_id,
                    default_slide_id=default_slide_id,
                    time_unit=time_unit,
                    lecture_start_ms=lecture_start_ms,
                )
                raw_json = _canonical(raw)
                existing = conn.execute(
                    "SELECT raw_json FROM events WHERE event_id = ?", (norm.event_id,)
                ).fetchone()
                if existing is not None:
                    if existing["raw_json"] == raw_json:
                        resp.duplicate_count += 1
                    else:
                        resp.conflict_count += 1
                        resp.conflict_event_ids.append(norm.event_id)
                    continue
                conn.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        norm.event_id,
                        norm.lecture_id,
                        norm.slide_id,
                        norm.shape_id,
                        norm.timestamp_seconds if norm.timestamp_valid else None,
                        int(norm.timestamp_valid),
                        int(norm.is_deleted and not norm.is_clear),
                        int(norm.is_clear),
                        time_unit,
                        lecture_start_ms,
                        raw_json,
                        _now(),
                    ),
                )
                touched.add((norm.lecture_id, norm.slide_id))
                resp.imported_count += 1
                resp.event_ids.append(norm.event_id)
                if norm.is_clear:
                    resp.clear_count += 1
                elif norm.is_deleted:
                    resp.deleted_count += 1
                elif norm.shape_type == ShapeType.UNKNOWN:
                    resp.unsupported_count += 1
                if not norm.timestamp_valid:
                    resp.unplaced_count += 1
            for lecture_id, slide_id in touched:
                self._invalidate(conn, lecture_id, slide_id)
        return resp

    def lecture_exists(self, lecture_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE lecture_id = ? LIMIT 1", (lecture_id,)
            ).fetchone()
        return row is not None

    def slide_ids(self, lecture_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT slide_id FROM events WHERE lecture_id = ? ORDER BY slide_id",
                (lecture_id,),
            ).fetchall()
        return [r["slide_id"] for r in rows]

    def timeline(self, lecture_id: str, slide_id: str) -> AnnotationTimeline | None:
        """Rebuild one slide's timeline from raw events; None if the slide has no events."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rowid, * FROM events WHERE lecture_id = ? AND slide_id = ? "
                "ORDER BY timestamp_valid DESC, timestamp_seconds, rowid",
                (lecture_id, slide_id),
            ).fetchall()
        if not rows:
            return None

        originals: dict[str, dict[str, Any]] = {}
        items: list[dict[str, Any] | NormalizedAnnotation] = []
        for row in rows:
            raw = json.loads(row["raw_json"])
            originals[row["event_id"]] = raw
            injected = {**raw, "event_id": row["event_id"]}
            if row["timestamp_valid"]:
                injected["timestamp_seconds"] = row["timestamp_seconds"]
                items.append(injected)
            else:
                items.append(
                    normalize_event(
                        injected,
                        default_lecture_id=lecture_id,
                        default_slide_id=slide_id,
                        time_unit=row["time_unit"],
                        lecture_start_ms=row["lecture_start_ms"],
                    )
                )
        sync_shape = self._sync_shape(lecture_id)
        if sync_shape:  # the cue mark drawn to line up speech is not lecture content
            items = [i for i in items if _shape_of(i) != sync_shape]
            if not items:
                return None
        timeline = build_timeline(items, lecture_id=lecture_id, slide_id=slide_id)
        for ann in [*timeline.events, *timeline.unplaced]:
            ann.raw_payload = originals.get(ann.event_id, ann.raw_payload)
        return timeline

    # ------------------------------------------------------------- groups etc.

    @staticmethod
    def _invalidate(conn: sqlite3.Connection, lecture_id: str, slide_id: str) -> None:
        conn.execute(
            "DELETE FROM groups WHERE lecture_id = ? AND slide_id = ?", (lecture_id, slide_id)
        )  # analyses cascade

    def replace_groups(
        self,
        lecture_id: str,
        slide_id: str,
        groups: list[AnnotationGroup],
        config: GroupingConfig,
    ) -> None:
        with self._connect() as conn:
            self._invalidate(conn, lecture_id, slide_id)
            for g in groups:
                conn.execute(
                    "INSERT INTO groups VALUES (?,?,?,?,?,?,?,?)",
                    (
                        g.group_id,
                        lecture_id,
                        slide_id,
                        g.start_seconds,
                        g.end_seconds,
                        config.model_dump_json(),
                        g.model_dump_json(),
                        _now(),
                    ),
                )

    def load_groups(self, lecture_id: str, slide_id: str | None = None) -> list[AnnotationGroup]:
        query = "SELECT group_json FROM groups WHERE lecture_id = ?"
        params: list[Any] = [lecture_id]
        if slide_id is not None:
            query += " AND slide_id = ?"
            params.append(slide_id)
        query += " ORDER BY start_seconds, slide_id, group_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AnnotationGroup.model_validate_json(r["group_json"]) for r in rows]

    def save_analyses(self, lecture_id: str, analyses: list[GroupAnalysis]) -> None:
        with self._connect() as conn:
            for a in analyses:
                conn.execute(
                    "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?)",
                    (a.context.group_id, lecture_id, a.model_dump_json(), _now()),
                )

    def load_analyses(
        self, lecture_id: str, since_seconds: float = 0.0
    ) -> list[tuple[str, float, float, GroupAnalysis]]:
        """(slide_id, start, end, analysis) for groups ending at or after `since_seconds`."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT g.slide_id, g.start_seconds, g.end_seconds, a.analysis_json "
                "FROM analyses a JOIN groups g ON g.group_id = a.group_id "
                "WHERE a.lecture_id = ? AND g.end_seconds >= ? "
                "ORDER BY g.start_seconds, g.slide_id, g.group_id",
                (lecture_id, since_seconds),
            ).fetchall()
        return [
            (
                r["slide_id"],
                r["start_seconds"],
                r["end_seconds"],
                GroupAnalysis.model_validate_json(r["analysis_json"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ slides

    def save_slide(self, lecture_id: str, slide_id: str, slide: SlideUpsert) -> SlideInfo:
        """Insert or replace one slide. Fields left out (None) keep their stored value."""
        with self._connect() as conn:
            old = conn.execute(
                "SELECT * FROM slides WHERE lecture_id = ? AND slide_id = ?",
                (lecture_id, slide_id),
            ).fetchone()

            def pick(name: str) -> Any:
                new = getattr(slide, name)
                return new if new is not None else (old[name] if old is not None else None)

            conn.execute(
                "INSERT OR REPLACE INTO slides VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    lecture_id,
                    slide_id,
                    pick("number"),
                    pick("width"),
                    pick("height"),
                    pick("scaled_width"),
                    pick("scaled_height"),
                    pick("text"),
                    pick("svg"),
                    _now(),
                ),
            )
        info = self.get_slide_info(lecture_id, slide_id)
        assert info is not None
        return info

    @staticmethod
    def _slide_info(row: sqlite3.Row) -> SlideInfo:
        return SlideInfo(
            lecture_id=row["lecture_id"],
            slide_id=row["slide_id"],
            number=row["number"],
            width=row["width"],
            height=row["height"],
            scaled_width=row["scaled_width"],
            scaled_height=row["scaled_height"],
            text=row["text"],
            has_svg=bool(row["svg"]),
        )

    def get_slide_info(self, lecture_id: str, slide_id: str) -> SlideInfo | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM slides WHERE lecture_id = ? AND slide_id = ?",
                (lecture_id, slide_id),
            ).fetchone()
        return self._slide_info(row) if row is not None else None

    def list_slides(self, lecture_id: str) -> list[SlideInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM slides WHERE lecture_id = ? ORDER BY number, slide_id",
                (lecture_id,),
            ).fetchall()
        return [self._slide_info(r) for r in rows]

    def get_slide_svg(self, lecture_id: str, slide_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT svg FROM slides WHERE lecture_id = ? AND slide_id = ?",
                (lecture_id, slide_id),
            ).fetchone()
        return row["svg"] if row is not None else None

    # ------------------------------------------------------------------ reset

    def delete_lecture(self, lecture_id: str) -> dict[str, int]:
        """Remove everything stored for one lecture; returns row counts per table."""
        counts: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses WHERE lecture_id = ?", (lecture_id,))
            for table in (
                "groups",
                "slides",
                "slide_words",
                "events",
                "speech_segments",
                "speech_meta",
            ):
                cur = conn.execute(f"DELETE FROM {table} WHERE lecture_id = ?", (lecture_id,))
                counts[table] = cur.rowcount
        return counts

    # ------------------------------------------------------------------ speech

    def _sync_shape(self, lecture_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sync_shape_id FROM speech_meta WHERE lecture_id = ?", (lecture_id,)
            ).fetchone()
        return row["sync_shape_id"] if row is not None else None

    def first_annotation(self, lecture_id: str) -> tuple[str, float] | None:
        """(shape id, meeting second) of the earliest drawn item: the sync mark, if one was made."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT shape_id, timestamp_seconds FROM events WHERE lecture_id = ? "
                "AND timestamp_valid = 1 AND is_deleted = 0 AND is_clear = 0 "
                "ORDER BY timestamp_seconds, rowid LIMIT 1",
                (lecture_id,),
            ).fetchone()
        return (row["shape_id"], row["timestamp_seconds"]) if row is not None else None

    def save_speech(
        self,
        lecture_id: str,
        segments: list[SpeechSegment],
        offset_seconds: float,
        method: str,
        source: str | None,
        sync_shape_id: str | None,
    ) -> SpeechInfo:
        """Replace all stored speech for a lecture (times stay in recording time)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM speech_segments WHERE lecture_id = ?", (lecture_id,))
            conn.executemany(
                "INSERT INTO speech_segments VALUES (?,?,?,?,?,?)",
                [
                    (lecture_id, i, str(seg.id), seg.start, seg.end, seg.text)
                    for i, seg in enumerate(segments)
                ],
            )
            conn.execute(
                "INSERT OR REPLACE INTO speech_meta VALUES (?,?,?,?,?,?)",
                (lecture_id, offset_seconds, method, source, sync_shape_id, _now()),
            )
            # cached groups and analyses were built from the old timeline/speech
            conn.execute("DELETE FROM groups WHERE lecture_id = ?", (lecture_id,))
        return SpeechInfo(
            lecture_id=lecture_id,
            segment_count=len(segments),
            offset_seconds=offset_seconds,
            method=method,
            source=source,
            sync_shape_id=sync_shape_id,
        )

    def load_speech(self, lecture_id: str) -> tuple[SpeechInfo, list[SpeechSegment]] | None:
        """Stored speech with every time moved onto the meeting clock (recording + offset)."""
        with self._connect() as conn:
            meta = conn.execute(
                "SELECT * FROM speech_meta WHERE lecture_id = ?", (lecture_id,)
            ).fetchone()
            if meta is None:
                return None
            rows = conn.execute(
                "SELECT * FROM speech_segments WHERE lecture_id = ? ORDER BY position",
                (lecture_id,),
            ).fetchall()
        shift = meta["offset_seconds"]
        segments = [
            SpeechSegment(
                id=r["segment_id"],
                start=max(0.0, r["start_seconds"] + shift),
                end=max(0.0, r["end_seconds"] + shift),
                text=r["text"],
            )
            for r in rows
        ]
        info = SpeechInfo(
            lecture_id=lecture_id,
            segment_count=len(segments),
            offset_seconds=shift,
            method=meta["method"],
            source=meta["source"],
            sync_shape_id=meta["sync_shape_id"],
        )
        return info, segments

    # ------------------------------------------------------------------ lectures

    def list_lectures(self) -> list[LectureSummary]:
        """All stored lectures, most recently imported first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.lecture_id AS lecture_id, COUNT(*) AS events, "
                "COUNT(DISTINCT e.slide_id) AS slides, "
                "MIN(CASE WHEN e.timestamp_valid = 1 THEN e.timestamp_seconds END) AS first_s, "
                "MAX(CASE WHEN e.timestamp_valid = 1 THEN e.timestamp_seconds END) AS last_s, "
                "MAX(e.imported_at) AS last_imported, "
                "EXISTS(SELECT 1 FROM speech_meta m WHERE m.lecture_id = e.lecture_id) AS speech "
                "FROM events e GROUP BY e.lecture_id ORDER BY last_imported DESC"
            ).fetchall()
        return [
            LectureSummary(
                lecture_id=r["lecture_id"],
                event_count=r["events"],
                slide_count=r["slides"],
                first_seconds=r["first_s"],
                last_seconds=r["last_s"],
                last_imported=r["last_imported"],
                has_speech=bool(r["speech"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ slide words

    def save_slide_words(self, lecture_id: str, slide_id: str, words: list[SlideWord]) -> int:
        """Replace the word boxes of one slide (whiteboard coordinates). Returns the count."""
        payload = json.dumps([w.model_dump() for w in words])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO slide_words VALUES (?,?,?,?)",
                (lecture_id, slide_id, payload, _now()),
            )
            # strokes are described using these words, so cached analyses are stale
            conn.execute(
                "DELETE FROM groups WHERE lecture_id = ? AND slide_id = ?", (lecture_id, slide_id)
            )
        return len(words)

    def get_slide_words(self, lecture_id: str, slide_id: str) -> list[SlideWord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT words_json FROM slide_words WHERE lecture_id = ? AND slide_id = ?",
                (lecture_id, slide_id),
            ).fetchone()
        if row is None:
            return []
        return [SlideWord.model_validate(w) for w in json.loads(row["words_json"])]
