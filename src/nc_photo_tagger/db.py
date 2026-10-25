"""SQLite state: which files are analyzed / uploaded, their CLIP embeddings and faces."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

# File status values
PENDING = "pending"  # needs AI analysis
ANALYZED = "analyzed"  # analyzed, tags need to be uploaded / removed
DONE = "done"  # tags on the server are up to date
ERROR = "error"  # cannot be read / analyzed

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    status TEXT NOT NULL,
    clip BLOB,
    tags TEXT NOT NULL DEFAULT '[]',
    uploaded TEXT NOT NULL DEFAULT '[]',
    retry_at REAL NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS files_status ON files(status, path);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    x REAL, y REAL, w REAL, h REAL,
    score REAL,
    px INTEGER,
    emb BLOB NOT NULL,
    person TEXT,
    manual TEXT,
    ignored INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS faces_path ON faces(path);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, np.float16).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.float16).astype(np.float32)


class DB:
    def __init__(self, path: Path, readonly: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if readonly:
            self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        else:
            self.con = sqlite3.connect(path)
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.executescript(SCHEMA)
        self.con.execute("PRAGMA synchronous=NORMAL")

    # ---- meta ----
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    # ---- scanning ----
    def known_files(self) -> dict[str, tuple[float, int]]:
        return {p: (m, s) for p, m, s in self.con.execute("SELECT path, mtime, size FROM files")}

    def upsert_pending(self, items: list[tuple[str, float, int]]) -> None:
        with self.con:
            self.con.executemany("DELETE FROM faces WHERE path=?", [(p,) for p, _, _ in items])
            self.con.executemany(
                """INSERT INTO files (path, mtime, size, status) VALUES (?, ?, ?, 'pending')
                   ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size,
                   status='pending', clip=NULL, error=NULL, retry_at=0""",
                items,
            )

    def remove(self, paths: list[str]) -> None:
        with self.con:
            self.con.executemany("DELETE FROM faces WHERE path=?", [(p,) for p in paths])
            self.con.executemany("DELETE FROM files WHERE path=?", [(p,) for p in paths])

    # ---- work queue ----
    def next_pending(self) -> str | None:
        row = self.con.execute(
            "SELECT path FROM files WHERE status='pending' ORDER BY path DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def next_upload(self, now: float) -> tuple[str, list[str], list[str]] | None:
        row = self.con.execute(
            """SELECT path, tags, uploaded FROM files WHERE status='analyzed' AND retry_at<=?
               ORDER BY path DESC LIMIT 1""", (now,)).fetchone()
        return (row[0], json.loads(row[1]), json.loads(row[2])) if row else None

    def upload_batch(self, now: float, limit: int = 200) -> list[tuple[str, list[str], list[str]]]:
        """Files waiting for upload that live in the same folder as the next one in the queue."""
        first = self.next_upload(now)
        if not first:
            return []
        prefix = first[0].rsplit("/", 1)[0] + "/"
        rows = self.con.execute(
            """SELECT path, tags, uploaded FROM files
               WHERE status='analyzed' AND retry_at<=? AND substr(path, 1, ?)=? ORDER BY path DESC""",
            (now, len(prefix), prefix))
        batch = []
        for path, tags, uploaded in rows:
            if "/" not in path[len(prefix):]:
                batch.append((path, json.loads(tags), json.loads(uploaded)))
                if len(batch) >= limit:
                    break
        return batch

    def save_analysis(self, path: str, clip: np.ndarray, faces: list) -> list[int]:
        with self.con:
            self.con.execute("DELETE FROM faces WHERE path=?", (path,))
            ids = []
            for f in faces:
                cur = self.con.execute(
                    "INSERT INTO faces (path, x, y, w, h, score, px, emb) VALUES (?,?,?,?,?,?,?,?)",
                    (path, f.x, f.y, f.w, f.h, f.score, f.px, to_blob(f.emb)))
                ids.append(cur.lastrowid)
            self.con.execute("UPDATE files SET clip=?, status='analyzed', error=NULL WHERE path=?",
                             (to_blob(clip), path))
        return ids

    def set_error(self, path: str, error: str) -> None:
        with self.con:
            self.con.execute("UPDATE files SET status='error', error=? WHERE path=?", (error[:500], path))

    def set_tags(self, items: list[tuple[str, list[str]]]) -> int:
        """Store the wanted tags; (re-)queue the upload if they differ from those on the server.

        Returns the number of files that need an upload."""
        queued = 0
        with self.con:
            for path, tags in items:
                row = self.con.execute("SELECT status, uploaded FROM files WHERE path=?", (path,)).fetchone()
                if not row:
                    continue
                status = row[0]
                if status == DONE and set(tags) != set(json.loads(row[1])):
                    status = ANALYZED  # freshly analyzed files (ANALYZED) always get synced once
                if status == ANALYZED:
                    queued += 1
                self.con.execute("UPDATE files SET tags=?, status=?, retry_at=0 WHERE path=?",
                                 (json.dumps(sorted(tags)), status, path))
        return queued

    def clips_for(self, paths: list[str]) -> dict[str, np.ndarray]:
        result = {}
        for i in range(0, len(paths), 500):
            part = paths[i:i + 500]
            for p, b in self.con.execute(
                    f"SELECT path, clip FROM files WHERE clip IS NOT NULL AND path IN ({','.join('?' * len(part))})",
                    part):
                result[p] = from_blob(b)
        return result

    def set_uploaded(self, path: str, uploaded: list[str], done: bool, retry_at: float = 0) -> None:
        with self.con:
            self.con.execute("UPDATE files SET uploaded=?, status=?, retry_at=? WHERE path=?",
                             (json.dumps(sorted(uploaded)), DONE if done else ANALYZED, retry_at, path))

    def iter_clip(self, chunk: int = 5000):
        last = ""
        while True:
            rows = self.con.execute(
                "SELECT path, clip FROM files WHERE clip IS NOT NULL AND path>? ORDER BY path LIMIT ?",
                (last, chunk)).fetchall()
            if not rows:
                return
            last = rows[-1][0]
            yield [r[0] for r in rows], np.stack([from_blob(r[1]) for r in rows])

    def counts(self) -> dict[str, int]:
        return dict(self.con.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall())

    # ---- faces ----
    def people_by_path(self, paths: list[str] | None = None) -> dict[str, set[str]]:
        q = """SELECT path, COALESCE(manual, person) FROM faces
               WHERE ignored=0 AND COALESCE(manual, person) IS NOT NULL"""
        result: dict[str, set[str]] = {}
        if paths is None:
            rows = self.con.execute(q).fetchall()
        else:
            rows = []
            for i in range(0, len(paths), 500):
                part = paths[i:i + 500]
                rows += self.con.execute(q + f" AND path IN ({','.join('?' * len(part))})", part).fetchall()
        for path, person in rows:
            result.setdefault(path, set()).add(person)
        return result

    def face_embedding(self, face_id: int) -> np.ndarray | None:
        row = self.con.execute("SELECT emb FROM faces WHERE id=?", (face_id,)).fetchone()
        return from_blob(row[0]) if row else None

    def unknown_face_count(self) -> int:
        return self.con.execute(
            "SELECT COUNT(*) FROM faces WHERE ignored=0 AND manual IS NULL AND person IS NULL").fetchone()[0]
