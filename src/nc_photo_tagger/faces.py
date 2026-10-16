"""People: reference faces, matching and clustering of unknown faces."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np

from .config import Config
from .db import DB, from_blob
from .models import FaceEngine, load_image

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
SAMPLE_RE = re.compile(r"face_(\d+)\.jpg")
CLUSTER_PREFIX = "cluster_"
CLUSTER_INFO = ".cluster.json"
SAMPLES_PER_CLUSTER = 16
IGNORE_NAMES = {"ignore", "ignored", "unknown", "nobody"}


class People:
    def __init__(self, cfg: Config, db: DB, engine: FaceEngine | None):
        self.cfg, self.db, self.engine = cfg, db, engine
        self.known_dir, self.unknown_dir = cfg.known_dir, cfg.unknown_dir
        self.names: list[str] = []
        self._R = np.zeros((0, 128), np.float32)
        self._owner = np.zeros(0, int)
        self._sig: str | None = None
        self._ref_cache: dict[tuple[str, float], np.ndarray | None] = {}

    # ---- named clusters -> known people ----------------------------------------------

    def process_named_clusters(self) -> set[str]:
        """Handle cluster folders the user renamed. Returns the photo paths whose people changed."""
        affected: set[str] = set()
        if not self.unknown_dir.is_dir():
            return affected
        for d in sorted(self.unknown_dir.iterdir()):
            if not d.is_dir() or d.name.startswith(CLUSTER_PREFIX) or d.name.startswith("."):
                continue
            try:
                info = json.loads((d / CLUSTER_INFO).read_text())
            except (OSError, ValueError):
                info = {}
            present = {int(m.group(1)) for f in d.iterdir() if (m := SAMPLE_RE.fullmatch(f.name))}
            removed = set(info.get("samples", [])) - present  # samples the user deleted = wrong faces
            members = [i for i in info.get("members", []) if i not in removed]
            name = d.name.strip()
            ignore = name.startswith("_") or name.lower() in IGNORE_NAMES
            with self.db.con:
                for i in range(0, len(members), 500):
                    part = members[i:i + 500]
                    marks = ",".join("?" * len(part))
                    affected.update(p for (p,) in self.db.con.execute(
                        f"SELECT DISTINCT path FROM faces WHERE id IN ({marks})", part))
                    if ignore:
                        self.db.con.execute(f"UPDATE faces SET ignored=1 WHERE id IN ({marks})", part)
                    else:
                        self.db.con.execute(f"UPDATE faces SET manual=? WHERE id IN ({marks})", [name, *part])
            if ignore:
                log.info("ignoring %d faces of %s", len(members), d.name)
            else:
                target = self.known_dir / name
                target.mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    if f.is_file() and not f.name.startswith("."):
                        dest = target / f.name
                        if not dest.exists():
                            shutil.move(str(f), dest)
                log.info("learned person %r from %s (%d faces)", name, d.name, len(members))
            shutil.rmtree(d, ignore_errors=True)
        return affected

    # ---- reference faces ----------------------------------------------------------------

    def _ref_files(self) -> list[tuple[str, Path]]:
        out = []
        if self.known_dir.is_dir():
            for person in sorted(self.known_dir.iterdir()):
                if person.is_dir() and not person.name.startswith("."):
                    out += [(person.name, f) for f in sorted(person.rglob("*"))
                            if f.is_file() and f.suffix.lower() in IMAGE_EXT]
        return out

    def _ref_embedding(self, f: Path) -> np.ndarray | None:
        m = SAMPLE_RE.fullmatch(f.name)
        if m and self.db is not None:
            emb = self.db.face_embedding(int(m.group(1)))
            if emb is not None:
                return emb
        key = (str(f), f.stat().st_mtime)
        if key not in self._ref_cache:
            emb = None
            if self.engine is not None:
                try:
                    faces = self.engine.faces(load_image(f))
                    if faces:
                        emb = max(faces, key=lambda x: x.w * x.h).emb
                    else:
                        log.warning("no face found in reference photo %s", f)
                except Exception as e:
                    log.warning("cannot read reference photo %s: %s", f, e)
            self._ref_cache[key] = emb
        return self._ref_cache[key]

    def load_refs(self) -> bool:
        files = self._ref_files()
        sig = hashlib.sha1(json.dumps(
            [(n, str(f), f.stat().st_mtime, f.stat().st_size) for n, f in files]).encode()).hexdigest()
        if sig == self._sig:
            return False
        names: list[str] = []
        embs, owner = [], []
        for name, f in files:
            emb = self._ref_embedding(f)
            if emb is None:
                continue
            if name not in names:
                names.append(name)
            embs.append(emb)
            owner.append(names.index(name))
        self.names = names
        self._R = np.stack(embs).astype(np.float32) if embs else np.zeros((0, 128), np.float32)
        self._owner = np.array(owner, int)
        self._sig = sig
        log.info("loaded %d reference faces of %d people", len(embs), len(names))
        return True

    def match(self, embs: np.ndarray) -> list[str | None]:
        embs = np.atleast_2d(embs)
        if not len(self._R):
            return [None] * len(embs)
        sims = embs @ self._R.T
        best = sims.argmax(axis=1)
        score = sims[np.arange(len(embs)), best]
        thr = self.cfg.match_threshold
        return [self.names[self._owner[b]] if s >= thr else None for b, s in zip(best, score)]

    def assign(self, face_ids: list[int], embs: np.ndarray) -> None:
        names = self.match(embs)
        with self.db.con:
            self.db.con.executemany("UPDATE faces SET person=? WHERE id=?", list(zip(names, face_ids)))

    def rematch_all(self) -> set[str]:
        affected: set[str] = set()
        last = 0
        while True:
            rows = self.db.con.execute(
                "SELECT id, path, emb, person FROM faces WHERE id>? ORDER BY id LIMIT 20000", (last,)).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            names = self.match(np.stack([from_blob(r[2]) for r in rows]))
            updates = [(n, r[0]) for r, n in zip(rows, names) if n != r[3]]
            affected.update(r[1] for r, n in zip(rows, names) if n != r[3])
            with self.db.con:
                self.db.con.executemany("UPDATE faces SET person=? WHERE id=?", updates)
        return affected

    def sync(self) -> set[str]:
        """Pick up renamed clusters and changed reference photos. Returns affected photo paths."""
        affected = self.process_named_clusters()
        if self.load_refs():
            affected |= self.rematch_all()
        return affected

    def pending_clusters(self) -> int:
        if not self.unknown_dir.is_dir():
            return 0
        return sum(1 for d in self.unknown_dir.iterdir() if d.is_dir() and d.name.startswith(CLUSTER_PREFIX))

    # ---- clustering of unknown faces ---------------------------------------------------------

    def cluster(self) -> int:
        rows = self.db.con.execute(
            """SELECT id, path, x, y, w, h, emb FROM faces
               WHERE ignored=0 AND manual IS NULL AND person IS NULL AND px>=? AND score>=0.9
               ORDER BY id""", (self.cfg.cluster_min_face_px,)).fetchall()
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        for d in self.unknown_dir.iterdir():
            if d.is_dir() and d.name.startswith(CLUSTER_PREFIX):
                shutil.rmtree(d, ignore_errors=True)
        if not rows:
            return 0

        E = np.stack([from_blob(r[6]) for r in rows])
        labels, C = _greedy_cluster(E, self.cfg.cluster_threshold)
        sizes = np.bincount(labels)
        big = [c for c in np.argsort(-sizes) if sizes[c] >= self.cfg.min_cluster_size][: self.cfg.max_clusters]

        for rank, c in enumerate(big, 1):
            idx = np.flatnonzero(labels == c)
            order = idx[np.argsort(-(E[idx] @ C[c]))]
            d = self.unknown_dir / f"{CLUSTER_PREFIX}{rank:04d}_{len(idx)}_faces"
            d.mkdir()
            samples = []
            for i in order[:SAMPLES_PER_CLUSTER]:
                fid, path, x, y, w, h, _ = rows[i]
                try:
                    _save_crop(path, x, y, w, h, d / f"face_{fid}.jpg")
                    samples.append(int(fid))
                except Exception as e:
                    log.debug("cannot crop face %s from %s: %s", fid, path, e)
            (d / CLUSTER_INFO).write_text(json.dumps(
                {"members": [int(rows[i][0]) for i in idx], "samples": samples}))
        log.info("clustered %d unknown faces into %d groups", len(rows), len(big))
        return len(big)


def _greedy_cluster(E: np.ndarray, thr: float, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Leader clustering in blocks: join the most similar centroid if similar enough, else start a new one."""
    n, dim = E.shape
    sums = np.zeros((n, dim), np.float32)
    C = np.zeros((n, dim), np.float32)
    labels = np.empty(n, int)
    k = 0
    for start in range(0, n, block):
        B = E[start:start + block]
        if k:
            sims = B @ C[:k].T
            best = sims.argmax(axis=1)
            ok = sims[np.arange(len(B)), best] >= thr
        else:
            best = np.zeros(len(B), int)
            ok = np.zeros(len(B), bool)
        labels[start:start + len(B)][ok] = best[ok]
        np.add.at(sums, best[ok], B[ok])
        k0 = k
        for i in np.flatnonzero(~ok):
            e = B[i]
            if k > k0:
                s = C[k0:k] @ e
                j = int(s.argmax())
                if s[j] >= thr:
                    labels[start + i] = k0 + j
                    sums[k0 + j] += e
                    C[k0 + j] = sums[k0 + j] / np.linalg.norm(sums[k0 + j])
                    continue
            labels[start + i] = k
            sums[k] = C[k] = e
            k += 1
        C[:k] = sums[:k] / np.maximum(np.linalg.norm(sums[:k], axis=1, keepdims=True), 1e-12)
    return labels, C[:k]


def _save_crop(path: str, x: float, y: float, w: float, h: float, dest: Path, margin: float = 0.35) -> None:
    img = load_image(path)
    W, H = img.size
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    half = max(w * W, h * H) * (0.5 + margin)
    box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half)))
    crop = img.crop(box)
    crop.thumbnail((256, 256))
    crop.save(dest, "JPEG", quality=90)
