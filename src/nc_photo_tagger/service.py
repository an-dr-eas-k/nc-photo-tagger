"""The background service: scan -> analyze -> upload tags, and report progress for the top bar."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import __version__
from .config import (ANALYSIS_OFF_FLAG, CACHE_DIR, DB_FILE, RECLUSTER_FLAG, RESCAN_FLAG, STATUS_FILE,
                     SYNC_OFF_FLAG, Config)
from .credentials import get_password
from .db import ANALYZED, DB, DONE, ERROR, PENDING
from .nextcloud import AuthError, Nextcloud, NotOnServer, ServerError, TagForbidden

log = logging.getLogger(__name__)

NOT_SYNCED_RETRY = 30 * 60
TAG_FORBIDDEN_RETRY = 60 * 60
# Re-cluster unknown faces when this many new unknown faces appeared (while busy / when idle)
RECLUSTER_BUSY = 2000
RECLUSTER_IDLE = 100


def set_low_priority() -> None:
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError) as e:
        log.debug("SCHED_IDLE not available: %s", e)
    try:
        cur = os.nice(0)
        if cur < 19:
            os.nice(19 - cur)
    except OSError:
        pass


class Service:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(DB_FILE)
        self.stop = False
        self.state = "starting"
        self.message = "Starting…"
        self.nc: Nextcloud | None = None
        self.server_msg = ""
        self.server_retry = 0.0
        self.server_backoff = 60.0
        self.forbidden: dict[str, float] = {}
        self.tagger = None
        self.people = None
        self._executor: ThreadPoolExecutor | None = None
        self._last_status = 0.0

    # ---- status for the GNOME extension ----------------------------------------------------

    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message

    def write_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_status < 2:
            return
        self._last_status = now
        counts = self.db.counts()
        total = sum(counts.values())
        done = total - counts.get(PENDING, 0)  # analyzed (uploaded or still waiting for upload)
        data = {
            "version": __version__,
            "state": self.state,
            "message": self.message,
            "total": total,
            "done": done,
            "percent": int(done * 100 / total) if total else 100,
            "pending": counts.get(PENDING, 0),
            "waiting": counts.get(ANALYZED, 0),
            "errors": counts.get(ERROR, 0),
            "people": len(self.people.names) if self.people else 0,
            "clusters": self.people.pending_clusters() if self.people else 0,
            "unknown_dir": str(self.cfg.unknown_dir),
            "known_dir": str(self.cfg.known_dir),
            "server": self.server_msg,
            "analysis_enabled": not ANALYSIS_OFF_FLAG.exists(),
            "sync_enabled": not SYNC_OFF_FLAG.exists(),
            "updated": now,
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, STATUS_FILE)

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while not self.stop and time.time() < end:
            time.sleep(min(0.5, end - time.time()))

    def _rel(self, path: str) -> str:
        for f in self.cfg.folders:
            try:
                return Path(path).relative_to(f.local).as_posix()
            except ValueError:
                pass
        return path

    # ---- main loop ------------------------------------------------------------------------------

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        set_low_priority()

        self._set("starting", "Loading AI models (first start downloads ~400 MB)…")
        self.write_status(force=True)
        from .faces import People
        from .models import ClipTagger, FaceEngine

        self.tagger = ClipTagger(self.cfg.threads)
        self.tagger.configure(self.cfg.tags, self.cfg.background_prompts,
                              self.cfg.aesthetic_tag, self.cfg.aesthetic_threshold)
        engine = FaceEngine(self.cfg.min_face_px, threads=self.cfg.threads) if self.cfg.faces_enabled else None
        if engine:
            self.people = People(self.cfg, self.db, engine)
            self.cfg.known_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.unknown_dir.mkdir(parents=True, exist_ok=True)
        self.apply_tag_config()
        if self.people:
            self.retag(self.people.sync())

        last_scan = last_people = last_cluster_check = 0.0
        while not self.stop:
            now = time.time()
            if RESCAN_FLAG.exists():
                RESCAN_FLAG.unlink(missing_ok=True)
                last_scan = 0
            if now - last_scan >= self.cfg.rescan_minutes * 60:
                self.scan()
                last_scan = time.time()
            if self.people and now - last_people >= 20:
                last_people = now
                self.retag(self.people.sync())

            analysis_on = not ANALYSIS_OFF_FLAG.exists()
            sync_on = not SYNC_OFF_FLAG.exists()
            worked = False
            path = self.db.next_pending() if analysis_on else None
            if path:
                self.analyze(path)
                worked = True
            if analysis_on and self.people and (RECLUSTER_FLAG.exists() or now - last_cluster_check >= 60):
                last_cluster_check = now
                if self._cluster_due(idle=not path):
                    self.cluster()
            if sync_on and self.upload_one():
                worked = True

            if not worked:
                self._set_idle_message()
                self.write_status()
                self._sleep(2)
            else:
                self.write_status()
        self._set("stopped", "Service stopped")
        self.write_status(force=True)

    def _set_idle_message(self) -> None:
        counts = self.db.counts()
        pending, waiting = counts.get(PENDING, 0), counts.get(ANALYZED, 0)
        analysis_on = not ANALYSIS_OFF_FLAG.exists()
        sync_on = not SYNC_OFF_FLAG.exists()
        if not analysis_on and not sync_on:
            self._set("paused", "Local tagging and Nextcloud sync are switched off")
        elif not analysis_on and not (sync_on and waiting):
            self._set("paused", f"Local tagging is switched off ({pending:,} photos not analyzed)")
        elif sync_on and self.server_msg:
            self._set("waiting", self.server_msg)
        elif not sync_on and waiting:
            self._set("paused", f"Nextcloud sync is switched off ({waiting:,} photos waiting)")
        elif waiting:
            self._set("waiting", f"{waiting:,} photos waiting to be synced to Nextcloud")
        else:
            self._set("idle", "All photos tagged – watching for new ones")

    # ---- tags configuration -----------------------------------------------------------------

    def apply_tag_config(self) -> None:
        """Re-apply tags to all analyzed photos if tags / threshold / prefix changed."""
        cfg = self.cfg
        sig = hashlib.sha1(json.dumps([cfg.tags, cfg.background_prompts, cfg.threshold, cfg.tag_prefix,
                                       cfg.tag_suffix, cfg.aesthetic_tag, cfg.aesthetic_threshold,
                                       self.tagger.IMAGE_MODEL]).encode()).hexdigest()
        if self.db.get_meta("tag_config") == sig:
            return
        self._set("working", "Applying changed tag settings…")
        self.write_status(force=True)
        people = self.db.people_by_path()
        queued = 0
        for paths, embs in self.db.iter_clip():
            tags = self.tagger.tags_for(embs, cfg.threshold)
            queued += self.db.set_tags([(p, self._server_tags(t, people.get(p))) for p, t in zip(paths, tags)])
        self.db.set_meta("tag_config", sig)
        log.info("tag settings applied, %d photos need new tags uploaded", queued)

    def _server_tags(self, tags: list[str], people) -> list[str]:
        """Tag names as they appear in Nextcloud: CLIP tags + people, all with the suffix."""
        sfx = self.cfg.tag_suffix
        return [t + sfx for t in tags] + [self.cfg.tag_prefix + n + sfx for n in sorted(people or ())]

    def retag(self, paths: set[str]) -> None:
        if not paths:
            return
        paths = sorted(paths)
        clips = self.db.clips_for(paths)
        people = self.db.people_by_path(paths)
        items = []
        for p in paths:
            if p in clips:
                tags = self.tagger.tags_for(clips[p], self.cfg.threshold)[0]
                items.append((p, self._server_tags(tags, people.get(p))))
        queued = self.db.set_tags(items)
        log.info("people changed on %d photos, %d need new tags uploaded", len(paths), queued)

    # ---- scanning ------------------------------------------------------------------------------

    def _excluded(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel + "/", pat) for pat in self.cfg.exclude)

    def scan(self) -> None:
        self._set("scanning", "Scanning photo folders…")
        self.write_status(force=True)
        found: dict[str, tuple[float, int]] = {}
        scanned_roots: list[str] = []
        for folder in self.cfg.folders:
            root = folder.local
            if not root.is_dir():
                log.warning("photo folder %s not found - skipping", root)
                continue
            scanned_roots.append(str(root).rstrip("/") + "/")
            for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: log.debug("scan: %s", e)):
                if self.stop:
                    return
                rel_dir = os.path.relpath(dirpath, root)
                rel_dir = "" if rel_dir == "." else rel_dir + "/"
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and not self._excluded(rel_dir + d)]
                for fn in filenames:
                    if fn.startswith(".") or fn.rsplit(".", 1)[-1].lower() not in self.cfg.extensions:
                        continue
                    if self._excluded(rel_dir + fn):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    found[p] = (st.st_mtime, st.st_size)

        known = self.db.known_files()
        changed = [(p, m, s) for p, (m, s) in found.items() if known.get(p) != (m, s)]
        removed = [p for p in known if p not in found and any(p.startswith(r) for r in scanned_roots)]
        removed += [p for p in known if not any(p.startswith(str(f.local).rstrip("/") + "/") for f in self.cfg.folders)]
        for i in range(0, len(changed), 5000):
            self.db.upsert_pending(changed[i:i + 5000])
        if removed:
            self.db.remove(removed)
        log.info("scan: %d photos, %d new/changed, %d removed", len(found), len(changed), len(removed))

    # ---- analysis ------------------------------------------------------------------------------

    def analyze(self, path: str) -> None:
        from .models import load_image

        self._set("working", f"Analyzing {self._rel(path)}")
        self.write_status()
        try:
            img = load_image(path)
            emb = self.tagger.embed(img)
            faces = self.people.engine.faces(img) if self.people else []
        except Exception as e:
            log.warning("cannot analyze %s: %s", path, e)
            self.db.set_error(path, str(e))
            return
        ids = self.db.save_analysis(path, emb, faces)
        names: set[str] = set()
        if faces:
            import numpy as np

            embs = np.stack([f.emb for f in faces])
            self.people.assign(ids, embs)
            names = self.db.people_by_path([path]).get(path, set())
        tags = self._server_tags(self.tagger.tags_for(emb, self.cfg.threshold)[0], names)
        self.db.set_tags([(path, tags)])
        log.debug("%s: %s", path, tags)

    def _cluster_due(self, idle: bool) -> bool:
        if RECLUSTER_FLAG.exists():
            return True
        last = int(self.db.get_meta("clustered_unknown", "0"))
        return self.db.unknown_face_count() - last >= (RECLUSTER_IDLE if idle else RECLUSTER_BUSY)

    def cluster(self) -> None:
        RECLUSTER_FLAG.unlink(missing_ok=True)
        self._set("working", "Grouping unknown faces…")
        self.write_status(force=True)
        try:
            self.people.cluster()
        except Exception:
            log.exception("clustering faces failed")
        self.db.set_meta("clustered_unknown", str(self.db.unknown_face_count()))

    # ---- upload ------------------------------------------------------------------------------

    def _client(self) -> Nextcloud | None:
        if self.nc is None:
            password = get_password(self.cfg)
            if not password:
                self.server_msg = "No Nextcloud app password – run 'nc-photo-tagger setup'"
                self.server_retry = time.time() + 120
                return None
            self.nc = Nextcloud(self.cfg.url, self.cfg.user, password, connections=self.cfg.sync_connections)
        return self.nc

    def upload_one(self) -> bool:
        """Sync the tags of all waiting photos of one folder. Returns False if there was nothing to do."""
        now = time.time()
        if now < self.server_retry:
            return False
        batch = self.db.upload_batch(now)
        if not batch:
            return False
        nc = self._client()
        if nc is None:
            return False

        folder = os.path.dirname(batch[0][0])
        remote_dir = self.cfg.remote_path(folder + "/x")
        if remote_dir is None:
            for path, _, _ in batch:
                self.db.set_error(path, "not inside a configured folder")
            return True
        remote_dir = remote_dir.rsplit("/", 1)[0]

        self._set("working", f"Syncing tags of {self._rel(folder)}")
        self.write_status(force=True)
        try:
            on_server = nc.list_dir(remote_dir)
            futures = {}
            pool = self._pool()
            for path, tags, uploaded in batch:
                entry = on_server.get(os.path.basename(path))
                if entry is None:  # not synced to the server yet
                    self.db.set_uploaded(path, uploaded, done=False, retry_at=now + NOT_SYNCED_RETRY)
                    continue
                futures[pool.submit(self._sync_file, nc, entry[0], entry[1], tags)] = (path, uploaded)

            failure: ServerError | None = None
            for fut in as_completed(futures):
                if fut.cancelled():  # sync switched off: the photo simply stays queued
                    continue
                path, uploaded = futures[fut]
                try:
                    up, ok = fut.result()
                except NotOnServer:
                    self.db.set_uploaded(path, uploaded, done=False, retry_at=now + NOT_SYNCED_RETRY)
                except ServerError as e:  # retried with the next batch, after a pause
                    failure = failure or e
                    self.db.set_uploaded(path, uploaded, done=False)
                else:
                    self.db.set_uploaded(path, list(up), done=ok, retry_at=0 if ok else now + TAG_FORBIDDEN_RETRY)
                if SYNC_OFF_FLAG.exists():  # switched off meanwhile: finish what is already running
                    for f in futures:
                        f.cancel()
                self.write_status()
            if failure:
                raise failure
            self.server_msg = ""
            self.server_backoff = 60
        except AuthError as e:
            self._server_failed(str(e), 600)
            self.nc = None
        except ServerError as e:
            self._server_failed(f"Nextcloud not reachable – retrying ({e})", self.server_backoff)
            self.server_backoff = min(self.server_backoff * 2, 1800)
        return True

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.cfg.sync_connections, thread_name_prefix="sync")
        return self._executor

    def _sync_file(self, nc: Nextcloud, fid: int, on_server: dict[str, int], tags: list[str]) -> tuple[set[str], bool]:
        """Make the tags of one file match `tags`. Runs in a worker thread, touches only the server.

        Only tags with our suffix are managed (added / removed); all other tags stay untouched.
        Returns (wanted tags that are on the server, everything in sync)."""
        now = time.time()
        sfx = self.cfg.tag_suffix.casefold()
        want = {t.casefold(): t for t in tags}
        ours = {n: i for n, i in on_server.items() if sfx and n.casefold().endswith(sfx)}
        present = {n.casefold() for n in ours}
        up = {t for k, t in want.items() if k in present}
        stuck = False
        for name, tid in ours.items():
            if name.casefold() not in want:
                try:
                    nc.unassign(fid, tid)
                    log.debug("file %d: removed tag %r", fid, name)
                except TagForbidden as e:
                    log.warning("%s", e)
                    stuck = True
        for key, tag in sorted(want.items()):
            if key in present or self.forbidden.get(tag, 0) > now:
                continue
            try:
                nc.assign(fid, nc.tag_id(tag))
                up.add(tag)
            except TagForbidden as e:
                log.warning("%s", e)
                self.forbidden[tag] = now + TAG_FORBIDDEN_RETRY
        return up, not (set(tags) - up) and not stuck

    def _server_failed(self, msg: str, delay: float) -> None:
        log.warning("%s", msg)
        self.server_msg = msg[:200]
        self.server_retry = time.time() + delay
