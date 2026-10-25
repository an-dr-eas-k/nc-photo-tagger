"""Minimal Nextcloud WebDAV client for collaborative (system) tags."""

from __future__ import annotations

import logging
import random
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote, urlparse

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)

NS = {"d": "DAV:", "oc": "http://owncloud.org/ns", "nc": "http://nextcloud.org/ns"}

PROPFIND_FILEID = """<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:prop><oc:fileid/></d:prop></d:propfind>"""

# File ids and system tags of all files in a folder with one request
PROPFIND_DIR_TAGS = """<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
<d:prop><oc:fileid/><nc:system-tags/></d:prop></d:propfind>"""

PROPFIND_TAGS = """<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:prop>
<oc:id/><oc:display-name/><oc:user-visible/><oc:user-assignable/>
</d:prop></d:propfind>"""


class ServerError(Exception):
    """Server not reachable / temporary failure - retry later."""


class AuthError(ServerError):
    """Login failed."""


class NotOnServer(Exception):
    """The file is not (yet) on the server, e.g. not synced yet."""


class TagForbidden(Exception):
    """The tag cannot be created or assigned by this user."""


class _Gate:
    """Limits parallel requests. Halves the limit when the server struggles (HTTP 5xx / 429) and
    raises it again, one by one, after a streak of successful requests."""

    GROW_AFTER = 25

    def __init__(self, maximum: int):
        self.max = self.limit = max(1, maximum)
        self._active = 0
        self._streak = 0
        self._cond = threading.Condition()

    def acquire(self) -> None:
        with self._cond:
            while self._active >= self.limit:
                self._cond.wait()
            self._active += 1

    def release(self, ok: bool) -> None:
        with self._cond:
            self._active -= 1
            if not ok:
                self._streak = 0
                if self.limit > 1:
                    self.limit = max(1, self.limit // 2)
                    log.info("server is busy - using %d parallel connections", self.limit)
            else:
                self._streak += 1
                if self._streak >= self.GROW_AFTER and self.limit < self.max:
                    self.limit += 1
                    self._streak = 0
                    log.info("server is fine - using %d parallel connections", self.limit)
            self._cond.notify_all()


class Nextcloud:
    def __init__(self, url: str, user: str, password: str, timeout: float = 30, connections: int = 16):
        self.base = url.rstrip("/")
        self.user = user
        self.timeout = timeout
        self.s = requests.Session()
        self.s.auth = (user, password)
        self.s.headers["User-Agent"] = "nc-photo-tagger"
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=connections)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)
        self._tags: dict[str, int] | None = None
        self._tag_lock = threading.Lock()
        self._gate = _Gate(connections)

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        # Short retries for overloaded servers (HTTP 5xx / 429, e.g. database locks under load).
        for attempt in range(4):
            self._gate.acquire()
            overloaded = False
            try:
                r = self.s.request(method, url, timeout=self.timeout, **kw)
                overloaded = r.status_code >= 500 or r.status_code == 429
            except requests.RequestException as e:
                raise ServerError(str(e)) from e
            finally:
                self._gate.release(not overloaded)
            if r.status_code == 401:
                raise AuthError("Nextcloud login failed (HTTP 401) - run 'nc-photo-tagger setup'")
            if not overloaded:
                return r
            if attempt < 3:
                time.sleep(0.5 * 2 ** attempt + random.random() * 0.5)
        raise ServerError(f"{method} {url}: HTTP {r.status_code}")

    def _files_url(self, remote_path: str) -> str:
        return f"{self.base}/remote.php/dav/files/{quote(self.user, safe='')}{quote(remote_path)}"

    def check(self) -> None:
        r = self._req("PROPFIND", self._files_url("/"), data=PROPFIND_FILEID, headers={"Depth": "0"})
        if r.status_code != 207:
            raise ServerError(f"unexpected answer from server: HTTP {r.status_code}")

    # ---- files ---------------------------------------------------------------------------

    def list_dir(self, remote_dir: str) -> dict[str, tuple[int, dict[str, int]]]:
        """Files of a folder: name -> (file id, {tag name: tag id}). One request."""
        url = self._files_url(remote_dir.rstrip("/") + "/")
        r = self._req("PROPFIND", url, data=PROPFIND_DIR_TAGS, headers={"Depth": "1"})
        if r.status_code == 404:
            return {}
        if r.status_code != 207:
            raise ServerError(f"PROPFIND {remote_dir}: HTTP {r.status_code}")
        own = unquote(urlparse(url).path).rstrip("/")
        result: dict[str, tuple[int, dict[str, int]]] = {}
        for resp in ET.fromstring(r.content).findall("d:response", NS):
            href = unquote(resp.findtext("d:href", "", NS)).rstrip("/")
            if href == own:
                continue
            prop = resp.find("d:propstat/d:prop", NS)
            fid = prop.findtext("oc:fileid", None, NS) if prop is not None else None
            if not fid:
                continue
            tags = {}
            for t in prop.findall("nc:system-tags/nc:system-tag", NS):
                tid = t.get(f"{{{NS['oc']}}}id")
                if tid and t.text:
                    tags[t.text] = int(tid)
            result[href.rsplit("/", 1)[-1]] = (int(fid), tags)
        return result

    # ---- tags ----------------------------------------------------------------------------

    def _load_tags(self) -> dict[str, int]:
        r = self._req("PROPFIND", f"{self.base}/remote.php/dav/systemtags/", data=PROPFIND_TAGS, headers={"Depth": "1"})
        if r.status_code != 207:
            raise ServerError(f"listing tags failed: HTTP {r.status_code}")
        tags: dict[str, int] = {}
        for resp in ET.fromstring(r.content).findall("d:response", NS):
            prop = resp.find("d:propstat/d:prop", NS)
            if prop is None:
                continue
            tid, name = prop.findtext("oc:id", None, NS), prop.findtext("oc:display-name", None, NS)
            if tid and name:
                tags[name.casefold()] = int(tid)
        self._tags = tags
        return tags

    def tag_id(self, name: str) -> int:
        with self._tag_lock:
            return self._tag_id(name)

    def _tag_id(self, name: str) -> int:
        tags = self._tags if self._tags is not None else self._load_tags()
        key = name.casefold()
        if key in tags:
            return tags[key]
        r = self._req(
            "POST",
            f"{self.base}/remote.php/dav/systemtags/",
            json={"name": name, "userVisible": True, "userAssignable": True, "canAssign": True},
        )
        if r.status_code == 201:
            tid = int(r.headers.get("Content-Location", "").rstrip("/").rsplit("/", 1)[-1])
            tags[key] = tid
            log.info("created tag %r (id %d)", name, tid)
            return tid
        if r.status_code == 409:  # created meanwhile or not visible to us
            tags = self._load_tags()
            if key in tags:
                return tags[key]
        if r.status_code in (403, 409):
            raise TagForbidden(f"cannot create tag {name!r} (HTTP {r.status_code}); "
                               "create it in Nextcloud or allow users to create tags")
        raise ServerError(f"creating tag {name!r} failed: HTTP {r.status_code}")

    def unassign(self, file_id: int, tag_id: int) -> None:
        r = self._req("DELETE", f"{self.base}/remote.php/dav/systemtags-relations/files/{file_id}/{tag_id}")
        if r.status_code in (204, 200, 404):  # 404 = not assigned (anymore)
            return
        if r.status_code == 403:
            raise TagForbidden(f"tag {tag_id} cannot be removed (HTTP 403)")
        raise ServerError(f"removing tag failed: HTTP {r.status_code}")

    def assign(self, file_id: int, tag_id: int) -> None:
        r = self._req("PUT", f"{self.base}/remote.php/dav/systemtags-relations/files/{file_id}/{tag_id}")
        if r.status_code in (201, 204, 409):  # 409 = already assigned
            return
        if r.status_code == 404:  # file or tag vanished - reload tags on the next attempt
            self._tags = None
            raise NotOnServer(f"file id {file_id} / tag id {tag_id}")
        if r.status_code == 403:
            raise TagForbidden(f"tag {tag_id} is not assignable (HTTP 403)")
        raise ServerError(f"assigning tag failed: HTTP {r.status_code}")
