"""Minimal Nextcloud WebDAV client for collaborative (system) tags."""

from __future__ import annotations

import logging
import posixpath
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from urllib.parse import quote, unquote, urlparse

import requests

log = logging.getLogger(__name__)

NS = {"d": "DAV:", "oc": "http://owncloud.org/ns", "nc": "http://nextcloud.org/ns"}

PROPFIND_FILEID = """<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:prop><oc:fileid/></d:prop></d:propfind>"""

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


class Nextcloud:
    def __init__(self, url: str, user: str, password: str, timeout: float = 30):
        self.base = url.rstrip("/")
        self.user = user
        self.timeout = timeout
        self.s = requests.Session()
        self.s.auth = (user, password)
        self.s.headers["User-Agent"] = "nc-photo-tagger"
        self._tags: dict[str, int] | None = None
        self._dirs: OrderedDict[str, tuple[float, dict[str, int]]] = OrderedDict()

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        try:
            r = self.s.request(method, url, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            raise ServerError(str(e)) from e
        if r.status_code == 401:
            raise AuthError("Nextcloud login failed (HTTP 401) - run 'nc-photo-tagger setup'")
        if r.status_code >= 500 or r.status_code == 429:
            raise ServerError(f"{method} {url}: HTTP {r.status_code}")
        return r

    def _files_url(self, remote_path: str) -> str:
        return f"{self.base}/remote.php/dav/files/{quote(self.user, safe='')}{quote(remote_path)}"

    def check(self) -> None:
        r = self._req("PROPFIND", self._files_url("/"), data=PROPFIND_FILEID, headers={"Depth": "0"})
        if r.status_code != 207:
            raise ServerError(f"unexpected answer from server: HTTP {r.status_code}")

    # ---- files ---------------------------------------------------------------------------

    def _list_dir(self, remote_dir: str) -> dict[str, int]:
        url = self._files_url(remote_dir.rstrip("/") + "/")
        r = self._req("PROPFIND", url, data=PROPFIND_FILEID, headers={"Depth": "1"})
        if r.status_code == 404:
            return {}
        if r.status_code != 207:
            raise ServerError(f"PROPFIND {remote_dir}: HTTP {r.status_code}")
        own = unquote(urlparse(url).path).rstrip("/")
        result: dict[str, int] = {}
        for resp in ET.fromstring(r.content).findall("d:response", NS):
            href = unquote(resp.findtext("d:href", "", NS)).rstrip("/")
            if href == own:
                continue
            fid = resp.findtext("d:propstat/d:prop/oc:fileid", None, NS)
            if fid:
                result[href.rsplit("/", 1)[-1]] = int(fid)
        return result

    def file_id(self, remote_path: str) -> int:
        parent, name = posixpath.split(remote_path)
        entry = self._dirs.get(parent)
        if entry is None or (name not in entry[1] and time.time() - entry[0] > 60):
            entry = (time.time(), self._list_dir(parent))
            self._dirs[parent] = entry
        self._dirs.move_to_end(parent)
        while len(self._dirs) > 16:
            self._dirs.popitem(last=False)
        fid = entry[1].get(name)
        if fid is None:
            raise NotOnServer(remote_path)
        return fid

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

    def file_tags(self, file_id: int) -> dict[str, int]:
        """All tags assigned to a file: name -> tag id."""
        r = self._req("PROPFIND", f"{self.base}/remote.php/dav/systemtags-relations/files/{file_id}",
                      data=PROPFIND_TAGS, headers={"Depth": "1"})
        if r.status_code == 404:
            self._dirs.clear()
            raise NotOnServer(f"file id {file_id}")
        if r.status_code != 207:
            raise ServerError(f"listing tags of file {file_id} failed: HTTP {r.status_code}")
        tags: dict[str, int] = {}
        for resp in ET.fromstring(r.content).findall("d:response", NS):
            prop = resp.find("d:propstat/d:prop", NS)
            if prop is None:
                continue
            tid, name = prop.findtext("oc:id", None, NS), prop.findtext("oc:display-name", None, NS)
            if tid and name:
                tags[name] = int(tid)
        return tags

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
            self._dirs.clear()
            raise NotOnServer(f"file id {file_id} / tag id {tag_id}")
        if r.status_code == 403:
            raise TagForbidden(f"tag {tag_id} is not assignable (HTTP 403)")
        raise ServerError(f"assigning tag failed: HTTP {r.status_code}")
