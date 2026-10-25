"""Paths, configuration file handling and the default configuration."""

from __future__ import annotations

import configparser
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import APP


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / APP
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share") / APP
CACHE_DIR = _xdg("XDG_CACHE_HOME", ".cache") / APP

CONFIG_FILE = CONFIG_DIR / "config.toml"
PASSWORD_FILE = CONFIG_DIR / "app-password"  # fallback if no keyring is available
DB_FILE = DATA_DIR / "state.db"
MODEL_DIR = CACHE_DIR / "models"

# Files shared with the GNOME Shell extension
STATUS_FILE = CACHE_DIR / "status.json"
# Switches (flag file present = switched off), persisted across restarts
ANALYSIS_OFF_FLAG = CACHE_DIR / "analysis-off"  # no local AI tagging
SYNC_OFF_FLAG = CACHE_DIR / "sync-off"  # no communication with Nextcloud
RESCAN_FLAG = CACHE_DIR / "rescan"
RECLUSTER_FLAG = CACHE_DIR / "recluster"

NEXTCLOUD_CLIENT_CFG = _xdg("XDG_CONFIG_HOME", ".config") / "Nextcloud" / "nextcloud.cfg"


@dataclass
class Folder:
    local: Path
    remote: str


@dataclass
class Config:
    url: str
    user: str
    folders: list[Folder]
    extensions: set[str]
    exclude: list[str]
    rescan_minutes: float
    threads: int
    sync_connections: int
    threshold: float
    background_prompts: list[str]
    aesthetic_tag: str
    aesthetic_threshold: float
    tags: dict[str, list[str]]
    faces_enabled: bool
    known_dir: Path
    unknown_dir: Path
    tag_prefix: str
    tag_suffix: str
    match_threshold: float
    min_face_px: int
    cluster_threshold: float
    cluster_min_face_px: int
    min_cluster_size: int
    max_clusters: int
    raw: dict = field(default_factory=dict, repr=False)

    def remote_path(self, local: str | Path) -> str | None:
        p = Path(local)
        for f in self.folders:
            try:
                rel = p.relative_to(f.local)
            except ValueError:
                continue
            return f.remote.rstrip("/") + "/" + rel.as_posix()
        return None


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def load(path: Path = CONFIG_FILE) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist - run '{APP} setup' first")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    nc = raw.get("nextcloud", {})
    tagging = raw.get("tagging", {})
    faces = raw.get("faces", {})
    scan = raw.get("scan", {})
    template = tagging.get("prompt_template", "a photo of {}")

    tags: dict[str, list[str]] = {}
    for name, prompts in raw.get("tags", {}).items():
        if isinstance(prompts, str):
            prompts = [prompts] if prompts.strip() else []
        prompts = [p for p in prompts if p.strip()] or [template.format(name)]
        tags[name] = prompts

    return Config(
        url=nc.get("url", "").rstrip("/"),
        user=nc.get("user", ""),
        folders=[Folder(_expand(f["local"]), "/" + f["remote"].strip("/")) for f in raw.get("folders", [])],
        extensions={e.lower().lstrip(".") for e in scan.get("extensions", ["jpg", "jpeg", "png", "heic", "heif", "webp", "tif", "tiff", "bmp"])},
        exclude=list(scan.get("exclude", [])),
        rescan_minutes=float(scan.get("rescan_minutes", 30)),
        threads=int(scan.get("threads", 0)),
        sync_connections=max(1, int(raw.get("nextcloud", {}).get("connections", 8))),
        threshold=float(tagging.get("threshold", 0.30)),
        background_prompts=list(tagging.get("background_prompts", ["a photo", "an image", "a blurry photo"])),
        aesthetic_tag=str(tagging.get("aesthetic_tag", "highlight")),
        aesthetic_threshold=float(tagging.get("aesthetic_threshold", 6.0)),
        tags=tags,
        faces_enabled=bool(faces.get("enabled", True)),
        known_dir=_expand(faces.get("known_dir", str(DATA_DIR / "faces" / "known"))),
        unknown_dir=_expand(faces.get("unknown_dir", str(DATA_DIR / "faces" / "unknown"))),
        tag_prefix=faces.get("tag_prefix", "Person: "),
        tag_suffix=tagging.get("suffix", " (pt)"),
        match_threshold=float(faces.get("match_threshold", 0.40)),
        min_face_px=int(faces.get("min_face_px", 30)),
        cluster_threshold=float(faces.get("cluster_threshold", 0.50)),
        cluster_min_face_px=int(faces.get("cluster_min_face_px", 60)),
        min_cluster_size=int(faces.get("min_cluster_size", 5)),
        max_clusters=int(faces.get("max_clusters", 40)),
        raw=raw,
    )


def detect_nextcloud_client(photo_dir: Path | None) -> tuple[str, str, str | None, Path | None]:
    """Read url, user and the remote folder for photo_dir from the Nextcloud desktop client."""
    url = user = ""
    remote = local = None
    if not NEXTCLOUD_CLIENT_CFG.exists():
        return url, user, remote, local
    cp = configparser.RawConfigParser(strict=False)
    cp.read(NEXTCLOUD_CLIENT_CFG)
    if not cp.has_section("Accounts"):
        return url, user, remote, local
    acc = cp["Accounts"]
    url = acc.get("0\\url", "")
    user = acc.get("0\\dav_user", "") or acc.get("0\\webflow_user", "")
    folders: dict[str, dict[str, str]] = {}
    for key, value in acc.items():
        parts = key.split("\\")
        if len(parts) == 4 and parts[0] == "0" and parts[1].lower() == "folders":
            folders.setdefault(parts[2], {})[parts[3].lower()] = value
    if photo_dir is not None:
        photo_dir = photo_dir.expanduser().absolute()
        resolved = photo_dir.resolve()
        for f in folders.values():
            lp, tp = f.get("localpath"), f.get("targetpath")
            if not lp or tp is None:
                continue
            lp_path = Path(lp).resolve()
            try:
                rel = resolved.relative_to(lp_path)
            except ValueError:
                continue
            remote = "/" + "/".join(p for p in [tp.strip("/"), rel.as_posix().strip(".").strip("/")] if p)
            local = photo_dir
            break
    return url, user, remote, local


DEFAULT_TAGS = {
    # Nextcloud tag name: English description(s) for the AI ("" = "a photo of <name>").
    # The tag name can be anything (e.g. German), the descriptions should stay English.
    "beach": "a photo of a sandy beach at the sea",
    "sea": "a photo of the sea or ocean",
    "lake": "a photo of a lake",
    "mountains": "a photo of mountains",
    "snow": "a photo of a snowy landscape",
    "forest": "a photo of a forest",
    "landscape": "a landscape photo of nature",
    "city": "a photo of a city street",
    "architecture": "a photo of a building",
    "church": "a photo of a church",
    "museum": "a photo inside a museum",
    "sunset": "a photo of a sunset",
    "night": "a photo taken at night",
    "flowers": "a close-up photo of flowers",
    "garden": "a photo of a garden",
    "party": "a photo of people at a party",
    "birthday": ["a photo of a birthday cake with candles", "a photo of a birthday party"],
    "christmas": ["a photo of a christmas tree", "a photo of a christmas celebration"],
    "wedding": "a photo of a wedding",
    "concert": "a photo of a concert",
    "sports": "a photo of people doing sports",
    "food": "a photo of a meal on a plate",
    "drinks": "a photo of drinks",
    "dog": "a photo of a dog",
    "cat": "a photo of a cat",
    "horse": "a photo of a horse",
    "animals": "a photo of an animal",
    "car": "a photo of a car",
    "bicycle": "a photo of a bicycle",
    "boat": "a photo of a boat",
    "airplane": "a photo of an airplane",
    "train": "a photo of a train",
    "baby": "a photo of a baby",
    "children": "a photo of children playing",
    "group photo": "a group photo of many people",
    "portrait": "a portrait photo of a person",
    "document": "a photo of a paper document with text",
}


def default_config_text(url: str, user: str, folders: list[tuple[str, str]]) -> str:
    folder_text = "\n".join(
        f'[[folders]]\nlocal = "{loc}"\nremote = "{rem}"\n' for loc, rem in folders
    ) or '[[folders]]\nlocal = "~/Pictures"\nremote = "/Photos"\n'

    def toml_str(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    tag_lines = []
    for name, prompt in DEFAULT_TAGS.items():
        value = "[" + ", ".join(toml_str(p) for p in prompt) + "]" if isinstance(prompt, list) else toml_str(prompt)
        tag_lines.append(f"{toml_str(name)} = {value}")
    tags_text = "\n".join(tag_lines)

    return f"""# nc-photo-tagger configuration
# After editing: restart the service (top bar menu -> "Restart service" or
#   systemctl --user restart nc-photo-tagger). Changed tags / thresholds are re-applied to
#   already analyzed photos without re-running the AI on them.

[nextcloud]
url = {toml_str(url)}
user = {toml_str(user)}
# The app password is stored in the GNOME keyring: run `nc-photo-tagger setup`.
# Parallel requests when syncing tags (higher = faster, but more load on the server)
connections = 8

# Local folder (synced by the Nextcloud desktop client) and the matching path on the server.
{folder_text}
[scan]
extensions = ["jpg", "jpeg", "png", "heic", "heif", "webp", "tif", "tiff", "bmp"]
# Glob patterns relative to the folder, e.g. ["archive/*", "*/thumbnails/*"]
exclude = []
rescan_minutes = 30
# CPU threads for the AI model, 0 = all cores (the service runs with idle CPU priority anyway)
threads = 0

[tagging]
# A tag is assigned when its probability is >= threshold (0..1). Lower = more tags.
# Use `nc-photo-tagger test <image>` to see the scores for a photo.
threshold = 0.30
# Appended to every tag in Nextcloud (also people). The app manages ONLY tags with this suffix:
# it adds them and removes them again when they no longer apply. Other tags are never touched.
suffix = " (pt)"
prompt_template = "a photo of {{}}"
# Catch-all descriptions that "win" when none of the tags fit. Never uploaded.
background_prompts = ["a photo", "an image", "a blurry photo"]
# Photos rated >= aesthetic_threshold (about 1..10, most photos are 4.5-6) by a small aesthetic
# model get this tag. Empty string = off. `nc-photo-tagger test <image>` shows the score.
aesthetic_tag = "highlight"
aesthetic_threshold = 6.0

[faces]
enabled = true
# One sub folder per person with example photos: known/Anna/*.jpg
known_dir = "{DATA_DIR / 'faces' / 'known'}"
# Groups of unrecognized faces. Rename a "cluster_0001" folder to the person's name to
# teach the app this person; rename it to "_ignore" to ignore these faces.
unknown_dir = "{DATA_DIR / 'faces' / 'unknown'}"
tag_prefix = "Person: "
# Cosine similarity needed to recognize a person (0.363 is the SFace default, higher = stricter)
match_threshold = 0.40
min_face_px = 30
cluster_threshold = 0.50
cluster_min_face_px = 60
min_cluster_size = 5
max_clusters = 40

[tags]
{tags_text}
"""
