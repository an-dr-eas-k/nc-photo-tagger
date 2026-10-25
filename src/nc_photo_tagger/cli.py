"""Command line interface."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
import time
from pathlib import Path

from . import APP, __version__
from . import config as C


def _setup_logging(verbose: bool) -> None:
    fmt = "%(levelname)s %(name)s: %(message)s" if os.environ.get("JOURNAL_STREAM") else \
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt)
    for noisy in ("urllib3", "httpx", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_setup(args) -> int:
    photos = Path(args.photos).expanduser().absolute() if args.photos else None
    if not C.CONFIG_FILE.exists() or args.force_config:
        url, user, remote, local = C.detect_nextcloud_client(photos)
        if photos and not remote:
            print(f"Could not find {photos} in the Nextcloud desktop client config.")
            remote = input("Path of this folder on the Nextcloud server (e.g. /Photos): ").strip() or "/Photos"
            local = photos
        url = args.url or url or input("Nextcloud URL (https://…): ").strip()
        user = args.user or user or input("Nextcloud user name: ").strip()
        folders = [(str(local), remote)] if local else []
        C.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        C.CONFIG_FILE.write_text(C.default_config_text(url, user, folders))
        print(f"Wrote {C.CONFIG_FILE}")
    else:
        print(f"Using existing {C.CONFIG_FILE}")

    cfg = C.load()
    print(f"  Nextcloud: {cfg.user} @ {cfg.url}")
    for f in cfg.folders:
        print(f"  Folder:    {f.local}  ->  {f.remote}")
    if args.no_password:
        return 0

    from .credentials import get_password, store_password
    from .nextcloud import Nextcloud, ServerError

    if get_password(cfg) and not args.password:
        print("An app password is already stored (use --password to change it).")
        return 0
    if not sys.stdin.isatty():
        print("Run 'nc-photo-tagger setup' in a terminal to enter the Nextcloud app password.")
        return 0
    print("\nCreate an app password in Nextcloud: Settings → Security → 'Create new app password'.")
    while True:
        pw = getpass.getpass("Nextcloud app password: ").strip()
        if not pw:
            return 1
        try:
            Nextcloud(cfg.url, cfg.user, pw).check()
        except ServerError as e:
            print(f"  ✗ {e}")
            continue
        print(f"  ✓ login ok, password stored in {store_password(cfg, pw)}")
        break
    os.system("systemctl --user try-restart nc-photo-tagger.service >/dev/null 2>&1")
    return 0


def cmd_run(args) -> int:
    from .service import Service

    Service(C.load()).run()
    return 0


def cmd_status(args) -> int:
    try:
        st = json.loads(C.STATUS_FILE.read_text())
    except (OSError, ValueError):
        print("no status yet - is the service running?  systemctl --user status nc-photo-tagger")
        return 1
    age = time.time() - st.get("updated", 0)
    print(f"{st['percent']}% of {st['total']:,} files done  ({st['state']}, updated {age:.0f}s ago)")
    print(f"  {st['message']}")
    print(f"  local tagging: {'on' if st.get('analysis_enabled', True) else 'OFF'}   "
          f"Nextcloud sync: {'on' if st.get('sync_enabled', True) else 'OFF'}")
    print(f"  pending analysis: {st['pending']:,}   waiting for upload: {st['waiting']:,}   errors: {st['errors']:,}")
    print(f"  known people: {st['people']}   unknown face groups to name: {st['clusters']}  ({st['unknown_dir']})")
    if st.get("server"):
        print(f"  server: {st['server']}")
    return 0


def cmd_test(args) -> int:
    """Show tag probabilities and recognized faces for some images (no upload, no state change)."""
    import numpy as np

    from .db import DB
    from .faces import People
    from .models import ClipTagger, FaceEngine, load_image

    cfg = C.load()
    tagger = ClipTagger(cfg.threads)
    tagger.configure(cfg.tags, cfg.background_prompts, cfg.aesthetic_tag, cfg.aesthetic_threshold)
    people = None
    if cfg.faces_enabled:
        engine = FaceEngine(cfg.min_face_px, threads=cfg.threads)
        db = DB(C.DB_FILE, readonly=True) if C.DB_FILE.exists() else None
        people = People(cfg, db, engine)
        people.load_refs()
    for path in args.images:
        try:
            img = load_image(path)
        except Exception as e:
            print(f"{path}: cannot read: {e}")
            continue
        emb = tagger.embed(img)
        probs = tagger.probabilities(emb)[0]
        order = np.argsort(-probs)[: args.top]
        tags = [t + cfg.tag_suffix for t in tagger.tags_for(emb, cfg.threshold)[0]]
        print(f"\n{path}")
        print("  scores: " + ", ".join(f"{tagger.names[i]} {probs[i]:.2f}" for i in order))
        aes = tagger.aesthetic_scores(emb)
        if aes is not None:
            print(f"  aesthetic: {aes[0]:.2f} (tag '{cfg.aesthetic_tag}' at >= {cfg.aesthetic_threshold})")
        print(f"  tags (>= {cfg.threshold}): {', '.join(tags) or '-'}")
        if cfg.faces_enabled:
            faces = people.engine.faces(img) if people else []
            names = people.match(np.stack([f.emb for f in faces])) if faces else []
            print(f"  faces: {len(faces)}" + (" → " + ", ".join(n or "unknown" for n in names) if faces else ""))
    return 0


def _flag(path: Path, on: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if on:
        path.touch()
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(prog=APP, description="Tag Nextcloud photos with a local AI model.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="create the config and store the Nextcloud app password")
    s.add_argument("--photos", help="local photo folder (synced by the Nextcloud client)")
    s.add_argument("--url")
    s.add_argument("--user")
    s.add_argument("--force-config", action="store_true", help="overwrite an existing config file")
    s.add_argument("--no-password", action="store_true", help="do not ask for the password")
    s.add_argument("--password", action="store_true", help="change the stored password")
    s.set_defaults(func=cmd_setup)

    sub.add_parser("run", help="run the service (used by systemd)").set_defaults(func=cmd_run)
    sub.add_parser("status", help="show progress").set_defaults(func=cmd_status)

    t = sub.add_parser("test", help="show tags and faces for images, to tune the config")
    t.add_argument("images", nargs="+")
    t.add_argument("--top", type=int, default=8)
    t.set_defaults(func=cmd_test)

    for name, flag, what in (("analysis", C.ANALYSIS_OFF_FLAG, "local AI tagging of photos"),
                             ("sync", C.SYNC_OFF_FLAG, "syncing tags with Nextcloud")):
        sw = sub.add_parser(name, help=f"switch {what} on or off")
        sw.add_argument("state", choices=["on", "off"])
        sw.set_defaults(func=lambda a, f=flag: _flag(f, a.state == "off"))
    sub.add_parser("pause", help="switch analysis and sync off").set_defaults(
        func=lambda a: (_flag(C.ANALYSIS_OFF_FLAG, True), _flag(C.SYNC_OFF_FLAG, True)))
    sub.add_parser("resume", help="switch analysis and sync on").set_defaults(
        func=lambda a: (_flag(C.ANALYSIS_OFF_FLAG, False), _flag(C.SYNC_OFF_FLAG, False)))
    sub.add_parser("rescan", help="rescan the photo folders now").set_defaults(
        func=lambda a: _flag(C.RESCAN_FLAG, True))
    sub.add_parser("recluster", help="regroup unknown faces now").set_defaults(
        func=lambda a: _flag(C.RECLUSTER_FLAG, True))

    args = p.parse_args()
    _setup_logging(args.verbose)
    sys.exit(args.func(args) or 0)
