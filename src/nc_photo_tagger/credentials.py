"""Nextcloud app password storage (GNOME keyring, with a file fallback)."""

from __future__ import annotations

import logging
import os

from . import APP
from .config import PASSWORD_FILE, Config

log = logging.getLogger(__name__)


def _account(cfg: Config) -> str:
    return f"{cfg.user}@{cfg.url}"


def get_password(cfg: Config) -> str | None:
    if pw := os.environ.get("NC_PHOTO_TAGGER_PASSWORD"):
        return pw
    try:
        import keyring

        if pw := keyring.get_password(APP, _account(cfg)):
            return pw
    except Exception as e:
        log.debug("keyring not available: %s", e)
    if PASSWORD_FILE.exists():
        return PASSWORD_FILE.read_text().strip() or None
    return None


def store_password(cfg: Config, password: str) -> str:
    """Store the password; returns a description of where it was stored."""
    try:
        import keyring

        keyring.set_password(APP, _account(cfg), password)
        if PASSWORD_FILE.exists():
            PASSWORD_FILE.unlink()
        return "GNOME keyring"
    except Exception as e:
        log.warning("keyring not available (%s), storing password in %s", e, PASSWORD_FILE)
        PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(PASSWORD_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(password)
        return str(PASSWORD_FILE)
