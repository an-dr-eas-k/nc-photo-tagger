#!/usr/bin/env bash
# Installs nc-photo-tagger for the current user:
#   venv + package, config, systemd user service (autostart at login), GNOME Shell extension.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

APP=nc-photo-tagger
UUID=nc-photo-tagger@nc-photo-tagger
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
VENV="$DATA/$APP/venv"
PHOTOS="${1:-}"

echo "==> Installing Python package into $VENV"
if command -v uv >/dev/null; then
    [ -x "$VENV/bin/python" ] || uv venv --quiet --python python3 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install --quiet --upgrade --reinstall-package "$APP" .
else
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade .
fi
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/$APP" "$HOME/.local/bin/$APP"

echo "==> Configuration"
if [ -n "$PHOTOS" ]; then
    "$VENV/bin/$APP" setup --no-password --photos "$PHOTOS"
else
    "$VENV/bin/$APP" setup --no-password
fi

echo "==> systemd user service"
mkdir -p "$HOME/.config/systemd/user"
cp systemd/$APP.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable $APP.service
systemctl --user restart $APP.service

echo "==> GNOME Shell extension"
EXT="$DATA/gnome-shell/extensions/$UUID"
rm -rf "$EXT"
mkdir -p "$EXT"
cp extension/$UUID/* "$EXT/"
if ! gnome-extensions enable "$UUID" 2>/dev/null; then
    # A new extension is only picked up after the next login on Wayland: enable it for then.
    python3 - "$UUID" <<'EOF'
import ast, subprocess, sys
uuid = sys.argv[1]
cur = subprocess.run(["gsettings", "get", "org.gnome.shell", "enabled-extensions"],
                     capture_output=True, text=True).stdout.strip()
lst = ast.literal_eval(cur.removeprefix("@as ")) if cur else []
if uuid not in lst:
    lst.append(uuid)
    subprocess.run(["gsettings", "set", "org.gnome.shell", "enabled-extensions", str(lst)], check=True)
EOF
    echo "   Extension installed - log out and back in to see it in the top bar."
fi

echo
echo "Done. Next steps:"
echo "  1. Store your Nextcloud app password:   $APP setup"
echo "  2. Optional: edit tags in ~/.config/$APP/config.toml, then: systemctl --user restart $APP"
echo "  3. Progress:   $APP status    |   logs: journalctl --user -u $APP -f"
