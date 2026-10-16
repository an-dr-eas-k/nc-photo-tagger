#!/usr/bin/env bash
# Removes service, extension and program. Config, state database and faces are kept unless --purge.
set -uo pipefail
APP=nc-photo-tagger
UUID=nc-photo-tagger@nc-photo-tagger
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"

systemctl --user disable --now $APP.service 2>/dev/null
rm -f "$HOME/.config/systemd/user/$APP.service"
systemctl --user daemon-reload
gnome-extensions disable "$UUID" 2>/dev/null
rm -rf "$DATA/gnome-shell/extensions/$UUID" "$DATA/$APP/venv" "$HOME/.local/bin/$APP"

if [ "${1:-}" = "--purge" ]; then
    rm -rf "$DATA/$APP" "${XDG_CONFIG_HOME:-$HOME/.config}/$APP" "${XDG_CACHE_HOME:-$HOME/.cache}/$APP"
    secret-tool clear service $APP 2>/dev/null
    echo "Removed everything (tags already in Nextcloud stay there)."
else
    echo "Removed. Config, state and faces kept (use --purge to delete them)."
fi
