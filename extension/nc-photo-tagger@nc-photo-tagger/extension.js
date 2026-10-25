import GLib from 'gi://GLib';
import Gio from 'gi://Gio';
import GObject from 'gi://GObject';
import St from 'gi://St';
import Clutter from 'gi://Clutter';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

const APP = 'nc-photo-tagger';
const SERVICE = `${APP}.service`;
const POLL_SECONDS = 3;
const STALE_SECONDS = 120;

const cachePath = name => GLib.build_filenamev([GLib.get_user_cache_dir(), APP, name]);

function openPath(path) {
    GLib.mkdir_with_parents(path, 0o755);
    try {
        Gio.AppInfo.launch_default_for_uri(GLib.filename_to_uri(path, null),
            global.create_app_launch_context(0, -1));
    } catch (e) {
        logError(e, `${APP}: cannot open ${path}`);
    }
}

function setFlag(name, on) {
    const path = cachePath(name);
    if (on) {
        GLib.mkdir_with_parents(GLib.path_get_dirname(path), 0o755);
        GLib.file_set_contents(path, '');
    } else {
        try {
            Gio.File.new_for_path(path).delete(null);
        } catch {
            // already gone
        }
    }
}

const Indicator = GObject.registerClass(
class Indicator extends PanelMenu.Button {
    _init() {
        super._init(0.0, 'Photo Tagger');
        this._status = null;
        this._tooltipText = 'Photo Tagger';
        this._cancellable = new Gio.Cancellable();
        this._statusFile = Gio.File.new_for_path(cachePath('status.json'));

        this._icon = new St.Icon({icon_name: 'camera-photo-symbolic', style_class: 'system-status-icon'});
        this.add_child(this._icon);

        this._tooltip = new St.Label({style_class: 'nc-photo-tagger-tooltip', visible: false});
        Main.uiGroup.add_child(this._tooltip);
        this.connect('notify::hover', () => this._syncTooltip());
        this.menu.connect('open-state-changed', () => this._syncTooltip());

        this._buildMenu();
        this._refresh();
        this._timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, POLL_SECONDS, () => {
            this._refresh();
            return GLib.SOURCE_CONTINUE;
        });
    }

    _buildMenu() {
        const info = () => {
            const item = new PopupMenu.PopupMenuItem('', {reactive: false, can_focus: false});
            this.menu.addMenuItem(item);
            return item;
        };
        this._progressItem = info();
        this._messageItem = info();
        this._detailItem = info();
        this._peopleItem = info();
        this._serverItem = info();

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._analysisItem = new PopupMenu.PopupSwitchMenuItem('Tag photos locally', true);
        this._analysisItem.connect('toggled', (_item, state) => {
            setFlag('analysis-off', !state);
            this._refreshSoon();
        });
        this.menu.addMenuItem(this._analysisItem);
        this._syncItem = new PopupMenu.PopupSwitchMenuItem('Sync tags to Nextcloud', true);
        this._syncItem.connect('toggled', (_item, state) => {
            setFlag('sync-off', !state);
            this._refreshSoon();
        });
        this.menu.addMenuItem(this._syncItem);

        this._namesItem = this.menu.addAction('Name unknown faces…', () =>
            openPath(this._status?.unknown_dir ??
                GLib.build_filenamev([GLib.get_user_data_dir(), APP, 'faces', 'unknown'])));
        this.menu.addAction('Known people folder', () =>
            openPath(this._status?.known_dir ??
                GLib.build_filenamev([GLib.get_user_data_dir(), APP, 'faces', 'known'])));

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this.menu.addAction('Rescan photos now', () => {
            setFlag('rescan', true);
            this._refreshSoon();
        });
        this.menu.addAction('Regroup unknown faces now', () => {
            setFlag('recluster', true);
            this._refreshSoon();
        });
        this.menu.addAction('Settings…', () => {
            const path = GLib.build_filenamev([GLib.get_user_config_dir(), APP, 'config.toml']);
            try {
                Gio.AppInfo.launch_default_for_uri(GLib.filename_to_uri(path, null),
                    global.create_app_launch_context(0, -1));
            } catch (e) {
                logError(e, `${APP}: cannot open ${path}`);
            }
        });
        this.menu.addAction('Restart service', () => {
            try {
                Gio.Subprocess.new(['systemctl', '--user', 'restart', SERVICE], Gio.SubprocessFlags.NONE);
            } catch (e) {
                logError(e, `${APP}: cannot restart service`);
            }
            this._refreshSoon();
        });
    }

    _refreshSoon() {
        if (this._soon)
            return;
        this._soon = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 500, () => {
            this._soon = 0;
            this._refresh();
            return GLib.SOURCE_REMOVE;
        });
    }

    _refresh() {
        this._statusFile.load_contents_async(this._cancellable, (file, res) => {
            let status = null;
            try {
                const [, bytes] = file.load_contents_finish(res);
                status = JSON.parse(new TextDecoder().decode(bytes));
            } catch (e) {
                if (e instanceof GLib.Error && e.matches(Gio.IOErrorEnum, Gio.IOErrorEnum.CANCELLED))
                    return;
            }
            this._update(status);
        });
    }

    _update(st) {
        if (!this._tooltip)
            return;
        this._status = st;
        const fmt = n => Number(n ?? 0).toLocaleString();
        const stale = !st || st.state === 'stopped' || Date.now() / 1000 - (st.updated ?? 0) > STALE_SECONDS;

        let progress, message;
        if (!st) {
            progress = 'Service not running';
            message = `Start it with: systemctl --user start ${SERVICE}`;
        } else {
            progress = `${st.percent}% of ${fmt(st.total)} files done`;
            message = stale ? 'Service not running' : st.message;
        }
        this._tooltipText = `Photo Tagger\n${progress}\n${message}`;
        this._tooltip.text = this._tooltipText;

        this._progressItem.label.text = progress;
        this._messageItem.label.text = message;
        this._detailItem.label.text = st
            ? `${fmt(st.pending)} to analyze · ${fmt(st.waiting)} waiting for upload · ${fmt(st.errors)} unreadable`
            : '';
        this._detailItem.visible = !!st;
        this._peopleItem.label.text = st
            ? `${fmt(st.people)} known people · ${fmt(st.clusters)} face groups to name`
            : '';
        this._peopleItem.visible = !!st;
        this._serverItem.label.text = st?.server ?? '';
        this._serverItem.visible = !!st?.server;
        this._namesItem.label.text = st?.clusters ? `Name unknown faces (${st.clusters} groups)…` : 'Name unknown faces…';
        this._analysisItem.setToggleState(st?.analysis_enabled ?? true);
        this._syncItem.setToggleState(st?.sync_enabled ?? true);

        this._icon.opacity = stale || st?.state === 'paused' ? 120 : 255;
        this._icon.icon_name = st?.server && !stale ? 'dialog-warning-symbolic' : 'camera-photo-symbolic';
        this._positionTooltip();
    }

    _syncTooltip() {
        if (!this._tooltip)
            return;
        if (this.hover && !this.menu.isOpen) {
            this._tooltip.text = this._tooltipText;
            this._tooltip.opacity = 0;
            this._tooltip.show();
            this._positionTooltip();
            this._tooltip.ease({opacity: 255, duration: 150, mode: Clutter.AnimationMode.EASE_OUT_QUAD});
        } else {
            this._tooltip.remove_all_transitions();
            this._tooltip.hide();
        }
    }

    _positionTooltip() {
        if (!this._tooltip?.visible)
            return;
        const [x, y] = this.get_transformed_position();
        const [w, h] = this.get_transformed_size();
        const [, tw] = this._tooltip.get_preferred_width(-1);
        const monitor = Main.layoutManager.findMonitorForActor(this) ?? Main.layoutManager.primaryMonitor;
        let tx = Math.floor(x + w / 2 - tw / 2);
        tx = Math.max(monitor.x + 4, Math.min(tx, monitor.x + monitor.width - tw - 4));
        this._tooltip.set_position(tx, Math.floor(y + h + 6));
    }

    destroy() {
        this._cancellable.cancel();
        if (this._timer) {
            GLib.source_remove(this._timer);
            this._timer = 0;
        }
        if (this._soon) {
            GLib.source_remove(this._soon);
            this._soon = 0;
        }
        const tooltip = this._tooltip;
        this._tooltip = null;
        tooltip.destroy();
        super.destroy();
    }
});

export default class PhotoTaggerExtension extends Extension {
    enable() {
        this._indicator = new Indicator();
        Main.panel.addToStatusArea(this.uuid, this._indicator);
    }

    disable() {
        this._indicator?.destroy();
        this._indicator = null;
    }
}
