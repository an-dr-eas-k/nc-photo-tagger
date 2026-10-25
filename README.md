# nc-photo-tagger

Tags a local (Nextcloud-synced) photo library with a **small local AI model** and uploads the
tags to Nextcloud as *collaborative tags* — including **face recognition**. Shows progress
in the GNOME top bar ("36% of 135,448 files done" on mouse-over) and starts at login.

Everything runs locally on the CPU. No photo leaves the machine (only tag names are sent to
your Nextcloud server).

| Part | What |
|---|---|
| Tagging | [CLIP ViT-B/32](https://huggingface.co/Qdrant/clip-ViT-B-32-vision) (ONNX, ~340 MB) zero-shot: every tag in your list is compared against the photo |
| Faces | OpenCV [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) detector + [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) recognizer (~40 MB) |
| Upload | Nextcloud WebDAV system tags API (`/remote.php/dav/systemtags…`) |
| Service | systemd user service, idle CPU/IO priority, autostarts with the GNOME session |
| Top bar | small GNOME Shell extension (GNOME 45 – 50) |

Speed: ~0.2–0.3 s per photo on a 8-core CPU (~5 photos/s, ~1 GB RAM).

## Install

```bash
./install.sh ~/Pictures                  # your Nextcloud-synced photo folder; remote path is read from the Nextcloud client
nc-photo-tagger setup                       # enter a Nextcloud app password (Settings → Security)
```

The installer

- creates a venv in `~/.local/share/nc-photo-tagger/venv` and links `~/.local/bin/nc-photo-tagger`,
- writes `~/.config/nc-photo-tagger/config.toml` (Nextcloud URL, user and the local → server
  folder mapping are taken from the Nextcloud desktop client),
- installs and starts `~/.config/systemd/user/nc-photo-tagger.service` (autostart at login),
- installs the GNOME Shell extension `nc-photo-tagger@nc-photo-tagger`. On Wayland a new extension
  only appears after **logging out and in again** once.

The app password is stored in the GNOME keyring. Until it is set, the service already analyzes
photos; the uploads wait.

Uninstall: `./uninstall.sh` (keeps config, state and faces) or `./uninstall.sh --purge`.

## How it works

1. **Scan** the folders every 30 min (and on "Rescan photos now") for new / changed / removed photos.
2. **Analyze** each photo (newest years first): CLIP image embedding + faces. Embeddings are
   stored in `~/.local/share/nc-photo-tagger/state.db`, so changing tags, thresholds or people
   later does **not** require running the AI over the photos again.
3. **Sync** the tags to Nextcloud. All tags of this app end with **` (pt)`** (e.g. `beach (pt)`,
   `Person: Anna (pt)`). The app owns these tags: per photo it adds the missing ones (creating the tag
   in Nextcloud if needed) and **removes `(pt)` tags that no longer apply** – e.g. after you removed
   a tag from the list, raised the threshold or renamed a person. Tags without the ` (pt)` suffix are
   never touched. Photos that are not on the server yet (sync pending) are retried later.
   For speed the sync works folder by folder: one request lists the files and their current tags,
   then the changes are sent over several parallel connections (`connections` in `[nextcloud]`,
   default 8). If the server answers with errors (HTTP 5xx / 429) the app automatically backs off
   to fewer connections and speeds up again when the server copes.

## Tags

Edit the `[tags]` section of `~/.config/nc-photo-tagger/config.toml` and restart the service
(top bar menu → *Restart service*). Format: `"tag name in Nextcloud" = "English description for the AI"`
(or a list of descriptions, or `""` for "a photo of <name>"). The tag *names* can be German.

The suffix ` (pt)` is added automatically (`suffix` in `[tagging]`).
A tag is assigned if its probability is ≥ `threshold` (default 0.30; lower = more tags).
The `background_prompts` win when nothing fits, so a photo may get no tag at all.
To tune, look at the scores of some photos:

```bash
nc-photo-tagger test ~/Pictures/2001/12/*/*.jpg
```

## Faces / people

People are tagged as `Person: <Name> (pt)` (prefix and suffix configurable).

- **Known people**: `~/.local/share/nc-photo-tagger/faces/known/<Name>/` contains example photos
  of each person. You can drop your own photos there (one clear face per photo).
- **Unknown faces** are grouped automatically into
  `~/.local/share/nc-photo-tagger/faces/unknown/cluster_0001_123_faces/` etc. (top bar menu →
  *Name unknown faces…*). To name a group:
  1. delete the face crops that do not belong to the person (optional),
  2. **rename the folder to the person's name**, e.g. `Anna`.

  Within ~20 s the service moves the crops to `known/Anna/`, assigns all faces of that group
  (except the deleted ones) to Anna, re-matches all faces of the library and uploads the new tags.
  Rename a group to `_ignore` to ignore these faces (strangers, posters, …).
- Groups are rebuilt every 2,000 new unknown faces while working, when idle, or via
  *Regroup unknown faces now*.

Note: Nextcloud collaborative tags are global — other users of the server can see the tag
*names* (but only on files they have access to).

## Top bar

Hover the camera icon for the progress, click it for details and actions: the two switches
below, name faces, rescan, regroup faces, open settings, restart service. The icon is dimmed
when the service is not running and shows a warning sign on Nextcloud problems.

Two independent switches (they persist across restarts):

- **Tag photos locally** – the AI analysis. Off: no photo is analyzed (no CPU use), new photos are only counted.
- **Sync tags to Nextcloud** – the upload / removal of tags. Off: the app never talks to the server;
  analyzed photos queue up and are synced as soon as you switch it on.

E.g. analyze everything first, check the results with `nc-photo-tagger test`, then switch sync on.

## Commands

```
nc-photo-tagger status            progress in the terminal
nc-photo-tagger analysis on|off   local AI tagging
nc-photo-tagger sync on|off       tag sync with Nextcloud
nc-photo-tagger pause | resume    both switches off / on
nc-photo-tagger rescan | recluster
nc-photo-tagger test IMAGE...     tag scores + recognized people (no changes)
nc-photo-tagger setup --password  change the stored app password
journalctl --user -u nc-photo-tagger -f
```

## Files

| Path | Content |
|---|---|
| `~/.config/nc-photo-tagger/config.toml` | settings and tag list |
| `~/.local/share/nc-photo-tagger/state.db` | per-photo state, embeddings, faces |
| `~/.local/share/nc-photo-tagger/faces/` | `known/` people, `unknown/` face groups |
| `~/.cache/nc-photo-tagger/models/` | AI models (downloaded on first start) |
| `~/.cache/nc-photo-tagger/status.json` | progress, read by the GNOME extension |

## License

[MIT](LICENSE). The AI models are downloaded separately and keep their own licenses
(CLIP ViT-B/32: MIT, YuNet: MIT, SFace: Apache-2.0).
