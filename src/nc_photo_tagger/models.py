"""The AI parts: image loading, CLIP zero-shot tagging and face detection/recognition."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import MODEL_DIR

log = logging.getLogger(__name__)

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = 300_000_000

# Faces are detected on (and their boxes stored relative to) the image scaled to this size.
MAX_SIDE = 1280


def load_image(path: str | Path, max_side: int = MAX_SIDE) -> Image.Image:
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))  # fast JPEG downscaling while decoding
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return im


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype(np.float32)


class ClipTagger:
    IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"
    TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
    LOGIT_SCALE = 100.0

    def __init__(self, threads: int = 0):
        from fastembed import ImageEmbedding, TextEmbedding

        def make(cls, name):
            kw = dict(cache_dir=str(MODEL_DIR), threads=threads or None)
            try:
                return cls(name, local_files_only=True, **kw)
            except Exception:
                log.info("downloading model %s ...", name)
                return cls(name, **kw)

        self._img = make(ImageEmbedding, self.IMAGE_MODEL)
        self._txt = make(TextEmbedding, self.TEXT_MODEL)
        self.names: list[str] = []
        self._T = np.zeros((0, 512), np.float32)
        self._owner = np.zeros(0, int)

    def configure(self, tags: dict[str, list[str]], background: list[str]) -> None:
        prompts, owner = [], []
        for i, ps in enumerate(tags.values()):
            prompts += ps
            owner += [i] * len(ps)
        prompts += background
        owner += [len(tags)] * len(background)
        self.names = list(tags)
        self._T = _normalize(np.array(list(self._txt.embed(prompts)), np.float32))
        self._owner = np.array(owner)

    def embed(self, image: Image.Image) -> np.ndarray:
        return _normalize(np.asarray(next(iter(self._img.embed([image]))), np.float32))

    def probabilities(self, embs: np.ndarray) -> np.ndarray:
        """(N, 512) image embeddings -> (N, len(names)) probabilities (softmax incl. background)."""
        embs = np.atleast_2d(embs).astype(np.float32)
        logits = self.LOGIT_SCALE * embs @ self._T.T
        n_cls = len(self.names) + 1
        cls = np.full((embs.shape[0], n_cls), -np.inf, np.float32)
        for c in range(n_cls):
            cols = self._owner == c
            if cols.any():
                cls[:, c] = logits[:, cols].max(axis=1)
        cls -= cls.max(axis=1, keepdims=True)
        p = np.exp(cls)
        p /= p.sum(axis=1, keepdims=True)
        return p[:, : len(self.names)]

    def tags_for(self, embs: np.ndarray, threshold: float) -> list[list[str]]:
        probs = self.probabilities(embs)
        return [[self.names[j] for j in np.flatnonzero(row >= threshold)] for row in probs]


@dataclass
class Face:
    x: float  # box relative to image size (0..1)
    y: float
    w: float
    h: float
    score: float
    px: int  # face size in pixels on the MAX_SIDE image
    emb: np.ndarray  # 128-d, L2 normalized


class FaceEngine:
    MODELS = {
        "face_detection_yunet_2023mar.onnx":
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx":
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    }

    def __init__(self, min_px: int = 30, score: float = 0.85, threads: int = 0):
        import cv2

        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
        if threads:
            cv2.setNumThreads(threads)
        self.cv2 = cv2
        det_path, rec_path = (self._model(name) for name in self.MODELS)
        self.min_px = min_px
        self.det = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def _model(self, name: str) -> Path:
        path = MODEL_DIR / name
        if not path.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s ...", name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(self.MODELS[name], tmp)
            if tmp.stat().st_size < 100_000:
                tmp.unlink()
                raise RuntimeError(f"download of {name} failed")
            tmp.rename(path)
        return path

    def faces(self, image: Image.Image) -> list[Face]:
        cv2 = self.cv2
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, dets = self.det.detect(bgr)
        if dets is None:
            return []
        out = []
        for d in dets:
            x, y, fw, fh = (float(v) for v in d[:4])
            px = int(min(fw, fh))
            if px < self.min_px:
                continue
            feat = self.rec.feature(self.rec.alignCrop(bgr, d)).flatten()
            out.append(Face(x / w, y / h, fw / w, fh / h, float(d[-1]), px, _normalize(feat)))
        return out
