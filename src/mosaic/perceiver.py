"""Perceiver: image -> class-agnostic objects -> one embedding per object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from kornia_rs import imgproc as k_imgproc
from PIL import Image

from mosaic.encoder import Encoder

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)

# CLIP's normalisation mean: this colour becomes exactly zero after preprocessing
NEUTRAL = (122, 117, 104)


@dataclass
class Region:
    box: Box
    embedding: np.ndarray  # (D,)
    objectness: float = 1.0
    fallback: bool = False  # True if no object was found and the whole image was used
    area: float = 1.0       # object area (mask if segmented, else box) as a fraction of the image
    mask: np.ndarray | None = None  # (H, W) bool, if a segmenter was used


class Detector(Protocol):
    def detect(self, images: Sequence[Any]) -> list[list[Any]]: ...  # items have .box, .score


class Segmenter(Protocol):
    def segment(self, images: Sequence[Any], boxes: Sequence[Sequence[Box]]) -> list[list[np.ndarray]]: ...


def cutout(image: Any, box: Box, mask: np.ndarray, pad: float = 0.1, dilate: int = 5, keep: float = 0.5) -> Any:
    """The masked object, cropped to `box` (+pad), with the background faded toward neutral.

    `keep` is how much of the background survives: 0 removes it entirely (a flat
    cut-out), 1 is a plain box crop. CLIP was trained on natural photos and
    recognises flat cut-outs noticeably worse, so the default keeps a faint background.
    """
    alpha_arr = mask.astype(np.uint8) * 255
    if dilate > 1:  # keep a thin rim of edge pixels
        alpha_arr = k_imgproc.dilate(alpha_arr[..., None], kernel="box", size=(dilate, dilate))[..., 0]
    alpha = Image.fromarray(alpha_arr)
    background = Image.blend(Image.new("RGB", image.size, NEUTRAL), image, keep)
    return crop(Image.composite(image, background, alpha), box, pad)


def crop(image: Any, box: Box, pad: float = 0.1) -> Any:
    """Crop `box` with `pad` (fraction of box size) of surrounding context."""
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    w, h = image.size
    return image.crop((max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py))))


class Perceiver:
    """Finds objects without knowing what they are, and embeds each one.

    With no detector, the whole image is the single region. With a segmenter,
    each object is cut out along its mask and the background is faded
    (`background_keep`), so background pixels contribute less.
    """

    def __init__(self, encoder: Encoder, detector: Detector | None = None,
                 segmenter: Segmenter | None = None, pad: float = 0.1, background_keep: float = 0.5):
        self.encoder = encoder
        self.detector = detector
        self.segmenter = segmenter
        self.pad = pad
        self.background_keep = background_keep

    def perceive(self, image: Any) -> list[Region]:
        return self.perceive_batch([image])[0]

    def perceive_batch(self, images: Sequence[Any]) -> list[list[Region]]:
        if self.detector is None:
            detections = [[] for _ in images]
        else:
            detections = self.detector.detect(images)
        if self.segmenter is not None:
            masks = self.segmenter.segment(images, [[d.box for d in dets] for dets in detections])
        else:
            masks = [[None] * len(dets) for dets in detections]
        crops, owners, meta = [], [], []
        for i, (image, dets, ms) in enumerate(zip(images, detections, masks)):
            if not dets:  # nothing object-like: fall back to the whole image
                w, h = image.size
                crops.append(image)
                meta.append(((0, 0, w, h), 1.0, self.detector is not None, None))
            for d, m in zip(dets, ms):
                crops.append(crop(image, d.box, self.pad) if m is None else cutout(image, d.box, m, self.pad, keep=self.background_keep))
                meta.append((d.box, d.score, False, m))
            owners += [i] * max(1, len(dets))
        embs = self.encoder.embed_images(crops)
        out: list[list[Region]] = [[] for _ in images]
        for i, e, (box, score, fallback, mask) in zip(owners, embs, meta):
            w, h = images[i].size
            pixels = mask.sum() if mask is not None else (box[2] - box[0]) * (box[3] - box[1])
            out[i].append(Region(box, e, score, fallback, float(pixels) / max(w * h, 1), mask))
        return out
