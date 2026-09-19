"""Class-agnostic object detector: finds "things" without knowing what they are.

Uses only OWLv2's vision tower, objectness head and box head. The text tower
and class head are never called, so detection involves no label set or text
query. Each patch token predicts one box plus an objectness score; we keep
confident, non-tiny boxes after non-maximum suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torchvision.ops import nms

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels


@dataclass
class Detection:
    box: Box
    score: float  # objectness probability


class ObjectnessDetector:
    def __init__(
        self,
        model: str = "google/owlv2-base-patch16-ensemble",
        device: str | None = None,
        fp16: bool = True,
        min_score: float = 0.1,
        max_objects: int = 8,
        nms_iou: float = 0.5,
        min_area: float = 0.01,
        min_side: int = 64,
    ):
        from transformers import AutoImageProcessor, Owlv2ForObjectDetection

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if fp16 and self.device.startswith("cuda") else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model, use_fast=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        self.detector_id = model
        self.min_score, self.max_objects, self.nms_iou, self.min_area = min_score, max_objects, nms_iou, min_area
        self.min_side = min_side  # pixels; smaller crops are too blurry for the encoder to recognise

    @torch.no_grad()
    def detect(self, images: Sequence[Any]) -> list[list[Detection]]:
        """Objects in each PIL image, highest objectness first."""
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        feature_map, _ = self.model.image_embedder(pixels.to(self.device, self.dtype))
        b, h, w, d = feature_map.shape
        feats = feature_map.reshape(b, h * w, d)
        scores = torch.sigmoid(self.model.objectness_predictor(feats).float())
        boxes = self.model.box_predictor(feats, feature_map).float()  # cxcywh on the padded square
        return [self._select(img, boxes[i], scores[i]) for i, img in enumerate(images)]

    def _select(self, image, cxcywh: torch.Tensor, scores: torch.Tensor) -> list[Detection]:
        w, h = image.size
        side = max(w, h)  # OWLv2 pads to a square (bottom/right) before resizing
        cx, cy, bw, bh = cxcywh.unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * side
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
        bw, bh = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
        keep = (scores >= self.min_score) & (bw * bh >= self.min_area * w * h)
        keep &= torch.minimum(bw, bh) >= self.min_side
        xyxy, scores = xyxy[keep], scores[keep]
        keep = nms(xyxy, scores, self.nms_iou)[: self.max_objects]
        return [Detection(tuple(int(round(v)) for v in xyxy[i].tolist()), float(scores[i])) for i in keep]
