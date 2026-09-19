"""Class-agnostic segmentation: turns each detected box into a pixel mask.

SAM (Segment Anything) is prompted with the detector's box and returns the
object's outline. Like the detector, it has no label set. The masked object
is then cut out onto a neutral background before embedding, so background
pixels (grass, sky, a table) no longer contribute to the object's embedding.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from mosaic.perceiver import NEUTRAL, Box, cutout  # noqa: F401  (re-exported)


class BoxSegmenter:
    def __init__(self, model: str = "facebook/sam-vit-base", device: str | None = None, fp16: bool = True,
                 encoder_batch: int = 2):
        from transformers import SamModel, SamProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if fp16 and self.device.startswith("cuda") else torch.float32
        self.processor = SamProcessor.from_pretrained(model)
        self.model = SamModel.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        self.segmenter_id = model
        self.encoder_batch = encoder_batch  # SAM's global attention is memory-hungry: ~1.5 GB per image in fp16

    @torch.no_grad()
    def segment(self, images: Sequence[Any], boxes: Sequence[Sequence[Box]]) -> list[list[np.ndarray]]:
        """For each image, one boolean (H, W) mask per box."""
        out: list[list[np.ndarray]] = [[] for _ in images]
        todo = [i for i, b in enumerate(boxes) if b]
        if not todo:
            return out
        pixels = self.processor(images=[images[i] for i in todo], return_tensors="pt")["pixel_values"]
        embeddings = torch.cat([
            self.model.get_image_embeddings(pixels[j : j + self.encoder_batch].to(self.device, self.dtype))
            for j in range(0, len(pixels), self.encoder_batch)
        ])
        longest = self.processor.image_processor.size["longest_edge"]
        for row, i in enumerate(todo):
            w, h = images[i].size
            scale = longest / max(w, h)  # SAM resizes the longest side to 1024; box prompts follow
            input_boxes = torch.tensor([[[c * scale for c in b] for b in boxes[i]]], device=self.device, dtype=self.dtype)
            pred = self.model(image_embeddings=embeddings[row : row + 1], input_boxes=input_boxes,
                              multimask_output=True)
            reshaped = torch.tensor([[int(h * scale + 0.5), int(w * scale + 0.5)]])
            masks = self.processor.image_processor.post_process_masks(
                pred.pred_masks.float().cpu(), torch.tensor([[h, w]]), reshaped)[0]
            best = pred.iou_scores.float().cpu()[0].argmax(dim=-1)  # SAM's own quality estimate per box
            out[i] = [masks[j, best[j]].numpy().astype(bool) for j in range(len(boxes[i]))]
        return out
