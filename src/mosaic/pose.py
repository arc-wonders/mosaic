"""Skeleton sensor: body keypoints for a person box (ViTPose, COCO 17 keypoints).

Like the detector and SAM, this is a frozen vision model with no text tower. It
answers only "where are the joints", in image pixels; what they mean (an arm is
raised, the elbow is bent) is decided by the rules in `mosaic.geometry`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# COCO order; left/right are the person's own sides, not the viewer's
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseEstimator:
    def __init__(self, model: str = "usyd-community/vitpose-base-simple", device: str | None = None,
                 fp16: bool = True):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if fp16 and self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = VitPoseForPoseEstimation.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        self.pose_id = model

    def estimate(self, image: Any, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
        """One (17, 3) array [x, y, confidence] in image pixels per [x0, y0, x1, y1] person box."""
        if not boxes:
            return []
        xywh = [[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]]
        inputs = self.processor(image, boxes=xywh, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model(pixel_values=inputs["pixel_values"].to(self.device, self.dtype))
        out.heatmaps = out.heatmaps.float()
        results = self.processor.post_process_pose_estimation(out, boxes=xywh)[0]
        return [np.concatenate([r["keypoints"].numpy(), r["scores"].numpy()[:, None]], axis=1).astype(np.float32)
                for r in results]
