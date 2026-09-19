"""Central encoder: maps images and text into one shared, frozen embedding space."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise along the last axis and cast to float32."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class Encoder(Protocol):
    """Anything that embeds images and text into the same D-dim space.

    `encoder_id` must uniquely identify the weights: modules built with one
    encoder are invalid under another.
    """

    encoder_id: str
    dim: int

    def embed_images(self, images: Sequence[Any]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ClipEncoder:
    """Frozen OpenCLIP encoder. Requires the `clip` extra."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        fp16: bool = False,
    ):
        """fp16=True runs the image/text towers under CUDA autocast: ~1.3x faster on GPU,
        and embeddings stay within cosine 0.9999 of fp32, so fp32-built modules still apply."""
        import open_clip
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.encoder_id = f"open_clip/{model_name}/{pretrained}"
        from PIL import Image

        self.dim = int(self.embed_images([Image.new("RGB", (32, 32))]).shape[1])  # works for any tower type

    def _autocast(self):
        return self._torch.autocast("cuda", dtype=self._torch.float16, enabled=self.fp16)

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        batch = self._torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._torch.no_grad(), self._autocast():
            out = self.model.encode_image(batch)
        return normalize(out.float().cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with self._torch.no_grad(), self._autocast():
            out = self.model.encode_text(tokens)
        return normalize(out.float().cpu().numpy())


class DinoEncoder:
    """Frozen DINOv2 (self-supervised, trained without labels or text): image embeddings only.

    Modules built with it cannot hold text entries, and text queries are unsupported.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None, fp16: bool = True):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if fp16 and self.device.startswith("cuda") else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval().requires_grad_(False)
        self.dtype = dtype
        self.encoder_id = f"hf/{model_name}"
        self.dim = self.model.config.hidden_size

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        with self._torch.no_grad():
            out = self.model(pixel_values=inputs.to(self.device, self.dtype)).pooler_output  # normalised CLS token
        return normalize(out.float().cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("DINOv2 has no text encoder")
