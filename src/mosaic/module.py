"""Knowledge module: a self-contained, attachable store for one domain."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mosaic.encoder import normalize


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Spherical k-means; returns (k, D) normalised centroids."""
    k = min(k, len(x))
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(x @ centroids.T, axis=1)
        for j in range(k):
            members = x[assign == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
        centroids = normalize(centroids)
    return centroids


@dataclass
class KnowledgeModule:
    name: str
    encoder_id: str
    embeddings: np.ndarray            # (N, D) normalised
    entries: list[dict[str, Any]]     # N records, each with at least "label"
    descriptors: np.ndarray           # (K, D) routing centroids
    accept_threshold: float | None = None  # min nearest-neighbour similarity to accept a query
    accept_percentile: float | None = None  # how accept_threshold was derived (see set_accept_percentile)
    part_of: str | None = None  # e.g. "people/person" for a hand; lets a reasoning model walk whole -> parts
    reject_margin: float | None = None  # with "negative" entries: accept only if nearest positive beats
    #                                     the nearest negative by at least this much
    version: str = "0.1.0"
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if len(self.embeddings) != len(self.entries):
            raise ValueError("embeddings and entries must have the same length")
        if self.embeddings.shape[1] != self.descriptors.shape[1]:
            raise ValueError("embeddings and descriptors must share a dimension")

    @cached_property
    def _kinds(self) -> np.ndarray:
        return np.array([e.get("kind", "image") for e in self.entries])

    @cached_property
    def _kind_masks(self) -> dict[str, np.ndarray]:
        return {k: self._kinds == k for k in np.unique(self._kinds)}

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def build(
        cls,
        name: str,
        encoder_id: str,
        embeddings: np.ndarray,
        entries: list[dict[str, Any]],
        n_descriptors: int = 4,
        accept_percentile: float | None = 5.0,
        **meta: Any,
    ) -> KnowledgeModule:
        """Descriptors are clustered per entry kind (image/text), since the two
        modalities occupy different regions of a shared CLIP-style space.

        The acceptance threshold is self-calibrated: the `accept_percentile`-th
        percentile of each image entry's similarity to its nearest *other* image
        entry. A query further from every entry than that is atypical for this module.
        """
        embeddings = normalize(embeddings)
        kinds = np.array([e.get("kind", "image") for e in entries])
        descriptors = np.concatenate(  # negatives describe what the leaf is *not*: never route by them
            [kmeans(embeddings[kinds == k], n_descriptors) for k in dict.fromkeys(kinds) if k != "negative"]
        )
        module = cls(
            name=name,
            encoder_id=encoder_id,
            embeddings=embeddings,
            entries=entries,
            descriptors=descriptors,
            **meta,
        )
        if accept_percentile is not None and "accept_threshold" not in meta:
            module.set_accept_percentile(accept_percentile)
        return module

    @cached_property
    def _leave_one_out(self) -> np.ndarray:
        """Each stored image entry's similarity to its nearest *other* image entry (sampled to 2000)."""
        images = self.embeddings[self._kind_masks.get("image", np.zeros(len(self), bool))]
        if len(images) < 2:
            return np.array([])
        sample = images[np.random.default_rng(0).permutation(len(images))[:2000]]
        sims = sample @ images.T
        sims[sims > 0.9999] = -1.0  # exclude each entry (and exact duplicates) matching itself
        return sims.max(axis=1)

    def nearest(self, query: np.ndarray, kind: str = "image") -> float:
        """Similarity of the closest entry of `kind` (-inf if the module has none)."""
        mask = self._kind_masks.get(kind)
        if mask is None or not mask.any():
            return -np.inf
        return float((self.embeddings[mask] @ normalize(query)).max())

    def set_accept_percentile(self, percentile: float) -> None:
        """Accept queries at least as typical as this percentile of the module's own members."""
        if len(self._leave_one_out):
            self.accept_threshold = float(np.percentile(self._leave_one_out, percentile))
            self.accept_percentile = float(percentile)

    def search(self, query: np.ndarray, k: int = 5, kind: str | None = None) -> list[tuple[int, float]]:
        """Cosine top-k over entries (optionally only one kind). Returns [(entry_index, score), ...]."""
        scores = self.embeddings @ normalize(query)
        if kind is not None:
            mask = self._kind_masks.get(kind)
            scores = np.where(mask, scores, -np.inf) if mask is not None else np.full_like(scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def save(self, root: str | Path) -> Path:
        path = Path(root) / self.name
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "encoder_id": self.encoder_id,
            "dim": self.dim,
            "size": len(self),
            "accept_threshold": self.accept_threshold,
            "accept_percentile": self.accept_percentile,
            "part_of": self.part_of,
            "reject_margin": self.reject_margin,
            "created": self.created,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        np.save(path / "embeddings.npy", self.embeddings)
        np.save(path / "descriptors.npy", self.descriptors)
        with open(path / "entries.jsonl", "w") as f:
            for e in self.entries:
                f.write(json.dumps(e) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeModule:
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        with open(path / "entries.jsonl") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        return cls(
            name=manifest["name"],
            encoder_id=manifest["encoder_id"],
            embeddings=np.load(path / "embeddings.npy"),
            entries=entries,
            descriptors=np.load(path / "descriptors.npy"),
            accept_threshold=manifest.get("accept_threshold"),
            accept_percentile=manifest.get("accept_percentile"),
            part_of=manifest.get("part_of"),
            reject_margin=manifest.get("reject_margin"),
            version=manifest.get("version", "0.1.0"),
            description=manifest.get("description", ""),
            created=manifest.get("created", ""),
        )
