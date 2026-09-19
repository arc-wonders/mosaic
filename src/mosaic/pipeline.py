"""Mosaic: the end-to-end pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from mosaic.encoder import Encoder
from mosaic.matcher import Match, Matcher
from mosaic.module import KnowledgeModule
from mosaic.perceiver import Box, Detector, Perceiver, Region, Segmenter
from mosaic.registry import Registry
from mosaic.router import Route, Router


@dataclass
class Result:
    box: Box | None
    routes: list[Route]   # empty -> router abstained
    matches: list[Match]  # empty -> nothing close enough in the routed modules

    @property
    def abstained(self) -> bool:
        return not self.routes or not self.matches


@dataclass
class Prediction:
    label: str | None     # None -> unknown
    module: str | None    # leaf that supplied the answer (or was closest, if rejected)
    votes: int
    nearest: float        # similarity of the closest stored example in `module`
    routes: list[Route]
    matches: list[Match]
    reason: str           # "known" | "router" | "match_threshold" | "leaf_threshold" | "negative_margin"

    @property
    def known(self) -> bool:
        return self.label is not None


class Mosaic:
    def __init__(
        self,
        encoder: Encoder,
        detector: Detector | None = None,
        segmenter: Segmenter | None = None,
        background_keep: float = 0.5,
        top_k_modules: int | list[int] = 2,
        route_threshold: float = 0.2,
        match_threshold: float = 0.0,
        hierarchical: bool = True,
        leaf_gate: bool = True,
    ):
        self.leaf_gate = leaf_gate
        self.encoder = encoder
        self.perceiver = Perceiver(encoder, detector, segmenter, background_keep=background_keep)
        self.registry = Registry(encoder.encoder_id, encoder.dim)
        self.router = Router(self.registry, top_k_modules, route_threshold, hierarchical)
        self.matcher = Matcher(self.registry, min_score=match_threshold)

    def attach(self, module: KnowledgeModule) -> None:
        """Plug in a module; its path name (e.g. "fruits/apple") places it in the tree."""
        self.registry.add(module)

    def detach(self, name: str) -> list[str]:
        """Plug out a leaf ("fruits/apple") or a whole group ("fruits")."""
        return self.registry.remove(name)

    def query_embedding(
        self,
        q: np.ndarray,
        k: int = 5,
        box: Box | None = None,
        kind: str | None = None,
        min_score: float | None = None,
    ) -> Result:
        routes = self.router.route(q)
        return Result(box, routes, self.matcher.match(q, routes, k, kind, min_score))

    def classify(self, q: np.ndarray, k: int = 5) -> Prediction:
        """Route, k-NN vote over image entries, then apply the winning leaf's own gate."""
        res = self.query_embedding(q, k, kind="image")
        if not res.routes:
            return Prediction(None, None, 0, -np.inf, res.routes, res.matches, "router")
        if not res.matches:
            return Prediction(None, res.routes[0].module, 0, -np.inf, res.routes, res.matches, "match_threshold")
        (module, label), votes = Counter((m.module, m.label) for m in res.matches).most_common(1)[0]
        nearest = max(m.score for m in res.matches if m.module == module)
        leaf = self.registry.get(module)
        if self.leaf_gate and leaf.accept_threshold is not None and nearest < leaf.accept_threshold:
            return Prediction(None, module, votes, nearest, res.routes, res.matches, "leaf_threshold")
        if self.leaf_gate and leaf.reject_margin is not None and nearest - leaf.nearest(q, "negative") < leaf.reject_margin:
            return Prediction(None, module, votes, nearest, res.routes, res.matches, "negative_margin")
        return Prediction(label, module, votes, nearest, res.routes, res.matches, "known")

    def classify_batch(self, queries: np.ndarray, k: int = 5, device: str | None = None):
        """Batched classify on GPU (or CPU) tensors; same decisions as classify().

        Returns a BatchPrediction with per-query labels, modules, votes and reasons.
        """
        from mosaic.engine import BatchEngine  # torch is an optional dependency

        if getattr(self, "_engine", None) is None or (device and self._engine.device.type != device):
            self._engine = BatchEngine(self.registry, device)
        return self._engine.classify(
            queries,
            top_k=self.router.top_k,
            route_threshold=self.router.threshold,
            match_threshold=self.matcher.min_score,
            k=k,
            leaf_gate=self.leaf_gate,
        )

    def recognize(self, image: Any, k: int = 5) -> list[tuple[Region, Prediction]]:
        """Find every object in the image (class-agnostic), then classify each one."""
        return [(r, self.classify(r.embedding, k)) for r in self.perceiver.perceive(image)]

    @staticmethod
    def main_label(objects: list[tuple[Region, Prediction]]) -> str | None:
        """Image-level answer: the known label covering the most image area (None if no object is known).

        Area weighting keeps small parts (a wheel that closely matches a stored wheel)
        from outvoting the object that fills the picture.
        """
        area: dict[str, float] = {}
        for region, pred in objects:
            if pred.known:
                area[pred.label] = area.get(pred.label, 0.0) + region.area
        return max(area, key=area.get) if area else None

    def identify(self, image: Any, k: int = 5) -> list[Result]:
        """Per-region: route, then retrieve the closest known entries."""
        return [self.query_embedding(r.embedding, k, r.box) for r in self.perceiver.perceive(image)]

    def describe(self, image: Any) -> list[Result]:
        """Per-region: best answer from each routed module (composition)."""
        results = []
        for region in self.perceiver.perceive(image):
            routes = self.router.route(region.embedding)
            results.append(Result(region.box, routes, self.matcher.compose(region.embedding, routes)))
        return results

    def search_text(self, text: str, k: int = 5) -> Result:
        """Text -> matching images. The match gate is calibrated for image queries
        (text-image similarity is much lower), so it is not applied here."""
        return self.query_embedding(self.encoder.embed_texts([text])[0], k, kind="image", min_score=-1.0)
