"""Router: coarse-to-fine MoE-style gating over the module tree, with abstention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mosaic.encoder import normalize
from mosaic.registry import Registry


@dataclass
class Route:
    module: str
    score: float


class Router:
    """Descends the registry tree level by level, keeping the top-k nodes at each level.

    `top_k` is an int (same k at every level) or a list, one k per level (the last
    value repeats for deeper levels), e.g. [1, 3]: best domain, then its 3 best classes.
    `threshold` gates only the first level: if no top-level node clears it, abstain.
    With `hierarchical=False` all leaves are scored at once (flat routing baseline).
    """

    def __init__(
        self,
        registry: Registry,
        top_k: int | list[int] = 2,
        threshold: float = 0.2,
        hierarchical: bool = True,
    ):
        self.registry = registry
        self.top_k = top_k
        self.threshold = threshold
        self.hierarchical = hierarchical

    def _k(self, level: int) -> int:
        if isinstance(self.top_k, int):
            return self.top_k
        return self.top_k[min(level, len(self.top_k) - 1)]

    def scores(self, query: np.ndarray, names: list[str] | None = None) -> list[Route]:
        """Score nodes (default: top level) by max cosine to their descriptors, best first."""
        q = normalize(query)
        names = self.registry.children("") if names is None else names
        routes = [Route(n, float(np.max(self.registry.descriptors(n) @ q))) for n in names]
        return sorted(routes, key=lambda r: r.score, reverse=True)

    def trace(self, query: np.ndarray) -> tuple[list[Route], list[list[Route]]]:
        """Returns (selected leaf routes, all scored candidates per level)."""
        if not self.hierarchical:
            scored = self.scores(query, self.registry.leaves())
            k = self._k(0) if isinstance(self.top_k, int) else self.top_k[-1]
            return [r for r in scored[:k] if r.score >= self.threshold], [scored]

        leaves, levels, frontier, level = [], [], [""], 0
        while frontier:
            candidates = [c for g in frontier for c in self.registry.children(g)]
            scored = self.scores(query, candidates)
            levels.append(scored)
            keep = scored[: self._k(level)]
            if level == 0:
                keep = [r for r in keep if r.score >= self.threshold]
            leaves += [r for r in keep if not self.registry.is_group(r.module)]
            frontier = [r.module for r in keep if self.registry.is_group(r.module)]
            level += 1
        return sorted(leaves, key=lambda r: r.score, reverse=True), levels

    def route(self, query: np.ndarray) -> list[Route]:
        """Selected leaf modules, best first; empty list means abstain."""
        return self.trace(query)[0]
