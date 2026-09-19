"""Matcher: similarity search inside routed modules, and cross-module composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mosaic.registry import Registry
from mosaic.router import Route


@dataclass
class Match:
    module: str
    label: str
    score: float
    entry: dict[str, Any]


class Matcher:
    def __init__(self, registry: Registry, min_score: float = 0.0):
        self.registry = registry
        self.min_score = min_score

    def match(
        self,
        query: np.ndarray,
        routes: list[Route],
        k: int = 5,
        kind: str | None = None,
        min_score: float | None = None,
    ) -> list[Match]:
        """Top-k entries across all routed modules scoring >= min_score, best first."""
        min_score = self.min_score if min_score is None else min_score
        matches = []
        for route in routes:
            module = self.registry.get(route.module)
            for idx, score in module.search(query, k, kind):
                if score >= min_score:
                    entry = module.entries[idx]
                    matches.append(Match(module.name, entry["label"], score, entry))
        return sorted(matches, key=lambda m: m.score, reverse=True)[:k]

    def compose(self, query: np.ndarray, routes: list[Route]) -> list[Match]:
        """v0 composition: the single best match from each routed module.

        E.g. an 'objects' module says `bolt` and a 'defects' module says
        `scratch` -> the region is a scratched bolt.
        """
        best = []
        for route in routes:
            top = self.match(query, [route], k=1)
            if top:
                best.append(top[0])
        return best
