"""Synthetic encoder + data: concepts are random unit vectors in D dims."""

import numpy as np
import pytest

from mosaic import KnowledgeModule, Mosaic, normalize

DIM = 64


class FakeImage:
    def __init__(self, vec):
        self.vec = vec
        self.size = (32, 32)

    def crop(self, box):
        return self


class FakeEncoder:
    encoder_id = "fake/v1"
    dim = DIM

    def __init__(self, concepts):
        self.concepts = concepts

    def embed_images(self, images):
        return normalize(np.stack([im.vec for im in images]))

    def embed_texts(self, texts):
        return normalize(np.stack([self.concepts[t] for t in texts]))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def concepts(rng):
    """Two domains in well-separated subspaces so routing is learnable."""
    def domain(offset, labels):
        out = {}
        for i, label in enumerate(labels):
            v = np.zeros(DIM)
            v[offset : offset + 16] = rng.normal(size=16) + 3.0  # shared domain direction
            v[offset + 16 + i] = 4.0                               # label-specific direction
            out[label] = normalize(v)
        return out

    return {
        **domain(0, ["sparrow", "robin", "crow"]),
        **domain(32, ["bolt", "nut", "washer"]),
    }


def make_module(name, labels, concepts, rng, per_label=10, encoder_id="fake/v1"):
    embs, entries = [], []
    for label in labels:
        for _ in range(per_label):
            embs.append(concepts[label] + rng.normal(scale=0.05, size=DIM))
            entries.append({"label": label, "kind": "image"})
    return KnowledgeModule.build(name, encoder_id, np.stack(embs), entries, n_descriptors=3)


@pytest.fixture
def birds(concepts, rng):
    return make_module("birds", ["sparrow", "robin", "crow"], concepts, rng)


@pytest.fixture
def hardware(concepts, rng):
    return make_module("hardware", ["bolt", "nut", "washer"], concepts, rng)


@pytest.fixture
def system(concepts):
    return Mosaic(FakeEncoder(concepts), top_k_modules=1, route_threshold=0.5)
