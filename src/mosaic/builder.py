"""Building a module from a set of example images of one concept.

Each example image may contain several detected objects (a cat *and* a teddy
bear). The labelled object is the one the images have in common. Each
candidate gets a recurrence score, its mean best-match similarity to the
*other* images. Starting from the most recurrent candidate per image, we then
repeatedly pick, per image, the candidate closest to the set's centroid. Only
the module's own images are used, with no class names or text.

Parts recur too: every car photo has wheels, and a wheel recurs more reliably
than a whole car, whose shape varies. `area_weight` adds sqrt(box area / image
area) to every score, so the choice leans toward the whole object rather than its parts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mosaic.encoder import normalize


def recurrence(candidates: Sequence[np.ndarray], max_reference: int = 500, seed: int = 0) -> list[np.ndarray]:
    """Per candidate: mean over other images (up to `max_reference`) of its best similarity to that image."""
    rng = np.random.default_rng(seed)
    ref = rng.permutation(len(candidates))[:max_reference]
    ref_embs = normalize(np.concatenate([candidates[j] for j in ref]))
    ref_owner = np.repeat(np.arange(len(ref)), [len(candidates[j]) for j in ref])
    starts = np.flatnonzero(np.r_[True, ref_owner[1:] != ref_owner[:-1]])
    position = {int(j): i for i, j in enumerate(ref)}
    scores = []
    for i, c in enumerate(candidates):
        best = np.maximum.reduceat(normalize(c) @ ref_embs.T, starts, axis=1)  # (n_i, n_ref)
        if i in position:
            best[:, position[i]] = np.nan  # an image does not vote for itself
        scores.append(np.nanmean(best, axis=1))
    return scores


def consensus_select(
    candidates: Sequence[np.ndarray],
    objectness: Sequence[np.ndarray] | None = None,
    areas: Sequence[np.ndarray] | None = None,
    area_weight: float = 0.5,
    iters: int = 10,
) -> list[int]:
    """For each image, the index of its candidate object shared across the set.

    candidates[i] is an (n_i, D) array of object embeddings for image i;
    areas[i] their box areas as a fraction of the image. `objectness` only
    breaks exact ties.
    """
    size = [area_weight * np.sqrt(a) for a in areas] if areas is not None else [0.0] * len(candidates)
    tie = [1e-6 * np.asarray(o) for o in objectness] if objectness is not None else [0.0] * len(candidates)
    if len(candidates) < 2:
        return [int(np.argmax(np.zeros(len(c)) + s + t)) for c, s, t in zip(candidates, size, tie)]
    choice = [int(np.argmax(r + s + t)) for r, s, t in zip(recurrence(candidates), size, tie)]
    for _ in range(iters):
        centroid = normalize(np.mean([c[j] for c, j in zip(candidates, choice)], axis=0))
        new = [int(np.argmax(normalize(c) @ centroid + s + t)) for c, s, t in zip(candidates, size, tie)]
        if new == choice:
            break
        choice = new
    return choice


def nearest_to(candidates: Sequence[np.ndarray], reference: np.ndarray) -> list[int]:
    """For each image, the candidate closest to any row of `reference` (e.g. a module's stored objects)."""
    ref = normalize(reference)
    return [int(np.argmax((normalize(c) @ ref.T).max(axis=1))) for c in candidates]
