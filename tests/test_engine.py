import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import DIM, FakeEncoder, make_module  # noqa: E402
from mosaic import Mosaic, normalize  # noqa: E402

DOMAINS = {"birds": ["sparrow", "robin", "crow"], "hardware": ["bolt", "nut", "washer"]}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def queries(concepts, rng):
    known = [concepts[c] + rng.normal(scale=0.08, size=DIM) for c in concepts for _ in range(20)]
    unknown = [rng.normal(size=DIM) for _ in range(20)]
    return normalize(np.stack(known + unknown))


def build(concepts, rng, hierarchical=True, **kw):
    system = Mosaic(FakeEncoder(concepts), hierarchical=hierarchical, **kw)
    for d, cs in DOMAINS.items():
        for c in cs:
            system.attach(make_module(f"{d}/{c}" if hierarchical else c, [c], concepts, rng))
    return system


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hierarchical", [True, False])
def test_batch_matches_single_query(concepts, rng, queries, device, hierarchical):
    system = build(concepts, rng, hierarchical, top_k_modules=[1, 2], route_threshold=0.5, match_threshold=0.1)
    batch = system.classify_batch(queries, device=device)
    single = [system.classify(q) for q in queries]
    assert batch.labels == [p.label for p in single]
    assert batch.reasons == [p.reason for p in single]
    assert "known" in batch.reasons and "router" in batch.reasons


@pytest.mark.parametrize("device", DEVICES)
def test_engine_follows_plug_out(concepts, rng, device):
    queries = np.stack(list(concepts.values()))  # clean prototypes: typical members of their leaf
    system = build(concepts, rng, top_k_modules=[1, 2], route_threshold=0.5)
    assert "robin" in system.classify_batch(queries, device=device).labels
    system.detach("birds/robin")
    after = system.classify_batch(queries, device=device)
    assert "robin" not in after.labels
    assert after.labels == [system.classify(q).label for q in queries]
