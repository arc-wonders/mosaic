import numpy as np
import pytest

from conftest import FakeEncoder, FakeImage, make_module
from mosaic import Mosaic

BIRDS = ["sparrow", "robin", "crow"]
HARDWARE = ["bolt", "nut", "washer"]


@pytest.fixture
def leaves(concepts, rng):
    return {f"{d}/{c}": make_module(f"{d}/{c}", [c], concepts, rng)
            for d, cs in [("birds", BIRDS), ("hardware", HARDWARE)] for c in cs}


@pytest.fixture
def tree(concepts, leaves):
    system = Mosaic(FakeEncoder(concepts), top_k_modules=[1, 2], route_threshold=0.5, match_threshold=0.9)
    for m in leaves.values():
        system.attach(m)
    return system


def test_tree_structure(tree):
    reg = tree.registry
    assert reg.children() == ["birds", "hardware"]
    assert reg.children("birds") == ["birds/crow", "birds/robin", "birds/sparrow"]
    assert reg.is_group("birds") and not reg.is_group("birds/crow")
    assert "birds" in reg and "birds/crow" in reg and len(reg) == 6


def test_routes_coarse_to_fine(tree, concepts):
    routes, levels = tree.router.trace(concepts["robin"])
    assert levels[0][0].module == "birds"
    assert {r.module for r in levels[1]} == {"birds/crow", "birds/robin", "birds/sparrow"}  # only birds' children
    assert routes[0].module == "birds/robin" and len(routes) == 2
    [result] = tree.identify(FakeImage(concepts["robin"]))
    assert result.matches[0].label == "robin"


def test_plug_out_and_in_a_leaf(tree, concepts, leaves):
    before = tree.registry.descriptors("birds").copy()
    assert tree.detach("birds/robin") == ["birds/robin"]
    assert "birds/robin" not in {r.module for r in tree.router.route(concepts["robin"])}
    assert not np.array_equal(before, tree.registry.descriptors("birds"))  # group summary refreshed
    assert tree.query_embedding(concepts["robin"]).abstained  # match gate: no close robin left

    tree.attach(leaves["birds/robin"])
    np.testing.assert_allclose(before, tree.registry.descriptors("birds"))
    assert tree.query_embedding(concepts["robin"]).matches[0].label == "robin"


def test_plug_out_a_whole_group(tree, concepts):
    assert tree.detach("birds") == ["birds/crow", "birds/robin", "birds/sparrow"]
    assert tree.registry.children() == ["hardware"]
    assert tree.query_embedding(concepts["crow"]).abstained
    assert tree.query_embedding(concepts["nut"]).matches[0].label == "nut"


def test_rejects_ambiguous_nodes(tree, concepts, rng):
    with pytest.raises(ValueError):
        tree.attach(make_module("birds", ["crow"], concepts, rng))  # "birds" is a group
    with pytest.raises(ValueError):
        tree.attach(make_module("birds/crow/young", ["crow"], concepts, rng))  # crow is a leaf


def test_flat_routing_baseline(concepts, leaves):
    flat = Mosaic(FakeEncoder(concepts), top_k_modules=2, route_threshold=0.5, hierarchical=False)
    for m in leaves.values():
        flat.attach(m)
    routes, levels = flat.router.trace(concepts["washer"])
    assert len(levels) == 1 and len(levels[0]) == 6
    assert routes[0].module == "hardware/washer"


def test_nested_save_and_load(leaves, concepts, tmp_path):
    for m in leaves.values():
        m.save(tmp_path)
    assert (tmp_path / "birds" / "robin" / "manifest.json").exists()
    system = Mosaic(FakeEncoder(concepts))
    assert len(system.registry.load_dir(tmp_path)) == 6
    assert system.registry.children() == ["birds", "hardware"]


def test_leaf_self_calibrates_and_gates(tree, concepts, leaves, tmp_path):
    robin = leaves["birds/robin"]
    assert robin.accept_threshold is not None and 0 < robin.accept_threshold < 1
    assert tree.classify(concepts["robin"]).label == "robin"

    tree.matcher.min_score = 0.0  # disable the global gate: the leaf gate alone must reject
    tree.detach("birds/robin")
    pred = tree.classify(concepts["robin"])
    assert not pred.known and pred.reason == "leaf_threshold"

    robin.save(tmp_path)
    from mosaic import KnowledgeModule
    assert KnowledgeModule.load(tmp_path / "birds" / "robin").accept_threshold == robin.accept_threshold


def test_negatives_reject_lookalikes_and_never_route(concepts, rng):
    from conftest import FakeEncoder
    from mosaic import KnowledgeModule, normalize

    # "robin" leaf that also stores counter-examples: things near robin that are not robins
    pos = [concepts["robin"] + rng.normal(scale=0.05, size=len(concepts["robin"])) for _ in range(20)]
    lookalike = normalize(concepts["robin"] + 1.5 * concepts["sparrow"])  # sparrow-ish, but only robin is plugged in
    neg = [lookalike + rng.normal(scale=0.02, size=len(lookalike)) for _ in range(20)]
    entries = [{"label": "robin", "kind": "image"}] * 20 + [{"label": "robin", "kind": "negative"}] * 20
    leaf = KnowledgeModule.build("birds/robin", "fake/v1", np.stack(pos + neg), entries, n_descriptors=2)
    assert len(leaf.descriptors) == 2  # only positives describe the leaf

    system = Mosaic(FakeEncoder(concepts), top_k_modules=[1, 1], route_threshold=0.1)
    system.attach(leaf)
    assert system.classify(lookalike).known  # without a margin, the look-alike is accepted
    leaf.reject_margin = 0.0
    system.registry.version += 1
    pred = system.classify(lookalike)
    assert not pred.known and pred.reason == "negative_margin"
    assert system.classify(concepts["robin"]).label == "robin"
    batch = system.classify_batch(np.stack([lookalike, concepts["robin"]]), device="cpu")
    assert batch.reasons == ["negative_margin", "known"]
    # the leaf's own accepts() applies the same two gates (used by geometry re-ranking and candidate lists)
    assert not leaf.accepts(lookalike) and leaf.accepts(concepts["robin"])
