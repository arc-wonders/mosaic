import numpy as np
import pytest

from conftest import DIM, FakeImage, make_module
from mosaic import EncoderMismatchError, KnowledgeModule, normalize


def test_routes_to_correct_module_and_matches(system, birds, hardware, concepts):
    system.attach(birds)
    system.attach(hardware)
    [result] = system.identify(FakeImage(concepts["robin"]))
    assert result.routes[0].module == "birds"
    assert result.matches[0].label == "robin"

    text = system.search_text("washer")
    assert text.routes[0].module == "hardware"
    assert text.matches[0].label == "washer"


def test_abstains_on_unknown(system, birds, hardware, rng):
    system.attach(birds)
    system.attach(hardware)
    unknown = normalize(rng.normal(size=DIM))
    assert system.query_embedding(unknown).abstained


def test_attach_and_detach_without_touching_encoder(system, birds, hardware, concepts):
    system.attach(birds)
    assert system.search_text("bolt").abstained  # no hardware knowledge yet

    system.attach(hardware)
    assert system.search_text("bolt").matches[0].label == "bolt"

    system.detach("hardware")
    assert system.search_text("bolt").abstained


def test_rejects_module_from_other_encoder(system, concepts, rng):
    stale = make_module("old", ["crow"], concepts, rng, encoder_id="fake/v0")
    with pytest.raises(EncoderMismatchError):
        system.attach(stale)


def test_compose_takes_best_from_each_module(concepts, birds, hardware):
    from conftest import FakeEncoder
    from mosaic import Mosaic

    system = Mosaic(FakeEncoder(concepts), top_k_modules=2, route_threshold=0.1)
    system.attach(birds)
    system.attach(hardware)
    query = normalize(concepts["crow"] + concepts["nut"])  # a region mixing both
    [result] = system.describe(FakeImage(query))
    assert {m.label for m in result.matches} == {"crow", "nut"}


def test_save_load_roundtrip(birds, tmp_path):
    path = birds.save(tmp_path)
    loaded = KnowledgeModule.load(path)
    assert loaded.name == birds.name and loaded.encoder_id == birds.encoder_id
    assert loaded.entries == birds.entries
    np.testing.assert_allclose(loaded.embeddings, birds.embeddings)
    np.testing.assert_allclose(loaded.descriptors, birds.descriptors)


def test_registry_load_dir(system, birds, hardware, tmp_path):
    birds.save(tmp_path)
    hardware.save(tmp_path)
    assert system.registry.load_dir(tmp_path) == ["birds", "hardware"]
    assert len(system.registry) == 2


def test_match_gate_abstains_when_nothing_is_close(concepts, birds):
    from conftest import FakeEncoder
    from mosaic import Mosaic

    # routes to birds (shares the domain direction) but matches no stored bird closely
    query = np.zeros(DIM)
    query[:16] = 1.0
    query = normalize(query)
    loose = Mosaic(FakeEncoder(concepts), top_k_modules=1, route_threshold=0.1)
    strict = Mosaic(FakeEncoder(concepts), top_k_modules=1, route_threshold=0.1, match_threshold=0.99)
    for s in (loose, strict):
        s.attach(birds)
    assert not loose.query_embedding(query).abstained
    result = strict.query_embedding(query)
    assert result.routes and result.abstained
