"""The web visualizer's HTTP layer: every route against the same fakes test_capability uses."""

import pytest

from test_capability import session  # noqa: F401  (fixture: VisionSession with fake encoder/detector/segmenter)
from test_geometry import FakePose

import mosaic.webui as webui


@pytest.fixture
def client(session, monkeypatch):
    session._pose_estimator = FakePose()  # a real ViTPose would see only flat colour rectangles, not anatomy
    monkeypatch.setattr(webui, "SESSION", session)
    return webui.app.test_client()


@pytest.fixture
def image_id(session):
    return session.load_image(str(session.images_path))["image_id"]


def test_index_page(client):
    assert client.get("/").status_code == 200


def test_samples_lists_without_models(client):
    assert client.get("/api/samples").status_code == 200


def test_find_objects_identifies_every_object(client, image_id):
    r = client.get(f"/api/find_objects?image_id={image_id}")
    assert r.status_code == 200
    objects = r.get_json()["objects"]
    assert {o["identify"]["module"] for o in objects} == {"people/person", "people/hand", "handheld/phone"}
    assert all(o["identify"]["known"] for o in objects)


def test_image_and_object_stage_images_are_served(client, image_id):
    assert client.get(f"/api/image/{image_id}").content_type == "image/jpeg"
    oid = client.get(f"/api/find_objects?image_id={image_id}").get_json()["objects"][0]["object_id"]
    assert client.get(f"/api/object/{oid}/mask.png").content_type == "image/png"
    assert client.get(f"/api/object/{oid}/cutout.png").content_type == "image/jpeg"


def test_explain_exposes_routing_and_gate(client, image_id):
    oid = client.get(f"/api/find_objects?image_id={image_id}").get_json()["objects"][0]["object_id"]
    ex = client.get(f"/api/object/{oid}/explain").get_json()
    assert ex["routing"]["levels"] and ex["gate"]["module"] == ex["module"]


def test_load_rejects_paths_outside_the_repo(client):
    r = client.post("/api/load", json={"path": "/etc/passwd"})
    assert r.status_code == 403


def test_unknown_object_is_a_clean_error_not_a_crash(client):
    r = client.get("/api/object/obj_999/explain")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_measure_on_a_person(client, image_id):
    client.get(f"/api/find_objects?image_id={image_id}")
    pid = client.get(f"/api/pose?image_id={image_id}").get_json()["people"][0]["person_id"]
    r = client.post("/api/measure", json={"person_id": pid, "expression": "visible(left_shoulder)"})
    assert r.get_json()["value"] in (True, False)
