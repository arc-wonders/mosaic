"""VisionSession logic with fake models: coordinates, zoom, stable ids, parts, relations, search."""

import numpy as np
import pytest
from PIL import Image

from conftest import DIM, make_module
from mosaic import normalize
from mosaic.capability import VisionSession

# Scene: a "person" rectangle holding a "phone" rectangle, drawn in distinct colours.
COLOURS = {"person": (200, 30, 30), "hand": (30, 200, 30), "phone": (30, 30, 200)}
SCENE = {"person": (100, 50, 300, 450), "hand": (240, 250, 290, 300), "phone": (270, 230, 330, 280)}


def scene_image():
    img = Image.new("RGB", (400, 500), (255, 255, 255))
    for name in ["person", "hand", "phone"]:
        img.paste(COLOURS[name], SCENE[name])
    return img


class ColourEncoder:
    """Embeds a crop by its dominant non-white, non-neutral colour."""
    encoder_id, dim = "colour", DIM

    def __init__(self, concepts):
        self.vec = {c: concepts[k] for c, k in [((200, 30, 30), "robin"), ((30, 200, 30), "crow"), ((30, 30, 200), "bolt")]}

    def embed_images(self, images):
        out = []
        for im in images:
            px = np.asarray(im.convert("RGB")).reshape(-1, 3)
            counts = {c: int((np.abs(px - c).sum(1) < 30).sum()) for c in self.vec}
            out.append(self.vec[max(counts, key=counts.get)])
        return normalize(np.stack(out))

    def embed_texts(self, texts):
        return normalize(np.ones((len(texts), DIM)))


class Det:
    def __init__(self, box, score):
        self.box, self.score = box, score


class ColourDetector:
    """Finds each coloured rectangle fully inside the view, in view pixels."""

    def detect(self, images):
        out = []
        for im in images:
            a = np.asarray(im.convert("RGB")).astype(int)
            dets = []
            for c in COLOURS.values():
                ys, xs = np.nonzero(np.abs(a - c).sum(2) < 30)
                if len(xs) > 50:
                    dets.append(Det((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), 0.9))
            out.append(dets)
        return out


class BoxMaskSegmenter:
    def segment(self, images, boxes):
        out = []
        for im, bxs in zip(images, boxes):
            ms = []
            for x0, y0, x1, y1 in bxs:
                m = np.zeros((im.size[1], im.size[0]), bool)
                m[y0:y1, x0:x1] = True
                ms.append(m)
            out.append(ms)
        return out


@pytest.fixture
def session(concepts, rng, tmp_path):
    for name, label in [("people/person", "robin"), ("people/hand", "crow"), ("handheld/phone", "bolt")]:
        m = make_module(name, [label], concepts, rng, encoder_id="colour")
        m.part_of = "people/person" if name == "people/hand" else None
        m.save(tmp_path)
    s = VisionSession(tmp_path, route_threshold=0.5, zoom_side=800, encoder=ColourEncoder(concepts),
                      detector=ColourDetector(), segmenter=BoxMaskSegmenter())
    s.images_path = tmp_path / "scene.png"
    scene_image().save(s.images_path)
    return s


def close(box, expected, tol=3):
    return all(abs(a - b) <= tol for a, b in zip(box, expected))


def test_find_objects_in_image_coordinates(session):
    img = session.load_image(str(session.images_path))
    assert img == {"image_id": "img_1", "width": 400, "height": 500}
    objs = session.find_objects("img_1")["objects"]
    assert len(objs) == 3
    boxes = sorted(o["box"] for o in objs)
    assert any(close(b, SCENE["person"]) for b in boxes)


def test_zoomed_region_maps_back_and_keeps_ids(session):
    session.load_image(str(session.images_path))
    first = {tuple(o["box"]): o["object_id"] for o in session.find_objects("img_1")["objects"]}
    zoom = session.find_objects("img_1", region=[200, 150, 360, 320])  # x and y offsets differ
    assert zoom["zoom"] > 1
    phone = next(o for o in zoom["objects"] if close(o["box"], SCENE["phone"]))
    assert phone["object_id"] in first.values()  # same object, same id
    for o in zoom["objects"]:  # box and mask agree after mapping back from the zoomed view
        assert close(o["box"], o["mask_box"], tol=2)


def test_identify_and_part_of(session):
    session.load_image(str(session.images_path))
    objs = session.find_objects("img_1")["objects"]
    ids = {session.identify(o["object_id"])["module"]: o["object_id"] for o in objs}
    assert set(ids) == {"people/person", "people/hand", "handheld/phone"}
    hand = session.identify(ids["people/hand"])
    assert hand["known"] and hand["part_of"] == "people/person"
    assert hand["candidates"][0]["module"] == "people/hand"


def test_identify_arbitrary_box(session):
    session.load_image(str(session.images_path))
    r = session.identify(image_id="img_1", box=list(SCENE["phone"]))
    assert r["module"] == "handheld/phone"


def test_find_parts_and_relate(session):
    session.load_image(str(session.images_path))
    person = next(o for o in session.find_objects("img_1")["objects"] if close(o["box"], SCENE["person"]))
    parts = session.find_parts(person["object_id"])["parts"]
    modules = {p["module"] for p in parts}
    assert "people/hand" in modules and "people/person" not in modules
    hand = next(p for p in parts if p["module"] == "people/hand")
    phone = next(p for p in parts if p["module"] == "handheld/phone")
    rel = session.relate(hand["object_id"], phone["object_id"])
    assert rel["touching"] and rel["overlap_fraction_of_a"] > 0
    assert "right of" in rel["b_is"] or "above" in rel["b_is"] or rel["b_is"].startswith("at about")
    far = session.relate(person["object_id"], phone["object_id"])
    assert far["overlap_fraction_of_b"] > 0.3


def test_search_by_module_within_region(session):
    session.load_image(str(session.images_path))
    hits = session.search("img_1", "handheld", region=[200, 200, 350, 320])["results"]
    assert hits[0]["module"] == "handheld/phone" and hits[0]["match"]
    assert not any(h["match"] and h["identified_as"] == "people/person" for h in hits)


def test_list_modules_and_render(session):
    tree = session.list_modules()
    groups = {c["group"] for c in tree["children"]}
    assert groups == {"people", "handheld"}
    hand = next(c for g in tree["children"] if g["group"] == "people" for c in g["children"] if c["module"] == "people/hand")
    assert hand["part_of"] == "people/person"
    session.load_image(str(session.images_path))
    session.find_objects("img_1")
    assert session.render("img_1")[:8] == b"\x89PNG\r\n\x1a\n"
