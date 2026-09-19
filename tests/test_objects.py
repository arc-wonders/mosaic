import numpy as np

from conftest import DIM, FakeEncoder, make_module
from mosaic import Mosaic, normalize
from mosaic.builder import consensus_select, nearest_to
from mosaic.perceiver import Perceiver, crop


class Canvas:
    """A fake image made of labelled rectangles; crops keep the objects they overlap most."""

    def __init__(self, objects, size=(100, 100)):
        self.objects, self.size = objects, size  # [(box, vec)]

    def crop(self, box):
        x0, y0, x1, y1 = box
        inside = [(b, v) for b, v in self.objects if b[0] >= x0 - 1 and b[1] >= y0 - 1 and b[2] <= x1 + 1 and b[3] <= y1 + 1]
        return Canvas(inside or self.objects, (x1 - x0, y1 - y0))

    @property
    def vec(self):
        return normalize(np.sum([v for _, v in self.objects], axis=0))


class Det:
    def __init__(self, box, score):
        self.box, self.score = box, score


class FakeDetector:
    def detect(self, images):
        return [[Det(b, 0.9 - 0.1 * i) for i, (b, _) in enumerate(im.objects)] for im in images]


def test_crop_adds_context_and_stays_inside():
    img = Canvas([((10, 10, 30, 30), np.ones(DIM))], size=(40, 40))
    c = crop(img, (10, 10, 30, 30), pad=0.1)
    assert c.size == (24, 24)
    c = crop(img, (0, 0, 40, 40), pad=0.5)
    assert c.size == (40, 40)


def test_perceiver_embeds_each_object_separately(concepts):
    img = Canvas([((0, 0, 40, 40), concepts["robin"]), ((50, 50, 90, 90), concepts["bolt"])])
    regions = Perceiver(FakeEncoder(concepts), FakeDetector()).perceive(img)
    assert [r.box for r in regions] == [(0, 0, 40, 40), (50, 50, 90, 90)]
    assert regions[0].embedding @ concepts["robin"] > 0.99
    assert regions[1].embedding @ concepts["bolt"] > 0.99


def test_perceiver_falls_back_to_whole_image(concepts):
    class Nothing:
        def detect(self, images):
            return [[] for _ in images]

    [region] = Perceiver(FakeEncoder(concepts), Nothing()).perceive(Canvas([((0, 0, 9, 9), concepts["nut"])]))
    assert region.fallback and region.box == (0, 0, 100, 100)


def test_recognize_labels_every_object(concepts, rng):
    system = Mosaic(FakeEncoder(concepts), detector=FakeDetector(), top_k_modules=[1, 2], route_threshold=0.5)
    for d, cs in {"birds": ["robin", "crow"], "hardware": ["bolt", "nut"]}.items():
        for c in cs:
            system.attach(make_module(f"{d}/{c}", [c], concepts, rng))
    img = Canvas([((0, 0, 40, 40), concepts["crow"]), ((50, 50, 90, 90), concepts["nut"])])
    assert [p.label for _, p in system.recognize(img)] == ["crow", "nut"]


def test_consensus_finds_the_shared_object(concepts, rng):
    """Each 'cat' image also contains a random distractor that is *more* object-like."""
    images, objectness = [], []
    for _ in range(30):
        distractor = normalize(rng.normal(size=DIM))
        cat = normalize(concepts["robin"] + rng.normal(scale=0.05, size=DIM))
        images.append(np.stack([distractor, cat]))
        objectness.append(np.array([0.9, 0.6]))  # distractor wins on objectness alone
    assert consensus_select(images, objectness) == [1] * 30
    assert nearest_to(images, concepts["robin"][None]) == [1] * 30


class FakeSegmenter:
    """Masks each box's left half, so the masked area is half the box."""

    def segment(self, images, boxes):
        out = []
        for im, bxs in zip(images, boxes):
            ms = []
            for x0, y0, x1, y1 in bxs:
                m = np.zeros((im.size[1], im.size[0]), bool)
                m[y0:y1, x0 : (x0 + x1) // 2] = True
                ms.append(m)
            out.append(ms)
        return out


def test_perceiver_uses_mask_area_and_cutout():
    from PIL import Image

    class OneBox:
        def detect(self, images):
            return [[Det((0, 0, 40, 40), 0.9)] for _ in images]

    class Recorder:
        encoder_id, dim = "rec", DIM
        seen = []

        def embed_images(self, images):
            self.seen.extend(images)
            return normalize(np.ones((len(images), DIM)))

    img = Image.new("RGB", (100, 100), (255, 0, 0))
    enc = Recorder()
    boxed = Perceiver(enc, OneBox()).perceive(img)[0]
    masked = Perceiver(enc, OneBox(), FakeSegmenter()).perceive(img)[0]
    assert boxed.mask is None and abs(boxed.area - 0.16) < 1e-9
    assert masked.mask is not None and abs(masked.area - 0.08) < 1e-9  # left half of the box
    box_crop, cut = enc.seen
    assert box_crop.getpixel((35, 20)) == (255, 0, 0)       # plain crop keeps every pixel
    assert cut.getpixel((42, 20)) != (255, 0, 0)            # right of the mask (plus rim) is blanked


def test_cutout_keeps_only_masked_pixels():
    from PIL import Image
    from mosaic.perceiver import NEUTRAL, cutout

    img = Image.new("RGB", (40, 20), (255, 0, 0))
    mask = np.zeros((20, 40), bool)
    mask[:, :10] = True
    out = cutout(img, (0, 0, 40, 20), mask, pad=0.0, dilate=1, keep=0.0)
    assert out.size == (40, 20)
    assert out.getpixel((5, 5)) == (255, 0, 0)
    assert out.getpixel((30, 5)) == NEUTRAL
    soft = cutout(img, (0, 0, 40, 20), mask, pad=0.0, dilate=1, keep=0.5)
    assert soft.getpixel((5, 5)) == (255, 0, 0)
    assert soft.getpixel((30, 5)) == tuple((255 * (i == 0) + n) // 2 for i, n in enumerate(NEUTRAL))
