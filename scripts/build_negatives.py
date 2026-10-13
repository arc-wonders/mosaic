"""Teach the people leaves what they are *not*: stored counter-examples and a calibrated reject margin.

Every people leaf's own gate (accept_threshold) is calibrated on human parts only, and SigLIP puts an orange
close enough to a nose, or a dog close enough to a person, to pass it (the Ollama benchmark accepted 13 of 21
non-human photos). This adds "negative" entries to each leaf; a leaf then accepts an object only if its nearest
stored example beats its nearest stored counter-example by the leaf's reject_margin (pipeline.Mosaic.classify).

Counter-examples, from the build split only:
  oi    non-human objects annotated in the Open Images people photos (cars, food, furniture, animals, tools ...;
        clothing, accessories, pictures and anything overlapping a person are left out), soft-mask cut-outs of
        the annotated box, exactly as the human instances were cut out (cache_human_crops.py)
  raw   everything MOSAIC's own detector finds in photos without people (cats, dogs, elephants, vehicles,
        fruit), cut out the way a live session sees them. Detections SigLIP's text tower thinks look more like a
        person or a body part than like the photo's category are dropped (a rider on a scooter).
Held out entirely, to measure rejection of things no counter-example resembles: birds, zebras, pizza, lemons.

Each leaf keeps the counter-examples it could confuse (nearest-example similarity within 0.05 of its gate).
reject_margin per leaf: the --percentile-th percentile of (nearest positive - nearest negative) over the leaf's
calibration instances, so about that share of real parts is given up.

    python scripts/build_negatives.py [--percentile 3] [--per-class 60]

Writes data/cache/negatives_siglip-so400m.npz, updates modules_people/ in place (re-running replaces the
negatives), and results/negatives.json (test split: humans still accepted, non-humans rejected).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from build_people import split_of
from human_classes import LEAVES
from mosaic.module import KnowledgeModule
from mosaic.perceiver import cutout

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/cache/negatives_siglip-so400m.npz"
SEEN = ["animals/cat", "animals/dog", "animals/elephant", "vehicles/airplane", "vehicles/pickup_truck",
        "vehicles/school_bus", "vehicles/scooter", "vehicles/sports_car", "fruits/apple", "fruits/banana",
        "fruits/orange", "fruits/pineapple", "fruits/strawberry"]
HELD_OUT = ["birds/eagle", "birds/owl", "birds/parrot", "birds/flamingo", "birds/penguin", "animals/zebra",
            "ood/pizza", "ood/lemon"]
# Open Images labels that are people, worn by people, or pictures / figures of people
NOT_NEGATIVE = {
    "Human body", "Mammal", "Clothing", "Footwear", "Fashion accessory", "Dress", "Glasses", "Sports uniform",
    "Beard", "Jeans", "Suit", "Swimwear", "Trousers", "Hat", "Shorts", "Helmet", "Sunglasses", "Sandal", "Goggles",
    "Fedora", "Brassiere", "Jacket", "High heels", "Lipstick", "Tie", "Shirt", "Sun hat", "Roller skates",
    "Cowboy hat", "Boot", "Necklace", "Bicycle helmet", "Earring", "Glove", "Coat", "Personal flotation device",
    "Crown", "Sock", "Miniskirt", "Swim cap", "Football helmet", "Baseball glove", "Tiara", "Sombrero", "Scarf",
    "Watch", "Headphones", "Mule", "Doll", "Toy", "Sculpture", "Bronze sculpture", "Bust", "Skull", "Poster",
    "Picture frame", "Television", "Computer monitor", "Billboard", "Cosmetics", "Teddy bear",
}
HUMAN_TEXTS = ["a photo of a person", "a photo of a human face", "a photo of a human head", "a photo of human hair",
               "a photo of a human hand", "a photo of a human arm", "a photo of a human leg", "a photo of a human foot"]
MARGIN_BAND = 0.05  # a leaf keeps counter-examples whose similarity to it is within this of its gate


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# ---------------------------------------------------------------- collect and embed counter-examples

def collect(per_class: int, bench_images: set[str]) -> tuple[np.ndarray, list[dict]]:
    from mosaic.capability import VisionSession

    session = VisionSession(REPO / "modules_people", taught_dir=None)
    X, meta = [], []

    # oi: annotated non-human objects in the people photos (all splits; the split is kept per item)
    ann = json.loads((REPO / "data/human/annotations.json").read_text())
    ann.pop("_files", None)
    for n, (path, boxes) in enumerate(sorted(ann.items())):
        human = [b[:4] for b in boxes if b[4] in LEAVES]
        todo = [b for b in boxes if b[4].startswith("oi:") and b[4][3:] not in NOT_NEGATIVE
                and min(b[2] - b[0], b[3] - b[1]) >= 32 and all(iou(b[:4], h) < 0.5 for h in human)]
        if not todo:
            continue
        img = Image.open(REPO / path).convert("RGB")
        masks = session.segmenter.segment([img], [[tuple(b[:4]) for b in todo]])[0]
        crops = []
        for b, m in zip(todo, masks):
            c = cutout(img, tuple(b[:4]), m, keep=0.5)
            c.thumbnail((256, 256))
            crops.append(c)
        X.append(session.encoder.embed_images(crops))
        meta += [{"source": "oi", "image": path, "box": b[:4], "category": b[4][3:], "split": split_of(path),
                  "held_out": False} for b in todo]
        if n % 300 == 0:
            print(f"  oi {n}/{len(ann)}: {len(meta)} objects", flush=True)

    # raw: what the detector finds in photos without people
    T = session.encoder.embed_texts(HUMAN_TEXTS)
    dropped = Counter()
    for cat in SEEN + HELD_OUT:
        files = sorted((REPO / "data/raw" / cat).glob("*.jpg"))
        held = cat in HELD_OUT
        by_split = defaultdict(list)
        for f in files:
            rel = str(f.relative_to(REPO))
            if str(f) not in bench_images:
                by_split[split_of(rel)].append(f)
        # held-out categories are only ever tested; seen ones give build + calibration + test images
        take = {"test": by_split["test"][:per_class // 2]} if held else \
               {s: by_split[s][: per_class if s == "build" else per_class // 2] for s in ("build", "calib", "test")}
        t_cat = session.encoder.embed_texts([f"a photo of a {cat.rsplit('/', 1)[1].replace('_', ' ')}"])[0]
        for split, fs in take.items():
            for f in fs:
                iid = session.load_image(str(f))["image_id"]
                objs = session.find_objects(iid)["objects"]
                for o in objs:
                    e = session.objects[o["object_id"]].embedding
                    if (T @ e).max() > e @ t_cat:  # looks more like a person / body part than like the category
                        dropped[cat] += 1
                        continue
                    X.append(e[None])
                    meta.append({"source": "raw", "image": str(f.relative_to(REPO)), "box": o["box"],
                                 "category": cat, "split": split, "held_out": held})
                for oid in session.images.pop(iid).objects:  # keep the session small (masks are full-size)
                    session.objects.pop(oid)
        print(f"  raw {cat}: {sum(m['category'] == cat for m in meta)} objects "
              f"({dropped[cat]} dropped as person-like)", flush=True)
    return np.concatenate(X).astype(np.float32), meta


# ---------------------------------------------------------------- attach, calibrate, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=3.0,
                    help="share of calibration human instances each leaf may give up (percent)")
    ap.add_argument("--per-class", type=int, default=60, help="raw build photos per seen category")
    ap.add_argument("--modules", default=str(REPO / "modules_people"))
    a = ap.parse_args()

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        N, meta = z["X"], json.loads(str(z["meta"]))
    else:
        from bench_ollama import questions  # never learn from the benchmark's own photos

        bench = {q["image"] for q in questions(25, 0, ["presence"])}
        N, meta = collect(a.per_class, bench)
        np.savez(CACHE, X=N, meta=json.dumps(meta))
    split = np.array([m["split"] for m in meta])
    source = np.array([m["source"] for m in meta])
    held = np.array([m["held_out"] for m in meta])
    print("counter-examples:", Counter(zip(source, split, held)))

    # the human instances (ground-truth soft-mask crops, as the leaves were built from)
    crops = json.loads((REPO / "data/human/crops.json").read_text())
    keys = sorted(crops)
    H = np.load(REPO / "data/cache/human_crops_siglip-so400m.npz")["X"]
    ann = json.loads((REPO / "data/human/annotations.json").read_text())
    hlab = np.array([ann[k.split("#")[0]][int(k.split("#")[1])][4] for k in keys])
    hsplit = np.array([split_of(k.split("#")[0]) for k in keys])

    root = Path(a.modules)
    build_neg = np.flatnonzero(split == "build")
    report = {"percentile": a.percentile, "band": MARGIN_BAND, "leaves": {}}
    for leaf in LEAVES:
        m = KnowledgeModule.load(root / leaf)
        pos = m._kind_masks["image"]
        P = m.embeddings[pos]
        entries = [e for e, p in zip(m.entries, pos) if p]
        # counter-examples this leaf could confuse
        near = (N[build_neg] @ P.T).max(axis=1)
        keep = build_neg[near >= m.accept_threshold - MARGIN_BAND]
        Nl = N[keep]
        # margin: nearest positive minus nearest negative, over this leaf's calibration instances
        cal = H[(hsplit == "calib") & (hlab == leaf)]
        d = (cal @ P.T).max(axis=1) - ((cal @ Nl.T).max(axis=1) if len(Nl) else -1.0)
        margin = float(np.percentile(d, a.percentile))
        neg_entries = [{"label": f"not_{leaf.rsplit('/', 1)[1]}", "kind": "negative", "image": meta[i]["image"],
                        "box": meta[i]["box"], "source": f"{meta[i]['source']}:{meta[i]['category']}"} for i in keep]
        new = KnowledgeModule(
            name=m.name, encoder_id=m.encoder_id, embeddings=np.concatenate([P, Nl]) if len(Nl) else P,
            entries=entries + neg_entries, descriptors=m.descriptors, accept_threshold=m.accept_threshold,
            accept_percentile=m.accept_percentile, part_of=m.part_of, reject_margin=margin if len(Nl) else None,
            geometry=m.geometry, version=m.version,
            description=m.description.split(" [")[0] + f" [{len(Nl)} counter-examples, reject margin {margin:.3f}]",
            created=m.created)
        new.save(root)
        report["leaves"][leaf] = {"counter_examples": len(Nl), "reject_margin": round(margin, 4),
                                  "gate": round(m.accept_threshold, 4)}
        print(f"{leaf:<28} {len(Nl):>5} counter-examples, reject margin {margin:+.3f}", flush=True)

    # ---- test: the full classify (router, k-NN vote, leaf gate, negative margin), before and after
    from build_people import Stub
    from mosaic import Mosaic

    cfg = json.loads((root / "registry.json").read_text())

    def system(with_negatives: bool) -> Mosaic:
        s = Mosaic(Stub(), top_k_modules=cfg["top_k"], route_threshold=cfg["route_threshold"])
        for leaf in LEAVES:
            m = KnowledgeModule.load(root / leaf)
            if not with_negatives:
                m.reject_margin = None
            s.attach(m)
        return s

    def accepted(s: Mosaic, Q: np.ndarray) -> np.ndarray:
        return np.array([s.classify(q).known for q in Q])

    sets = {
        "human parts (test, ground-truth crops)": H[hsplit == "test"],
        "non-human, Open Images objects (test)": N[(source == "oi") & (split == "test")],
        "non-human, seen categories (test photos)": N[(source == "raw") & (split == "test") & ~held],
        "non-human, held-out categories": N[(source == "raw") & held],
    }
    report["test"] = {}
    for name, Q in sets.items():
        before, after = accepted(system(False), Q), accepted(system(True), Q)
        key = "accepted" if name.startswith("human") else "rejected"
        f = (lambda v: v.mean()) if key == "accepted" else (lambda v: 1 - v.mean())
        report["test"][name] = {"n": len(Q), f"{key}_before": round(float(f(before)), 4),
                                f"{key}_after": round(float(f(after)), 4)}
        print(f"{name:<45} n={len(Q):>5}  {key} {f(before) * 100:5.1f}% -> {f(after) * 100:5.1f}%")
    (REPO / "results/negatives.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
