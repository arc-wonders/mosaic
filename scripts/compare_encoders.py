"""Which frozen encoder identifies body parts best? Same crops, same split, same model.

Every annotated human instance (data/human/crops.json, soft-mask cut-outs) is embedded by each
encoder. Build-split instances become the people/* leaves (image entries only, so text-free
encoders compete fairly); test-split instances are identified by
  knn        flat 7-NN over all stored instances
  routed     the MOSAIC tree: people -> {person, face_region, limbs} -> part, then 5-NN vote
Reported: accuracy, per part, region accuracy, and accuracy counting anatomically adjacent
answers (face for head, hand for arm) as right.

Writes results/encoders.json (embeddings cached in data/cache/human_crops_<encoder>.npz).
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from human_classes import LEAVES, NAME, PART_OF
from mosaic import KnowledgeModule, Mosaic
from mosaic.encoder import ClipEncoder, DinoEncoder

ENCODERS = {
    "clip-b32 (current)": lambda: ClipEncoder("ViT-B-32", "laion2b_s34b_b79k", fp16=True),
    "clip-l14 dfn2b": lambda: ClipEncoder("ViT-L-14", "dfn2b", fp16=True),
    "eva02-l14": lambda: ClipEncoder("EVA02-L-14", "merged2b_s4b_b131k", fp16=True),
    "siglip-so400m": lambda: ClipEncoder("ViT-SO400M-14-SigLIP", "webli", fp16=True),
    "clip-h14 dfn5b": lambda: ClipEncoder("ViT-H-14", "dfn5b", fp16=True),
    "dinov2-base": lambda: DinoEncoder("facebook/dinov2-base"),
    "dinov2-large": lambda: DinoEncoder("facebook/dinov2-large"),
}


def split_of(path: str) -> str:
    h = int(hashlib.md5(path.encode()).hexdigest(), 16) % 10
    return "build" if h < 7 else ("calib" if h < 8 else "test")


def ancestors(x):
    out = []
    while x in PART_OF:
        x = PART_OF[x]
        out.append(x)
    return out


def adjacent(a, b):
    return a == b or a in ancestors(b) or b in ancestors(a)


class Stub:
    def __init__(self, encoder_id, dim):
        self.encoder_id, self.dim = encoder_id, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of encoder names")
    args = ap.parse_args()
    ann = json.loads(Path("data/human/annotations.json").read_text())
    crops = json.loads(Path("data/human/crops.json").read_text())
    keys = sorted(crops)
    labels = np.array([ann[k.split("#")[0]][int(k.split("#")[1])][4] for k in keys])
    splits = np.array([split_of(k.split("#")[0]) for k in keys])
    build, test = splits == "build", splits == "test"
    out_path = Path("results/encoders.json")
    R = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name, make in ENCODERS.items():
        if args.only and name not in args.only:
            continue
        cache = Path("data/cache") / f"human_crops_{name.split()[0]}.npz"
        t0 = time.perf_counter()
        if cache.exists():
            X = np.load(cache)["X"]
            speed = R.get(name, {}).get("images_per_s")
        else:
            enc = make()
            X = np.concatenate([enc.embed_images([Image.open(crops[k]).convert("RGB") for k in keys[i : i + 64]])
                                for i in range(0, len(keys), 64)])
            speed = len(keys) / (time.perf_counter() - t0)
            np.savez(cache, X=X)
            del enc
            import torch

            torch.cuda.empty_cache()
        Xb, yb, Xt, yt = X[build], labels[build], X[test], labels[test]

        # flat 7-NN
        S = Xt @ Xb.T
        top = np.argpartition(-S, 7, axis=1)[:, :7]
        knn = np.array([Counter(yb[t]).most_common(1)[0][0] for t in top])

        # MOSAIC tree
        stub = Stub(f"cmp/{name}", X.shape[1])
        system = Mosaic(stub, top_k_modules=[1, 2, 3], route_threshold=-1.0, leaf_gate=False)
        for leaf in LEAVES:
            E = Xb[yb == leaf]
            system.attach(KnowledgeModule.build(leaf, stub.encoder_id, E, [{"label": leaf, "kind": "image"}] * len(E),
                                                n_descriptors=4, part_of=PART_OF.get(leaf)))
        routed, region_ok, kept = [], [], []
        for q, y in zip(Xt, yt):
            routes, levels = system.router.trace(q)
            want_region = y.rsplit("/", 1)[0] if y.count("/") == 2 else y
            region_ok.append(len(levels) > 1 and levels[1][0].module == want_region)
            kept.append(y in {r.module for r in routes})
            routed.append(system.classify(q).module)
        routed = np.array(routed, dtype=object)

        def summary(pred):
            return {"accuracy": round(float((pred == yt).mean()), 4),
                    "accuracy_anatomy_credit": round(float(np.mean([adjacent(p, y) for p, y in zip(pred, yt)])), 4),
                    "per_part": {NAME[l]: round(float((pred[yt == l] == l).mean()), 3) for l in LEAVES}}

        R[name] = {"dim": int(X.shape[1]), "images_per_s": round(speed, 1) if speed else None,
                   "knn": summary(knn), "routed": {**summary(routed),
                                                   "region_correct": round(float(np.mean(region_ok)), 4),
                                                   "part_among_kept": round(float(np.mean(kept)), 4)},
                   "n_build": int(build.sum()), "n_test": int(test.sum())}
        r = R[name]
        print(f"{name:<20} dim {r['dim']:<5} knn {r['knn']['accuracy']*100:5.1f}  routed {r['routed']['accuracy']*100:5.1f}"
              f"  (anatomy credit {r['routed']['accuracy_anatomy_credit']*100:5.1f})  region {r['routed']['region_correct']*100:5.1f}"
              f"  kept {r['routed']['part_among_kept']*100:5.1f}  {r['images_per_s']} img/s", flush=True)
        out_path.write_text(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
