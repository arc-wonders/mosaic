"""Multi-object test: can the system name *every* object in an image?

Builds collages of two held-out test images from different classes, placed side
by side, and runs the live pipelines end to end:
  objects: detector -> box crops -> every object classified (modules_objects/)
  masks:   detector -> SAM masks -> cut-outs -> every object classified (modules_masks/)
  whole:   one embedding for the whole collage (modules_whole/)
A whole-image system can name at most one of the two by construction; this
measures what object-level recognition buys.

Writes results/multi_object.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from classes import LABELS
from mosaic import Mosaic
from mosaic.detector import ObjectnessDetector
from mosaic.encoder import ClipEncoder
from mosaic.segmenter import BoxSegmenter
from run_experiment import N_BUILD, split

TOP_K = [1, 3]


def collage(a: Path, b: Path, side: int = 256) -> Image.Image:
    out = Image.new("RGB", (2 * side, side))
    out.paste(Image.open(a).convert("RGB").resize((side, side)), (0, 0))
    out.paste(Image.open(b).convert("RGB").resize((side, side)), (side, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--data", default="data/raw")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    splits = {c: split(Path(args.data), c, rng) for c in LABELS}  # same split as run_experiment
    pick = np.random.default_rng(1)
    pairs = []
    for _ in range(args.n):
        a, b = pick.choice(len(LABELS), 2, replace=False)
        ca, cb = LABELS[a], LABELS[b]
        pairs.append((ca, cb, splits[ca][2][pick.integers(200)], splits[cb][2][pick.integers(200)]))
    images = [collage(pa, pb) for _, _, pa, pb in pairs]

    encoder = ClipEncoder(fp16=True)
    kinds = [k for k in ("objects", "masks", "whole") if Path(f"results/results_{k}.json").exists()]
    results = {k: json.loads(Path(f"results/results_{k}.json").read_text()) for k in kinds}
    detector = ObjectnessDetector(min_side=results["objects"]["config"]["min_side"])
    found = {}
    for kind in kinds:
        system = Mosaic(encoder, detector=None if kind == "whole" else detector,
                        segmenter=BoxSegmenter() if kind == "masks" else None,
                        top_k_modules=TOP_K, route_threshold=results[kind]["route_threshold"])
        system.registry.load_dir(f"modules_{kind}")
        found[kind] = []
        for i in range(0, len(images), 16):
            for regions in system.perceiver.perceive_batch(images[i : i + 16]):
                found[kind].append({p.label for p in (system.classify(r.embedding) for r in regions) if p.known})

    def score(found):
        truth = [{ca, cb} for ca, cb, _, _ in pairs]
        return {
            "both_objects_named": float(np.mean([t <= f for f, t in zip(found, truth)])),
            "at_least_one_named": float(np.mean([bool(t & f) for f, t in zip(found, truth)])),
            "mean_correct_per_collage": float(np.mean([len(t & f) for f, t in zip(found, truth)])),
            "wrong_label_rate": float(np.mean([bool(f - t) for f, t in zip(found, truth)])),
            "nothing_named": float(np.mean([not f for f in found])),
        }

    R = {"n_collages": len(pairs), **{k: score(v) for k, v in found.items()}}
    Path("results/multi_object.json").write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=2))

    sample = Image.new("RGB", (4 * 256, 3 * 138), "white")
    from PIL import ImageDraw
    for k in range(12):
        im = images[k].resize((256, 128))
        sample.paste(im, ((k % 4) * 256, (k // 4) * 138))
        ImageDraw.Draw(sample).text(((k % 4) * 256 + 2, (k // 4) * 138 + 126),
                                    " | ".join(f"{kind[0]}:{','.join(sorted(found[kind][k])) or '-'}" for kind in found),
                                    fill=(0, 0, 0))
    sample.save("results/multi_object_sample.jpg")


if __name__ == "__main__":
    main()
