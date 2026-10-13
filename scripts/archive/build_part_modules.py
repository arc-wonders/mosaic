"""Build leaves for people / body parts / handheld objects from box-annotated images.

Every annotated instance (min side >= --min-side) is segmented with SAM from its
ground-truth box, cut out with a soft background, and embedded. Per class, up
to --per-class instances are split by *image* into 70% build / 10% calibration / 20% test,
so two hands from the same photo never straddle build and test.

Leaves go to modules_masks/<domain>/<class> (next to the 19 existing leaves) with
`part_of` set for body parts. Test instances are then classified against the
full registry. Writes results/parts.json.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic import KnowledgeModule, Mosaic
from mosaic.encoder import ClipEncoder
from mosaic.perceiver import cutout
from mosaic.segmenter import BoxSegmenter
from parts_classes import PART_DOMAIN_OF, PART_DOMAINS, PART_LABELS, PART_OF
from run_experiment import MODULE_TEMPLATES

CACHE = Path("data/cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--min-side", type=int, default=40)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--modules", default="modules_masks")
    args = ap.parse_args()

    ann = json.loads(Path("data/parts/annotations.json").read_text())
    # instances per class, grouped by image
    by_class: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for path, boxes in ann.items():
        for j, b in enumerate(boxes):
            if b[4] in PART_DOMAIN_OF and min(b[2] - b[0], b[3] - b[1]) >= args.min_side:
                by_class[b[4]][path].append(j)

    rng = np.random.default_rng(0)
    splits: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for c in PART_LABELS:
        images = list(by_class[c])
        rng.shuffle(images)
        chosen = []
        for p in images:
            chosen += [(p, j) for j in by_class[c][p]]
            if len(chosen) >= args.per_class:
                break
        chosen = chosen[: args.per_class]
        n = len(chosen)
        cut1, cut2 = int(0.7 * n), int(0.8 * n)
        while 0 < cut1 < n and chosen[cut1][0] == chosen[cut1 - 1][0]:  # don't split an image across sets
            cut1 += 1
        while cut1 < cut2 < n and chosen[cut2][0] == chosen[cut2 - 1][0]:
            cut2 += 1
        splits[c] = {"build": chosen[:cut1], "calib": chosen[cut1:cut2], "test": chosen[cut2:]}

    # ---- segment + embed every selected instance (cached) --------------------------------
    encoder = ClipEncoder(fp16=True)
    cache_path = CACHE / f"parts_k{args.keep}_{encoder.encoder_id.replace('/', '_')}.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    needed: dict[str, list[int]] = defaultdict(list)
    for c in PART_LABELS:
        for items in splits[c].values():
            for p, j in items:
                if f"{p}#{j}" not in cache:
                    needed[p].append(j)
    if needed:
        segmenter = BoxSegmenter()
        paths = sorted(needed)
        print(f"segmenting {sum(len(v) for v in needed.values())} instances in {len(paths)} images", flush=True)
        for i in range(0, len(paths), 8):
            chunk = paths[i : i + 8]
            images = [Image.open(p).convert("RGB") for p in chunk]
            boxes = [[tuple(ann[p][j][:4]) for j in needed[p]] for p in chunk]
            masks = segmenter.segment(images, boxes)
            crops, keys = [], []
            for p, img, bxs, ms in zip(chunk, images, boxes, masks):
                for j, b, m in zip(needed[p], bxs, ms):
                    crops.append(cutout(img, b, m, keep=args.keep))
                    keys.append(f"{p}#{j}")
            for k, e in zip(keys, encoder.embed_images(crops)):
                cache[k] = e.astype(np.float32)
            if (i // 8) % 25 == 0:
                print(f"  {i + len(chunk)}/{len(paths)} images", flush=True)
                np.savez(cache_path, **cache)
        np.savez(cache_path, **cache)

    emb = lambda items: np.stack([cache[f"{p}#{j}"] for p, j in items])

    # ---- build leaves ----------------------------------------------------------------------
    leaves = {}
    for c in PART_LABELS:
        items = splits[c]["build"]
        texts = [t.format(c) for t in MODULE_TEMPLATES]
        entries = [{"label": c, "kind": "image", "image": p, "box": ann[p][j][:4]} for p, j in items]
        entries += [{"label": c, "kind": "text", "text": t} for t in texts]
        leaves[c] = KnowledgeModule.build(
            f"{PART_DOMAIN_OF[c]}/{c}", encoder.encoder_id,
            np.concatenate([emb(items), encoder.embed_texts(texts)]), entries,
            n_descriptors=4, accept_percentile=5, part_of=PART_OF.get(c),
            description=f"{c} ({PART_DOMAIN_OF[c]})",
        )
        leaves[c].save(args.modules)

    # ---- registry-wide domain gate: 1st percentile over every domain's calibration objects --
    base = json.loads(Path("results/results_masks.json").read_text())["route_threshold"]
    system = Mosaic(encoder, top_k_modules=[1, 3], route_threshold=-1.0)
    system.registry.load_dir(args.modules)
    calib = np.concatenate([emb(splits[c]["calib"]) for c in PART_LABELS if splits[c]["calib"]])
    parts_pct1 = float(np.percentile([system.router.trace(q)[1][0][0].score for q in calib], 1))
    route_threshold = min(base, parts_pct1)
    config_path = Path(args.modules) / "registry.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update({"route_threshold": route_threshold, "top_k": [1, 3], "k": 5,
                   "calibration": f"1st percentile of the top domain score over held-out calibration objects "
                                  f"(19 ImageNet classes: {base:.3f}; people/handheld: {parts_pct1:.3f})"})
    config_path.write_text(json.dumps(config, indent=2))
    system.router.threshold = route_threshold

    # ---- evaluate test instances against the full registry ------------------------------
    R = {"n_leaves_in_registry": len(system.registry), "route_threshold": route_threshold,
         "route_threshold_imagenet_calibration": base, "route_threshold_parts_calibration": parts_pct1,
         "instances": {c: {s: len(v) for s, v in splits[c].items()} for c in PART_LABELS}, "per_class": {}}
    all_true, all_pred = [], []
    for c in PART_LABELS:
        items = splits[c]["test"]
        if not items:
            continue
        gated = system.classify_batch(emb(items), device="cuda")
        system.leaf_gate, system.router.threshold = False, -1.0
        forced = system.classify_batch(emb(items), device="cuda")
        system.leaf_gate, system.router.threshold = True, route_threshold
        R["per_class"][c] = {
            "n_test": len(items),
            "accuracy_forced": float(np.mean([l == c for l in forced.labels])),
            "accuracy_with_gates": float(np.mean([l == c for l in gated.labels])),
            "unknown_rate": float(np.mean([l is None for l in gated.labels])),
            "confused_with": dict(Counter(l for l in forced.labels if l != c).most_common(3)),
        }
        all_true += [c] * len(items)
        all_pred += forced.labels
    R["mean_accuracy_forced"] = float(np.mean([v["accuracy_forced"] for v in R["per_class"].values()]))
    R["mean_accuracy_with_gates"] = float(np.mean([v["accuracy_with_gates"] for v in R["per_class"].values()]))
    Path("results/parts.json").write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
