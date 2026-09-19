"""MOSAIC experiment: object-level (default) or whole-image regions, same pipeline.

  --regions objects   every image -> class-agnostic objects (scripts/extract_objects.py);
                      modules store only object embeddings, each chosen by consensus
                      across that class's images; every test object is classified
  --regions masks     like objects, but each object is segmented (SAM) and embedded as a
                      cut-out on a neutral background (scripts/extract_masks.py)
  --regions whole     one region per image (the earlier pipeline), for comparison

Module tree: L1 groups = domains, L2 leaves = one plug-in module per class.
Per class: 700 images build the leaf, 100 calibrate the route gate, 200 test.
Image-level decision (Mosaic.main_label): the known label covering the most
image area across accepted objects; no accepted object -> unknown.

Writes results/results_<regions>.json.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from classes import DOMAIN_OF, DOMAINS, LABELS, OOD, class_dir
from mosaic import KnowledgeModule, Mosaic, normalize
from mosaic.builder import consensus_select, nearest_to
from mosaic.encoder import ClipEncoder
from PIL import Image

N_BUILD, N_CALIB, N_TEST = 700, 100, 200
MODULE_TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a picture of the {}.", "a {} in the wild.", "a cropped photo of a {}."]
QUERY_TEMPLATE = "an image showing a {}"
KNN = 5
TOP_K = [1, 3]
CACHE = Path("data/cache")


def leaf(c: str) -> str:
    return f"{DOMAIN_OF[c]}/{c.replace(' ', '_')}"


def split(root: Path, cls: str, rng):
    files = sorted(class_dir(root, cls).glob("*.jpg"))
    files = [files[i] for i in rng.permutation(len(files))]
    return files[:N_BUILD], files[N_BUILD : N_BUILD + N_CALIB], files[N_BUILD + N_CALIB : N_BUILD + N_CALIB + N_TEST]


class Regions:
    """Per-image candidate regions: (n, D) embeddings, objectness, boxes, area fractions."""

    def __init__(self, mode: str, encoder_id: str, min_side: int = 0, mask_keep: float = 0.5):
        self.min_side = min_side
        sizes_path = CACHE / "image_sizes.json"
        self.sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
        self._sizes_path = sizes_path
        slug = encoder_id.replace("/", "_")
        self.whole = np.load(CACHE / f"{slug}.npz")
        if mode == "objects":
            self.embs = np.load(CACHE / f"objects_{slug}.npz")
            self.meta = json.loads((CACHE / "objects.json").read_text())
        elif mode == "masks":
            self.embs = np.load(CACHE / f"masked_{min_side}_k{mask_keep}_{slug}.npz")
            self.meta = json.loads((CACHE / f"masks_{min_side}.json").read_text())
        else:
            self.embs, self.meta = self.whole, None
        self.mode = mode

    def get(self, path):
        e = self.embs[str(path)] if self.mode != "masks" else None
        if self.mode == "whole":
            return e[None], np.ones(1), [None], np.ones(1)
        if str(path) not in self.sizes:
            self.sizes[str(path)] = Image.open(path).size
        w, h = self.sizes[str(path)]
        if self.mode == "masks":
            if str(path) not in self.meta:  # no object >= min_side: whole image
                return self.whole[str(path)][None], np.array([-1.0]), [[0, 0, w, h]], np.ones(1)
            m = self.meta[str(path)]
            return self.embs[str(path)], np.array([r[4] for r in m]), [r[:4] for r in m], np.array([r[5] for r in m])
        m = self.meta[str(path)]
        keep = [i for i, r in enumerate(m) if min(r[2] - r[0], r[3] - r[1]) >= self.min_side]
        if not keep:  # nothing large enough to recognise: fall back to the whole image
            return self.whole[str(path)][None], np.array([-1.0]), [[0, 0, w, h]], np.ones(1)
        m = [m[i] for i in keep]
        area = np.array([(r[2] - r[0]) * (r[3] - r[1]) / (w * h) for r in m])
        return e[keep], np.array([r[4] for r in m]), [r[:4] for r in m], area

    def save_sizes(self):
        self._sizes_path.write_text(json.dumps(self.sizes))


def build_leaf(cls, encoder, regions: Regions, paths):
    cands = [regions.get(p) for p in paths]
    chosen = consensus_select([c[0] for c in cands], [c[1] for c in cands], [c[3] for c in cands])
    embs = np.stack([c[0][j] for c, j in zip(cands, chosen)])
    texts = [t.format(cls) for t in MODULE_TEMPLATES]
    entries = [{"label": cls, "kind": "image", "image": str(p), "box": c[2][j]}
               for p, c, j in zip(paths, cands, chosen)]
    entries += [{"label": cls, "kind": "text", "text": t} for t in texts]
    module = KnowledgeModule.build(
        leaf(cls), encoder.encoder_id, np.concatenate([embs, encoder.embed_texts(texts)]), entries,
        n_descriptors=4, description=f"{cls} ({DOMAIN_OF[cls]})",
    )
    changed = float(np.mean([j != int(np.argmax(c[1])) for c, j in zip(cands, chosen)]))
    return module, changed


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--regions", choices=["objects", "masks", "whole"], default="objects")
    ap.add_argument("--min-side", type=int, default=64, help="ignore objects smaller than this many pixels")
    ap.add_argument("--mask-keep", type=float, default=0.5,
                    help="--regions masks: background kept around each mask (0 = flat cut-out)")
    ap.add_argument("--unknown-rate", type=float, default=0.05,
                    help="calibrate leaf gates so this fraction of calibration images come out unknown")
    ap.add_argument("--modules", default=None, help="where to save leaves (default: modules/ for objects)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modules_dir = args.modules or f"modules_{args.regions}"
    rng = np.random.default_rng(0)
    R: dict = {"config": vars(args) | {"n_build": N_BUILD, "n_calib": N_CALIB, "n_test": N_TEST, "knn": KNN,
                                        "top_k_per_level": TOP_K}}

    encoder = ClipEncoder()
    regions = Regions(args.regions, encoder.encoder_id, args.min_side, args.mask_keep)
    splits = {c: split(root, c, rng) for c in LABELS}
    ood_files = {c: sorted(class_dir(root, c).glob("*.jpg")) for c in OOD}

    # ---- region statistics --------------------------------------------------------
    all_paths = [p for c in LABELS for s in splits[c] for p in s] + [p for c in OOD for p in ood_files[c]]
    n_obj = [len(regions.get(p)[1]) for p in all_paths]
    regions.save_sizes()
    R["regions"] = {"mean_per_image": float(np.mean(n_obj)), "distribution": dict(sorted(Counter(n_obj).items())),
                    "no_object_fallback_rate": float(np.mean([regions.get(p)[1][0] < 0 for p in all_paths]))}

    # ---- build leaves (only object embeddings, chosen by consensus) --------------------
    system = Mosaic(encoder, top_k_modules=TOP_K, route_threshold=-1.0, leaf_gate=False)
    leaves, consensus_changed = {}, {}
    for c in LABELS:
        leaves[c], consensus_changed[c] = build_leaf(c, encoder, regions, splits[c][0])
        leaves[c].save(modules_dir)
        system.attach(leaves[c])
    R["consensus_picked_other_than_top_objectness"] = consensus_changed
    reg = system.registry

    # ---- test set: every object of every test image --------------------------------
    def flatten(paths):
        embs, owner, area = [], [], []
        for i, p in enumerate(paths):
            e, _, _, a = regions.get(p)
            embs.append(e)
            area.append(a)
            owner += [i] * len(e)
        return np.concatenate(embs), np.array(owner), np.concatenate(area)

    test_paths = [p for c in LABELS for p in splits[c][2]]
    test_y = [c for c in LABELS for _ in range(N_TEST)]
    test_X, test_owner, test_area = flatten(test_paths)
    ood_X = {c: flatten(ood_files[c]) for c in OOD}

    def image_level(obj_labels, owner, area, n_images):
        """Per image: the known label covering the most area (as Mosaic.main_label), and all known labels."""
        votes = [Counter() for _ in range(n_images)]
        for lab, i, a in zip(obj_labels, owner, area):
            if lab is not None:
                votes[i][lab] += a
        return [v.most_common(1)[0][0] if v else None for v in votes], [set(v) for v in votes]

    def run(X, owner, area, n, route=-1.0, gate=False):
        system.router.threshold, system.leaf_gate = route, gate
        return image_level(system.classify_batch(X, k=KNN, device="cuda").labels, owner, area, n)

    acc = lambda p, mask=None: float(np.mean([a == y for a, y, m in zip(p, test_y, mask if mask is not None else [1] * len(p)) if m]))

    # ---- calibrate route gate on calib images' reference objects ---------------------
    def reference_objects(paths, c):
        stored = leaves[c].embeddings[leaves[c]._kind_masks["image"]]
        cands = [regions.get(p)[0] for p in paths]
        return np.stack([e[j] for e, j in zip(cands, nearest_to(cands, stored))])

    calib_ref = np.concatenate([reference_objects(splits[c][1], c) for c in LABELS])
    route_threshold = float(np.percentile([system.router.trace(q)[1][0][0].score for q in calib_ref], 1))
    R["route_threshold"] = route_threshold

    # leaf gates: one shared percentile, chosen so `unknown_rate` of calibration *images* come out unknown
    calib_paths = [p for c in LABELS for p in splits[c][1]]
    calib_X = flatten(calib_paths)
    sweep = {}
    for pct in [1, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30]:
        for m in leaves.values():
            m.set_accept_percentile(pct)
        reg.version += 1  # thresholds changed: repack the GPU engine
        sweep[pct] = float(np.mean([p is None for p in run(*calib_X, len(calib_paths), route_threshold, True)[0]]))
    pct = min(sweep, key=lambda k: abs(sweep[k] - args.unknown_rate))
    for m in leaves.values():
        m.set_accept_percentile(pct)
        m.save(modules_dir)
    reg.version += 1
    R["leaf_gate_percentile"], R["calib_unknown_rate_by_percentile"] = pct, sweep
    R["leaf_accept_thresholds"] = {c: leaves[c].accept_threshold for c in LABELS}

    # ---- E1 routing of each test image's labelled object (diagnostic) ------------------
    test_ref = np.concatenate([reference_objects(splits[c][2], c) for c in LABELS])
    system.router.threshold = -1.0  # diagnostics: always route
    traces = [system.router.trace(q) for q in test_ref]
    R["E1_routing_of_labelled_object"] = {
        "L1_domain_accuracy": float(np.mean([t[1][0][0].module == DOMAIN_OF[y] for t, y in zip(traces, test_y)])),
        "L2_class_top1": float(np.mean([t[0][0].module == leaf(y) for t, y in zip(traces, test_y)])),
        "L2_class_in_top3": float(np.mean([leaf(y) in {r.module for r in t[0]} for t, y in zip(traces, test_y)])),
    }

    # ---- E2 image-level accuracy, no abstention, + baselines on the same regions ----------
    primary, labelsets = run(test_X, test_owner, test_area, len(test_paths))
    R["E2_accuracy"] = {"mosaic": acc(primary)}
    R["E2_true_label_among_objects"] = float(np.mean([y in s for s, y in zip(labelsets, test_y)]))
    R["E2_per_domain"] = {d: acc(primary, [DOMAIN_OF[y] == d for y in test_y]) for d in DOMAINS}
    R["E2_per_class"] = {c: acc(primary, [y == c for y in test_y]) for c in LABELS}
    R["E2_confusions"] = dict(Counter(f"{y} -> {p}" for p, y in zip(primary, test_y) if p != y).most_common(15))

    flat_X = np.concatenate([leaves[c].embeddings[leaves[c]._kind_masks["image"]] for c in LABELS])
    flat_y = np.array([c for c in LABELS for _ in range(N_BUILD)])

    def per_object_to_image(obj_labels, _conf):
        return image_level(obj_labels, test_owner, test_area, len(test_paths))[0]

    S = test_X @ flat_X.T
    top = np.argpartition(-S, KNN, axis=1)[:, :KNN]
    knn_lab = [Counter(flat_y[t]).most_common(1)[0][0] for t in top]
    R["E2_accuracy"]["flat_knn_no_router"] = acc(per_object_to_image(knn_lab, S.max(1)))
    zs_W = normalize(np.stack([encoder.embed_texts([t.format(c) for t in MODULE_TEMPLATES]).mean(0) for c in LABELS]))
    Z = test_X @ zs_W.T
    R["E2_accuracy"]["zero_shot_clip"] = acc(per_object_to_image([LABELS[i] for i in Z.argmax(1)], Z.max(1)))
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    probe = LogisticRegression(max_iter=3000, C=10).fit(flat_X, flat_y)
    probe_ms = 1000 * (time.perf_counter() - t0)
    P = probe.predict_proba(test_X)
    R["E2_accuracy"]["linear_probe"] = acc(per_object_to_image(list(probe.classes_[P.argmax(1)]), P.max(1)))

    # ---- E3 unknowns: route gate + per-leaf gates ------------------------------------
    primary_g, labelsets_g = run(test_X, test_owner, test_area, len(test_paths), route_threshold, True)
    R["E3_abstention"] = {
        "in_domain_image_unknown_rate": float(np.mean([p is None for p in primary_g])),
        "in_domain_accuracy_with_gates": acc(primary_g),
        "in_domain_extra_wrong_label_rate": float(np.mean([bool(s - {y}) for s, y in zip(labelsets_g, test_y)])),
        **{f"{c}_image_rejected": float(np.mean([p is None for p in run(*ood_X[c], len(ood_files[c]), route_threshold, True)[0]]))
           for c in OOD},
    }

    # ---- E4 plug out / plug in (gated) ----------------------------------------------
    def plug_test(node, mine):
        removed = system.detach(node)
        without = run(test_X, test_owner, test_area, len(test_paths), route_threshold, True)[0]
        for name in removed:
            system.attach(next(m for m in leaves.values() if m.name == name))
        again = run(test_X, test_owner, test_area, len(test_paths), route_threshold, True)[0]
        return {
            "its_images_unknown": float(np.mean([p is None for p, m in zip(without, mine) if m])),
            "its_images_mislabelled": float(np.mean([p is not None for p, m in zip(without, mine) if m])),
            "mislabelled_as": dict(Counter(p for p, m in zip(without, mine) if m and p).most_common(3)),
            "other_images_changed": int(sum(a != b for a, b, m in zip(primary_g, without, mine) if not m)),
            "identical_after_replug": again == primary_g,
        }

    E4 = {"plug_out_class": {c: plug_test(leaf(c), [y == c for y in test_y]) for c in LABELS},
          "plug_out_domain": {d: plug_test(d, [DOMAIN_OF[y] == d for y in test_y]) for d in DOMAINS}}
    E4["class_plug_out_mislabelled_mean"] = float(np.mean([v["its_images_mislabelled"] for v in E4["plug_out_class"].values()]))
    E4["linear_probe_retrain_ms"] = probe_ms
    R["E4_plug"] = E4

    # ---- E5 text queries retrieve stored objects ---------------------------------------
    system.router.threshold = -1.0
    text = {c: system.search_text(QUERY_TEMPLATE.format(c), k=50) for c in LABELS}
    R["E5_text"] = {
        "leaf_route_accuracy": float(np.mean([bool(r.routes) and r.routes[0].module == leaf(c) for c, r in text.items()])),
        "mean_precision_at_50": float(np.mean([np.mean([m.label == c for m in r.matches]) for c, r in text.items()])),
    }

    (out / f"results_{args.regions}.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({k: v for k, v in R.items() if k not in ("E2_per_class", "config", "E4_plug", "leaf_accept_thresholds",
                                                              "consensus_picked_other_than_top_objectness")}, indent=2))
    print("class plug-out mislabelled mean:", E4["class_plug_out_mislabelled_mean"])


if __name__ == "__main__":
    main()
