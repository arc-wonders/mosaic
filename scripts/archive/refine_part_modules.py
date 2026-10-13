"""Rebuild the people/* leaves with detector-style positives and negatives, and evaluate.

Uses one global image split (by image hash) for everything, so no stored example can
come from an image used to test any class. Three versions of each people leaf:
  A  ground-truth-box cut-outs only (what build_part_modules.py makes)
  B  A + detector proposals that match a ground-truth part (IoU >= 0.5)
  C  B + negatives: background proposals and other annotated parts from the same
     source images (e.g. legs are negatives for arm), with a reject margin per leaf
     calibrated on calibration images to maximise F1
Evaluation: every detector proposal in the test images (from extract_part_proposals.py)
is classified against the full registry; per-part precision / recall / F1, and the
fraction of background proposals accepted as a people part.

Writes results/parts_refined.json; with --install, saves version C into modules_masks/people.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mosaic import KnowledgeModule, Mosaic
from mosaic.encoder import ClipEncoder
from parts_classes import PART_DOMAINS, PART_OF
from run_experiment import MODULE_TEMPLATES

CACHE = Path("data/cache")
PEOPLE = list(PART_DOMAINS["people"])
SOURCE_OF = {c: srcs[0][0] for c, srcs in PART_DOMAINS["people"].items()}  # humanparts / openimages
ANNOTATED = {s: [c for c in PEOPLE if SOURCE_OF[c] == s] for s in set(SOURCE_OF.values())}


def split_of(path: str) -> str:
    h = int(hashlib.md5(path.encode()).hexdigest(), 16) % 10
    return "build" if h < 7 else ("calib" if h < 8 else "test")


def source_of(path: str) -> str:
    return Path(path).name.split("_")[0]


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="modules_masks")
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--max-negatives", type=int, default=3000)
    ap.add_argument("--install", action="store_true", help="write version C into --modules")
    args = ap.parse_args()

    encoder = ClipEncoder(fp16=True)
    slug = encoder.encoder_id.replace("/", "_")
    ann = json.loads(Path("data/parts/annotations.json").read_text())
    gt_cache = np.load(CACHE / f"parts_k0.5_{slug}.npz")  # ground-truth box cut-outs (build_part_modules.py)
    props = json.loads((CACHE / "part_proposals.json").read_text())
    prop_embs = np.load(CACHE / f"part_proposals_{slug}.npz")
    rng = np.random.default_rng(0)

    # ---- collect examples per class and split --------------------------------------------
    gt_pos = defaultdict(lambda: defaultdict(list))    # class -> split -> [emb]
    det_pos = defaultdict(lambda: defaultdict(list))
    negatives = defaultdict(lambda: defaultdict(list))  # class -> split -> [emb]
    eval_items = defaultdict(list)                      # split -> [(emb, label, source)]
    for key in gt_cache.files:
        path, j = key.split("#")
        c = ann[path][int(j)][4]
        if c in PEOPLE:
            gt_pos[c][split_of(path)].append(gt_cache[key])
    for path, rows in props.items():
        if path not in prop_embs.files:
            continue
        E, src, split = prop_embs[path], source_of(path), split_of(path)
        for row, e in zip(rows, E):
            label = row["label"]
            if label == "ambiguous":
                continue
            eval_items[split].append((e, label, src))
            for c in ANNOTATED[src]:  # this image annotates c, so we know whether the proposal is a c
                (det_pos if label == c else negatives)[c][split].append(e)

    def cap(xs, n):
        return [xs[i] for i in rng.permutation(len(xs))[:n]] if len(xs) > n else xs

    # ---- build leaf versions -------------------------------------------------------------------
    def build(c, version):
        pos = cap(gt_pos[c]["build"], args.per_class)
        if version in ("B", "C"):
            pos = pos + cap(det_pos[c]["build"], args.per_class)
        texts = [t.format(c) for t in MODULE_TEMPLATES]
        embs = [np.stack(pos), encoder.embed_texts(texts)]
        entries = [{"label": c, "kind": "image"}] * len(pos) + [{"label": c, "kind": "text", "text": t} for t in texts]
        if version == "C":
            neg = cap(negatives[c]["build"], args.max_negatives)
            embs.append(np.stack(neg))
            entries += [{"label": c, "kind": "negative"}] * len(neg)
        return KnowledgeModule.build(f"people/{c}", encoder.encoder_id, np.concatenate(embs), entries,
                                     n_descriptors=4, accept_percentile=5, part_of=PART_OF.get(c),
                                     description=f"{c} (people), version {version}")

    config = json.loads((Path(args.modules) / "registry.json").read_text())
    R = {"split": {s: len(v) for s, v in eval_items.items()},
         "stored": {c: {"gt_positives": len(gt_pos[c]["build"]), "detector_positives": len(det_pos[c]["build"]),
                        "negatives": len(negatives[c]["build"])} for c in PEOPLE}, "versions": {}}

    def evaluate(system, split):
        items = eval_items[split]
        X = np.stack([e for e, _, _ in items])
        pred = system.classify_batch(X, device="cuda").labels
        out = {}
        for c in PEOPLE:
            src = SOURCE_OF[c]
            rel = [(p, lab) for p, (_, lab, s) in zip(pred, items) if s == src]
            tp = sum(p == c and lab == c for p, lab in rel)
            fp = sum(p == c and lab != c for p, lab in rel)
            fn = sum(p != c and lab == c for p, lab in rel)
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1(prec, rec), 3),
                      "n_true": tp + fn, "n_predicted": tp + fp}
        bg = [p for p, (_, lab, _) in zip(pred, items) if lab == "background"]
        out["_background_accepted_as_people_part"] = round(float(np.mean([p in PEOPLE for p in bg])), 3)
        out["_background_accepted_as_anything"] = round(float(np.mean([p is not None for p in bg])), 3)
        out["_mean_f1"] = round(float(np.mean([out[c]["f1"] for c in PEOPLE])), 3)
        return out

    for version in ("A", "B", "C"):
        system = Mosaic(encoder, top_k_modules=[1, 3], route_threshold=config["route_threshold"])
        system.registry.load_dir(args.modules)
        leaves = {c: build(c, version) for c in PEOPLE}
        for c, leaf in leaves.items():
            system.detach(leaf.name)
            system.attach(leaf)
        if version == "C":  # calibrate each leaf's reject margin on calibration proposals (maximise F1)
            items = eval_items["calib"]
            X = np.stack([e for e, _, _ in items])
            for leaf in leaves.values():
                leaf.reject_margin = None
            system.registry.version += 1
            base = system.classify_batch(X, device="cuda")
            margins = {}
            for c, leaf in leaves.items():
                idx = [i for i, (p, (_, _, s)) in enumerate(zip(base.labels, items)) if p == c and s == SOURCE_OF[c]]
                n_true = sum(lab == c and s == SOURCE_OF[c] for _, lab, s in items)
                if not idx or not n_true:
                    continue
                gap = np.array([base.nearest[i] - leaf.nearest(X[i], "negative") for i in idx])
                is_c = np.array([items[i][1] == c for i in idx])
                best = max((f1(is_c[gap >= t].mean() if (gap >= t).any() else 0.0, is_c[gap >= t].sum() / n_true), t)
                           for t in np.unique(np.r_[gap, -1.0]))
                leaf.reject_margin = margins[c] = float(best[1])
            system.registry.version += 1
            R["reject_margins"] = margins
        R["versions"][version] = evaluate(system, "test")
        print(version, json.dumps(R["versions"][version]), flush=True)
        if version == "C" and args.install:
            for leaf in leaves.values():
                leaf.save(args.modules)

    Path("results/parts_refined.json").write_text(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
