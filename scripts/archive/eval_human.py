"""Build the human identification model (people/* leaves) and measure how good routing is.

Leaf versions (same global image split for everything):
  A  ground-truth instance cut-outs only
  B  A + detector proposals matching a ground-truth instance of the leaf
  C  B + negatives (true non-elements only: unannotated background and clearly non-human
     objects; clothing/accessories are NOT negatives) with a per-leaf reject margin
     calibrated on calibration proposals

The model evaluated is the full registry: the new people/ tree (3 levels:
people -> {person, face_region, limbs} -> parts) plus the non-human domains from
modules_masks (animals, birds, vehicles, fruits, handheld).

Evaluation:
  gt        test ground-truth instances: domain / region / part routing, final label,
            confusion, and how many errors are anatomically adjacent (face<->head, hand<->arm ...)
  proposals every detector proposal in test images: per-part precision / recall / F1,
            background accepted as a body part, what clothing gets called
  nonhuman  non-human test objects: fraction wrongly routed to / accepted as people
  flat      the same with flat routing over all leaves (no region level)

Writes results/human.json; --install saves the chosen version's leaves to modules_human/.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from human_classes import LEAVES, NAME, PART_OF
from mosaic import KnowledgeModule, Mosaic
from mosaic.encoder import ClipEncoder
from run_experiment import MODULE_TEMPLATES

CACHE = Path("data/cache")
NOT_NEGATIVE = ("Human", "Clothing", "Footwear", "Shirt", "Jeans", "Dress", "Jacket", "Suit", "Shorts",
                "Skirt", "Sunglasses", "Glasses", "Hat", "Tie", "Glove", "Sock", "Scarf", "Fashion accessory",
                "Mammal", "Person", "Man", "Woman", "Boy", "Girl", "Helmet", "Earrings", "Necklace",
                "Goggles", "Backpack", "Handbag", "Swimwear", "Brassiere", "Coat", "Sandal", "Boot",
                "High heels", "Roller skates", "Cowboy hat", "Sombrero", "Fedora", "Sun hat",
                "Bicycle helmet", "Football helmet", "Belt", "Watch", "Doll", "Toy", "Sculpture", "Statue")


def split_of(path: str) -> str:
    h = int(hashlib.md5(path.encode()).hexdigest(), 16) % 10
    return "build" if h < 7 else ("calib" if h < 8 else "test")


def is_negative(label: str) -> bool:
    if label == "background":
        return True
    return label.startswith("oi:") and not any(k in label for k in NOT_NEGATIVE)


def adjacent(a: str, b: str) -> bool:
    """a and b are linked in the anatomy (one is an ancestor of the other)."""
    def ancestors(x):
        out = []
        while x in PART_OF:
            x = PART_OF[x]
            out.append(x)
        return out
    return a in ancestors(b) or b in ancestors(a)


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--max-negatives", type=int, default=2000)
    ap.add_argument("--base", default="modules_masks", help="non-human domains are loaded from here")
    ap.add_argument("--install", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    encoder = ClipEncoder(fp16=True)
    slug = encoder.encoder_id.replace("/", "_")
    ann = json.loads(Path("data/human/annotations.json").read_text())
    gt_cache = np.load(CACHE / f"human_gt_{slug}.npz")
    props = json.loads((CACHE / "human_props.json").read_text())
    prop_embs = np.load(CACHE / f"human_props_{slug}.npz")

    # ---- gather examples by split ------------------------------------------------------------
    gt = defaultdict(lambda: defaultdict(list))        # leaf -> split -> [emb]
    det_pos = defaultdict(lambda: defaultdict(list))   # leaf -> split -> [emb]
    negatives = defaultdict(list)                      # split -> [emb]
    proposals = defaultdict(list)                      # split -> [(emb, label)]
    for key in gt_cache.files:
        path, j = key.split("#")
        gt[ann[path][int(j)][4]][split_of(path)].append(gt_cache[key])
    for path, rows in props.items():
        if path not in prop_embs.files:
            continue
        for row, e in zip(rows, prop_embs[path]):
            label, split = row["label"], split_of(path)
            if label == "ambiguous":
                continue
            proposals[split].append((e, label))
            if label in LEAVES:
                det_pos[label][split].append(e)
            elif is_negative(label):
                negatives[split].append(e)

    def cap(xs, n):
        return [xs[i] for i in rng.permutation(len(xs))[:n]] if len(xs) > n else list(xs)

    R = {"counts": {NAME[l]: {"gt_build": len(gt[l]["build"]), "gt_test": len(gt[l]["test"]),
                              "detector_positives_build": len(det_pos[l]["build"])} for l in LEAVES},
         "negatives_build": len(negatives["build"]),
         "proposals": {s: len(v) for s, v in proposals.items()}}
    print(json.dumps(R["counts"]), flush=True)
    neg_pool = cap(negatives["build"], args.max_negatives)

    def build(leaf, version):
        pos = cap(gt[leaf]["build"], args.per_class)
        if version in ("B", "C"):
            pos += cap(det_pos[leaf]["build"], args.per_class)
        texts = [t.format(NAME[leaf]) for t in MODULE_TEMPLATES]
        embs = [np.stack(pos), encoder.embed_texts(texts)]
        entries = ([{"label": NAME[leaf], "kind": "image"}] * len(pos)
                   + [{"label": NAME[leaf], "kind": "text", "text": t} for t in texts])
        if version == "C" and neg_pool:
            embs.append(np.stack(neg_pool))
            entries += [{"label": NAME[leaf], "kind": "negative"}] * len(neg_pool)
        return KnowledgeModule.build(leaf, encoder.encoder_id, np.concatenate(embs), entries, n_descriptors=4,
                                     accept_percentile=5, part_of=PART_OF.get(leaf),
                                     description=f"{NAME[leaf]} (human model, version {version})")

    base_config = json.loads((Path(args.base) / "registry.json").read_text())

    def model(leaves, hierarchical=True):
        system = Mosaic(encoder, top_k_modules=[1, 2, 3], route_threshold=base_config["route_threshold"],
                        hierarchical=hierarchical)
        system.registry.load_dir(args.base)
        if "people" in system.registry:
            system.detach("people")  # replaced by the human model
        for leaf in leaves.values():
            system.attach(leaf)
        return system

    # non-human test objects, held out (never stored in any leaf): for each 19-class photo, its candidate
    # object closest to its own class leaf (the dog, not the person walking it), plus held-out handheld
    # instances. Anything identical to a stored example is dropped, so nothing can match itself.
    from mosaic import Registry
    base_reg = Registry(encoder.encoder_id, encoder.dim)
    base_reg.load_dir(args.base)
    stored = {n: base_reg.get(n).embeddings[base_reg.get(n)._kind_masks["image"]]
              for n in base_reg.leaves() if not n.startswith("people/")}
    all_stored = np.concatenate(list(stored.values()))
    nonhuman = []
    masked = CACHE / f"masked_64_k0.5_{slug}.npz"
    if masked.exists():
        E = np.load(masked)
        keys = [k for k in E.files if "/ood/" not in k]
        for k in [keys[i] for i in rng.permutation(len(keys))[:1500]]:
            domain, cls = k.split("/")[2:4]
            leaf_embs = stored.get(f"{domain}/{cls}")
            if leaf_embs is None:
                continue
            cands = E[k]
            nonhuman.append(cands[int(np.argmax((cands @ leaf_embs.T).max(axis=1)))])
    parts_cache = CACHE / f"parts_k0.5_{slug}.npz"
    if parts_cache.exists():
        P = np.load(parts_cache)
        pann = json.loads(Path("data/parts/annotations.json").read_text())
        hh = [k for k in P.files if pann[k.split("#")[0]][int(k.split("#")[1])][4] in
              ("phone", "bottle", "cup", "knife", "ball", "book")]
        nonhuman += [P[k] for k in [hh[i] for i in rng.permutation(len(hh))[:800]]]
    nonhuman = [e for e in nonhuman if float((all_stored @ e).max()) < 0.9999]

    def evaluate(system, leaves):
        out = {}
        # -- ground-truth instances
        rows = [(e, leaf) for leaf in LEAVES for e in gt[leaf]["test"]]
        dom = reg_ok = leaf_top1 = leaf_kept = 0
        preds, forced = [], []
        gate = system.router.threshold
        for e, leaf in rows:
            routes, levels = system.router.trace(e)
            dom += levels[0][0].module == "people"
            if len(levels) > 1 and system.router.hierarchical:
                reg_ok += levels[1][0].module == (leaf.rsplit("/", 1)[0] if leaf.count("/") == 2 else leaf)
            leaf_top1 += bool(routes) and routes[0].module == leaf
            leaf_kept += leaf in {r.module for r in routes}
            preds.append(system.classify(e).module)
            system.leaf_gate, system.router.threshold = False, -1.0
            forced.append(system.classify(e).module)
            system.leaf_gate, system.router.threshold = True, gate
        n = len(rows)
        truth = [leaf for _, leaf in rows]
        wrong = [(t, p) for t, p in zip(truth, forced) if p != t]
        out["gt"] = {
            "n": n,
            "domain_is_people": round(dom / n, 4),
            "region_correct": round(reg_ok / n, 4) if system.router.hierarchical else None,
            "part_ranked_first_by_router": round(leaf_top1 / n, 4),
            "part_among_kept_candidates": round(leaf_kept / n, 4),
            "accuracy_forced": round(float(np.mean([p == t for p, t in zip(forced, truth)])), 4),
            "accuracy_with_gates": round(float(np.mean([p == t for p, t in zip(preds, truth)])), 4),
            "unknown_rate": round(float(np.mean([p is None for p in preds])), 4),
            "errors_anatomically_adjacent": round(float(np.mean([adjacent(t, p) for t, p in wrong if p])), 4) if wrong else None,
            "errors_outside_people": round(float(np.mean([not (p or "").startswith("people/") for t, p in wrong])), 4) if wrong else None,
            "per_part_accuracy_forced": {NAME[l]: round(float(np.mean([p == l for p, t in zip(forced, truth) if t == l])), 3)
                                          for l in LEAVES if any(t == l for t in truth)},
            "top_confusions": dict(Counter(f"{NAME[t]} -> {NAME.get(p, p)}" for t, p in wrong).most_common(12)),
        }
        # -- detector proposals
        items = proposals["test"]
        pr = [system.classify(e).module for e, _ in items]
        per = {}
        for l in LEAVES:
            tp = sum(p == l and lab == l for p, (_, lab) in zip(pr, items))
            fp = sum(p == l and lab != l for p, (_, lab) in zip(pr, items))
            fn = sum(p != l and lab == l for p, (_, lab) in zip(pr, items))
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            per[NAME[l]] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1(prec, rec), 3),
                            "n_true": tp + fn}
        neg_idx = [i for i, (_, lab) in enumerate(items) if is_negative(lab)]
        clothing = [pr[i] for i, (_, lab) in enumerate(items) if lab.startswith("oi:") and not is_negative(lab)]
        out["proposals"] = {
            "n": len(items), "per_part": per,
            "mean_f1": round(float(np.mean([v["f1"] for v in per.values()])), 3),
            "non_elements_accepted_as_body_part": round(float(np.mean([(pr[i] or "").startswith("people/") for i in neg_idx])), 3),
            "clothing_and_accessories_called": dict(Counter(NAME.get(p, p or "unknown") for p in clothing).most_common(6)),
        }
        # -- non-human objects
        if nonhuman:
            nh = [system.classify(e) for e in nonhuman]
            out["nonhuman"] = {"n": len(nh),
                               "routed_to_people": round(float(np.mean([system.router.trace(e)[1][0][0].module == "people" for e in nonhuman])), 3),
                               "accepted_as_people": round(float(np.mean([(p.module or "").startswith("people/") and p.known for p in nh])), 3)}
        return out

    R["versions"] = {}
    chosen = {}
    for version in ("A", "B", "C"):
        leaves = {l: build(l, version) for l in LEAVES}
        system = model(leaves)
        if version == "C":  # reject margin per leaf, maximising F1 on calibration proposals
            items = proposals["calib"]
            base = [system.classify(e) for e, _ in items]
            for l, leaf in leaves.items():
                idx = [i for i, p in enumerate(base) if p.module == l and p.known]
                n_true = sum(lab == l for _, lab in items)
                if not idx or not n_true:
                    continue
                gap = np.array([base[i].nearest - leaf.nearest(items[i][0], "negative") for i in idx])
                ok = np.array([items[i][1] == l for i in idx])
                best = max((f1(ok[gap >= t].mean() if (gap >= t).any() else 0.0, ok[gap >= t].sum() / n_true), t)
                           for t in np.unique(np.r_[gap, -1.0]))
                leaf.reject_margin = float(best[1])
            system.registry.version += 1
            R["reject_margins"] = {NAME[l]: leaves[l].reject_margin for l in LEAVES}
        R["versions"][version] = evaluate(system, leaves)
        print(version, json.dumps({k: (v if k != "gt" else {kk: vv for kk, vv in v.items() if kk not in ("per_part_accuracy_forced",)})
                                   for k, v in R["versions"][version].items()}), flush=True)
        chosen[version] = leaves
    flat = model(chosen["C"], hierarchical=False)
    flat.router.top_k = 3
    R["flat_routing_C"] = evaluate(flat, chosen["C"])

    Path("results/human.json").write_text(json.dumps(R, indent=2))
    if args.install:
        for leaf in chosen[args.install].values():
            leaf.save("modules_human")
        (Path("modules_human") / "registry.json").write_text(json.dumps(
            {"route_threshold": base_config["route_threshold"], "top_k": [1, 2, 3], "k": 5,
             "version": args.install}, indent=2))


if __name__ == "__main__":
    main()
