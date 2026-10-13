"""Build the people domain on SigLIP SO400M, with anatomical signatures, and measure what geometry adds.

Inputs (from earlier steps):
  data/human/annotations.json               Open Images boxes (fetch_human.py)
  data/human/crops.json + crops/            soft-mask cut-out of every instance (cache_human_crops.py)
  data/cache/human_crops_siglip-so400m.npz  their SigLIP embeddings (compare_encoders.py --only siglip-so400m)
  data/human/poses.json                     ViTPose keypoints per annotated person (extract_poses.py)

Same image-hash split as experiments 6-7 (70 build / 10 calibrate / 20 test). Each leaf stores
its build instances (image entries, with source photo and box) and a geometry signature fitted
on the same instances. The registry gate is the 1st percentile of calibration instances'
domain score; the geometry weight is chosen on calibration instances.

Test instances are identified three ways (no gates, as in experiment 7):
  routed    the MOSAIC tree, 5-NN vote (embedding only)
  leafsim   best leaf by mean of its 3 closest stored examples (embedding only)
  geometry  leafsim + w * signature log-likelihood, when a skeleton owns the instance

Writes modules_people/ (with registry.json) and results/people.json.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from human_classes import LEAVES, NAME, PART_OF
from mosaic import KnowledgeModule, Mosaic
from mosaic.geometry import Skeleton, fit_signature, owner, part_features, signature_loglik

ENCODER_ID = "open_clip/ViT-SO400M-14-SigLIP/webli"


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
    encoder_id, dim = ENCODER_ID, 1152


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="modules_people")
    args = ap.parse_args()

    ann = json.loads(Path("data/human/annotations.json").read_text())
    crops = json.loads(Path("data/human/crops.json").read_text())
    poses = json.loads(Path("data/human/poses.json").read_text())
    keys = sorted(crops)
    X = np.load("data/cache/human_crops_siglip-so400m.npz")["X"]
    assert len(X) == len(keys)
    boxes = [ann[k.split("#")[0]][int(k.split("#")[1])][:4] for k in keys]
    labels = np.array([ann[k.split("#")[0]][int(k.split("#")[1])][4] for k in keys])
    splits = np.array([split_of(k.split("#")[0]) for k in keys])

    skeletons = {p: [Skeleton(np.array(r["keypoints"], dtype=np.float32), r["box"]) for r in rs]
                 for p, rs in poses.items()}
    feats = []
    for k, box in zip(keys, boxes):
        s = owner(box, skeletons.get(k.split("#")[0], []))
        feats.append(part_features(box, s) if s is not None and s.n_visible >= 4 else None)

    # ---- leaves --------------------------------------------------------------------------------
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out / "people", ignore_errors=True)
    system = Mosaic(Stub(), top_k_modules=[1, 2, 3], route_threshold=-1.0, leaf_gate=False)
    build = splits == "build"
    for leaf in LEAVES:
        idx = np.flatnonzero(build & (labels == leaf))
        entries = [{"label": NAME[leaf], "kind": "image", "image": keys[i].split("#")[0], "box": boxes[i],
                    "source": "openimages"} for i in idx]
        signature = fit_signature([feats[i] for i in idx if feats[i] is not None])
        m = KnowledgeModule.build(leaf, ENCODER_ID, X[idx], entries, n_descriptors=4, part_of=PART_OF.get(leaf),
                                  geometry=signature, description=f"human {NAME[leaf]} (Open Images)")
        m.save(out)
        system.attach(m)
        print(f"{leaf:<28} {len(idx):>5} examples, accepts >= {m.accept_threshold:.3f}, "
              f"geometry from {signature['n']}", flush=True)

    # ---- per-leaf scores for calibration and test instances ------------------------------------------
    def leaf_sims(Q):
        cols = []
        for leaf in LEAVES:
            S = Q @ system.registry.get(leaf).embeddings.T
            cols.append(np.sort(S, axis=1)[:, -3:].mean(axis=1))
        return np.stack(cols, axis=1)

    def geo_scores(idx):
        G = np.zeros((len(idx), len(LEAVES)))
        has = np.zeros(len(idx), bool)
        for r, i in enumerate(idx):
            if feats[i] is not None:
                has[r] = True
                G[r] = [signature_loglik(system.registry.get(l).geometry, feats[i]) for l in LEAVES]
        return G, has

    calib = np.flatnonzero(splits == "calib")
    route_threshold = float(np.percentile([system.router.trace(q)[1][0][0].score for q in X[calib]], 1))

    Sc, (Gc, hc) = leaf_sims(X[calib]), geo_scores(calib)
    yc = labels[calib]
    best_w, best_acc = 0.0, -1.0
    for w in [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]:
        pred = np.array(LEAVES)[np.argmax(Sc + w * Gc, axis=1)]
        acc = float((pred == yc).mean())
        print(f"  calib w={w:<6} accuracy {acc * 100:.1f}")
        if acc > best_acc:
            best_w, best_acc = w, acc

    config = {"route_threshold": route_threshold, "top_k": [1, 2, 3], "k": 5, "geometry_weight": best_w,
              "encoder_id": ENCODER_ID, "pose": "usyd-community/vitpose-base-simple",
              "calibration": "route_threshold: 1st percentile of calibration instances' domain score; "
                             "geometry_weight: best calibration accuracy of leafsim + w * log-likelihood"}
    (out / "registry.json").write_text(json.dumps(config, indent=2))

    # ---- test --------------------------------------------------------------------------------------------
    test = np.flatnonzero(splits == "test")
    yt = labels[test]
    St, (Gt, ht) = leaf_sims(X[test]), geo_scores(test)
    routed = np.array([system.classify(q).module for q in X[test]], dtype=object)
    leafsim = np.array(LEAVES)[np.argmax(St, axis=1)]
    fused = np.where(ht, np.array(LEAVES)[np.argmax(St + best_w * Gt, axis=1)], leafsim)
    geometry_only = np.array(LEAVES)[np.argmax(Gt, axis=1)]

    def summary(pred, mask=None):
        m = np.ones(len(yt), bool) if mask is None else mask
        p, y = pred[m], yt[m]
        return {"n": int(m.sum()),
                "accuracy": round(float((p == y).mean()), 4),
                "accuracy_anatomy_credit": round(float(np.mean([adjacent(a, b) for a, b in zip(p, y)])), 4),
                "per_part": {NAME[l]: round(float((p[y == l] == l).mean()), 3) for l in LEAVES if (y == l).any()}}

    R = {"encoder": ENCODER_ID, "route_threshold": route_threshold, "geometry_weight": best_w,
         "n_build": int(build.sum()), "n_test": len(test),
         "test_with_skeleton": round(float(ht.mean()), 4),
         "all_test": {"routed": summary(routed), "leafsim": summary(leafsim), "geometry": summary(fused)},
         "with_skeleton": {"routed": summary(routed, ht), "leafsim": summary(leafsim, ht),
                           "geometry": summary(fused, ht), "geometry_alone": summary(geometry_only, ht)},
         "confusions_leafsim": Counter(f"{NAME[a]}->{NAME[b]}" for a, b in zip(yt, leafsim) if a != b).most_common(8),
         "confusions_geometry": Counter(f"{NAME[a]}->{NAME[b]}" for a, b in zip(yt, fused) if a != b).most_common(8)}
    Path("results/people.json").write_text(json.dumps(R, indent=2))
    for part, v in R["all_test"].items():
        print(f"{part:<10} accuracy {v['accuracy'] * 100:5.1f}  anatomy credit {v['accuracy_anatomy_credit'] * 100:5.1f}")
    print(f"skeleton covers {ht.mean() * 100:.1f}% of test instances; geometry weight {best_w}")
    print(json.dumps(R["with_skeleton"], indent=1)[:3000])


if __name__ == "__main__":
    main()
