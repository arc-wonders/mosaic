"""Segment + embed every annotated human instance, and every detector proposal, in data/human.

Per image, views = the whole image plus zoomed crops of its two largest annotated people
(upscaled to --zoom-side, as the model zooms at query time). In each view the detector
proposes objects; SAM segments proposals (and, in the whole-image view, the ground-truth
human boxes) and every object is embedded as a soft cut-out.

Proposal labels, by IoU with *all* annotated boxes of the image:
    >= 0.5 -> that box's label (a human leaf path, or "oi:<name>" e.g. "oi:Jeans")
    <  0.3 with every box -> "background"         else -> "ambiguous"

Writes (resumable):
  data/cache/human_gt_<enc>.npz         {"path#j": embedding of ground-truth box j}
  data/cache/human_props.json           {path: [{view, box, objectness, label, iou}, ...]}
  data/cache/human_props_<enc>.npz      {path: (n, D) proposal embeddings}
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from human_classes import LEAVES
from mosaic.detector import ObjectnessDetector
from mosaic.encoder import ClipEncoder
from mosaic.perceiver import cutout
from mosaic.segmenter import BoxSegmenter

CACHE = Path("data/cache")


def split_of(path: str) -> str:
    h = int(hashlib.md5(path.encode()).hexdigest(), 16) % 10
    return "build" if h < 7 else ("calib" if h < 8 else "test")


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom-side", type=int, default=768)
    ap.add_argument("--min-side", type=int, default=32, help="ground-truth instances smaller than this are skipped")
    ap.add_argument("--keep", type=float, default=0.5)
    args = ap.parse_args()

    ann = json.loads(Path("data/human/annotations.json").read_text())
    encoder, detector, segmenter = ClipEncoder(fp16=True), ObjectnessDetector(), BoxSegmenter()
    slug = encoder.encoder_id.replace("/", "_")
    gt_path, meta_path, props_path = (CACHE / f"human_gt_{slug}.npz", CACHE / "human_props.json",
                                      CACHE / f"human_props_{slug}.npz")
    gt = dict(np.load(gt_path)) if gt_path.exists() else {}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    props = dict(np.load(props_path)) if props_path.exists() else {}
    # test and calibration images first, so evaluation data exists even if the run is cut short
    todo = sorted((p for p in ann if p not in meta), key=lambda p: (split_of(p) == "build", p))
    print(f"{len(todo)} images", flush=True)

    def save():
        np.savez(gt_path, **gt)
        meta_path.write_text(json.dumps(meta))
        np.savez(props_path, **props)

    for n, path in enumerate(todo):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        boxes = ann[path]
        people = sorted((b for b in boxes if b[4] == "people/person" and min(b[2] - b[0], b[3] - b[1]) >= 64),
                        key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
        meta[path], vecs = [], []
        for v, region in enumerate([[0, 0, w, h]] + [b[:4] for b in people]):
            view = img.crop(region)
            scale = max(1.0, args.zoom_side / max(view.size))
            if scale > 1:
                view = view.resize((round(view.width * scale), round(view.height * scale)), Image.BICUBIC)
            to_view = lambda b: tuple(int(round((b[i] - region[i % 2]) * scale)) for i in range(4))
            dets = detector.detect([view])[0]
            gt_idx = [j for j, b in enumerate(boxes) if v == 0 and b[4] in LEAVES
                      and min(b[2] - b[0], b[3] - b[1]) >= args.min_side and f"{path}#{j}" not in gt]
            prompts = [d.box for d in dets] + [to_view(boxes[j]) for j in gt_idx]
            if not prompts:
                continue
            masks = segmenter.segment([view], [prompts])[0]
            embs = encoder.embed_images([cutout(view, b, m, keep=args.keep) for b, m in zip(prompts, masks)])
            for d, e in zip(dets, embs[: len(dets)]):
                x0, y0, x1, y1 = d.box
                box = [round(region[0] + x0 / scale), round(region[1] + y0 / scale),
                       round(region[0] + x1 / scale), round(region[1] + y1 / scale)]
                best = max(((iou(box, b[:4]), b[4]) for b in boxes), default=(0.0, None))
                label = best[1] if best[0] >= 0.5 else ("background" if best[0] < 0.3 else "ambiguous")
                meta[path].append({"view": v, "box": box, "objectness": round(d.score, 3),
                                   "label": label, "iou": round(best[0], 3)})
                vecs.append(e.astype(np.float32))
            for j, e in zip(gt_idx, embs[len(dets):]):
                gt[f"{path}#{j}"] = e.astype(np.float32)
        if vecs:
            props[path] = np.stack(vecs)
        if n % 100 == 0 or n == len(todo) - 1:
            print(f"{n + 1}/{len(todo)}", flush=True)
            save()
    save()


if __name__ == "__main__":
    main()
