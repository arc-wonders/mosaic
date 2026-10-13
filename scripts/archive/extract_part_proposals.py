"""Detector proposals on the people images, labelled against the box annotations.

For every Human-Parts / Open Images image in data/parts: the whole image and zoomed
views (each annotated person, or the region around the annotated arms/legs) are
upscaled to --zoom-side, run through the class-agnostic detector, segmented with
SAM, cut out with a soft background and embedded, just as the model sees objects
at query time. Each proposal is labelled by IoU with the image's annotated boxes:
    >= 0.5 with a box of class c  -> c
    <  0.3 with every box         -> "background"
    otherwise                     -> "ambiguous" (not used)

Writes data/cache/part_proposals.json (metadata) and part_proposals_<encoder>.npz.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.detector import ObjectnessDetector
from mosaic.encoder import ClipEncoder
from mosaic.perceiver import cutout
from mosaic.segmenter import BoxSegmenter

CACHE = Path("data/cache")


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def views(path, boxes, w, h):
    """Whole image plus zoomed regions where parts are expected."""
    out = [[0, 0, w, h]]
    if Path(path).name.startswith("humanparts"):
        people = [b for b in boxes if b[4] == "person" and min(b[2] - b[0], b[3] - b[1]) >= 64]
        out += [b[:4] for b in sorted(people, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:3]]
    else:
        limbs = [b for b in boxes if b[4] in ("arm", "leg")]
        if limbs:
            x0, y0 = min(b[0] for b in limbs), min(b[1] for b in limbs)
            x1, y1 = max(b[2] for b in limbs), max(b[3] for b in limbs)
            px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            region = [max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py))]
            if region != [0, 0, w, h]:
                out.append(region)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom-side", type=int, default=768)
    ap.add_argument("--keep", type=float, default=0.5)
    args = ap.parse_args()

    ann = json.loads(Path("data/parts/annotations.json").read_text())
    paths = sorted(p for p in ann if not Path(p).name.startswith("coco"))  # people images only
    encoder, detector, segmenter = ClipEncoder(fp16=True), ObjectnessDetector(), BoxSegmenter()
    meta_path = CACHE / "part_proposals.json"
    embs_path = CACHE / f"part_proposals_{encoder.encoder_id.replace('/', '_')}.npz"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    embs = dict(np.load(embs_path)) if embs_path.exists() else {}
    todo = [p for p in paths if p not in meta]
    print(f"{len(todo)} images", flush=True)

    for n, path in enumerate(todo):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        gt = ann[path]
        meta[path] = []
        vecs = []
        for v, region in enumerate(views(path, gt, w, h)):
            view = img.crop(region)
            scale = max(1.0, args.zoom_side / max(view.size))
            if scale > 1:
                view = view.resize((round(view.width * scale), round(view.height * scale)), Image.BICUBIC)
            dets = detector.detect([view])[0]
            if not dets:
                continue
            masks = segmenter.segment([view], [[d.box for d in dets]])[0]
            crops = [cutout(view, d.box, m, keep=args.keep) for d, m in zip(dets, masks)]
            for d, e in zip(dets, encoder.embed_images(crops)):
                x0, y0, x1, y1 = d.box
                box = [round(region[0] + x0 / scale), round(region[1] + y0 / scale),
                       round(region[0] + x1 / scale), round(region[1] + y1 / scale)]
                best = max(((iou(box, g[:4]), g[4]) for g in gt), default=(0.0, None))
                label = best[1] if best[0] >= 0.5 else ("background" if best[0] < 0.3 else "ambiguous")
                meta[path].append({"view": v, "box": box, "objectness": round(d.score, 3),
                                   "label": label, "iou": round(best[0], 3)})
                vecs.append(e.astype(np.float32))
        if vecs:
            embs[path] = np.stack(vecs)
        if n % 100 == 0 or n == len(todo) - 1:
            print(f"{n + 1}/{len(todo)}", flush=True)
            meta_path.write_text(json.dumps(meta))
            np.savez(embs_path, **embs)


if __name__ == "__main__":
    main()
