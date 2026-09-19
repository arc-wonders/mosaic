"""Segment every cached detection (boxes >= --min-side px) with SAM and embed the masked cut-outs.

Reads data/cache/objects.json (from extract_objects.py). Writes, resumable, keyed by image path:
  data/cache/masks_<min_side>.json               {path: [[x0,y0,x1,y1,objectness,mask_area_fraction], ...]}
  data/cache/masked_<min_side>_k<keep>_<encoder>.npz  {path: (n, D) embeddings of the masked objects}
  data/cache/maskbits_<min_side>.npz             {path#i: packed bool mask cropped to its box}
Images without a large-enough box are skipped (callers fall back to the whole image).
Masks already in maskbits_<min_side>.npz are reused, so changing --keep only re-embeds.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.encoder import ClipEncoder
from mosaic.perceiver import cutout

CACHE = Path("data/cache")


def unpack(packed: np.ndarray, box, size) -> np.ndarray:
    """Inverse of the np.packbits(mask[y0:y1, x0:x1]) stored per object."""
    x0, y0, x1, y1 = box
    inner = np.unpackbits(packed)[: (y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
    mask = np.zeros((size[1], size[0]), bool)
    mask[y0:y1, x0:x1] = inner
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--keep", type=float, default=0.5, help="background kept around the mask (0 = flat cut-out)")
    args = ap.parse_args()

    encoder, segmenter = ClipEncoder(fp16=True), None
    boxes = json.loads((CACHE / "objects.json").read_text())
    meta_path = CACHE / f"masks_{args.min_side}.json"
    embs_path = CACHE / f"masked_{args.min_side}_k{args.keep}_{encoder.encoder_id.replace('/', '_')}.npz"
    bits_path = CACHE / f"maskbits_{args.min_side}.npz"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    embs = dict(np.load(embs_path)) if embs_path.exists() else {}
    bits = dict(np.load(bits_path)) if bits_path.exists() else {}

    todo = {}
    for path, rows in boxes.items():
        keep = [r for r in rows if r[4] >= 0 and min(r[2] - r[0], r[3] - r[1]) >= args.min_side]
        if keep and path not in embs:
            todo[path] = keep
    paths = sorted(todo)
    print(f"{len(paths)} images to segment", flush=True)

    def save():
        meta_path.write_text(json.dumps(meta))
        np.savez(embs_path, **embs)
        np.savez(bits_path, **bits)

    t0 = time.perf_counter()
    for n, i in enumerate(range(0, len(paths), args.batch)):
        chunk = paths[i : i + args.batch]
        images = [Image.open(p).convert("RGB") for p in chunk]
        chunk_boxes = [[tuple(int(v) for v in r[:4]) for r in todo[p]] for p in chunk]
        if all(f"{p}#0" in bits for p in chunk):  # masks already computed: decode them
            masks = [[unpack(bits[f"{p}#{j}"], b, img.size) for j, b in enumerate(bxs)]
                     for p, img, bxs in zip(chunk, images, chunk_boxes)]
        else:
            if segmenter is None:
                from mosaic.segmenter import BoxSegmenter

                segmenter = BoxSegmenter()
            masks = segmenter.segment(images, chunk_boxes)
        crops, owners = [], []
        for p, image, bxs, ms in zip(chunk, images, chunk_boxes, masks):
            w, h = image.size
            meta[p] = []
            for j, (box, mask) in enumerate(zip(bxs, ms)):
                crops.append(cutout(image, box, mask, keep=args.keep))
                owners.append(p)
                meta[p].append([*box, todo[p][j][4], round(float(mask.sum()) / (w * h), 5)])
                x0, y0, x1, y1 = box
                bits[f"{p}#{j}"] = np.packbits(mask[y0:y1, x0:x1])
        out = encoder.embed_images(crops)
        for p in chunk:
            embs[p] = np.stack([e for e, o in zip(out, owners) if o == p]).astype(np.float32)
        if n % 50 == 0 or i + args.batch >= len(paths):
            done = i + len(chunk)
            rate = done / (time.perf_counter() - t0)
            print(f"{done}/{len(paths)}  {rate:.1f} img/s  eta {(len(paths) - done) / rate / 60:.1f} min", flush=True)
            save()
    save()


if __name__ == "__main__":
    main()
