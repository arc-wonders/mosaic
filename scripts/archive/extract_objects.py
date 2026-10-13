"""Detect every object in every image under data/raw and embed each object crop.

Writes (resumable, keyed by image path):
  data/cache/objects.json                       {path: [[x0,y0,x1,y1,objectness], ...]}
  data/cache/objects_<encoder_id>.npz           {path: (n_objects, D) crop embeddings}
Images with no detected object get one whole-image entry with objectness -1.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.detector import ObjectnessDetector
from mosaic.encoder import ClipEncoder
from mosaic.perceiver import Perceiver

CACHE = Path("data/cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    encoder = ClipEncoder(fp16=True)
    perceiver = Perceiver(encoder, ObjectnessDetector())
    boxes_path = CACHE / "objects.json"
    embs_path = CACHE / f"objects_{encoder.encoder_id.replace('/', '_')}.npz"
    boxes = json.loads(boxes_path.read_text()) if boxes_path.exists() else {}
    embs = dict(np.load(embs_path)) if embs_path.exists() else {}

    paths = [str(p) for p in sorted(Path(args.data).rglob("*.jpg")) if str(p) not in embs]
    print(f"{len(paths)} images to process", flush=True)
    t0 = time.perf_counter()
    for n, i in enumerate(range(0, len(paths), args.batch)):
        chunk = paths[i : i + args.batch]
        regions = perceiver.perceive_batch([Image.open(p).convert("RGB") for p in chunk])
        for p, rs in zip(chunk, regions):
            boxes[p] = [[*r.box, -1.0 if r.fallback else round(r.objectness, 4)] for r in rs]
            embs[p] = np.stack([r.embedding for r in rs]).astype(np.float32)
        if n % 50 == 0 or i + args.batch >= len(paths):
            done = i + len(chunk)
            rate = done / (time.perf_counter() - t0)
            print(f"{done}/{len(paths)}  {rate:.1f} img/s  eta {(len(paths) - done) / rate / 60:.1f} min", flush=True)
            boxes_path.write_text(json.dumps(boxes))
            np.savez(embs_path, **embs)
    boxes_path.write_text(json.dumps(boxes))
    np.savez(embs_path, **embs)


if __name__ == "__main__":
    main()
