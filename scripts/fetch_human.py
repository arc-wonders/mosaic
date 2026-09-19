"""Download Open Images validation images containing people, keeping *every* annotated box.

Writes data/human/images/oi_<n>.jpg (long side <= 1024) and data/human/annotations.json:
    {path: [[x0, y0, x1, y1, label], ...]}
label is the leaf path for human classes (e.g. "people/limbs/hand") and "oi:<name>" for
everything else (clothing, furniture, ...), so later a proposal matching nothing
annotated can be called background. Stops once every leaf has --per-class boxes of
min side >= --min-side, or after --max-shards shards.
"""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

from fetch_parts import category_names
from human_classes import LEAVES, OI_TO_LEAF

OUT = Path("data/human")
REPO = "finedet/openimages"
MAX_SIDE = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--min-side", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=8)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ann_path = OUT / "annotations.json"
    ann = json.loads(ann_path.read_text()) if ann_path.exists() else {}
    big = lambda b: min(b[2] - b[0], b[3] - b[1]) >= args.min_side
    counts = Counter(b[4] for boxes in ann.values() for b in boxes if b[4] in LEAVES and big(b))
    done_files = {p.split("#")[0] for p in ann.get("_files", [])}
    names = category_names(REPO)
    fs = HfFileSystem()
    files = sorted(f for f in fs.ls(f"datasets/{REPO}/data", detail=False) if "/validation-" in f)[: args.max_shards]

    for f in files:
        if all(counts[c] >= args.per_class for c in LEAVES):
            break
        if f in done_files:
            continue
        pf = pq.ParquetFile(fs.open(f))
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg, columns=["image", "objects"]).to_pylist():
                objs = row["objects"]
                boxes = []
                for (x, y, w, h), c in zip(objs["bbox"], objs["category"]):
                    label = OI_TO_LEAF.get(names[c], f"oi:{names[c]}")
                    boxes.append([x, y, x + w, y + h, label])
                human = [b[4] for b in boxes if b[4] in LEAVES and big(b)]
                if not any(counts[c] < args.per_class for c in human):
                    continue
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                s = min(1.0, MAX_SIDE / max(img.size))
                if s < 1:
                    img = img.resize((round(img.width * s), round(img.height * s)))
                path = OUT / "images" / f"oi_{len(ann):05d}.jpg"
                img.save(path, quality=90)
                ann[str(path)] = [[round(b[0] * s), round(b[1] * s), round(b[2] * s), round(b[3] * s), b[4]] for b in boxes]
                counts.update(human)
            print(Path(f).name, f"rg {rg}", {c.rsplit('/', 1)[1]: counts[c] for c in LEAVES}, flush=True)
            ann_path.write_text(json.dumps(ann))
            if all(counts[c] >= args.per_class for c in LEAVES):
                break
    ann_path.write_text(json.dumps(ann))
    print("images:", len(ann), {c.rsplit('/', 1)[1]: counts[c] for c in LEAVES})


if __name__ == "__main__":
    main()
