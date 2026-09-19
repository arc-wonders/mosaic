"""Download box-annotated images for the classes in parts_classes.py.

Writes data/parts/images/<source>_<n>.jpg (long side <= 1024) and
data/parts/annotations.json {path: [[x0, y0, x1, y1, class], ...]} with every
target-class box in each saved image, so images can also be used to test
finding parts. Stops once every class has --per-class boxes of min side >= --min-side.
"""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

from parts_classes import PART_DOMAINS

OUT = Path("data/parts")
MAX_SIDE = 1024


def wanted(source: str) -> dict[str, str]:
    """source category name -> our class name."""
    return {name: cls for d in PART_DOMAINS.values() for cls, srcs in d.items()
            for src, names in srcs if src == source for name in names}


SOURCES = [  # (name, repo, split, box format)
    ("humanparts", "finedet/humanparts", "train", "xywh"),
    ("coco", "detection-datasets/coco", "train", "xyxy"),
    ("openimages", "finedet/openimages", "validation", "xywh"),
]


def category_names(repo: str) -> list[str]:
    import urllib.request

    info = json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/info?dataset={repo}"))
    objects = info["dataset_info"]["default"]["features"]["objects"]
    category = objects["feature"]["category"] if "feature" in objects else objects["category"][0]
    return category["names"]


def save(img_bytes: bytes, boxes, path: Path):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    s = min(1.0, MAX_SIDE / max(img.size))
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)))
    img.save(path, quality=90)
    return [[round(x0 * s), round(y0 * s), round(x1 * s), round(y1 * s), c] for (x0, y0, x1, y1), c in boxes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--min-side", type=int, default=40)
    ap.add_argument("--max-shards", type=int, default=8, help="per source")
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ann_path = OUT / "annotations.json"
    ann = json.loads(ann_path.read_text()) if ann_path.exists() else {}
    counts = Counter(b[4] for boxes in ann.values() for b in boxes
                     if min(b[2] - b[0], b[3] - b[1]) >= args.min_side)
    fs = HfFileSystem()

    for source, repo, split, fmt in SOURCES:
        want = wanted(source)
        if not want:
            continue
        names = category_names(repo)
        files = sorted(f for f in fs.ls(f"datasets/{repo}/data", detail=False) if f"/{split}-" in f)[: args.max_shards]
        for f in files:
            if all(counts[c] >= args.per_class for c in set(want.values())):
                break
            pf = pq.ParquetFile(fs.open(f))
            for rg in range(pf.metadata.num_row_groups):
                for row in pf.read_row_group(rg, columns=["image", "objects"]).to_pylist():
                    objs = row["objects"]
                    boxes = [(list(b) if fmt == "xyxy" else [b[0], b[1], b[0] + b[2], b[1] + b[3]], want[names[c]])
                             for b, c in zip(objs["bbox"], objs["category"]) if names[c] in want]
                    big = [c for b, c in boxes if min(b[2] - b[0], b[3] - b[1]) >= args.min_side]
                    if not any(counts[c] < args.per_class for c in big):
                        continue
                    path = OUT / "images" / f"{source}_{len(ann):05d}.jpg"
                    ann[str(path)] = save(row["image"]["bytes"], boxes, path)
                    counts.update(big)
                print(Path(f).name, f"rg {rg}", dict(counts), flush=True)
                ann_path.write_text(json.dumps(ann))
                if all(counts[c] >= args.per_class for c in set(want.values())):
                    break
            ann_path.write_text(json.dumps(ann))
    ann_path.write_text(json.dumps(ann))
    print("images:", len(ann), "boxes:", dict(counts))


if __name__ == "__main__":
    main()
