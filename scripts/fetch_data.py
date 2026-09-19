"""Download the images defined in classes.py into data/raw/<domain>/<class>/.

Files are named <source>_<nnnn>.jpg. Classes that already have enough images
from a source are skipped, so the script can be re-run after adding classes.
"""

import argparse
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from PIL import Image

from classes import DOMAINS, OOD, class_dir

IMAGENET = "datasets/benjamin-paine/imagenet-1k-256x256/data"
FRUITS360 = "datasets/PedroSampaio/fruits-360/data"


def wanted_by_source(root: Path) -> dict[str, dict[int, list]]:
    """source -> {label: [(out_dir, remaining_count)]}, skipping completed work."""
    plan = defaultdict(lambda: defaultdict(list))
    for cls, sources in [(c, s) for d in DOMAINS.values() for c, s in d.items()] + list(OOD.items()):
        out = class_dir(root, cls)
        out.mkdir(parents=True, exist_ok=True)
        for source, labels, count in sources:
            have = len(list(out.glob(f"{source}_*.jpg")))
            if have < count:
                plan[source][tuple(labels)].append((out, have, count))
    return plan


def save(img, out: Path, source: str, i: int) -> None:
    img.convert("RGB").save(out / f"{source}_{i:04d}.jpg", quality=92)


def label_col(pf) -> int:
    return [pf.metadata.schema.column(i).path for i in range(pf.metadata.num_columns)].index("label")


def fetch_imagenet(plan) -> None:
    targets = {labels[0]: specs[0] for labels, specs in plan.items()}  # one ImageNet class per entry
    counts = {k: have for k, (_, have, _) in targets.items()}
    fs = HfFileSystem()
    files = sorted(f for f in fs.ls(IMAGENET, detail=False) if "/train-" in f)
    for path in files:  # train shards are sorted by label
        pending = {k for k, (_, _, n) in targets.items() if counts[k] < n}
        if not pending:
            break
        pf = pq.ParquetFile(fs.open(path))
        col = label_col(pf)
        for rg in range(pf.metadata.num_row_groups):
            s = pf.metadata.row_group(rg).column(col).statistics
            lo, hi = (s.min, s.max) if s is not None and s.has_min_max else (0, 1000)
            if not any(lo <= k <= hi for k in pending):
                continue
            for row in pf.read_row_group(rg, columns=["image", "label"]).to_pylist():
                k = row["label"]
                if k in pending and counts[k] < targets[k][2]:
                    save(Image.open(io.BytesIO(row["image"]["bytes"])), targets[k][0], "imagenet", counts[k])
                    counts[k] += 1
        print(path.rsplit("/", 1)[1], {targets[k][0].name: counts[k] for k in targets}, flush=True)


def fetch_fruits360(plan) -> None:
    fs = HfFileSystem()
    [path] = [f for f in fs.ls(FRUITS360, detail=False) if "/train-" in f]
    table = pq.read_table(fs.open(path), columns=["image", "label"])
    labels = np.array(table.column("label").to_pylist())
    images = table.column("image")
    rng = np.random.default_rng(0)
    for group, [(out, have, count)] in plan.items():
        # spread picks evenly over varieties (e.g. 8 apple kinds)
        per = [rng.permutation(np.flatnonzero(labels == l)) for l in group]
        picks = [idx[i] for i in range(max(map(len, per))) for idx in per if i < len(idx)][have:count]
        for i, row in enumerate(picks, start=have):
            save(Image.open(io.BytesIO(images[int(row)].as_py()["bytes"])), out, "fruits360", i)
        print("fruits360", out.name, count)


def fetch_cats_dogs(plan) -> None:
    targets = {labels[0]: specs[0] for labels, specs in plan.items()}
    counts = {k: have for k, (_, have, _) in targets.items()}
    ds = load_dataset("microsoft/cats_vs_dogs", split="train", streaming=True).shuffle(seed=0, buffer_size=5000)
    for row in ds:
        k = row["labels"]
        if k in targets and counts[k] < targets[k][2]:
            save(row["image"], targets[k][0], "cats_vs_dogs", counts[k])
            counts[k] += 1
        if all(counts[k] >= targets[k][2] for k in targets):
            break
    print("cats_vs_dogs", {targets[k][0].name: counts[k] for k in targets})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    root = Path(ap.parse_args().root)
    plan = wanted_by_source(root)
    for source, fetch in [("imagenet", fetch_imagenet), ("fruits360", fetch_fruits360), ("cats_vs_dogs", fetch_cats_dogs)]:
        if plan.get(source):
            fetch(plan[source])
