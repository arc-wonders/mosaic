"""Save every annotated human instance (min side >= 32 px) as a soft-mask cut-out image.

The crops are what any encoder sees, so encoders can be compared on identical input:
  data/human/crops/<image stem>_<j>.jpg   (longest side 256)
  data/human/crops.json                   {"<path>#<j>": crop path}
"""

import json
from pathlib import Path

from PIL import Image

from human_classes import LEAVES
from mosaic.perceiver import cutout
from mosaic.segmenter import BoxSegmenter

OUT = Path("data/human/crops")


def main():
    ann = json.loads(Path("data/human/annotations.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = Path("data/human/crops.json")
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    segmenter = BoxSegmenter()
    paths = sorted(ann)
    for n, path in enumerate(paths):
        todo = [j for j, b in enumerate(ann[path]) if b[4] in LEAVES and min(b[2] - b[0], b[3] - b[1]) >= 32
                and f"{path}#{j}" not in index]
        if not todo:
            continue
        img = Image.open(path).convert("RGB")
        masks = segmenter.segment([img], [[tuple(ann[path][j][:4]) for j in todo]])[0]
        for j, m in zip(todo, masks):
            crop = cutout(img, tuple(ann[path][j][:4]), m, keep=0.5)
            crop.thumbnail((256, 256))
            out = OUT / f"{Path(path).stem}_{j}.jpg"
            crop.save(out, quality=92)
            index[f"{path}#{j}"] = str(out)
        if n % 200 == 0:
            print(f"{n + 1}/{len(paths)}", flush=True)
            index_path.write_text(json.dumps(index))
    index_path.write_text(json.dumps(index))
    print("crops:", len(index))


if __name__ == "__main__":
    main()
