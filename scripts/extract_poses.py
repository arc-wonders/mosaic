"""Run the skeleton sensor (ViTPose) on every annotated person in the human photos.

    data/human/poses.json   {"<image path>": [{"box": [x0, y0, x1, y1], "keypoints": [[x, y, conf] x 17]}, ...]}

The person boxes are Open Images ground truth, so the geometry experiments measure
what the skeleton adds given a person box, not how well persons are detected.
"""

import json
from pathlib import Path

from PIL import Image

from mosaic.pose import PoseEstimator

OUT = Path("data/human/poses.json")


def main():
    ann = json.loads(Path("data/human/annotations.json").read_text())
    poses = json.loads(OUT.read_text()) if OUT.exists() else {}
    estimator = PoseEstimator()
    todo = [p for p in sorted(ann) if p not in poses]
    for n, path in enumerate(todo):
        boxes = [b[:4] for b in ann[path] if b[4] == "people/person"]
        kps = estimator.estimate(Image.open(path).convert("RGB"), boxes) if boxes else []
        poses[path] = [{"box": b, "keypoints": [[round(float(v), 2) for v in row] for row in k]}
                       for b, k in zip(boxes, kps)]
        if n % 200 == 0:
            print(f"{n + 1}/{len(todo)}", flush=True)
            OUT.write_text(json.dumps(poses))
    OUT.write_text(json.dumps(poses))
    print("images:", len(poses), "people:", sum(len(v) for v in poses.values()))


if __name__ == "__main__":
    main()
