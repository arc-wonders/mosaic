"""Why did the scripted `count` task fall from 68% (run 1) to 32% (run 2)?

Run 2's only change between runs is `build_negatives.py`'s non-human rejection gate (reject_margin):
identify() now accepts a person/limb only if it beats its nearest counter-example by a margin. If that
gate rejects real people, `count` (which counts `pose()`'s people, which only considers objects whose raw
lookup already says "people/*") would undercount even though the skeleton model itself didn't change.

This reruns the exact run-2 count questions (same seed/n, from bench_ollama.questions) on two module
sets that share one set of frozen models (one GPU load): the pre-negatives backup and the current
(gated) modules, so any count difference is attributable only to the gate.

    PYTHONPATH=scripts python scripts/diag_count_regression.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from bench_ollama import questions  # noqa: E402

from mosaic.capability import VisionSession  # noqa: E402
from mosaic.detector import ObjectnessDetector  # noqa: E402
from mosaic.encoder import siglip  # noqa: E402
from mosaic.segmenter import BoxSegmenter  # noqa: E402

#  questions() draws from one seeded RNG sequentially across presence -> count -> locate, so asking for
#  "count" alone (skipping presence's draws) would consume the RNG differently and silently sample a
#  different, wrong set of images. Always regenerate with the full task list the real run used, then filter.
qs = [q for q in questions(25, seed=0, tasks=["presence", "count", "locate"]) if q["task"] == "count"]
print(f"{len(qs)} count questions (run 2, seed 0)")

encoder, detector, segmenter = siglip(), ObjectnessDetector(), BoxSegmenter()

sessions = {
    "pre_negatives": VisionSession(ROOT / "modules_people_pre_negatives", encoder=encoder, detector=detector,
                                   segmenter=segmenter),
    "current": VisionSession(ROOT / "modules_people", encoder=encoder, detector=detector, segmenter=segmenter),
}

rows = []
for q in qs:
    row = {"image": Path(q["image"]).name, "truth": q["truth"]}
    for name, s in sessions.items():
        iid = s.load_image(q["image"])["image_id"]
        objs = s.find_objects(iid)["objects"]
        idents = [s.identify(o["object_id"]) for o in objs]
        people = s.pose(image_id=iid)["people"]
        row[f"{name}_pred"] = len(people)
        row[f"{name}_raw_person_like"] = sum(1 for k in idents if k["known"] and k["module"].startswith("people/"))
    rows.append(row)
    print(f"  {row['image']:>20s}  truth={row['truth']}  pre_negatives={row['pre_negatives_pred']}  "
          f"current={row['current_pred']}  "
          f"(raw person-like objects: pre={row['pre_negatives_raw_person_like']} cur={row['current_raw_person_like']})")

acc = lambda key: sum(r[key] == r["truth"] for r in rows) / len(rows)
mae = lambda key: sum(abs(r[key] - r["truth"]) for r in rows) / len(rows)
print()
print(f"pre_negatives: exact={acc('pre_negatives_pred'):.0%}  mae={mae('pre_negatives_pred'):.2f}")
print(f"current:       exact={acc('current_pred'):.0%}  mae={mae('current_pred'):.2f}")

out = ROOT / "results/diag_count_regression.json"
out.write_text(json.dumps(rows, indent=2))
print(f"\nwrote {out}")
