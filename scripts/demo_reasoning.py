"""A scripted reasoning loop over the MOSAIC capability: "how is each person posed, and what are they holding?"

This plays the part a reasoning model would: every step is a call on
VisionSession (mosaic.capability), with the JSON-able dict it gets back. The
policy is fixed here; a reasoning model would choose the calls itself.

    python scripts/demo_reasoning.py IMAGE [--out results/demo.png]
"""

import argparse
import asyncio
import json
from pathlib import Path

from mosaic.capability import VisionSession

REPO = Path(__file__).resolve().parents[1]
session: VisionSession | None = None


async def call(name, **args):
    content = getattr(session, name)(**args)
    brief = {k: v for k, v in content.items() if k != "candidates"} if isinstance(content, dict) else content
    print(f"\n> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    print("  " + json.dumps(brief, default=str)[:600])
    return content


async def main(image: str, out: str):
    global session
    session = VisionSession(REPO / "modules_people")  # thresholds from modules_people/registry.json
    img = await call("load_image", path=image)
    W, H, iid = img["width"], img["height"], img["image_id"]

    # 1. what is there, who is there, and how are they posed?
    everything = (await call("find_objects", image_id=iid))["objects"]
    people = (await call("pose", image_id=iid))["people"]
    answer = []
    for person in people:
        pid, facts = person["person_id"], person["facts"]
        pose_words = [f for f in ("standing", "sitting", "squatting", "on_all_fours", "hands_up", "arms_crossed",
                                  "hand_near_face", "left_arm_L", "right_arm_L") if facts.get(f) is True]
        answer.append(f"{pid}: facing {facts.get('facing', '?')}; " + (", ".join(pose_words) or "no named pose"))

        # 2. is something in each hand? zoom where the skeleton says the hand is, and also consider
        #    everything found in the whole image (a held object is often bigger than the zoom window)
        for side, (hx, hy) in person["hands_estimated_at"].items():
            r = max(32, int(0.6 * person["body_unit_px"]))
            region = [max(0, hx - r), max(0, hy - r), min(W, hx + r), min(H, hy + r)]
            found = (await call("find_objects", image_id=iid, region=region))["objects"]
            seen = {o["object_id"] for o in found}
            found += [o for o in everything if o["object_id"] not in seen and o["object_id"] != person["object_id"]]
            held = []
            for o in found:
                ident = await call("identify", object_id=o["object_id"])
                if ident["known"] and ident["module"].startswith("people/"):
                    continue  # a body part (the hand itself, an arm)
                touch = await call("measure", person_id=pid, expression=f"touches({side}, {o['object_id']})")
                if touch["value"]:
                    held.append((o["object_id"], ident["module"] or "an object the registry doesn't know"))
            answer.append(f"  {side}: " + ("holding " + "; ".join(f"{w} ({oid})" for oid, w in held)
                                           if held else "nothing found in it"))

    Path(out).write_bytes(session.render(iid))
    print(f"\n  rendered for a human to check -> {out}")
    print("\nANSWER:\n  " + "\n  ".join(answer or ["no person found"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="results/demo.png")
    a = ap.parse_args()
    asyncio.run(main(a.image, a.out))
