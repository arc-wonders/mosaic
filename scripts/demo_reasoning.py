"""A scripted reasoning loop over the MOSAIC capability: "is the person holding something?"

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


def expand(box, factor, w, h):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) * factor / 2, (box[3] - box[1]) * factor / 2
    return [max(0, int(cx - hw)), max(0, int(cy - hh)), min(w, int(cx + hw)), min(h, int(cy + hh))]


async def call(name, **args):
    content = getattr(session, name)(**args)
    brief = {k: v for k, v in content.items() if k != "candidates"} if isinstance(content, dict) else content
    print(f"\n> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    print("  " + json.dumps(brief, default=str)[:600])
    return content


async def main(image: str, out: str):
    global session
    modules = REPO / "modules_masks"
    session = VisionSession(modules, route_threshold=json.loads((modules / "registry.json").read_text())["route_threshold"])
    img = await call("load_image", path=image)
    W, H, iid = img["width"], img["height"], img["image_id"]
    await call("list_modules", path="people")

    # 1. what is in the picture?
    objects = (await call("find_objects", image_id=iid))["objects"]
    people, known_handheld, identities = [], [], {}
    for o in objects:
        ident = await call("identify", object_id=o["object_id"])
        identities[o["object_id"]] = ident["module"]
        if ident["known"] and ident["module"] == "people/person":
            people.append(o)
        if ident["known"] and (ident["module"] or "").startswith("handheld/"):
            known_handheld.append(o["object_id"])
    if not people:  # nothing confidently a person: look at the largest object
        people = sorted(objects, key=lambda o: -o["area_fraction"])[:1]
        print("\n  (no confident person; inspecting the largest object)")

    answer = []
    for person in people:
        # 2. where are its hands?
        parts = (await call("find_parts", object_id=person["object_id"]))["parts"]
        print(f"\n  person {person['object_id']} parts: "
              + ", ".join(f"{p['module'] or 'unknown'}@{p['box']}" for p in parts))
        hands = [p for p in parts if p["module"] == "people/hand"]
        if not hands:  # the detector may have merged a hand into what it holds: ask the registry directly
            found = (await call("search", image_id=iid, query="people/hand", region=person["box"]))["results"]
            hands = [h for h in found if h["match"]]
        if not hands:
            answer.append(f"{person['object_id']}: no hand found")

        # 3. is something in each hand?
        for hand in hands:
            held = []
            for oid in known_handheld:  # anything already identified as handheld touching the hand?
                rel = await call("relate", object_a=hand["object_id"], object_b=oid)
                if rel["touching"]:
                    held.append((oid, rel))
            if not held:  # otherwise ask the registry to look around the hand
                region = expand(hand["box"], 4, W, H)
                hits = (await call("search", image_id=iid, query="handheld", region=region))["results"]
                for h in [h for h in hits if h["match"]][:1]:
                    rel = await call("relate", object_a=hand["object_id"], object_b=h["object_id"])
                    held.append((h["object_id"], rel))
            for oid, rel in held[:1]:
                what = identities.get(oid) or (await call("identify", object_id=oid))["module"]
                verdict = "touching" if rel["touching"] else "near"
                answer.append(f"hand {hand['object_id']} at {hand['box']} is {verdict} {what} ({oid}), "
                              f"overlapping {rel['overlap_fraction_of_a']:.0%} of the hand")
            if not held:
                answer.append(f"hand {hand['object_id']} at {hand['box']}: nothing from 'handheld' found touching it")

    Path(out).write_bytes(session.render(iid))
    print(f"\n  rendered -> {out}")
    print("\nANSWER:\n  " + "\n  ".join(answer or ["no person found"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="results/demo.png")
    a = ap.parse_args()
    asyncio.run(main(a.image, a.out))
