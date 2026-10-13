#!/usr/bin/env python3
"""See what the MCP tools would tell a reasoning model about an image, from the terminal.

    python mosaic_cli.py photo.jpg                        # objects, identities, skeletons and pose facts
    python mosaic_cli.py photo.jpg --save out.png         # draw masks, boxes and labels (for you, not the model)
    python mosaic_cli.py photo.jpg --rule "angle(left_hip, left_knee, left_ankle)"   # measure on each person
    python mosaic_cli.py photo.jpg --without people/face_region   # plug a group out for this run
    python mosaic_cli.py --tree                           # the module tree
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mosaic.capability import VisionSession  # noqa: E402


def print_tree(session: VisionSession, node: dict | None = None, depth: int = 0) -> None:
    for child in (node or session.list_modules())["children"]:
        if "group" in child:
            print("  " * depth + child["group"] + "/")
            print_tree(session, child, depth + 1)
        else:
            m = session.mosaic.registry.get(child["module"])
            extra = f", part of {m.part_of}" if m.part_of else ""
            print("  " * depth + f"{child['module'].rsplit('/', 1)[-1]:<10} {child['examples']} examples, "
                  f"accepts >= {m.accept_threshold:.3f}{extra}")


def main():
    ap = argparse.ArgumentParser(description="Objects, identities and pose facts for an image, as the MCP tools see them.")
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--modules", type=Path, default=ROOT / "modules_people")
    ap.add_argument("--taught", type=Path, default=ROOT / "taught", help="modules and rules taught in MCP sessions")
    ap.add_argument("--without", action="append", default=[], metavar="MODULE", help="plug out a module or group")
    ap.add_argument("--rule", action="append", default=[], metavar="EXPR", help="geometry expression to measure")
    ap.add_argument("--save", type=Path, help="write an annotated copy of the (last) image here")
    ap.add_argument("--tree", action="store_true", help="print the module tree")
    args = ap.parse_args()
    if not args.images and not args.tree:
        ap.error("give an image, or --tree")
    if not (args.modules / "registry.json").exists():
        sys.exit(f"no modules in {args.modules}; run scripts/build_people.py first")

    session = VisionSession(args.modules, taught_dir=args.taught)
    for name in args.without:
        print("plugged out:", ", ".join(session.forget(name).get("removed", [])))
    if args.tree:
        print_tree(session)

    for path in args.images:
        if not path.exists():
            print(f"\n{path}: file not found")
            continue
        iid = session.load_image(str(path))["image_id"]
        objects = session.find_objects(iid)["objects"]
        print(f"\n{path}: {len(objects)} object(s)")
        for o in objects:
            r = session.identify(o["object_id"])
            geo = r.get("geometry") or {}
            what = r["module"] if r["known"] else f"unknown ({r['reason']})"
            note = f"  [skeleton changed it from {geo['appearance_alone_said']}]" if geo.get("changed") else ""
            print(f"  {o['object_id']:<7} {str(o['box']):<24} {what}  sim {r['similarity']}{note}")
        for p in session.pose(iid)["people"]:
            facts = p["facts"]
            print(f"  {p['person_id']} (object {p['object_id']}): {len(p['joints'])} joints seen")
            for k, v in facts.items():
                if v is True or (not isinstance(v, bool) and v != "none"):
                    print(f"     {k:<26} {v}")
            if p["constraints_broken"]:
                print(f"     implausible: {', '.join(p['constraints_broken'])}")
            for expr in args.rule:
                m = session.measure(p["person_id"], expr)
                print(f"     {expr} = {m['value']}" + (f"  (not seen: {', '.join(m['missing'])})" if m["missing"] else ""))
        if args.save:
            args.save.write_bytes(session.render(iid))
            print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
