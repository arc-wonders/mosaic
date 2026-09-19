#!/usr/bin/env python3
"""Find every object in an image (without knowing what they are) and match each one.

    python mosaic_cli.py test.jpg [more.jpg ...] [--k 5]
    python mosaic_cli.py test.jpg --save out.jpg           # draw masks, boxes and labels
    python mosaic_cli.py test.jpg --regions objects        # box crops instead of masks
    python mosaic_cli.py test.jpg --without fruits/apple   # plug a module out for this run
    python mosaic_cli.py --tree                            # show the module tree
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from mosaic import Mosaic  # noqa: E402
from mosaic.encoder import ClipEncoder  # noqa: E402

TOP_K = [1, 3]  # domains kept at L1, classes kept at L2
DEFAULT_ROUTE_THRESHOLD = 0.5  # overridden by <modules>/registry.json, then results/results_<regions>.json


def calibrated_route_threshold(regions: str, modules: Path) -> float:
    for f in (modules / "registry.json", ROOT / "results" / f"results_{regions}.json"):
        if f.exists():
            return json.loads(f.read_text()).get("route_threshold", DEFAULT_ROUTE_THRESHOLD)
    return DEFAULT_ROUTE_THRESHOLD


def print_tree(system: Mosaic) -> None:
    reg = system.registry
    for group in reg.children():
        print(group)
        for child in reg.children(group):
            m = reg.get(child)
            print(f"  └ {child.split('/')[-1]:<14} {len(m)} entries, accepts >= {m.accept_threshold:.3f}")


def explain(system: Mosaic, pred, route_t: float) -> str:
    if pred.reason == "router":
        return f"UNKNOWN (no domain scores above {route_t:.3f})"
    if pred.reason == "leaf_threshold":
        threshold = system.registry.get(pred.module).accept_threshold
        return f"UNKNOWN (closest is {pred.module} at {pred.nearest:.3f}; it accepts >= {threshold:.3f})"
    return f"{pred.module}  (similarity {pred.nearest:.3f}, {pred.votes}/{len(pred.matches)} neighbours agree)"


def main():
    ap = argparse.ArgumentParser(description="Detect objects and match each against the MOSAIC module tree.")
    ap.add_argument("images", nargs="*", type=Path, help="image file(s) to check")
    ap.add_argument("-i", "--image", dest="extra", action="append", type=Path, default=[],
                    help="image file (alternative to the positional form)")
    ap.add_argument("--k", type=int, default=5, help="nearest neighbours to vote over")
    ap.add_argument("--route-threshold", type=float, default=None, help="min domain score to consider an object")
    ap.add_argument("--min-objectness", type=float, default=0.1, help="detector confidence cut-off")
    ap.add_argument("--no-leaf-gate", action="store_true", help="never answer UNKNOWN at the class level")
    ap.add_argument("--without", action="append", default=[], metavar="MODULE",
                    help="plug out a class (fruits/apple) or domain (fruits) for this run")
    ap.add_argument("--save", type=Path, help="write an annotated copy of the (last) image here")
    ap.add_argument("--tree", action="store_true", help="print the module tree")
    ap.add_argument("--regions", choices=["masks", "objects"], default="masks",
                    help="masks: SAM cut-outs (default); objects: box crops")
    ap.add_argument("--modules", type=Path, default=None, help="default: modules_<regions>/")
    args = ap.parse_args()
    args.images += args.extra
    if not args.images and not args.tree:
        ap.error("give an image, e.g.  python mosaic_cli.py test.jpg   or   python mosaic_cli.py --image test.jpg")

    from mosaic.detector import ObjectnessDetector

    modules = args.modules or ROOT / f"modules_{args.regions}"
    route_t = args.route_threshold if args.route_threshold is not None else calibrated_route_threshold(args.regions, modules)
    segmenter = None
    if args.regions == "masks":
        from mosaic.segmenter import BoxSegmenter

        segmenter = BoxSegmenter()
    system = Mosaic(ClipEncoder(fp16=True), detector=ObjectnessDetector(min_score=args.min_objectness),
                    segmenter=segmenter, top_k_modules=TOP_K, route_threshold=route_t,
                    leaf_gate=not args.no_leaf_gate)
    if not system.registry.load_dir(modules):
        sys.exit(f"no modules found in {modules}; run scripts/run_experiment.py --regions {args.regions} first")
    for name in args.without:
        print(f"plugged out: {', '.join(system.detach(name))}")
    if args.tree:
        print_tree(system)

    for path in args.images:
        if not path.exists():
            print(f"\n{path}: file not found")
            continue
        image = Image.open(path).convert("RGB")
        results = system.recognize(image, k=args.k)
        print(f"\n{path}: {len(results)} object(s)" + (" (none detected; used whole image)" if results[0][0].fallback else ""))
        for n, (region, pred) in enumerate(results, start=1):
            shape = f"mask {region.area * 100:.0f}% of image" if region.mask is not None else f"box {region.box}"
            print(f"  #{n} {shape}  objectness {region.objectness:.2f}")
            print(f"     -> {explain(system, pred, route_t)}")

        if args.save:
            for region, pred in results:  # tint each mask: green = known, grey = unknown
                if region.mask is not None:
                    tint = Image.new("RGB", image.size, (40, 200, 60) if pred.known else (160, 160, 160))
                    alpha = Image.fromarray((region.mask * 90).astype("uint8"))
                    image = Image.composite(tint, image, alpha)
            draw = ImageDraw.Draw(image)
            for n, (region, pred) in enumerate(results, start=1):
                colour = (40, 200, 60) if pred.known else (160, 160, 160)
                draw.rectangle(region.box, outline=colour, width=2)
                draw.text((region.box[0] + 4, region.box[1] + 2), f"#{n} {pred.label or 'unknown'}", fill=colour)
            image.save(args.save)
            print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
