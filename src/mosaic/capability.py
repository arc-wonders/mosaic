"""MOSAIC as the eyes of a text-only reasoning model.

The reasoning model never sees pixels. It looks at a scene step by step with
these calls and reasons over the structured answers, all in original-image
pixel coordinates [x0, y0, x1, y1]:

    load_image(path)                       -> image id and size
    find_objects(image, region=None)       -> class-agnostic objects (box, mask area, id).
                                              With a region, the region is cropped and upscaled
                                              first ("zoom"), so small things (a hand) are found.
    identify(object | image+box)           -> registry lookup: module path, known/unknown,
                                              similarity vs the module's own acceptance threshold,
                                              runner-up candidates, and what it is a part of
    find_parts(object)                     -> objects inside an object (zoomed), each identified
    search(image, query, region=None)      -> objects matching a module path ("handheld/phone",
                                              "handheld") or free text, optionally within a region
    relate(object_a, object_b)             -> mask overlap, containment, touching, direction, distance
    list_modules(path="")                  -> what the registry knows
    pose(image | object)                   -> people's skeletons (joints) and the facts the geometry
                                              rules derive from them (arm raised, elbow angle, sitting ...)
    measure(person, expression)            -> evaluate any geometry expression on a skeleton
    define_rule / list_rules               -> add a named geometry concept at run time
    teach(label, objects)                  -> plug in a new module from objects in the conversation
    forget(name)                           -> plug a module out (a taught one is also deleted from disk)
    explain(object)                        -> why identify answered what it did: the routing trace,
                                              the gate, the stored examples it matched, the geometry
    render(image, objects)                 -> annotated PNG, for a human checking the session (not an
                                              MCP tool: the reasoning model works from text)

Objects keep their ids across calls: a new detection that overlaps an existing
object in the same image (IoU >= 0.85) reuses that object's id.
"""

from __future__ import annotations

import io
import itertools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mosaic.geometry import DEFAULT_RULES, RuleBook, Skeleton, owner, part_features, signature_loglik
from mosaic.module import KnowledgeModule
from mosaic.perceiver import NEUTRAL, cutout
from mosaic.pipeline import Mosaic

BoxT = list[int]


@dataclass
class _Image:
    id: str
    path: str
    image: Any
    objects: list[str] = field(default_factory=list)


@dataclass
class _Object:
    id: str
    image_id: str
    box: BoxT                 # detector box, original-image pixels
    mask: np.ndarray | None   # (H, W) bool in original-image pixels
    objectness: float
    embedding: np.ndarray
    found_in: BoxT | None     # the region it was found in (None = whole image)
    identity: dict | None = None  # identify() answer, geometry included
    lookup: dict | None = None    # embedding-only answer (what person boxes for pose come from)
    geometry: dict | None = None  # how the skeleton changed or confirmed the answer

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    def distance_to(self, point) -> float:
        """Pixels from a point to the object's mask (0 if on it); the box stands in without a mask."""
        x, y = float(point[0]), float(point[1])
        if self.mask is None:
            dx = max(self.box[0] - x, 0.0, x - self.box[2])
            dy = max(self.box[1] - y, 0.0, y - self.box[3])
            return float(np.hypot(dx, dy))
        h, w = self.mask.shape
        if 0 <= int(y) < h and 0 <= int(x) < w and self.mask[int(y), int(x)]:
            return 0.0
        ys, xs = np.nonzero(self.mask)
        if not len(xs):
            return float("inf")
        return float(np.sqrt(((xs - x) ** 2 + (ys - y) ** 2).min()))

    @property
    def mask_box(self) -> BoxT:
        if self.mask is None or not self.mask.any():
            return self.box
        ys, xs = np.nonzero(self.mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _iou(a: BoxT, b: BoxT) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _clip(box, w, h) -> BoxT:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return [max(0, min(w, x0)), max(0, min(h, y0)), max(0, min(w, x1)), max(0, min(h, y1))]


@dataclass
class _Person:
    id: str
    image_id: str
    object_id: str | None  # the person object the skeleton was estimated in (None: the whole image)
    skeleton: Skeleton


class VisionSession:
    def __init__(
        self,
        modules: str | Path | list[str | Path],
        route_threshold: float | None = None,
        top_k: list[int] | None = None,
        k: int = 5,
        zoom_side: int = 768,
        background_keep: float = 0.5,
        use_masks: bool = True,
        device: str | None = None,
        encoder=None,
        detector=None,
        segmenter=None,
        pose_estimator=None,
        geometry_weight: float | None = None,
        taught_dir: str | Path | None = None,
        rules: str | Path | None = DEFAULT_RULES,
    ):
        """encoder / detector / segmenter / pose_estimator default to SigLIP SO400M, OWLv2 objectness, SAM and
        ViTPose; pass your own to swap them. route_threshold, top_k and geometry_weight default to the values
        calibrated in the first module directory's registry.json. Modules taught in the session (and rules
        defined in it) are saved to taught_dir, and loaded from it on start."""
        dirs = [Path(m) for m in ([modules] if isinstance(modules, (str, Path)) else modules)]
        config_path = dirs[0] / "registry.json" if dirs else None
        config = json.loads(config_path.read_text()) if config_path and config_path.exists() else {}
        route_threshold = config.get("route_threshold", 0.5) if route_threshold is None else route_threshold
        top_k = list(config.get("top_k", [1, 3]) if top_k is None else top_k)
        self.geometry_weight = config.get("geometry_weight", 0.0) if geometry_weight is None else geometry_weight
        if encoder is None:
            from mosaic.encoder import siglip

            encoder = siglip(device)
        if detector is None:
            from mosaic.detector import ObjectnessDetector

            detector = ObjectnessDetector(device=device)
        if segmenter is None and use_masks:
            from mosaic.segmenter import BoxSegmenter

            segmenter = BoxSegmenter(device=device)
        self.encoder, self.detector, self.segmenter = encoder, detector, segmenter
        self._pose_estimator, self.device = pose_estimator, device
        self.mosaic = Mosaic(self.encoder, top_k_modules=top_k, route_threshold=route_threshold)
        self.taught_dir = Path(taught_dir) if taught_dir else None
        for root in dirs:
            self.mosaic.registry.load_dir(root)
        self._taught_modules: set[str] = set()
        if self.taught_dir and self.taught_dir.exists():
            self._taught_modules.update(self.mosaic.registry.load_dir(self.taught_dir))
        self.rules = RuleBook.load(*[p for p in [rules] if p])
        self._taught_rules: set[str] = set()
        if self.taught_dir and (self.taught_dir / "rules.json").exists():
            taught = RuleBook.load(self.taught_dir / "rules.json")
            for name, r in taught.rules.items():
                self.rules.add(name, r["expression"], r["description"], r["kind"], check=False)
                self._taught_rules.add(name)
        self.k, self.zoom_side, self.background_keep = k, zoom_side, background_keep
        self.images: dict[str, _Image] = {}
        self.objects: dict[str, _Object] = {}
        self.people: dict[str, _Person] = {}
        self._posed: dict[str, list[str]] = {}  # image id -> person ids, once pose has run on it
        self._ids = itertools.count(1)
        self._person_ids = itertools.count(1)

    @property
    def pose_estimator(self):
        if self._pose_estimator is None:
            from mosaic.pose import PoseEstimator

            self._pose_estimator = PoseEstimator(device=self.device)
        return self._pose_estimator

    # ---- images ------------------------------------------------------------------
    def load_image(self, path: str) -> dict:
        image = Image.open(path).convert("RGB")
        image_id = f"img_{len(self.images) + 1}"
        self.images[image_id] = _Image(image_id, str(path), image)
        return {"image_id": image_id, "width": image.width, "height": image.height}

    # ---- perception ----------------------------------------------------------------
    def find_objects(self, image_id: str, region: BoxT | None = None, max_objects: int = 8) -> dict:
        """Class-agnostic objects in the image, or zoomed into `region`."""
        img = self.images[image_id]
        W, H = img.image.size
        region = _clip(region, W, H) if region is not None else [0, 0, W, H]
        rx0, ry0, rx1, ry1 = region
        if rx1 - rx0 < 4 or ry1 - ry0 < 4:
            return {"image_id": image_id, "region": region, "objects": [], "note": "region too small"}
        view = img.image.crop(region)
        scale = max(1.0, self.zoom_side / max(view.size))  # upscale small regions ("zoom in")
        if scale > 1:
            view = view.resize((round(view.width * scale), round(view.height * scale)), Image.BICUBIC)
        detections = self.detector.detect([view])[0][:max_objects]
        if not detections:
            return {"image_id": image_id, "region": region, "zoom": round(scale, 2), "objects": []}
        masks = (self.segmenter.segment([view], [[d.box for d in detections]])[0]
                 if self.segmenter else [None] * len(detections))
        crops = [cutout(view, d.box, m, keep=self.background_keep) if m is not None else view.crop(d.box)
                 for d, m in zip(detections, masks)]
        embeddings = self.encoder.embed_images(crops)

        found = []
        for d, m, e in zip(detections, masks, embeddings):
            x0, y0, x1, y1 = d.box
            box = _clip([rx0 + x0 / scale, ry0 + y0 / scale, rx0 + x1 / scale, ry0 + y1 / scale], W, H)
            full_mask = None
            if m is not None:
                small = Image.fromarray(m.astype(np.uint8) * 255).resize((rx1 - rx0, ry1 - ry0), Image.NEAREST)
                full_mask = np.zeros((H, W), bool)
                full_mask[ry0:ry1, rx0:rx1] = np.asarray(small) > 127
            whole_image = region == [0, 0, W, H]
            clipped = not whole_image and (box[0] <= rx0 + 1 or box[1] <= ry0 + 1 or box[2] >= rx1 - 1 or box[3] >= ry1 - 1)
            obj = self._register(image_id, box, full_mask, d.score, e, None if whole_image else region, clipped)
            found.append(self._summary(obj))
        return {"image_id": image_id, "region": region, "zoom": round(scale, 2), "objects": found}

    def _register(self, image_id, box, mask, objectness, embedding, found_in, clipped=False) -> _Object:
        for oid in self.images[image_id].objects:  # same thing seen again: keep its id
            existing = self.objects[oid]
            if _iou(existing.box, box) >= 0.85:
                # prefer the zoomed (sharper) view, but never one where the region cut the object off
                if found_in is not None and existing.found_in is None and not clipped:
                    existing.box, existing.mask, existing.embedding, existing.found_in = box, mask, embedding, found_in
                    existing.identity = existing.lookup = existing.geometry = None
                return existing
        obj = _Object(f"obj_{next(self._ids)}", image_id, box, mask, float(objectness), embedding, found_in)
        self.objects[obj.id] = obj
        self.images[image_id].objects.append(obj.id)
        return obj

    def _summary(self, obj: _Object) -> dict:
        W, H = self.images[obj.image_id].image.size
        area = float(obj.mask.sum()) if obj.mask is not None else (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1])
        return {"object_id": obj.id, "box": obj.box, "mask_box": obj.mask_box,
                "area_fraction": round(area / (W * H), 4), "objectness": round(obj.objectness, 3)}

    # ---- identification -------------------------------------------------------------
    def identify(self, object_id: str | None = None, image_id: str | None = None, box: BoxT | None = None) -> dict:
        """What is this? Give an object id, or an image id plus a box."""
        if object_id is None:
            if image_id is None or box is None:
                raise ValueError("identify needs object_id, or image_id and box")
            obj = self._object_at(image_id, box)
        else:
            obj = self.objects[object_id]
        if obj.identity is None:
            if obj.lookup is None:
                obj.lookup = self._lookup(obj.embedding)
            obj.identity = self._apply_geometry(obj, dict(obj.lookup))
        return {**self._summary(obj), **obj.identity}

    def _apply_geometry(self, obj: _Object, ident: dict) -> dict:
        """Re-rank body-part leaves by where the object sits on its person's skeleton.

        Only leaves with an anatomical signature take part, and only those that would accept the object
        on appearance alone (plus the embedding's own answer), so geometry picks between plausible parts
        (head vs face, arm vs hand) but never makes an unknown object known."""
        reg = self.mosaic.registry
        if not self.geometry_weight or not ident["known"] or not getattr(reg.get(ident["module"]), "geometry", None):
            return ident
        person = self._owner(obj)
        if person is None:
            why = "no skeleton covers this object"
            if self._posed.get(obj.image_id) and getattr(reg.get(ident["module"]), "part_of", None):
                why = "no person's skeleton covers it: a body part outside every person is suspect"
            obj.geometry = {"applied": False, "why": why}
            ident["geometry"] = obj.geometry
            return ident
        f = part_features(obj.box, person.skeleton)
        scored = []
        for name in reg.leaves():
            m = reg.get(name)
            if not getattr(m, "geometry", None):
                continue
            sims = m.embeddings[m._kind_masks["image"]] @ obj.embedding
            top3 = float(np.sort(sims)[-3:].mean())
            nearest = float(sims.max())
            accepts = m.accepts(obj.embedding, nearest)
            ll = signature_loglik(m.geometry, f)
            scored.append({"module": name, "score": top3 + self.geometry_weight * ll, "nearest": nearest,
                           "eligible": accepts or name == ident["module"], "fit": ll})
        best = max((c for c in scored if c["eligible"]), key=lambda c: c["score"])
        fits = sorted(scored, key=lambda c: -c["fit"])[:3]
        obj.geometry = {
            "applied": True, "person_id": person.id,
            "joints_inside": {g: c for g, c in f["counts"].items() if c},
            "size_vs_person": round(float(np.exp(f["log_size"])), 3),
            "height_on_person": round(f["rel_y"], 3),
            "best_anatomical_fits": [{"module": c["module"], "log_likelihood": round(c["fit"], 1)} for c in fits],
            "changed": best["module"] != ident["module"],
        }
        if obj.geometry["changed"]:
            obj.geometry["appearance_alone_said"] = ident["module"]
            leaf = reg.get(best["module"])
            ident.update(module=best["module"], label=leaf.entries[0]["label"], similarity=round(best["nearest"], 3),
                         part_of=leaf.part_of, votes=None,
                         reason="matched a registered module; the skeleton chose between look-alike parts")
        ident["geometry"] = obj.geometry
        return ident

    def _object_at(self, image_id: str, box: BoxT) -> _Object:
        """An object for an arbitrary box: segment it (zoomed) and embed it."""
        img = self.images[image_id]
        W, H = img.image.size
        box = _clip(box, W, H)
        for oid in img.objects:
            if _iou(self.objects[oid].box, box) >= 0.85:
                return self.objects[oid]
        pad = [box[0] - 0.1 * (box[2] - box[0]), box[1] - 0.1 * (box[3] - box[1]),
               box[2] + 0.1 * (box[2] - box[0]), box[3] + 0.1 * (box[3] - box[1])]
        region = _clip(pad, W, H)
        view = img.image.crop(region)
        scale = max(1.0, self.zoom_side / max(view.size))
        view = view.resize((round(view.width * scale), round(view.height * scale)), Image.BICUBIC)
        local = [int((box[0] - region[0]) * scale), int((box[1] - region[1]) * scale),
                 int((box[2] - region[0]) * scale), int((box[3] - region[1]) * scale)]
        m = self.segmenter.segment([view], [[local]])[0][0] if self.segmenter else None
        crop = cutout(view, local, m, keep=self.background_keep) if m is not None else view.crop(local)
        full_mask = None
        if m is not None:
            small = Image.fromarray(m.astype(np.uint8) * 255).resize((region[2] - region[0], region[3] - region[1]))
            full_mask = np.zeros((H, W), bool)
            full_mask[region[1]:region[3], region[0]:region[2]] = np.asarray(small) > 127
        return self._register(image_id, box, full_mask, 0.0, self.encoder.embed_images([crop])[0], region)

    def _lookup(self, q: np.ndarray) -> dict:
        pred = self.mosaic.classify(q, self.k)
        reg = self.mosaic.registry
        # runner-ups: every leaf's closest stored example, best first
        scores = []
        for name in reg.leaves():
            m = reg.get(name)
            hits = m.search(q, 1, kind="image")
            if hits:
                scores.append((name, hits[0][1], m.accept_threshold, m.accepts(q, hits[0][1])))
        scores.sort(key=lambda s: -s[1])
        candidates = [{"module": n, "similarity": round(s, 3), "accepts_from": round(t, 3) if t else None,
                       "would_accept": a} for n, s, t, a in scores[:3]]
        module = pred.module or (candidates[0]["module"] if candidates else None)
        leaf = reg.get(module) if module else None
        return {
            "label": pred.label,
            "module": pred.module if pred.known else None,
            "known": pred.known,
            "reason": {"known": "matched a registered module",
                       "router": "no domain in the registry is close enough",
                       "match_threshold": "nothing stored is close enough",
                       "leaf_threshold": f"closest module {pred.module} does not consider it typical enough",
                       "negative_margin": f"closer to {pred.module}'s stored counter-examples than to the real thing"}[pred.reason],
            "similarity": round(float(pred.nearest), 3) if np.isfinite(pred.nearest) else None,
            "votes": f"{pred.votes}/{len(pred.matches)}" if pred.matches else None,
            "part_of": getattr(leaf, "part_of", None) if pred.known else None,
            "candidates": candidates,
        }

    def find_parts(self, object_id: str, max_parts: int = 8) -> dict:
        """Objects inside `object_id` (zoomed in), each identified."""
        whole = self.objects[object_id]
        result = self.find_objects(whole.image_id, region=whole.box, max_objects=max_parts + 1)
        parts = []
        for o in result["objects"]:
            if o["object_id"] == object_id or _iou(o["box"], whole.box) >= 0.7:
                continue  # the object itself, re-detected
            parts.append(self.identify(o["object_id"]))
        return {"object_id": object_id, "box": whole.box, "zoom": result.get("zoom"), "parts": parts}

    def search(self, image_id: str, query: str, region: BoxT | None = None, scan: bool = True) -> dict:
        """Objects matching a registry module path ("handheld/phone", "handheld") or free text.

        Two sources of candidates: objects the detector finds in the region, and (for module queries,
        scan=True) a window scan of the region scored directly against the queried modules, which
        finds things the detector merged into something else (a hand holding a phone) or missed.
        Every result is re-identified, so "match" means the registry itself agrees.
        """
        reg = self.mosaic.registry
        if query not in reg:
            return self._search_text(image_id, query, region)
        leaves = reg.leaves(query)
        detected = self.find_objects(image_id, region)["objects"]
        candidates = {o["object_id"] for o in detected}
        if scan:
            for box in self._scan(image_id, leaves, region):
                candidates.add(self._object_at(image_id, box).id)
        hits = []
        for oid in candidates:
            ident = self.identify(oid)
            best_sim, best_leaf = max((reg.get(n).search(self.objects[oid].embedding, 1, kind="image")[0][1], n)
                                      for n in leaves)
            hits.append({**self._summary(self.objects[oid]), "module": best_leaf, "similarity": round(best_sim, 3),
                         "identified_as": ident["module"],
                         "match": bool(ident["known"] and (ident["module"] == query
                                                           or (ident["module"] or "").startswith(query + "/")))})
        hits.sort(key=lambda h: (-h["match"], -h["similarity"]))
        return {"image_id": image_id, "query": query, "mode": "module", "region": region, "results": hits}

    def _scan(self, image_id: str, leaves: list[str], region: BoxT | None, top: int = 5) -> list[BoxT]:
        """Slide square windows (4 scales, half-window stride) over the region and return the boxes where a
        queried leaf is the registry's best answer and clears its own acceptance threshold, ranked by how
        far it beats every other leaf. (Ranking by raw similarity alone favours skin, sky or clothing,
        which resemble many modules at once.)"""
        img = self.images[image_id].image
        W, H = img.size
        rx0, ry0, rx1, ry1 = _clip(region, W, H) if region else [0, 0, W, H]
        side = max(rx1 - rx0, ry1 - ry0)
        boxes = []
        for frac in (0.1, 0.15, 0.25, 0.4):
            s_ = max(16, int(frac * side))
            step = max(8, s_ // 2)
            for y in range(ry0, max(ry0 + 1, ry1 - s_ + 1), step):
                for x in range(rx0, max(rx0 + 1, rx1 - s_ + 1), step):
                    boxes.append([x, y, min(W, x + s_), min(H, y + s_)])
        if not boxes:
            return []
        from mosaic.perceiver import crop

        embs = np.concatenate([self.encoder.embed_images([crop(img, b) for b in boxes[i : i + 32]])
                               for i in range(0, len(boxes), 32)])  # 32: SigLIP's activations stay < 0.5 GB
        reg = self.mosaic.registry
        names = reg.leaves()
        sims = np.stack([(embs @ reg.get(n).embeddings[reg.get(n)._kind_masks["image"]].T).max(axis=1)
                         for n in names], axis=1)  # (windows, leaves): nearest stored example per leaf
        accept = np.array([reg.get(n).accept_threshold or -1.0 for n in names])
        target = np.array([n in leaves for n in names])
        best_target = np.where(target, sims, -np.inf).max(axis=1)
        best_other = np.where(~target, sims, -np.inf).max(axis=1) if (~target).any() else np.full(len(boxes), -1.0)
        target_ok = (np.where(target, sims - accept, -np.inf) >= 0).any(axis=1)
        lead = best_target - best_other
        order = [i for i in np.argsort(-lead) if lead[i] > 0 and target_ok[i]]
        kept: list[BoxT] = []
        for i in order:  # greedy non-maximum suppression
            if all(_iou(boxes[i], k) < 0.3 for k in kept):
                kept.append(boxes[i])
            if len(kept) == top:
                break
        return kept

    def _search_text(self, image_id: str, query: str, region: BoxT | None) -> dict:
        found = self.find_objects(image_id, region)["objects"]
        t = self.encoder.embed_texts([f"a photo of a {query}"])[0]
        sims = np.array([float(self.objects[o["object_id"]].embedding @ t) for o in found])
        prob = getattr(self.encoder, "text_match_probability", lambda s: None)(sims)
        hits = []
        for i, o in enumerate(found):
            hit = {**o, "similarity": round(float(sims[i]), 3)}
            if prob is not None:  # SigLIP: a calibrated probability that the object matches the text
                hit["probability"] = round(float(prob[i]), 3)
                hit["match"] = bool(prob[i] >= 0.5)
            else:
                hit["match"] = bool(sims[i] >= 0.25 and sims[i] == sims.max())
            hits.append(hit)
        hits.sort(key=lambda h: -h["similarity"])
        return {"image_id": image_id, "query": query, "mode": "text (encoder text tower; weaker than a registry module)",
                "region": region, "results": hits}

    # ---- geometry ------------------------------------------------------------------
    def relate(self, object_a: str, object_b: str) -> dict:
        a, b = self.objects[object_a], self.objects[object_b]
        W, H = self.images[a.image_id].image.size
        ma = a.mask if a.mask is not None else _box_mask(a.box, W, H)
        mb = b.mask if b.mask is not None else _box_mask(b.box, W, H)
        inter = float((ma & mb).sum())
        grow = max(3, int(0.02 * max(W, H)))
        x0, y0 = max(0, a.box[0] - grow), max(0, a.box[1] - grow)  # dilate only around a
        x1, y1 = min(W, a.box[2] + grow), min(H, a.box[3] + grow)
        touching = inter > 0 or bool((_dilate(ma[y0:y1, x0:x1], grow) & mb[y0:y1, x0:x1]).any())
        ca = np.array([(a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2])
        cb = np.array([(b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2])
        dx, dy = cb - ca
        direction = []
        if abs(dx) > 0.1 * W:
            direction.append("right of" if dx > 0 else "left of")
        if abs(dy) > 0.1 * H:
            direction.append("below" if dy > 0 else "above")
        return {
            "a": object_a, "b": object_b,
            "overlap_fraction_of_a": round(inter / max(ma.sum(), 1), 3),
            "overlap_fraction_of_b": round(inter / max(mb.sum(), 1), 3),
            "b_inside_a": inter / max(mb.sum(), 1) > 0.8,
            "a_inside_b": inter / max(ma.sum(), 1) > 0.8,
            "touching": touching,
            "b_is": " and ".join(direction) + " a" if direction else "at about the same place as a",
            "center_distance_px": round(float(np.hypot(dx, dy)), 1),
            "center_distance_image_fraction": round(float(np.hypot(dx / W, dy / H)), 3),
            "uses_masks": a.mask is not None and b.mask is not None,
            "touch_tolerance_px": grow,
        }

    # ---- registry --------------------------------------------------------------------
    def list_modules(self, path: str = "") -> dict:
        reg = self.mosaic.registry

        def node(n):
            if reg.is_group(n):
                return {"group": n, "children": [node(c) for c in reg.children(n)]}
            m = reg.get(n)
            out = {"module": n, "examples": int(m._kind_masks.get("image", np.zeros(0)).sum())}
            if getattr(m, "part_of", None):
                out["part_of"] = m.part_of
            if n in self._taught_modules:
                out["taught"] = True
            return out

        return {"path": path or "/", "children": [node(c) for c in reg.children(path)]}

    # ---- skeletons and geometry rules --------------------------------------------------------
    def _person_leaves(self) -> set[str]:
        """Leaves that are the root of a part_of chain (people/person): their objects get skeletons."""
        reg = self.mosaic.registry
        parts = {m.name: m.part_of for m in reg if getattr(m, "part_of", None)}
        return {p for p in parts.values() if p in reg and p not in parts}

    def _pose_image(self, image_id: str) -> list[str]:
        """Skeletons for every person object in the image (or the whole image if none), estimated once."""
        if image_id in self._posed:
            return self._posed[image_id]
        img = self.images[image_id]
        if not any(self.objects[o].found_in is None for o in img.objects):
            self.find_objects(image_id)
        roots = self._person_leaves()
        persons = []
        for oid in list(img.objects):
            o = self.objects[oid]
            if o.lookup is None:
                o.lookup = self._lookup(o.embedding)
            top = o.lookup["module"] or (o.lookup["candidates"][0]["module"] if o.lookup["candidates"] else None)
            if top in roots:
                persons.append(o)
        W, H = img.image.size
        boxes = [o.box for o in persons] or [[0, 0, W, H]]
        owners = [o.id for o in persons] or [None]
        self._posed[image_id] = []
        for box, oid in zip(boxes, owners):
            self._add_person(image_id, oid, box)
        return self._posed[image_id]

    def _add_person(self, image_id: str, object_id: str | None, box: BoxT) -> _Person | None:
        img = self.images[image_id]
        kps = self.pose_estimator.estimate(img.image, [box])[0]
        skeleton = Skeleton(np.asarray(kps, dtype=np.float32), list(box))
        if skeleton.n_visible < 4:
            return None
        for pid in self._posed.setdefault(image_id, []):  # two boxes on one body give one skeleton
            other = self.people[pid].skeleton
            both = (skeleton.keypoints[:, 2] >= skeleton.min_confidence) & (other.keypoints[:, 2] >= other.min_confidence)
            if both.sum() >= 4:
                d = np.linalg.norm(skeleton.keypoints[both, :2] - other.keypoints[both, :2], axis=1).mean()
                if d < 0.1 * other.scale:
                    return self.people[pid]
        person = _Person(f"person_{next(self._person_ids)}", image_id, object_id, skeleton)
        self.people[person.id] = person
        self._posed[image_id].append(person.id)
        return person

    def _owner(self, obj: _Object) -> _Person | None:
        people = [self.people[p] for p in self._pose_image(obj.image_id)]
        s = owner(obj.box, [p.skeleton for p in people])
        return next((p for p in people if p.skeleton is s), None)

    def _resolver(self, image_id: str):
        def resolve(name: str):
            if name not in self.objects:
                raise ValueError(f"no object {name}")
            if self.objects[name].image_id != image_id:
                raise ValueError(f"{name} is in another image")
            return self.objects[name]
        return resolve

    def _person_summary(self, p: _Person) -> dict:
        sk = p.skeleton
        hands = {}
        for side in ("left_hand", "right_hand"):
            if sk.visible(side):
                x, y = sk.point(side)
                hands[side] = [round(float(x)), round(float(y))]
        return {"person_id": p.id, "object_id": p.object_id, "box": [round(v) for v in sk.box],
                "body_unit_px": round(sk.scale, 1), "joints": sk.summary(), "hands_estimated_at": hands,
                **self.rules.evaluate_all(sk, self._resolver(p.image_id))}

    def pose(self, image_id: str | None = None, object_id: str | None = None) -> dict:
        """Skeletons of the people in an image (or of one object), with every rule evaluated on each."""
        if object_id is not None:
            obj = self.objects[object_id]
            image_id = obj.image_id
            self._pose_image(image_id)
            found = next((self.people[p] for p in self._posed[image_id] if self.people[p].object_id == object_id), None)
            found = found or self._add_person(image_id, object_id, obj.box)
            people = [found] if found else []
        else:
            people = [self.people[p] for p in self._pose_image(image_id)]
        return {"image_id": image_id,
                "coordinates": "image pixels; left_*/right_* are the person's own sides; directions are in the "
                               "image frame (up = top of the image, left = viewer's left); dist() is in body units "
                               "(1 = torso length)",
                "people": [self._person_summary(p) for p in people]}

    def measure(self, person_id: str, expression: str) -> dict:
        """Evaluate a geometry expression (see list_rules) on one person's skeleton."""
        p = self.people[person_id]
        return {"person_id": person_id, "expression": expression,
                **self.rules.evaluate(expression, p.skeleton, self._resolver(p.image_id))}

    def define_rule(self, name: str, expression: str, description: str = "", kind: str = "fact") -> dict:
        """Add (or replace) a named rule; it is evaluated in every later pose() and can be used in other rules."""
        self.rules.add(name, expression, description, kind)
        self._taught_rules.add(name)
        if self.taught_dir:
            self.rules.save(self.taught_dir / "rules.json", only=self._taught_rules)
        preview = {pid: self.rules.evaluate(expression, p.skeleton, self._resolver(p.image_id))
                   for pid, p in list(self.people.items())[:5]}
        return {"defined": name, "kind": kind, "saved": bool(self.taught_dir), "on_people_seen_so_far": preview}

    def list_rules(self) -> dict:
        from mosaic.geometry import FUNCTIONS_DOC, POINT_NAMES

        return {"joints": sorted(POINT_NAMES), "functions": FUNCTIONS_DOC,
                "rules": [{**r, **({"defined_in_session": True} if r["name"] in self._taught_rules else {})}
                          for r in self.rules.describe()]}

    # ---- plug in / plug out at run time ----------------------------------------------------------
    def _views(self, obj: _Object) -> list:
        """The object as the encoder sees it, plus a mirror image and two framings, so one example is not one point."""
        from PIL import ImageOps

        from mosaic.perceiver import crop

        img = self.images[obj.image_id].image
        base = cutout(img, obj.box, obj.mask, keep=self.background_keep) if obj.mask is not None else crop(img, obj.box)
        tight = cutout(img, obj.box, obj.mask, pad=0.02, keep=self.background_keep) if obj.mask is not None \
            else crop(img, obj.box, 0.02)
        loose = crop(img, obj.box, 0.3)
        return [base, ImageOps.mirror(base), tight, loose]

    def _forget_answers(self) -> None:
        """The registry changed: every cached answer is stale (skeletons are kept; they do not depend on it)."""
        for o in self.objects.values():
            o.identity = o.lookup = o.geometry = None

    def teach(self, label: str, object_ids: list[str], part_of: str | None = None, description: str = "") -> dict:
        """Plug in a new module built from objects seen in this session (or add examples to one taught before).

        `label` is a module path ("people/limbs/tattoo") or a bare name, which goes under "taught/". The
        accept threshold is calibrated across the examples when there are at least 3 of them; with fewer, it
        is the median threshold of the registry's other leaves."""
        reg = self.mosaic.registry
        name = label if "/" in label else f"taught/{label}"
        if name in reg and name not in self._taught_modules:
            raise ValueError(f"{name} is a built-in module; teach under another name, e.g. taught/{label}")
        objs = [self.objects[o] for o in object_ids]
        if not objs:
            raise ValueError("teach needs at least one object id")
        views, entries = [], []
        for o in objs:
            vs = self._views(o)
            views += vs
            entries += [{"label": name.rsplit("/", 1)[-1], "kind": "image", "image": self.images[o.image_id].path,
                         "box": o.box, "source": f"{self.images[o.image_id].path}#{o.id}", "view": v}
                        for v in ("cutout", "mirror", "tight", "context")][: len(vs)]
        E = self.encoder.embed_images(views)
        if name in reg:  # more examples for something taught before
            old = reg.get(name)
            E, entries = np.concatenate([old.embeddings, E]), old.entries + entries
            part_of = part_of or old.part_of
            description = description or old.description
            reg.remove(name)
        module = KnowledgeModule.build(name, self.encoder.encoder_id, E, entries, n_descriptors=4,
                                       accept_percentile=None, part_of=part_of,
                                       description=description or f"taught in a session: {label}")
        sources = np.array([e["source"] for e in entries])
        if len(set(sources)) >= 3:
            S = E @ E.T
            S[sources[:, None] == sources[None, :]] = -1.0  # compare only different examples
            module.accept_threshold = float(np.percentile(S.max(axis=1), 5))
            calibration = f"5th percentile of similarity between different examples ({len(set(sources))})"
        else:
            others = [m.accept_threshold for m in reg if m.accept_threshold is not None]
            module.accept_threshold = float(np.median(others)) if others else 0.7
            calibration = "median of the registry's other leaves (fewer than 3 examples; teach more to calibrate)"
        similar = sorted(((float((reg.get(n).embeddings @ E.mean(0)).max()), n) for n in reg.leaves()), reverse=True)[:3]
        reg.add(module)
        self._taught_modules.add(name)
        saved = None
        if self.taught_dir:
            saved = str(module.save(self.taught_dir))
        self._forget_answers()
        check = {o.id: self.identify(o.id)["module"] for o in objs}
        return {"module": name, "examples": len(set(sources)), "stored_views": len(entries),
                "accepts_from": round(module.accept_threshold, 3), "calibration": calibration,
                "saved_to": saved, "recognised_again": check,
                "closest_existing_modules": [{"module": n, "similarity": round(s, 3)} for s, n in similar]}

    def forget(self, name: str) -> dict:
        """Plug out a module or group, or drop a rule. Taught modules and rules are also deleted from disk;
        built-in modules are plugged out for this session only."""
        reg = self.mosaic.registry
        if name not in reg and name in self.rules.rules:
            self.rules.remove(name)
            was_taught = name in self._taught_rules
            self._taught_rules.discard(name)
            if was_taught and self.taught_dir:
                self.rules.save(self.taught_dir / "rules.json", only=self._taught_rules)
            return {"removed_rule": name, "deleted_from_disk": was_taught and bool(self.taught_dir)}
        if name not in reg:
            raise ValueError(f"no module, group or rule named {name}")
        removed = reg.remove(name)
        deleted = []
        for n in removed:
            if n in self._taught_modules:
                self._taught_modules.discard(n)
                path = self.taught_dir / n if self.taught_dir else None
                if path and (path / "manifest.json").exists():
                    shutil.rmtree(path)
                    deleted.append(n)
                    parent = path.parent
                    while parent != self.taught_dir and parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
        self._forget_answers()
        return {"removed": removed, "deleted_from_disk": deleted,
                "kept_on_disk": [n for n in removed if n not in deleted]}

    def explain(self, object_id: str) -> dict:
        """Why identify() answered what it did: routing per level, the gate, the stored examples it matched."""
        obj = self.objects[object_id]
        ident = self.identify(object_id)
        reg, router = self.mosaic.registry, self.mosaic.router
        routes, levels = router.trace(obj.embedding)
        module = ident["module"] or (ident["candidates"][0]["module"] if ident["candidates"] else None)
        out = {"object_id": object_id, "module": ident["module"], "known": ident["known"], "reason": ident["reason"],
               "routing": {"domain_gate": round(router.threshold, 3),
                           "levels": [[{"node": r.module, "score": round(r.score, 3)} for r in lvl[:3]] for lvl in levels],
                           "kept_leaves": [r.module for r in routes]}}
        if module:
            m = reg.get(module)
            hits = m.search(obj.embedding, 5, kind="image")
            out["gate"] = {"module": module, "nearest_similarity": round(hits[0][1], 3) if hits else None,
                           "accepts_from": round(m.accept_threshold, 3) if m.accept_threshold is not None else None,
                           "calibrated_as": f"{m.accept_percentile}th percentile of its own examples' nearest-neighbour "
                                            "similarity" if m.accept_percentile else "set when taught"}
            if m.reject_margin is not None:  # counter-examples: what this module is not
                i, sc = m.search(obj.embedding, 1, kind="negative")[0]
                out["gate"]["nearest_counter_example"] = {"similarity": round(sc, 3), "source": m.entries[i].get("source"),
                                                          "image": m.entries[i].get("image"), "box": m.entries[i].get("box")}
                out["gate"]["must_beat_it_by"] = round(m.reject_margin, 3)
            out["evidence"] = [{"similarity": round(sc, 3),
                                **{k: m.entries[i][k] for k in ("label", "image", "box", "source", "view") if k in m.entries[i]}}
                               for i, sc in hits]
        runner = next((c for c in ident["candidates"] if c["module"] != module), None)
        if runner:
            m = reg.get(runner["module"])
            i, sc = m.search(obj.embedding, 1, kind="image")[0]
            out["runner_up"] = {"module": runner["module"], "similarity": round(sc, 3),
                                **{k: m.entries[i][k] for k in ("image", "box") if k in m.entries[i]}}
        if obj.geometry:
            out["geometry"] = obj.geometry
        return out


    def render(self, image_id: str, object_ids: list[str] | None = None) -> bytes:
        """PNG of the image with the given (default: all) objects outlined and labelled."""
        img = self.images[image_id]
        out = img.image.copy()
        ids = object_ids or img.objects
        for oid in ids:
            o = self.objects[oid]
            if o.mask is not None:
                tint = Image.new("RGB", out.size, (40, 200, 60) if (o.identity or {}).get("known") else NEUTRAL)
                out = Image.composite(tint, out, Image.fromarray((o.mask * 80).astype(np.uint8)))
        draw = ImageDraw.Draw(out)
        for oid in ids:
            o = self.objects[oid]
            label = (o.identity or {}).get("module") or ("?" if o.identity else "")
            draw.rectangle(o.box, outline=(255, 60, 60), width=2)
            draw.text((o.box[0] + 3, o.box[1] + 2), f"{oid} {label}", fill=(255, 255, 0))
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()


def _box_mask(box, w, h) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    from PIL import ImageFilter

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * r + 1))) > 0
