"""MOSAIC as a perception capability for a reasoning model.

A reasoning model looks at a scene step by step with these calls, all in
original-image pixel coordinates [x0, y0, x1, y1]:

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
    render(image, objects)                 -> annotated PNG, so a multimodal model can check

Objects keep their ids across calls: a new detection that overlaps an existing
object in the same image (IoU >= 0.85) reuses that object's id.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
    identity: dict | None = None

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


class VisionSession:
    def __init__(
        self,
        modules: str | Path | list[str | Path],
        route_threshold: float = 0.5,
        top_k: list[int] = (1, 3),
        k: int = 5,
        zoom_side: int = 768,
        background_keep: float = 0.5,
        use_masks: bool = True,
        device: str | None = None,
        encoder=None,
        detector=None,
        segmenter=None,
    ):
        """encoder / detector / segmenter default to CLIP, OWLv2 objectness and SAM; pass your own to swap them."""
        if encoder is None:
            from mosaic.encoder import ClipEncoder

            encoder = ClipEncoder(fp16=True, device=device)
        if detector is None:
            from mosaic.detector import ObjectnessDetector

            detector = ObjectnessDetector(device=device)
        if segmenter is None and use_masks:
            from mosaic.segmenter import BoxSegmenter

            segmenter = BoxSegmenter(device=device)
        self.encoder, self.detector, self.segmenter = encoder, detector, segmenter
        self.mosaic = Mosaic(self.encoder, top_k_modules=list(top_k), route_threshold=route_threshold)
        for root in [modules] if isinstance(modules, (str, Path)) else modules:
            self.mosaic.registry.load_dir(root)
        self.k, self.zoom_side, self.background_keep = k, zoom_side, background_keep
        self.images: dict[str, _Image] = {}
        self.objects: dict[str, _Object] = {}
        self._ids = itertools.count(1)

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
                    existing.identity = None
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
            obj.identity = self._lookup(obj.embedding)
        return {**self._summary(obj), **obj.identity}

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
                scores.append((name, hits[0][1], m.accept_threshold))
        scores.sort(key=lambda s: -s[1])
        candidates = [{"module": n, "similarity": round(s, 3), "accepts_from": round(t, 3) if t else None,
                       "would_accept": t is None or s >= t} for n, s, t in scores[:3]]
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

        embs = np.concatenate([self.encoder.embed_images([crop(img, b) for b in boxes[i : i + 256]])
                               for i in range(0, len(boxes), 256)])
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
        sims = [float(self.objects[o["object_id"]].embedding @ t) for o in found]
        hits = [{**o, "similarity": round(s, 3), "match": s >= 0.25 and s == max(sims)} for o, s in zip(found, sims)]
        hits.sort(key=lambda h: -h["similarity"])
        return {"image_id": image_id, "query": query, "mode": "text (CLIP; weaker than a registry module)",
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
            return out

        return {"path": path or "/", "children": [node(c) for c in reg.children(path)]}

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
