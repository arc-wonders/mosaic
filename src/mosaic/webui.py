"""A local web UI for watching MOSAIC's pipeline decide, stage by stage.

Wraps one VisionSession in a Flask app: the browser drives find_objects / identify /
explain / pose / measure over HTTP, and gets back the mask, the soft cut-out, the
routing trace, the gate, and the stored evidence as images and JSON, so a person can
click an object and see exactly why the registry answered what it did -- the same
facts the LLM gets through the MCP tools, laid out for a human instead of reasoned
over by one.

    python -m mosaic.webui                      # modules_people, taught/, http://127.0.0.1:5050
    python -m mosaic.webui --modules modules_masks --port 5055

This process holds the frozen models (detector, SAM, encoder) for its whole life, and
answers one request at a time (threaded=False) -- the 8 GB GPU here barely holds one
copy of them; never point two of these, or one of these and an MCP session, at the
same GPU at once.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file
from PIL import Image
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[2]  # repo root: src/mosaic/webui.py -> ../../..
WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = ROOT / "data" / "uploads"

from mosaic.capability import VisionSession  # noqa: E402
from mosaic.perceiver import NEUTRAL, crop, cutout  # noqa: E402

app = Flask(__name__)
SESSION: VisionSession | None = None
LOCK = Lock()  # one GPU job at a time

MASK_TINT = (230, 150, 40)  # amber: matches "module / learned" in the architecture figure


def _session() -> VisionSession:
    if SESSION is None:
        raise RuntimeError("session not started")
    return SESSION


def _png(image: Image.Image) -> Response:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


def _jpeg(image: Image.Image, quality: int = 90) -> Response:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


def _safe_path(rel: str) -> Path:
    """A path under the repo root, rejecting anything that escapes it."""
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT) or not p.exists():
        abort(403 if not p.is_relative_to(ROOT) else 404)
    return p


# ---- pages / static ---------------------------------------------------------------------------
@app.get("/")
def index():
    return send_file(WEB_DIR / "index.html")


# ---- image lifecycle --------------------------------------------------------------------------
@app.get("/api/samples")
def api_samples():
    """A grab bag of images already in the repo, grouped, for quick picking without a file dialog."""
    out = []
    human = ROOT / "data" / "human" / "images"
    if human.exists():
        for p in sorted(human.glob("*.jpg"))[:40]:
            out.append({"path": str(p.relative_to(ROOT)), "group": "people"})
    raw = ROOT / "data" / "raw"
    if raw.exists():
        for sub in sorted(raw.glob("*/*")):
            if not sub.is_dir():
                continue
            imgs = sorted(q for q in sub.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))[:3]
            out += [{"path": str(q.relative_to(ROOT)), "group": f"{sub.parent.name}/{sub.name}"} for q in imgs]
    return jsonify({"samples": out})


@app.post("/api/load")
def api_load():
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"{int(time.time() * 1000)}_{secure_filename(f.filename)}"
        f.save(dest)
        path = dest
    else:
        payload = request.get_json(force=True, silent=True) or {}
        rel = payload.get("path") or request.form.get("path")
        if not rel:
            abort(400, "pass a sample 'path' or upload a 'file'")
        path = _safe_path(rel)
    with LOCK:
        info = _session().load_image(str(path))
    return jsonify(info)


@app.get("/api/image/<image_id>")
def api_image(image_id: str):
    img = _session().images[image_id].image
    return _jpeg(img)


# ---- stage 1-2: detect + mask + soft cut-out -----------------------------------------------------
@app.get("/api/find_objects")
def api_find_objects():
    """Detect + mask + encode every object, then identify each one independently: one object's
    identify() failing (a bad skeleton, a model hiccup) must not hide the detections that already
    succeeded for every other object in the image."""
    image_id = request.args["image_id"]
    with LOCK:
        s = _session()
        result = s.find_objects(image_id)
        for o in result["objects"]:
            try:
                o["identify"] = s.identify(o["object_id"])
            except Exception as e:
                o["identify"] = None
                o["identify_error"] = str(e)
    return jsonify(result)


@app.get("/api/object/<oid>/mask.png")
def api_mask(oid: str):
    s = _session()
    obj = s.objects[oid]
    img = s.images[obj.image_id].image
    w, h = img.size
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if obj.mask is not None:
        rgba[obj.mask] = (*MASK_TINT, 150)
    return _png(Image.fromarray(rgba, "RGBA"))


@app.get("/api/object/<oid>/cutout.png")
def api_cutout(oid: str):
    """Exactly what the encoder embeds: the masked object, background faded toward neutral."""
    s = _session()
    obj = s.objects[oid]
    img = s.images[obj.image_id].image
    view = cutout(img, obj.box, obj.mask, keep=s.background_keep) if obj.mask is not None else crop(img, obj.box)
    return _jpeg(view, quality=94)


# ---- stage 3-4: routing, gate, geometry ----------------------------------------------------------
@app.get("/api/object/<oid>/explain")
def api_explain(oid: str):
    with LOCK:
        return jsonify(_session().explain(oid))


@app.get("/api/evidence.png")
def api_evidence():
    """A thumbnail crop from a stored example's source photo (explain()'s evidence / runner-up / counter-example)."""
    rel, box = request.args["image"], request.args["box"]
    path = _safe_path(rel)
    b = [int(v) for v in box.split(",")]
    img = Image.open(path).convert("RGB")
    return _jpeg(crop(img, b, pad=0.15), quality=90)


# ---- stage 5: skeleton + rules -------------------------------------------------------------------
@app.get("/api/pose")
def api_pose():
    image_id = request.args["image_id"]
    with LOCK:
        return jsonify(_session().pose(image_id))


@app.post("/api/measure")
def api_measure():
    payload = request.get_json(force=True)
    with LOCK:
        return jsonify(_session().measure(payload["person_id"], payload["expression"]))


@app.get("/api/rules")
def api_rules():
    return jsonify(_session().list_rules())


@app.get("/api/tree")
def api_tree():
    return jsonify(_session().list_modules(request.args.get("path", "")))


# ---- errors ---------------------------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_error(e: Exception):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": str(e)}), 400


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", type=Path, default=ROOT / "modules_people")
    ap.add_argument("--taught", type=Path, default=ROOT / "taught", help="modules and rules taught in sessions")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()
    if not (args.modules / "registry.json").exists():
        sys.exit(f"no modules in {args.modules}; run scripts/build_people.py first")

    global SESSION
    print(f"loading models and {args.modules} ...", file=sys.stderr)
    SESSION = VisionSession(args.modules, taught_dir=args.taught)
    print(f"ready: http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == "__main__":
    main()
