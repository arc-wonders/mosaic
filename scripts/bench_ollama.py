"""Benchmark: a small text-only LLM + MOSAIC tools vs. a vision-language model, both local Qwen models on Ollama.

    slm   qwen2.5:7b-instruct   never sees pixels; calls the MOSAIC MCP tools (same schemas and functions as
                                mosaic.server, called in-process) and answers from what they return
    vlm   qwen2.5vl:7b          gets the image itself, no tools

Tasks, all scored against Open Images ground truth on the *test* split (images MOSAIC never stored):
    presence  "Is there a person?"        people photos (yes) vs. animals / vehicles / fruit / food (no)
    count     "How many people?"          exact match against the annotated person boxes
    locate    "Box of one <part>"         hit if IoU >= 0.5 with any annotated box of that part

    python scripts/bench_ollama.py [--n 25] [--systems vlm,slm] [--tasks presence,count,locate]

Answers are appended to results/bench_ollama.jsonl as they come (a rerun skips what is already there);
the summary goes to results/bench_ollama.json. The VLM runs first and is unloaded before MOSAIC's models
load, so both fit on one 8 GB GPU.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_people import split_of  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SLM, VLM = "qwen2.5:7b-instruct", "qwen2.5vl:7b"
OUT_LOG, OUT_SUM = REPO / "results/bench_ollama.jsonl", REPO / "results/bench_ollama.json"
PARTS = ["people/face_region/face", "people/face_region/head", "people/limbs/hand", "people/limbs/foot",
         "people/limbs/arm", "people/limbs/leg"]
NEGATIVE_DIRS = ["raw/animals", "raw/vehicles", "raw/fruits", "raw/ood/pizza"]
# "no person" photos that turned out to show people (checked by eye); never asked about
CONTAINS_PEOPLE = {"animals/dog/cats_vs_dogs_0864.jpg", "vehicles/scooter/imagenet_0312.jpg",
                   "vehicles/sports_car/imagenet_0817.jpg", "animals/dog/cats_vs_dogs_0866.jpg",
                   "vehicles/scooter/imagenet_0035.jpg", "fruits/pineapple/imagenet_0459.jpg",
                   "vehicles/scooter/imagenet_0041.jpg", "vehicles/scooter/imagenet_0993.jpg"}
MIN_SIDE = 48  # parts smaller than this are not asked about
MAX_STEPS = 10
RESULT_CHARS = 5000  # a tool result is cut here so a 7B model's context isn't flooded

ANSWER_RULE = ("End your reply with one line 'ANSWER: <answer>' where <answer> is {fmt}. Give your best answer "
               "even if unsure.")
TASKS = {
    "presence": ("Is there a person (a human being, or any human body part) in this image?", "yes or no"),
    "count": ("How many people are in this image? Count every person, including partly visible ones.",
              "a single integer"),
    "locate": ("Give the bounding box of one {part} in this image, in pixels of the image as [x0, y0, x1, y1] "
               "(origin top-left).", "the box as [x0, y0, x1, y1]"),
}


# ---------------------------------------------------------------- questions (fixed by --seed)

def questions(n: int, seed: int, tasks: list[str]) -> list[dict]:
    ann = json.loads((REPO / "data/human/annotations.json").read_text())
    ann.pop("_files", None)
    rng = random.Random(seed)
    test = sorted(p for p in ann if split_of(p) == "test")
    persons = lambda p: [b for b in ann[p] if b[4] == "people/person"]
    qs = []
    if "presence" in tasks:
        pos = rng.sample([p for p in test if persons(p)], n)
        # test split only: build_negatives.py learns counter-examples from the other photos of these folders
        neg_pool = [str(f) for d in NEGATIVE_DIRS for f in sorted((REPO / "data" / d).rglob("*.jpg"))
                    if split_of(str(f.relative_to(REPO))) == "test"]
        rng.shuffle(neg_pool)  # an excluded photo is replaced by the next one; the rest stay the same
        neg = [f for f in neg_pool if str(Path(f).relative_to(REPO / "data/raw")) not in CONTAINS_PEOPLE][:n]
        qs += [{"task": "presence", "image": str(Path(p).resolve()), "truth": "yes"} for p in pos]
        qs += [{"task": "presence", "image": str(Path(p).resolve()), "truth": "no"} for p in neg]
    if "count" in tasks:
        # 1-5 people, every person box a reasonable size (tiny background people are annotated unevenly)
        ok = [p for p in test if 1 <= len(persons(p)) <= 5
              and all(min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE for b in persons(p))]
        for p in rng.sample(ok, n):
            qs.append({"task": "count", "image": str(Path(p).resolve()), "truth": len(persons(p))})
    if "locate" in tasks:
        cands = [(p, part) for p in test for part in PARTS
                 if any(b[4] == part and min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE for b in ann[p])]
        for p, part in rng.sample(cands, n):
            gt = [b[:4] for b in ann[p] if b[4] == part]
            qs.append({"task": "locate", "image": str(Path(p).resolve()), "part": part.rsplit("/", 1)[1],
                       "truth": gt})
    for i, q in enumerate(qs):
        q["qid"] = f"{q['task']}_{i:03d}"
    return qs


def prompt_of(q: dict) -> str:
    text, fmt = TASKS[q["task"]]
    return text.format(part=q.get("part", "")) + " " + ANSWER_RULE.format(fmt=fmt)


# ---------------------------------------------------------------- scoring

def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse(task: str, text: str):
    m = re.findall(r"ANSWER:\s*(.+)", text or "", flags=re.I)
    tail = m[-1] if m else (text or "")[-200:]
    if task == "presence":
        w = re.findall(r"\b(yes|no)\b", tail.lower())
        return w[0] if w else None
    if task == "count":
        w = re.findall(r"\d+", tail)
        return int(w[0]) if w else None
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    return [float(x) for x in nums[:4]] if len(nums) >= 4 else None


def score(q: dict, pred) -> dict:
    if pred is None:
        return {"correct": False}
    if q["task"] == "locate":
        best = max(iou(pred, g) for g in q["truth"])
        return {"correct": best >= 0.5, "iou": round(best, 3)}
    return {"correct": pred == q["truth"]}


# ---------------------------------------------------------------- Ollama

def chat(model: str, messages: list, tools: list | None = None, keep_alive="10m") -> dict:
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_ctx": 16384, "seed": 0}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def unload(model: str) -> None:
    req = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError:  # not pulled / not loaded: nothing to free
        pass
    for _ in range(120):  # wait until its VRAM is actually released
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=30) as r:
            if not any(m["name"] == model for m in json.loads(r.read())["models"]):
                return
        time.sleep(0.5)


SWAP = False  # --swap: unload the LLM before every tool call (for GPUs too small to hold both)


def slm_turn(messages: list, tools: list | None = None) -> dict:
    """One SLM turn. MOSAIC peaks near 2.4 GB and the 7B model needs ~5.6 GB: on an 8 GB card they mostly fit
    together (Ollama keeps a few layers on the CPU). With SWAP they time-share the GPU instead."""
    r = chat(SLM, messages, tools)
    if SWAP:
        unload(SLM)
    return r


def warm_up(server) -> None:
    """Load MOSAIC's models and run each kind of call once, so its GPU memory is taken before Ollama sizes
    the LLM to what is left."""
    img = sorted((REPO / "data/human/images").glob("*.jpg"))[0]
    iid = server.load_image(str(img))["image_id"]
    objs = server.find_objects(iid)["objects"]
    server.pose(image_id=iid)
    server.search(iid, "people/limbs/hand")
    if objs:
        server.find_parts(objs[0]["object_id"])


# ---------------------------------------------------------------- the two systems

def run_vlm(q: dict) -> dict:
    img = Image.open(q["image"]).convert("RGB")
    # Qwen2.5-VL answers in pixels of the image it was given; hand it dims that are multiples of 28 (its patch
    # grid) so nothing is resized again inside Ollama, and map the box back to the original image
    w, h = (max(28, round(d / 28) * 28) for d in img.size)
    buf = io.BytesIO()
    img.resize((w, h)).save(buf, format="JPEG", quality=95)
    text = prompt_of(q) + f" The image is {w}x{h} pixels."
    msg = {"role": "user", "content": text, "images": [base64.b64encode(buf.getvalue()).decode()]}
    t = time.time()
    r = chat(VLM, [msg])
    out = r["message"]["content"]
    pred = parse(q["task"], out)
    if q["task"] == "locate" and pred:
        sx, sy = img.width / w, img.height / h
        pred = [pred[0] * sx, pred[1] * sy, pred[2] * sx, pred[3] * sy]
    return {"pred": pred, "reply": out, "seconds": round(time.time() - t, 1), "steps": 1}


SLM_SYSTEM = """You answer questions about an image you cannot see. You see it only through the MOSAIC tools, \
which return structured facts (objects, identities, skeletons, measurements). Call tools to gather evidence, then \
answer. Rules:
- The image is already on disk; start with load_image(path) to get its image_id.
- identify() tells you what an object is; known=false means the registry does not recognise it (not a body part \
it knows). Don't guess past that.
- pose(image_id) lists every person the skeleton model found (people with no skeleton are not listed).
- Coordinates are pixels [x0, y0, x1, y1] of the original image.
- Use as few tool calls as you need; don't repeat a call you've made.

MOSAIC's own instructions:
"""


def run_slm(q: dict, server, tools: list) -> dict:
    import torch
    messages = [{"role": "system", "content": SLM_SYSTEM + server.INSTRUCTIONS},
                {"role": "user", "content": f"Image file: {q['image']}\n\n{prompt_of(q)}"}]
    calls, t = [], time.time()
    out = ""
    for step in range(MAX_STEPS):
        last = step == MAX_STEPS - 1
        r = slm_turn(messages, None if last else tools)
        m = r["message"]
        messages.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")})
        if not m.get("tool_calls"):
            out = m.get("content", "")
            break
        for tc in m["tool_calls"]:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            try:
                res = json.dumps(getattr(server, name)(**args), default=str, separators=(",", ":"))
            except Exception as e:  # a bad call is information for the model, as over MCP
                res = json.dumps({"error": f"{type(e).__name__}: {e}"})
            torch.cuda.empty_cache()
            calls.append({"tool": name, "args": args, "chars": len(res)})
            if len(res) > RESULT_CHARS:
                res = res[:RESULT_CHARS] + f"... [truncated, {len(res)} chars]"
            messages.append({"role": "tool", "content": res, "tool_name": name})
        if last:
            messages.append({"role": "user", "content": "No more tool calls. " + ANSWER_RULE.format(
                fmt=TASKS[q["task"]][1])})
            out = slm_turn(messages)["message"].get("content", "")
    return {"pred": parse(q["task"], out), "reply": out, "seconds": round(time.time() - t, 1),
            "steps": len(calls), "calls": calls}


def run_script(q: dict, server) -> dict:
    """MOSAIC with a fixed policy and no LLM: what the tools can answer when used the obvious way.
    presence: any object identified as a people/* module; count: people pose() found;
    locate: the best result of search() for the part's module."""
    t = time.time()
    iid = server.load_image(q["image"])["image_id"]
    if q["task"] == "presence":
        objs = server.find_objects(iid)["objects"]
        known = [server.identify(o["object_id"]) for o in objs]
        pred = "yes" if any(k["known"] and k["module"].startswith("people/") for k in known) else "no"
    elif q["task"] == "count":
        pred = len(server.pose(image_id=iid)["people"])
    else:
        part = next(p for p in PARTS if p.endswith("/" + q["part"]))
        hits = server.search(iid, part)["results"]
        best = ([h for h in hits if h["match"]] or hits)[:1]
        pred = [float(v) for v in best[0]["box"]] if best else None
    return {"pred": pred, "reply": "", "seconds": round(time.time() - t, 1), "steps": 0}


def mcp_tools(server) -> list:
    """The MCP server's own tool list, in Ollama's function-calling format."""
    tools = asyncio.run(server.mcp.list_tools())
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_schema}} for t in tools]


# ---------------------------------------------------------------- main

def summarise(rows: list[dict]) -> dict:
    out = defaultdict(dict)
    for sys_ in sorted({r["system"] for r in rows}):
        for task in sorted({r["task"] for r in rows}):
            rs = [r for r in rows if r["system"] == sys_ and r["task"] == task]
            if not rs:
                continue
            s = {"n": len(rs), "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 3),
                 "unparsed": sum(r["pred"] is None for r in rs),
                 "mean_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1),
                 "mean_tool_calls": round(sum(r["steps"] for r in rs) / len(rs), 1)}
            if task == "presence":
                for truth in ("yes", "no"):
                    sub = [r for r in rs if r["truth"] == truth]
                    s[f"acc_truth_{truth}"] = round(sum(r["correct"] for r in sub) / max(1, len(sub)), 3)
            if task == "count":
                s["mean_abs_error"] = round(sum(abs((r["pred"] or 0) - r["truth"]) for r in rs) / len(rs), 2)
            if task == "locate":
                s["mean_best_iou"] = round(sum(r.get("iou", 0) for r in rs) / len(rs), 3)
            out[sys_][task] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="questions per task (presence: n yes + n no)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", default="vlm,script,slm", help="vlm, slm, script (MOSAIC, fixed policy)")
    ap.add_argument("--tasks", default="presence,count,locate")
    ap.add_argument("--swap", action="store_true", help="unload the LLM before every MOSAIC call")
    a = ap.parse_args()
    global SWAP
    SWAP = a.swap
    qs = questions(a.n, a.seed, a.tasks.split(","))
    done = {}
    if OUT_LOG.exists():
        for line in OUT_LOG.read_text().splitlines():
            r = json.loads(line)
            done[(r["system"], r["qid"], r["image"])] = r
    log = OUT_LOG.open("a")

    for system in a.systems.split(","):
        todo = [q for q in qs if (system, q["qid"], q["image"]) not in done]
        print(f"== {system}: {len(todo)} of {len(qs)} questions to run", flush=True)
        if system in ("slm", "script") and todo:
            unload(VLM)
            os.environ.setdefault("MOSAIC_TAUGHT", str(REPO / "results/bench_taught"))  # keep ./taught clean
            from mosaic import server
            tools = mcp_tools(server)
            warm_up(server)
        for i, q in enumerate(todo):
            try:
                r = {"vlm": lambda: run_vlm(q), "slm": lambda: run_slm(q, server, tools),
                     "script": lambda: run_script(q, server)}[system]()
            except Exception as e:
                r = {"pred": None, "reply": f"ERROR {type(e).__name__}: {e}", "seconds": 0, "steps": 0}
            row = {"system": system, **q, **r, **score(q, r["pred"])}
            done[(system, q["qid"], q["image"])] = row
            log.write(json.dumps(row, default=str) + "\n")
            log.flush()
            print(f"  [{i + 1}/{len(todo)}] {q['qid']} truth={q['truth'] if q['task'] != 'locate' else q['part']} "
                  f"pred={r['pred']} {'OK' if row['correct'] else 'x'} ({r['seconds']}s, {r['steps']} calls)",
                  flush=True)
        if system == "vlm":
            unload(VLM)

    keys = {(q["qid"], q["image"]) for q in qs}
    rows = [r for (s, qid, img), r in done.items() if (qid, img) in keys]
    summary = {"models": {"slm": SLM, "vlm": VLM, "script": "MOSAIC, fixed policy, no LLM"}, "n_per_task": a.n, "seed": a.seed, **summarise(rows)}
    OUT_SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
