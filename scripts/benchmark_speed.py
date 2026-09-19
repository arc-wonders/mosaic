"""Speed benchmark: single-query CPU path vs batched engine on CPU and GPU, plus the encoder.

Uses the modules in modules/ and the cached embeddings from run_experiment.py.
Writes results/speed.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mosaic import Mosaic
from mosaic.encoder import ClipEncoder

ENCODER_ID = "open_clip/ViT-B-32/laion2b_s34b_b79k"
R = {}


class Stub:  # routing/matching only needs the id and dim
    encoder_id, dim = ENCODER_ID, 512


def timed(fn, repeat=3, sync=False):
    best = float("inf")
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if sync:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    results = json.loads(Path("results/results.json").read_text())
    system = Mosaic(Stub(), top_k_modules=[1, 3], route_threshold=results["route_threshold"])
    system.registry.load_dir("modules")
    cache = np.load(f"data/cache/{ENCODER_ID.replace('/', '_')}.npz")
    Q = np.stack([cache[k] for k in cache.files]).astype(np.float32)  # all 19,400 cached images
    R["n_queries"], R["n_leaves"], R["n_entries"] = len(Q), len(system.registry), int(sum(len(m) for m in system.registry))
    R["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None

    # ---- agreement: engine vs single-query path -----------------------------------
    sub = Q[::5]
    single = [system.classify(q) for q in sub]
    for dev in ["cpu", "cuda"]:
        b = system.classify_batch(sub, device=dev)
        R[f"agreement_{dev}"] = float(np.mean([a == p.label and r == p.reason for a, r, p in zip(b.labels, b.reasons, single)]))

    # ---- classification speed (route + kNN + leaf gate), ms per query ----------------
    n_single = 2000
    t, _ = timed(lambda: [system.classify(q) for q in Q[:n_single]], repeat=1)
    R["single_query_numpy_ms"] = 1000 * t / n_single
    speed = {}
    for dev in ["cpu", "cuda"]:
        system.classify_batch(Q[:64], device=dev)  # warm up + pack
        for bs in [1, 64, 1024, len(Q)]:
            chunks = [Q[i : i + bs] for i in range(0, min(len(Q), max(bs, 2048)), bs)]
            n = sum(len(c) for c in chunks)
            t, _ = timed(lambda: [system.classify_batch(c, device=dev) for c in chunks], sync=dev == "cuda")
            speed[f"{dev}_batch_{bs}"] = {"ms_per_query": 1000 * t / n, "queries_per_s": n / t}
    R["batched_engine"] = speed

    # ---- routing only, and repack cost after a plug-out ------------------------------
    eng = system._engine
    t, _ = timed(lambda: (system.detach("fruits/apple"), eng.refresh()), repeat=1, sync=True)
    R["plug_out_plus_gpu_repack_ms"] = 1000 * t
    system.attach(__import__("mosaic").KnowledgeModule.load("modules/fruits/apple"))

    # ---- encoder: fp32 vs fp16, images per second ------------------------------------
    paths = sorted(Path("data/raw").rglob("*.jpg"))[::40][:512]
    images = [Image.open(p).convert("RGB") for p in paths]
    enc = {}
    for fp16 in [False, True]:
        e = ClipEncoder(fp16=fp16)
        e.embed_images(images[:32])  # warm up
        t, embs = timed(lambda: np.concatenate([e.embed_images(images[i : i + 128]) for i in range(0, len(images), 128)]),
                        sync=True)
        enc["fp16" if fp16 else "fp32"] = {"images_per_s": len(images) / t, "embs": embs}
    cos = np.sum(enc["fp16"]["embs"] * enc["fp32"]["embs"], axis=1)
    R["encoder"] = {k: {"images_per_s": v["images_per_s"]} for k, v in enc.items()}
    R["encoder"]["fp16_vs_fp32_cosine_min"] = float(cos.min())
    q32, q16 = enc["fp32"]["embs"], enc["fp16"]["embs"]
    b32, b16 = system.classify_batch(q32, device="cuda"), system.classify_batch(q16, device="cuda")
    R["encoder"]["fp16_same_decision_as_fp32"] = float(np.mean([a == b for a, b in zip(b32.labels, b16.labels)]))

    # preprocessing (PIL resize/normalise on CPU) is part of the encoder time
    e = ClipEncoder(fp16=True)
    t, _ = timed(lambda: torch.stack([e.preprocess(im) for im in images]), repeat=1)
    R["encoder"]["cpu_preprocess_images_per_s"] = len(images) / t

    Path("results/speed.json").write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
