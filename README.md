# MOSAIC

**M**odular **O**bject-agnostic **S**emantic **A**rchitecture with **I**ndexed **C**oncepts.

A vision system where a frozen, category-free encoder does the *seeing*, and
all knowledge of *what things are* lives in external, attachable **modules**
(the tiles of the mosaic). A router picks which tiles to consult per query.

Term paper: *Modular Object-Agnostic Vision Architecture Using Embedding-Based
Semantic Routing* — see `../synopsis/`.

## Architecture

```
image ─► Detector (class-agnostic objectness: finds "things", knows no labels)
           │   box #1        box #2        box #3 ...
           ▼
         Encoder (frozen) ─► one embedding per object crop
                                         │   (each object goes through the rest independently)
             ┌───────────── Router: coarse → fine over the module tree ─────────────┐
             │ L1  animals   birds   vehicles   [fruits]   ← best domain (ABSTAIN if  │
             │                                     │         below route threshold) │
             │ L2          apple  orange  banana  pineapple  strawberry             │
             │             [top-3 class leaves kept]                                │
             └──────────────────────────────────────────────────────────────────────┘
                                         │
             Matcher: 5-NN vote over the kept leaves' stored examples
                                         │
             Leaf gate: the winning leaf's own accept threshold (ABSTAIN if the
             nearest example is less typical than the leaf's own members)
                                         │
                              #1 fruits/apple   #2 unknown   ...
```

**Everything is object-level.** A query image is split into objects by a detector
that has no label set: OWLv2's vision tower, objectness head and box head, with no text
tower or class head. Modules store **only object embeddings**. When a module is built
from example images, each image's labelled object is the one the set has in common
(**consensus selection** in `builder.py`). That way, a teddy bear in one cat photo isn't
stored as "cat", even when it is the most object-like thing in that photo. The selection leans
toward whole objects, so a module doesn't fill up with wheels and headlights. Objects
under 64 px on a side are ignored, because they are too blurry to recognise. The image's
overall answer is the known label that covers the most image area (`Mosaic.main_label`).

**Leaves** are plug-in modules, one per class (`fruits/apple`). **Groups** (`fruits`)
hold no data. A group's routing descriptors summarise its children, and they are
recomputed whenever a child is plugged in or out. You can plug out one class
(`detach("fruits/apple")`) or a whole domain (`detach("fruits")`). The tree can be any
depth, and a module named without `/` is a top-level leaf.

The contract between components: **every vector is an L2-normalised `float32`
of dimension `D`, produced by one specific `encoder_id`.**

| Component   | File              | Responsibility |
|-------------|-------------------|----------------|
| Detector    | `detector.py`     | class-agnostic object boxes + objectness (OWLv2 heads, no text) |
| Encoder     | `encoder.py`      | image/text → shared embedding space. Frozen. |
| Perceiver   | `perceiver.py`    | image → detected objects → one embedding per crop |
| Segmenter   | `segmenter.py`    | optional: SAM mask per detected box (class-agnostic); soft cut-out |
| Builder     | `builder.py`      | picks each example image's labelled object by consensus |
| Module      | `module.py`       | a leaf: entries, embeddings, routing descriptors, self-calibrated accept threshold |
| Registry    | `registry.py`     | the plug-in tree; group summaries; enforces encoder compatibility |
| Router      | `router.py`       | coarse-to-fine top-k per level (or flat, as a baseline) |
| Matcher     | `matcher.py`      | NN search in routed leaves + composition across leaves |
| Mosaic      | `pipeline.py`     | wires it together; `classify()` applies the gates |
| BatchEngine | `engine.py`       | the same pipeline as batched GPU tensor ops; repacks on plug-in/out |

### Design decisions

1. **"Object-agnostic" encoder vs. text alignment.** Text queries need a shared
   image–text space (CLIP/SigLIP). We use such an encoder frozen, and define
   *object-agnostic* operationally: **the central model has no task heads or
   label sets, and is never updated when concepts change.**
2. **Hierarchy by path names.** Classes are plug-in leaves under domain groups.
   A group summarises its children's descriptors, compressed to 16 with k-means.
3. **Three decisions, three signals.** The router picks *where to look* from centroids.
   k-NN picks *which class*. The **leaf gate** decides *whether it is known at all*.
   Each leaf calibrates its own gate from its own images (the 5th percentile of
   leave-one-out nearest-neighbour similarity), so it needs no shared data and
   travels with the module.
4. **Encoder–module drift is a hard error.** Manifests record `encoder_id` and `dim`.
5. **Multi-centroid, per-modality descriptors.** Image and text entries each get
   their own k-means centroids, because the two sit in different regions of CLIP space.

### Module format (on disk)

```
modules/<domain>/<class>/
  manifest.json     # name, version, description, encoder_id, dim, accept_threshold, accept_percentile,
                    # part_of (e.g. hand -> people/limbs/arm), reject_margin (with negatives), created
  embeddings.npy    # (N, D) float32, L2-normalised
  entries.jsonl     # N lines: {"label", "kind": "image"|"text"|"negative", ...}
<registry>/registry.json   # route_threshold (domain gate) and top_k, calibrated for the whole registry
  descriptors.npy   # (K, D) float32 routing centroids
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # core: numpy only
.venv/bin/pip install -e ".[clip]"         # real encoder (torch + open_clip)
.venv/bin/pytest
```

## Usage

```bash
python scripts/extract_objects.py                  # detect + embed all objects in data/raw
python scripts/run_experiment.py                   # build object modules (modules/), evaluate
python scripts/run_experiment.py --regions whole   # whole-image comparison (modules_whole/)
python scripts/extract_masks.py                    # SAM masks for every box (+ soft cut-out embeddings)
python scripts/run_experiment.py --regions masks   # mask-based modules (modules_masks/)
python scripts/multi_object_test.py                # two-object collages, both pipelines
python mosaic_cli.py image.png --save out.jpg          # detect + segment objects, classify each, draw masks
python mosaic_cli.py image.png --regions objects       # box crops instead of masks
python mosaic_cli.py image.png --without fruits/apple  # plug out one class for this run
python mosaic_cli.py image.png --without fruits        # plug out a whole domain
python mosaic_cli.py --tree                            # list groups, leaves, accept thresholds
```

```python
system.attach(KnowledgeModule.build("fruits/mango", encoder.encoder_id, embs, entries))  # plug in
system.detach("fruits/mango")                                                            # plug out
pred = system.classify(query_embedding)   # pred.label, pred.module, pred.reason
batch = system.classify_batch(Q, device="cuda")  # same decisions, batched on the GPU
encoder = ClipEncoder(fp16=True)                 # ~1.3x faster encoding, same decisions
```

## Results

All experiments use frozen encoders; nothing is fine-tuned. The numbers below are from the runs
described. Raw files are in `results/`. Experiments 1–2 were run before the scripts were generalised,
and their JSON was overwritten by later runs, so their numbers are recorded here only.

### Datasets

| Set | Classes | Source | Size |
|---|---|---|---|
| Objects | animals (cat, dog, elephant, zebra), birds (eagle, owl, parrot, flamingo, penguin), vehicles (sports car, pickup truck, school bus, scooter, airplane), fruits (apple, orange, banana, strawberry, pineapple) | cats_vs_dogs, ImageNet-1k, Fruits-360 (300/fruit) | 1,000 images/class, 700 build / 100 calibrate / 200 test |
| OOD probes | lemon (near), pizza (far) | ImageNet-1k | 200 each |
| People + handheld (v1) | person, face, hand, arm, leg; phone, bottle, cup, knife, ball, book | Human-Parts, Open Images, COCO | ≤500 instances/class, split by image 70/10/20 |
| Human model | person, head, face, hair, eye, nose, mouth, ear, arm, hand, leg, foot | Open Images validation (all boxes kept) | 1,764 photos, 16.5k instances, split by image hash 70/10/20 |

### Experiment 1 — 4 classes, whole images, CLIP ViT-B/32

| Metric | Result |
|---|---|
| Routing to the right module | 100% |
| Accuracy: MOSAIC / flat kNN / zero-shot CLIP / linear probe | 98.5 / 98.5 / 98.5 / 99.25% |
| Detach `fruits` → fruit images unknown | 100% (none mislabelled); identical predictions after re-attach |
| Pizza / lemon refused | 95.5% / 2% |

### Experiment 2 — 19 classes in 4 domains, whole images

| Metric | Result |
|---|---|
| Routing to the right domain | 99.76% |
| Accuracy: MOSAIC / flat kNN / zero-shot / linear probe | 98.68 / 98.76 / 98.63 / 99.08% |
| Detach a domain → its images mislabelled, router gate only / + match gate | animals 31.5% → 3.5%, birds 33.1% → 2.1% |
| Pizza refused, router gate / router + match gate | 74% / 36.5% |

### Experiment 3 — hierarchy: domain → one plug-in leaf per class (`results/results_whole_image_v3.json`)

| Metric | Hierarchical | Flat |
|---|---|---|
| Right domain | 99.76% | — |
| Right class first / in top 3 | 98.47 / 99.66% | 98.53 / 99.82% |
| Descriptor comparisons per query | 102 | 152 |
| Accuracy | 98.66% | 98.74% |

| Unknown objects (≈5.4% of known images refused) | Lemon refused | Pizza refused |
|---|---|---|
| Router gate only | 2.0% | 97% |
| Router + one global match threshold | 4.5% | 57.5% |
| Router + per-leaf gate | 6.5% | 84% |

Plug-out: removing a domain sends 99.5–100% of its images to unknown. Removing one class
mislabels 28.5% of its images as a sibling on average (51% with a global threshold). Predictions are
identical after every re-plug.

**Speed** (`results/speed.json`, RTX 4060): single query (numpy) 0.26 ms. The batched GPU engine
does ~183k queries/s, with 100% agreement with the single-query path. Encoder: fp32 333 img/s,
fp16 426 img/s (cosine to fp32 ≥ 0.99996, same decisions).

### Experiment 4 — object-level: class-agnostic detector, object-only modules

Resolution floor sweep (`results/sweep/`; all rows calibrated to ≈5% of known images unknown):

| Min object side | Objects/img | Accuracy | Extra wrong label | Pizza refused | Class plug-out → mislabelled | Domain plug-out → mislabelled |
|---|---|---|---|---|---|---|
| 0 px | 3.60 | 96.3 | 32.2% | 41% | 46.4% | 22.4% |
| 32 px | 2.85 | 96.4 | 18.8% | 60.5% | 38.3% | 9.6% |
| 48 px | 2.22 | 96.8 | 10.3% | 76% | 37.1% | 4.9% |
| **64 px** | 1.78 | 97.1 | 4.3% | 74% | 30.6% | 1.3% |
| 96 px | 1.30 | 97.7 | 1.8% | 72.5% | 31.8% | 0.5% |
| whole image | 1 | 98.7 | 0.8% | 84% | 28.5% | 0.2% |

Fixes found along the way: consensus picked parts (headlights, wheels) for vehicles (79–91%) until it
was weighted toward whole objects. Picking the image's label by the most similar object gave 79.7%, and
picking it by the area-weighted vote gave 96.3%.

Multi-object test (`results/multi_object.json`, 600 two-object collages, live pipeline):

| | Boxes | Soft masks | Whole image |
|---|---|---|---|
| Both objects named | 83.0% | 85.3% | 0% |
| At least one named | 99.3% | 99.3% | 13.8% |
| Any wrong label | 9.5% | 15.2% | 0.3% |

### Experiment 4b — masks instead of boxes (SAM, box-prompted)

| | Boxes | Mask, flat grey background | Mask, 50% background | Whole image |
|---|---|---|---|---|
| Accuracy | 97.1 | 91.1 | 96.3 | 98.7 |
| Extra wrong label | 4.3% | 20.2% | 7.7% | 0.8% |
| Pizza refused | 74% | 68.5% | 70% | 84% |
| Class plug-out → mislabelled | 30.6% | 53.0% | 35.8% | 28.5% |

Zero-shot CLIP on 514 labelled objects: box crop 97.3%, flat cut-out 91.4%, blurred background 95.9%,
dimmed background 96.7%. CLIP needs some context, so cut-outs keep a 50% background.

### Experiment 5 — people and handheld objects, v1 (`results/parts.json`)

30 leaves in the registry. Test instances of the 11 new classes (forced / with gates): mean 76.1% / 70.5%.

| person | face | hand | arm | leg | phone | bottle | cup | knife | ball | book |
|---|---|---|---|---|---|---|---|---|---|---|
| 79.4 | 93.9 | 93.8 | 57.0 | 67.3 | 72.3 | 61.0 | 77.3 | 83.0 | 72.7 | 79.0 |

The domain gate was recalibrated over all domains: 0.528 → 0.449 (stored in `modules_masks/registry.json`).

### Experiment 6 — human identification model, routing quality (`results/human.json`)

Tree `people/ → {person, face_region/{head, face, hair, eye, nose, mouth, ear}, limbs/{arm, hand, leg, foot}}`,
evaluated in the full registry (with the non-human domains), CLIP ViT-B/32, 3,407 test instances.
Versions: A = ground-truth crops, B = + detector-style positives, C = + negatives (non-elements only) with a reject margin.

| | A | B | C | Flat (C) |
|---|---|---|---|---|
| Human instances routed to `people` | 94.9% | 92.7% | 95.5% | — |
| Right region (person / face region / limbs) | 73.3% | 73.9% | 75.0% | — |
| Router's first part choice | 51.4% | 51.2% | 49.4% | 50.2% |
| Right part among the 3 kept | 84.9% | 83.4% | 84.2% | 82.8% |
| Final part accuracy | 48.2% | 49.9% | 49.3% | 50.5% |
| Errors that are anatomically adjacent (head↔face, arm↔hand …) | 65% | 61% | 64% | — |
| Non-human objects routed to / accepted as people | 2.1 / 1.3% | 2.4 / 1.4% | 3.3 / 1.3% | — |

Per part (A): eye 86, nose 87, mouth 82, ear 70, hand 66, face 61, leg 50, hair 46, arm 38, foot 32, head 32, person 29%.
The detector-proposal evaluation (precision ≤ 0.13 per part) is unreliable, because Open Images doesn't label
every part in a photo, so many "background" proposals are unlabelled people or faces.

Geometry test (kNN on the part's box relative to its person, same split): embedding only 57.4%,
embedding + geometry 75.1%. Note that the geometry features included a flag that gives the person label away,
so the gain is overstated. Head (36 → 53%) and leg (33 → 60%) improved.

### Experiment 7 — which frozen encoder? (`results/encoders.json`)

Same 16.5k soft-mask crops, same split, people tree only, image entries only.

| Encoder | kNN | Routed | Anatomy credit | Right region | Part in top 3 | img/s |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (current) | 56.0 | 56.0 | 86.9 | 76.8 | 89.2 | 353 |
| CLIP ViT-L/14 (DFN-2B) | 57.8 | 58.0 | 89.2 | 81.3 | 91.4 | 57 |
| EVA02-L/14 | 55.6 | 56.1 | 90.2 | 80.0 | 91.7 | 61 |
| **SigLIP SO400M** | **62.2** | **61.9** | **91.2** | **84.0** | **92.8** | 38 |
| CLIP ViT-H/14 (DFN-5B) | 61.2 | 60.5 | 90.8 | 82.8 | 92.6 | 37 |
| DINOv2-B/14 | 56.7 | 56.4 | 86.9 | 77.2 | 88.9 | 208 |
| DINOv2-L/14 | 55.5 | 55.7 | 88.1 | 74.9 | 87.3 | 74 |

Per part, routed (%):

| Encoder | person | head | face | hair | eye | nose | mouth | ear | arm | hand | leg | foot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP B/32 | 55 | 35 | 58 | 49 | 88 | 87 | 78 | 62 | 50 | 67 | 42 | 27 |
| CLIP L/14 | 54 | 30 | 75 | 48 | 86 | 90 | 81 | 73 | 54 | 72 | 44 | 49 |
| EVA02-L | 50 | 30 | 54 | 45 | 91 | 91 | 81 | 75 | 56 | 75 | 54 | 51 |
| SigLIP SO400M | 58 | 35 | 62 | 53 | 91 | 94 | 79 | 82 | 66 | 76 | 51 | 54 |
| CLIP H/14 | 57 | 34 | 65 | 52 | 89 | 91 | 83 | 78 | 59 | 74 | 54 | 57 |
| DINOv2-B | 62 | 35 | 56 | 41 | 84 | 90 | 80 | 70 | 45 | 69 | 36 | 57 |
| DINOv2-L | 61 | 37 | 44 | 38 | 90 | 90 | 86 | 78 | 43 | 67 | 43 | 60 |

### What the results say

1. **Modularity works:** plugging in and out is exact and instant, and whole domains unplug cleanly.
   Removing a single class still leaks about 30% of its images to a sibling.
2. **On object-centric benchmarks MOSAIC matches CLIP, but doesn't beat it,** because every class there is
   already known to CLIP. Object-level recognition is what makes multi-object images work (83% vs 0%).
3. **Body parts are the hard case.** The domain level is solid (95% routed to people, 2–3% false human), but
   parts that contain each other (person ⊃ head ⊃ face, arm ⊃ hand) are separated by extent rather than
   appearance. The best encoder lifts part accuracy from 56% to 62%; with anatomy credit it reaches 91%.
   Relative size and position are the missing signal.

## Roadmap

- [x] Core: module format, registry, router, matcher, pipeline (numpy)
- [x] CLIP encoder + whole-image perceiver, build modules from image folders
- [x] Experiment 1: animals + fruits (4 classes)
- [x] Experiment 2: 19 classes / 4 modules, two-gate abstention
- [x] Experiment 3: hierarchical tree (domain → class leaves), per-leaf gates (see `results/report.md`)
- [x] Experiment 4: class-agnostic object detection, object-only modules (area-aware consensus),
      64 px resolution floor, image-level gate calibration, multi-object collage test (see `results/report.md`)
- [x] Experiment 4b: SAM masks (flat vs soft cut-outs) vs boxes
- [x] Batched GPU engine + fp16 encoder (see `results/speed.json`)
- [x] Experiment 5: people + handheld leaves (v1), `part_of`, registry-level domain gate
- [x] Experiment 6: human identification model (3-level people tree), routing quality, negatives
- [x] Experiment 7: frozen-encoder comparison on body parts (SigLIP SO400M best)
- [ ] Switch to SigLIP SO400M and rebuild all modules; add relative size/position as a routing signal for nested parts
- [ ] Query understanding + compositional checks ("red t-shirt" = `clothing/tshirt` + colour check),
      hierarchy of abstract → basic concepts — design only, see `docs/query_understanding.md`
- [ ] FAISS index for large modules
- [ ] Experiments: routing accuracy, retrieval quality, latency vs. #modules,
      cost of adding a concept vs. fine-tuned baseline
- [ ] Failure-mode study: misrouting, contradictory modules, drift, storage
