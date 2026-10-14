# MOSAIC

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-server-green)](#mcp-tools)
[![Status](https://img.shields.io/badge/status-research%20preview-orange)](#roadmap)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**M**odular **O**bject-agnostic **S**emantic **A**rchitecture with **I**ndexed **C**oncepts — an **MCP server**
that gives a **text-only LLM** sight without giving it pixels.

The model calls tools (find objects, identify, pose, measure, teach, forget, explain), gets back structured
facts it can check (names, boxes, similarities, joint angles), and does all the reasoning itself. No
vision-language model is in the loop, and no image ever leaves your machine. The domain shipped today is
**people**: the person, the body parts, and the body's geometry.

Benchmarked against Gemini and a local Qwen VLM — [jump to results](#benchmark-results). Term paper:
*Modular Object-Agnostic Vision Architecture Using Embedding-Based Semantic Routing* (see `../synopsis/`).

## Quick start

```bash
git clone <this repo> && cd mosaic
python3 -m venv .venv && .venv/bin/pip install -e ".[vision,mcp]"
python scripts/fetch_human.py && python scripts/cache_human_crops.py \
  && python scripts/compare_encoders.py --only siglip-so400m \
  && python scripts/extract_poses.py && python scripts/build_people.py   # builds modules_people/, ~10 min on a GPU

claude mcp add mosaic -- /path/to/mosaic/.venv/bin/python -m mosaic.server   # or any other MCP client
```

See [Setup](#setup) for the full build pipeline and [Usage](#usage) for the CLI / Python API / MCP registration.

## Why not just use a VLM?

A VLM compresses an image into a caption-like answer inside its weights. MOSAIC keeps perception outside the
model, in parts you can inspect, calibrate and swap. That makes some things possible that a VLM can't do:

| | A VLM | LLM + MOSAIC | Status |
|---|---|---|---|
| **Learn a new visual concept in a conversation** ("that's my mug") and keep it | Its weights are frozen; it forgets when the context ends | `teach` builds a module from objects in the session. It's recognised on the next call and saved | built |
| **Forget on demand** (privacy, a wrong concept) | It can't unlearn | `forget` plugs the module out and deletes it. Removal is exact, and nothing of it remains | built |
| **Say "I don't know"** | It hallucinates with confidence | Each module's own gate (calibrated on its own examples) answers *unknown*, and says which module came closest and by how much | built |
| **Show its evidence** | No provenance | `explain` returns the routing scores, the gate, and the stored examples (source photo + box) behind an answer | built |
| **Exact body geometry** (angles, directions, which way the elbow turns) | Can't measure pixels | A skeleton plus geometry rules: values like `left_elbow_angle = 92°`, computed, not guessed | built |
| **New geometric concepts without training** | Needs fine-tuning | `define_rule("waving", "...")`: the LLM names a concept once, and it's evaluated on every person from then on | built |
| **Small details** | The input is downsampled | `find_objects(region=…)` zooms into a region before detecting | built |
| **Privacy / cost** | The image goes to the model provider | Images stay local; only text reaches the LLM | built |
| **Reason over thousands of photos** ("every photo where she's sitting") | Limited by its context window | Index a collection once, then query it | planned |

The claim to test in the paper is a benchmark of **LLM + MOSAIC vs. a frontier VLM**, with one task group per
row above (see the roadmap).

### One real question, three answers

Run 2's `locate_075`: **"Give the box of one hand"**, on a photo of two people
(`data/human/images/oi_01037.jpg`). Ground truth has two valid hand boxes; a prediction counts if it clears
IoU ≥ 0.5 against either one.

**Gemini** (one call, sees the pixels):
```
ANSWER: [856, 765, 1000, 936]
```
IoU 0.0 — the wrong hand, and no way to tell it was wrong.

**Qwen2.5-VL** (one call, sees the pixels):
```
[575, 928, 710, 1086]
```
IoU 0.472 — the right hand, just outside the hit threshold, with no signal that it was close.

**Claude + MOSAIC** (7 tool calls, text only, no pixels):
```
1. load_image(path=".../oi_01037.jpg")                                     -> img_77
2. pose(image_id="img_77")                     -> hands_estimated_at left_hand=[598, 1010]
3. find_objects(image_id="img_77", region=[500,880,680,1024])              -> obj_624
4. identify(object_id="obj_624")               -> known=false (routed to arm, box too broad)
5. find_objects(image_id="img_77", region=[550,950,660,1024])    -> obj_651, obj_652
6. identify(object_id="obj_652")      -> known=true, "people/limbs/hand", similarity 0.777
7. identify(object_id="obj_651")      -> known=false (not relevant)

ANSWER: [551, 950, 658, 1016]
```
IoU 0.535 — a hit. `pose()`'s estimated hand position said *where* to zoom; the first candidate's `identify()`
gate rejected it as **unknown**, not as a wrong label, so the model zoomed tighter instead of guessing; the
second candidate matched the registered hand module and got named. Every step was a fact it could check, not a
caption read off a downsampled image.

Same comparison exists for all 25 `locate` questions in `results/bench_ollama.jsonl` (`qid: locate_075` here).

## Benchmark results

100 questions (presence / count / locate) against Open Images ground truth, the same images and questions for
every system. Full methodology and discussion: [`docs/benchmark_ollama.md`](docs/benchmark_ollama.md); raw rows
in `results/bench_ollama.jsonl`.

```
presence — "is there a person?"  (accuracy)
script           █████████████████████████████████████░░░   92%
slm + MOSAIC     ███████████████████████████████░░░░░░░░░   78%
vlm (pixels)     ██████████████████████████████████████░░   94%
gemini (pixels)  ████████████████████████████████████░░░░   90%
claude + MOSAIC  █████████████████████████████████████░░░   92%

count — "how many people?"  (exact match)
script           ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░   36%
slm + MOSAIC     ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   29%
vlm (pixels)     ███████████████████░░░░░░░░░░░░░░░░░░░░░   48%
gemini (pixels)  ███████████████████░░░░░░░░░░░░░░░░░░░░░   48%
claude + MOSAIC  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░   36%

locate — "box of one <part>"  (IoU ≥ 0.5 hit rate)
script           ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    4%
vlm (pixels)     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   20%
gemini (pixels)  ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   28%
claude + MOSAIC  ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   28%
```

- `vlm` / `slm` = local Qwen 2.5 7B family; `gemini` = `gemini-flash-latest`; `claude + MOSAIC` = Claude
  (claude-sonnet-5) calling MOSAIC's MCP tools, no pixels, one subagent per question; `script` = MOSAIC with a
  fixed policy and no LLM at all.
- **An LLM driving MOSAIC's tools matches a frontier VLM on `locate` and comes close on `presence`** — the
  clearest result so far for the paper's central claim — but spends far more tool calls and tokens to get there
  (≈12 tool calls/question vs. 1 pixel-only call).
- **`count` is still where tool-only reasoning loses to raw pixels**: both VLMs beat both tool-driving LLMs by
  a wide margin.
- `slm`'s `locate` rows haven't been run yet; see the doc for what's still open.

## Architecture

```
image ─► Detector (OWLv2 objectness + box heads only: finds "things", knows no labels)
           │  box #1   box #2   ...                       ┌──────────────────────────────────────┐
           ├──► SAM mask ─► soft cut-out ─► Encoder ──────► Router: people ─► face_region / limbs  │
           │                              (SigLIP SO400M,  │   ─► top-3 parts ─► 5-NN vote ─►     │
           │                               frozen)         │   leaf gate (known / unknown)        │
           │                                               └────────────────┬─────────────────────┘
           └──► person boxes ─► Skeleton (ViTPose, 17 joints) ──┐           │
                                                                ├─► geometry: which joints fall in each
                                                                │   part's box re-ranks look-alike parts
                                                                │   (head vs face, arm vs hand)
                                                                └─► rules: named expressions over joints
                                                                    (angles, directions, sitting, facing ...)
                   ▲
   text-only LLM ──┘  MCP tools: find_objects, identify, pose, measure, define_rule, teach, forget, explain ...
```

- **Nothing here has a label set except the registry.** The detector, SAM and the pose model are frozen
  class-agnostic sensors. What things *are* lives only in the modules, and what poses *mean* lives only in the rules.
- **Modules** are plug-in leaves (`people/limbs/hand`) under groups (`people/limbs`). Each stores object
  embeddings with provenance, its own accept gate, `part_of` (hand → arm → person), and an **anatomical
  signature**. The signature records where the part sits on a skeleton (which joints its box holds, its size,
  its height on the person) and is fitted on the module's own examples, so it travels with the module.
- **Rules** are data (`src/mosaic/people_rules.json`, plus rules taught in sessions), not code.
- The contract between components is unchanged: every vector is an L2-normalised `float32` from one `encoder_id`.
  A module built with another encoder is a hard error.

| Component | File | Responsibility |
|---|---|---|
| Detector | `detector.py` | class-agnostic boxes + objectness |
| Segmenter | `segmenter.py` | box → mask (SAM) |
| Encoder | `encoder.py` | frozen image/text embeddings (SigLIP SO400M by default; CLIP, DINOv2 also supported) |
| Pose | `pose.py` | person box → 17 joints (ViTPose), in image pixels |
| Geometry | `geometry.py` | skeletons, anatomical signatures, the rule language |
| Module / Registry | `module.py`, `registry.py` | plug-in leaves and the tree; group descriptors summarise children |
| Router / Matcher / Mosaic | `router.py`, `matcher.py`, `pipeline.py` | coarse-to-fine routing, k-NN, gates |
| VisionSession | `capability.py` | the stateful tool layer: stable object ids, zoom, identify + geometry, pose, teach/forget, explain |
| MCP server | `server.py` | exposes VisionSession as MCP tools |
| BatchEngine | `engine.py` | the same pipeline as batched GPU tensor ops |

## MCP tools

| Tool | What it returns |
|---|---|
| `load_image(path)` | image id, size |
| `find_objects(image_id, region?)` | class-agnostic objects with stable ids; a region is zoomed in first |
| `identify(object_id \| image_id+box)` | module, known/unknown and why, similarity vs. the gate, top-3 candidates, `part_of`, and what the skeleton contributed |
| `find_parts(object_id)` | objects inside an object, each identified |
| `search(image_id, query, region?)` | objects matching a module path (with a window scan), or free text (SigLIP match probability) |
| `relate(a, b)` | mask overlap, containment, touching, direction, distance |
| `pose(image_id \| object_id)` | per person: joints, estimated hand positions, every rule evaluated (facts / broken constraints / unknown) |
| `measure(person_id, expression)` | any geometry expression on one skeleton |
| `define_rule(name, expression, …)` / `list_rules()` | add a named concept at run time / the vocabulary |
| `teach(label, object_ids)` / `forget(name)` | plug a new module in from session objects / plug a module, group or rule out |
| `explain(object_id)` | routing scores per level, gates, the stored examples it matched, the runner-up, the geometry |
| `list_modules(path?)` | the tree |

## Geometry rules

A rule is a named expression over joints (`left_shoulder`, `left_elbow`, `left_wrist`, …, plus derived `neck`,
`mid_hip`, `left_hand`, `right_hand`). The body's "kinematic chain" reads naturally. For example, *"from the
shoulder the upper arm goes down, then at the elbow the arm turns"*:

```
left_upper_arm   direction(left_shoulder, left_elbow)                     -> "down"
left_forearm     direction(left_elbow, left_wrist)                        -> "right"
left_elbow_turn  turn(left_shoulder, left_elbow, left_wrist)              -> "left"   (bends left in the image)
left_elbow_angle angle(left_shoulder, left_elbow, left_wrist)             -> 91.0
left_arm_L       goes(left_shoulder, left_elbow, "down", 35) and between(left_elbow_angle, 60, 120)   -> true
```

- **Functions:** `angle`, `heading`, `direction`, `goes(a, b, word, tol)`, `turn`, `toward`, `above`/`below`/
  `left_of`/`right_of`, `dist` (in torso lengths), `px`, `x`/`y` (within the person box), `visible`,
  `between`, `touches(point, obj_N)`, `inside(point, obj_N)`, plus `abs`, `min`, `max` and `round`.
- **Rules can use other rules and objects:** `holding_phone = touches(right_hand, obj_7) and right_arm_L`.
- **Three-valued logic:** a rule that needs a joint the skeleton couldn't see is *unknown*, and it reports the
  missing joints. It is never silently false. An `or` that one visible side already decides is still answered.
- **Two kinds of rule.** *Facts* are reported as values. *Constraints* are anatomy checks, such as upper arm vs.
  forearm length, and they're reported only when they fail, which flags a skeleton that's probably wrong.
- **Safety:** expressions are parsed, not `eval`ed. Only whitelisted syntax and functions are allowed.
- **Coordinates:** directions are in the image frame (up = top of the image). `left_*` / `right_*` joints are
  the person's own sides, so for someone facing the camera, their left arm is on the image right.

49 rules ship by default: elbow and knee angles, arm directions, raised, hanging, L-shape, pointing and
direction, hands up, hand near face, arms crossed, hands on hips, facing, torso upright or horizontal, sitting,
standing, squatting, on all fours, arms spread, plus 7 anatomy constraints.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"                 # core: numpy only
.venv/bin/pip install -e ".[vision,mcp]"          # models (torch, open_clip, transformers) + MCP server
.venv/bin/pytest
```

Build the people domain (Open Images, 1,764 photos, 16.5k instances). Each step caches its output:

```bash
python scripts/fetch_human.py                                # photos + boxes -> data/human/
python scripts/cache_human_crops.py                          # SAM soft-mask cut-out of every instance
python scripts/compare_encoders.py --only siglip-so400m      # SigLIP embeddings of the cut-outs
python scripts/extract_poses.py                              # ViTPose skeleton per annotated person
python scripts/build_people.py                               # modules_people/ + results/people.json
```

(Run scripts from the repo root with `PYTHONPATH=scripts`.)

## Usage

Give an LLM the tools (Claude Code shown; any MCP client works):

```bash
claude mcp add mosaic -- /path/to/mosaic/.venv/bin/python -m mosaic.server
# env: MOSAIC_MODULES (default modules_people), MOSAIC_TAUGHT (default ./taught: taught modules + rules)
```

See what the tools would report, from the terminal:

```bash
python mosaic_cli.py photo.jpg                             # objects, identities, skeletons, pose facts
python mosaic_cli.py photo.jpg --rule "angle(left_hip, left_knee, left_ankle)"
python mosaic_cli.py photo.jpg --save out.png              # annotated image, for you (the LLM never sees it)
python mosaic_cli.py --tree
PYTHONPATH=scripts python scripts/demo_reasoning.py photo.jpg   # a scripted loop: pose, then what's in each hand
```

```python
from mosaic.capability import VisionSession
s = VisionSession("modules_people", taught_dir="taught")
img = s.load_image("photo.jpg")["image_id"]
objs = s.find_objects(img)["objects"]
s.identify(objs[0]["object_id"])            # module, known, candidates, geometry
people = s.pose(img)["people"]              # joints + facts per person
s.measure(people[0]["person_id"], "turn(left_shoulder, left_elbow, left_wrist)")
s.define_rule("waving", "left_arm_raised and left_elbow_angle < 150")
s.teach("my_mug", ["obj_5"]); s.forget("taught/my_mug")
```

## Identification accuracy

All encoders and sensors are frozen; nothing is fine-tuned. Raw numbers are in `results/`. Experiments 1–7
(five domains, CLIP, the hierarchy, masks, the encoder comparison) are in
[`docs/earlier_experiments.md`](docs/earlier_experiments.md).

### Experiment 8: the people domain on SigLIP, plus skeleton geometry (`results/people.json`)

Open Images, same image-hash split as experiments 6–7 (70% build / 10% calibration / 20% test; 11,434 build
and 3,407 test instances). There are 12 leaves. Accuracy is forced (no gates), so it's comparable to experiment 7.
The geometry weight was chosen on the calibration split.

| | Accuracy | Anatomy credit (head↔face, arm↔hand … counted right) |
|---|---|---|
| Routed tree, 5-NN vote (appearance only) | 61.9% | 91.2% |
| Best leaf by its 3 nearest examples (appearance only) | 61.4% | 91.0% |
| **+ anatomical signature on the skeleton** | **74.5%** | **94.0%** |
| Signature alone (no appearance; instances with a skeleton) | 68.2% | 85.8% |

| Per part (%) | person | head | face | hair | eye | nose | mouth | ear | arm | hand | leg | foot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Appearance (routed) | 58 | 35 | 62 | 53 | 91 | 94 | 79 | 82 | 66 | 76 | 51 | 54 |
| + geometry | **88** | **43** | **67** | 53 | **97** | **97** | **85** | **94** | **75** | **84** | **69** | **62** |

- **Appearance and geometry fail differently, so combining them works.** Parts that contain each other look
  alike but sit differently on the skeleton. The biggest gains are for person (+30 points), legs (+18) and ears (+12).
- **Head vs. face vs. hair is still the main confusion** (head→face 117 and head→hair 82 of the 870 errors).
  Their boxes hold the same joints and differ only in extent.
- **Caveats.** Skeletons here were estimated in the ground-truth person boxes, and 96.4% of test instances
  have one. In a live session, person boxes come from the detector and the appearance model. For a person
  cut off by the frame, the box-relative position can mislead (a hand at the bottom of a cropped portrait
  can read as a leg). Measuring from the neck in torso lengths avoids that, but it scored lower (71.3%), so
  the box-relative version ships. The domain gate for "is this human at all" is calibrated on human parts only
  (1st percentile, 0.578). Non-human rejection with SigLIP isn't measured yet.

## Roadmap

- [x] Core: module format, registry, router, matcher, gates; batched GPU engine
- [x] Experiments 1–7: five domains, hierarchy, class-agnostic detection, masks, people tree, encoder comparison
- [x] Scope: one domain (people), SigLIP SO400M, LLM access through MCP only (no VLM in the loop)
- [x] Skeleton sensor (ViTPose) and the geometry rule language, with 49 default rules
- [x] Anatomical signatures per leaf; geometry re-ranking in `identify` (experiment 8: 61.9 → 74.5%)
- [x] Run-time `teach` / `forget` / `define_rule` / `explain`
- [x] **Benchmark: LLM + MOSAIC vs. a frontier VLM — first result.** presence/count/locate on 100 questions;
      Claude + MOSAIC matches Gemini on `locate`, trails on `count` ([results](#benchmark-results)). Still
      missing: novel-instance recognition after `teach`, unlearning after `forget`, abstention on out-of-domain
      objects, exact pose geometry (angles, left/right), evidence quality
- [ ] Collection index: `index_collection` / `query_collection` over thousands of photos
- [ ] Handheld objects as taught modules, and a "holding" evaluation (hand–object contact)
- [ ] Head / face / hair: an extent-aware feature (the box's share of the head region)
- [ ] Non-human rejection with SigLIP (negatives / reject margin, as in experiment 6 version C)
- [ ] Attribute checks (colour of a mask, size, count), from [`docs/query_understanding.md`](docs/query_understanding.md)
- [ ] Deterministic body-plan gate: classify a mask's topology (limb count/symmetry via skeletonization or
      contour analysis) before embedding-based routing, independent of any learned module; route only the
      registry subtree consistent with that class, and boost/flag routing confidence when the embedding match
      and the deterministic class agree or disagree
- [ ] True 3D joint angles from multi-view input: ORB-based keypoint tracking across frames plus SLAM-style
      triangulation to recover depth, replacing the current 2D image-plane approximation in `geometry.py`
      (`angle()` on projected joints is foreshortening-prone single-view)

## Contributing

Issues and PRs welcome — the open items in the roadmap above are the best starting points. `pytest` covers
the core pipeline; run it before sending a PR. New domains follow the same module/registry contract as `people`
(see [Architecture](#architecture)).

## License

[MIT](LICENSE).
