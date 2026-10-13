# Benchmark: a small text-only LLM + MOSAIC vs. a local VLM (Ollama, Qwen)

Status: **in progress** (run 2 stopped part-way on 2026-09-28). Script: `scripts/bench_ollama.py`.

## Setup

| System | What it is |
|---|---|
| `vlm` | `qwen2.5vl:7b` given the image, no tools |
| `slm` | `qwen2.5:7b-instruct`, text only, calling MOSAIC's 14 MCP tools (same schemas and functions as `mosaic.server`, called in-process), max 10 steps |
| `script` | MOSAIC with a fixed policy and no LLM: presence = any object identified as `people/*`; count = people found by `pose`; locate = best `search()` result for the part's module |

Both LLMs are the same Qwen 2.5 7B family (Q4_K_M), temperature 0, on one RTX 4060 Laptop (8 GB).

Tasks, 25 questions each, scored against Open Images ground truth on the **test** split (images MOSAIC never stored):

- **presence**: "Is there a person?" on 25 people photos (yes) and 25 photos without people (no) from `data/raw` (animals,
  vehicles, fruit, pizza). Run 2 draws the no-person photos from the raw test split only and skips 8 photos checked by
  eye to contain people (`CONTAINS_PEOPLE` in the script).
- **count**: "How many people?" on photos with 1–5 annotated persons, each at least 48 px. Exact match.
- **locate**: "Give the box of one face / head / hand / foot / arm / leg." Hit if IoU ≥ 0.5 with any annotated box of that part.

Raw rows (every answer, the tool calls, the full reply) are in `results/bench_ollama*.jsonl`.

## Run 1: before the fixes (`results/bench_ollama_v1.*`, `results/bench_mosaic_scripted.json`)

Presence numbers use corrected labels: 3 of the no-person photos showed people and 1 showed a painted face (dropped).

| | vlm | slm + MOSAIC | script |
|---|---|---|---|
| Presence, people photos | 25/28 | 26/28 | 26/28 |
| Presence, no-person photos | **20/21** | 9/21 | 8/21 |
| Count, exact / mean abs. error | 64% / 1.56 | 48% / 0.92 | **68% / 0.52** |
| Locate, hit rate / mean IoU | **20%** / 0.32 | 4% / 0.09 | 8% / 0.16 |
| Seconds per question | 2 | 60–90 | — |

- **MOSAIC accepted almost anything as a person or part** (orange → nose, strawberry → mouth, dog → person, pizza → hand).
  The per-leaf gates were calibrated on human parts only. The script failed the same way as the SLM, so this is MOSAIC's
  fault, not the LLM's.
- **Counting from skeletons is MOSAIC's strength** (MAE 0.52 vs 1.56), but the 7B model rarely called `pose`. It summed
  `identify` results and counted arms as extra people.
- **Locating parts is weak on both sides.** Only 5 of 25 targets had any MOSAIC candidate with IoU ≥ 0.5. Head (10 of
  the 25 questions) almost always failed.
- **MOSAIC ran out of GPU memory** next to the LLM, and even alone in `search()`. Run 1 had to unload the LLM before
  every tool call, which is why it took 60–90 s per question.

## Fixes made after run 1

1. **Memory** (`encoder.py`, `capability.py`). SigLIP's image tower is now stored in fp16. Before, `fp16` meant autocast
   only and the weights stayed in fp32. The text tower moved to the CPU (1.8 GB, used only by free-text `search`). The
   window scan now embeds 32 crops per batch instead of 256. MOSAIC's peak went from **6.76 GB to 2.41 GB** (1.57 GB
   resident). Embeddings are unchanged: min cosine to the stored ones was 0.99986 on 256 crops.
2. **Non-human rejection** (`scripts/build_negatives.py`, `results/negatives.json`). Each people leaf now stores
   counter-examples (kind `negative`) and a `reject_margin`: it accepts an object only if its nearest example beats its
   nearest counter-example by the margin. Counter-examples come from the build split only: non-human objects annotated in
   the Open Images people photos, plus whatever MOSAIC's detector finds in `data/raw` photos without people. The margin
   is the 3rd percentile of each leaf's calibration instances. Birds, zebra, pizza and lemon are held out entirely.
   `identify`'s geometry re-ranking, its candidate list and `explain` now apply the same gate (`KnowledgeModule.accepts`).

   | Test split | Before | After |
   |---|---|---|
   | Human parts accepted (ground-truth crops) | 91.0% | 89.3% |
   | Non-human Open Images objects rejected | 38.2% | 65.3% |
   | Non-human, seen categories, rejected | 60.0% | 92.5% |
   | Non-human, held-out categories, rejected | 65.2% | 88.1% |

   Caveat: raw detections that SigLIP's text tower scored as more person-like than their category were dropped, to keep
   riders and owners out of the counter-examples. That filter also applied to the test and held-out sets, so the rejection
   rates may be optimistic. The image-level benchmark below doesn't have this bias.

## Run 2: after the fixes (partial, `results/bench_ollama.*`)

Questions differ from run 1 for count and locate: resampling the no-person photos shifted the random draw. Compare
systems within a run, not across runs. The `slm` system hasn't run yet.

| | vlm | script |
|---|---|---|
| Presence, people photos | 23/25 | **24/25** |
| Presence, no-person photos | **24/25** | 22/25 (run 1: 8/21) |
| Count, exact / mean abs. error | **48% / 0.96** | 32% / 1.16 |
| Locate, hit rate / mean IoU | **20% / 0.35** | 0% / 0.16 (21 of 25 done) |

## Open questions and next steps

- **Why the scripted count fell** from 68% (run 1) to 32%. The questions are different, but the counter-examples might
  also be rejecting real persons before `pose`. To check: run the run-2 count questions with the backup modules and the
  new ones. The pre-negatives modules are in `modules_people_pre_negatives/` (git-ignored, like every `modules_*`).
- **Finish run 2** (`slm`): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=scripts .venv/bin/python
  scripts/bench_ollama.py --n 25`. Finished rows are skipped. The LLM and MOSAIC now share the GPU (`--swap` restores
  time-sharing).
- **Guide the SLM's tool use** (not yet changed, so the runs stay comparable): count with `pose`, locate with
  `search(image, "people/limbs/<part>")`.
- **Part localisation** is the weakest capability: head / face / hair extents, and `search`'s window scan.
- **Tasks still to add**: pose geometry (needs keypoint ground truth, e.g. COCO), and `teach` / `forget`, where a VLM
  has no equivalent.
