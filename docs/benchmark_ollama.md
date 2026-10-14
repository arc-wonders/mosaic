# Benchmark: LLM + MOSAIC vs. vision-language models

Status: **run 2 complete** except `slm`'s `locate` task. Script: `scripts/bench_ollama.py`.

## Setup

| System | What it is |
|---|---|
| `vlm` | `qwen2.5vl:7b` (local, Ollama) given the image, no tools |
| `gemini` | `gemini-flash-latest` (cloud) given the image, no tools — a frontier VLM, not just a 7B local one |
| `slm` | `qwen2.5:7b-instruct` (local, Ollama), text only, calling MOSAIC's 14 MCP tools (same schemas and functions as `mosaic.server`, called in-process), max 10 steps |
| `claude_agent` | Claude (claude-sonnet-5), text only, calling the same MOSAIC MCP tools — one Claude Code subagent per question, run through the `Workflow` tool so 100 questions fan out concurrently |
| `script` | MOSAIC with a fixed policy and no LLM: presence = any object identified as `people/*`; count = people found by `pose`; locate = best `search()` result for the part's module |

`vlm`/`slm` are the same Qwen 2.5 7B family (Q4_K_M), temperature 0, on one RTX 4060 Laptop (8 GB).
`gemini`/`claude_agent` are remote API calls and need no local GPU budget beyond MOSAIC's own ~2.4 GB.

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

## Run 2: after the fixes, all 5 systems (`results/bench_ollama.*`)

Questions differ from run 1 for count and locate: resampling the no-person photos shifted the random draw. Compare
systems within a run, not across runs. 100 questions per system (50 presence, 25 count, 25 locate), same images and
ground truth for every system. `slm`'s `locate` rows were never run (see Open questions).

| | script | slm + MOSAIC | vlm (pixels) | gemini (pixels) | claude + MOSAIC |
|---|---|---|---|---|---|
| Presence, accuracy | 92% | 78% | **94%** | 90% | 92% |
| Presence, yes / no | 96% / 88% | 96% / 60% | 92% / 96% | 92% / 88% | **100%** / 84% |
| Count, exact match | 36% | 29% | **48%** | **48%** | 36% |
| Count, mean abs. error | 1.08 | 1.21 | 0.96 | **0.80** | 1.00 |
| Locate, hit rate (IoU≥0.5) | 4% | not run | 20% | **28%** | **28%** |
| Locate, mean best IoU | 0.161 | not run | 0.353 | 0.350 | **0.359** |
| Mean tool calls / question | 0 | 5.9 | 1 (none) | 1 (none) | 12.0 |

- **Claude driving the MOSAIC tools matches a frontier VLM on `locate`** (28% hit rate, same mean IoU) and is close
  on `presence` (92% vs 90%, with every "yes" answered right) — the clearest evidence yet for the paper's central
  claim. It used ~12 tool calls/question on average (up to 20+ on `count`) and far more tokens than one Gemini call,
  so this is accuracy parity bought with much more compute, not a cheaper path.
- **`count` is still where tool-only reasoning loses to raw pixels.** Both VLMs (`vlm`, `gemini`) beat both
  tool-driving LLMs (`slm`, `claude_agent`) by a wide margin. Reasoning from `pose`/`find_objects` facts seems to
  undercount people that direct vision catches, especially in crowded images — worth a closer look at whether the
  LLMs are calling `pose` enough, or MOSAIC's detector is merging adjacent people.
- **The fixed-policy `script` baseline is MOSAIC's floor, not its ceiling.** It ties `claude_agent` on `count` (36%)
  but is far behind on `locate` (4% vs 28%) — an LLM choosing *which* `search()`/`find_parts()` call to make, instead
  of always taking the first result, is most of what separates "the tools exist" from "the tools answer the
  question."

## Open questions and next steps

- **Why the scripted count fell** from 68% (run 1) to 36% (run 2) is still open, but one hypothesis is ruled
  out. `scripts/diag_count_regression.py` reran run 2's 25 count questions against both the pre-negatives
  modules (`modules_people_pre_negatives/`, git-ignored like every `modules_*`) and the current ones, after
  `pose()` was changed to filter people by `identify()`'s `routed_module` (the router's raw pick) instead of
  the gated label (so a rejected gate can no longer silently drop a real person from the count). Both module
  sets now score identically: 36% exact, MAE 1.08 (`results/diag_count_regression.json`). So the negatives
  gate isn't the cause of the regression after all — something else changed between run 1 and run 2.
- **A different, over-counting failure mode** (found outside the benchmark set, on a sprint-start photo with a
  dense stadium-crowd background): `find_objects` can return a spurious second "person" that isn't a stray
  detector box, but a three-stage chain. (1) OWLv2's class-agnostic objectness genuinely scores a wide,
  short band of crowd texture as object-like (0.24, above the 0.1 threshold) — not a NMS duplicate. (2) SAM,
  given that box, fails to isolate the sliver of real person inside it and masks most of the crowd as
  foreground too. (3) The resulting mixed crowd+person embedding still passes the `people/person` gate, but
  barely: similarity 0.761 against a required 0.675, beating the nearest stored negative ("sports equipment",
  0.773) by only 0.017. `pose()` then fits a skeleton to this blob, reusing the real runner's few visible
  joints, producing a near-duplicate "person" alongside the real one. This is the mirror image of the
  undercount hypothesis above (the gate rejecting a real person) — here it's over-accepting a background
  region — and is the kind of case the planned deterministic body-plan gate (see roadmap) is meant to catch:
  a mask shaped like a thin horizontal band isn't person-shaped regardless of what it embeds as.
- **Finish `slm`'s `locate` rows**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
  scripts/bench_ollama.py --systems slm --tasks locate`. Finished rows are skipped. The LLM and MOSAIC now share the
  GPU (`--swap` restores time-sharing).
- **Guide the SLM's tool use** (not yet changed, so the runs stay comparable): count with `pose`, locate with
  `search(image, "people/limbs/<part>")`.
- **Part localisation** is the weakest capability for every tool-driving system: head / face / hair extents, and
  `search`'s window scan.
- **Tasks still to add**: pose geometry (needs keypoint ground truth, e.g. COCO), and `teach` / `forget`, where a VLM
  has no equivalent.
