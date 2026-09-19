# MOSAIC — Experiment 4: object-level recognition (boxes and masks)

**Date:** 2026-09-19 · **Detector:** OWLv2 objectness + box heads only (`google/owlv2-base-patch16-ensemble`, no text tower, no class head)
**Encoder (frozen):** `open_clip/ViT-B-32/laion2b_s34b_b79k` · **GPU:** RTX 4060 Laptop
Raw numbers: `results_objects.json`, `results_whole.json`, `multi_object.json`, `sweep/` · Reproduce:
`fetch_data.py` → `extract_objects.py` → `run_experiment.py [--regions whole]` → `multi_object_test.py`

## What changed

Earlier experiments embedded the whole image, background included. MOSAIC is now object-level end to end:

1. **Detect objects without knowing what they are.** OWLv2's objectness head scores each patch token's box by how
   object-like it is. No label set or text query is involved. Boxes under 64 px on a side are dropped, and if
   nothing remains, the whole image is used (10.5% of images).
2. **Store only object embeddings.** When a class leaf is built, each example image's labelled object is chosen by
   **consensus**: the candidate that recurs across that class's images, leaning toward whole objects rather than their parts.
3. **Classify every object** through the same tree (domain → top-3 classes → 5-NN → leaf gate).
   The image-level answer is the known label covering the most image area.

## What went wrong on the way (and the fixes)

| Problem | Cause | Fix |
|---|---|---|
| Vehicle leaves stored headlights, wheels, mirrors (79–91% of picks) | Parts recur across a class more reliably than whole objects do | Consensus score adds 0.5·√(box area) |
| Accuracy 79.7% | "Most similar object" answered for the image, and parts matched stored parts best | Answer = label covering the most area (96.3%) |
| 32–38% of images got an extra wrong label; pizza refused only 34.5% | Tiny (about 40 px) blurry fragments look alike to CLIP and were accepted by some leaf | Ignore objects under 64 px on a side |
| Gates calibrated per object, but decisions are per image | More objects per image means more chances of a false accept | Leaf-gate strictness calibrated so that 5% of *calibration images* come out unknown |

Resolution-floor sweep (all rows at about 5% of known images unknown):

| Min side | obj/img | Accuracy | Extra wrong label | Pizza refused | Class plug-out → mislabelled | Domain plug-out → mislabelled |
|---|---|---|---|---|---|---|
| 0 px | 3.60 | 96.3 | 32.2% | 41% | 46.4% | 22.4% |
| 32 px | 2.85 | 96.4 | 18.8% | 60.5% | 38.3% | 9.6% |
| 48 px | 2.22 | 96.8 | 10.3% | 76% | 37.1% | 4.9% |
| **64 px** | **1.78** | **97.1** | **4.3%** | **74%** | **30.6%** | **1.3%** |
| 96 px | 1.30 | 97.7 | 1.8% | 72.5% | 31.8% | 0.5% |
| whole image | 1 | 98.7 | 0.8% | 84% | 28.5% | 0.2% |

At 96 px, 25% of images fall back to the whole image, so 64 px is the chosen trade-off.

## Results (64 px, 3,800 single-object test images)

| | Objects | Whole image |
|---|---|---|
| Accuracy (no abstention) | 97.1% | 98.7% |
| Accuracy with gates / unknown rate | 93.3% / 5.0% | 93.8% / 5.4% |
| True label among the image's objects | 98.2% | — |
| Routing of the labelled object: domain / class first / class in top-3 | 99.5 / 97.0 / 99.3% | 99.8 / 98.5 / 99.7% |
| Pizza refused / lemon refused | 74% / 7% | 84% / 6.5% |
| Class plug-out → mislabelled (mean of 19) | 30.6% | 28.5% |
| Domain plug-out → mislabelled | 0.3–2.9% | ~0% |
| Identical after every plug-out and replug | ✓ | ✓ |
| Text → objects: correct leaf / precision@50 | 19/19 / 100% | 19/19 / 100% |

Baselines on the same object regions: flat 5-NN 97.3%, zero-shot CLIP 97.1%, linear probe 97.7%.

## Multi-object test (600 collages of two classes, full live pipeline)

| | Objects | Whole image |
|---|---|---|
| **Both objects named** | **83.0%** | 0% (impossible by design) |
| At least one named | 99.3% | 13.8% |
| Correct labels per collage | 1.82 | 0.14 |
| Any wrong label | 9.5% | 0.3% |
| Nothing named | 0.7% | 85.8% |

A whole-image embedding of two objects looks like neither, so the gates mostly reject it. Most of the object
pipeline's misses are small or distant objects under the 64 px floor, such as far-off elephants or an eagle in the sky.
Sample: `multi_object_sample.jpg`.

## Masks instead of boxes (Experiment 4b)

Each box is segmented with SAM (`facebook/sam-vit-base`, box-prompted and class-agnostic). The object is embedded as a
cut-out, and mask area replaces box area in consensus and in the image-level vote.
Pipeline: `extract_masks.py [--keep K]` → `run_experiment.py --regions masks --mask-keep K`.

| (3,800 single-object images; 600 collages) | Boxes | Masks, flat grey background | **Masks, 50% background** | Whole image |
|---|---|---|---|---|
| Accuracy | **97.1** | 91.1 | 96.3 | 98.7 |
| Accuracy with gates / unknown | 93.3 / 5.0 | 89.3 / 4.3 | 93.2 / 4.8 | 93.8 / 5.4 |
| Extra wrong label | **4.3%** | 20.2% | 7.7% | 0.8% |
| Pizza refused | **74%** | 68.5% | 70% | 84% |
| Class plug-out → mislabelled | **30.6%** | 53.0% | 35.8% | 28.5% |
| Domain plug-out → mislabelled | **1.3%** | 26.7% | 5.3% | 0.2% |
| Collages: both objects named | 83.0% | — | **85.3%** | 0% |
| Collages: any wrong label | **9.5%** | — | 15.2% | 0.3% |

**Why flat cut-outs fail.** Zero-shot CLIP accuracy on 514 labelled test objects was 97.3% for box crops, but only
91.4% for flat cut-outs on grey. It was 95.9% with the background blurred and 96.7% with the background dimmed 50%.
Which of SAM's three masks is used barely matters (91.4 vs 91.6%). CLIP was trained on natural photos, so it
recognises a floating object on flat grey noticeably worse, and it also loses useful context (water around a flamingo).
The soft mask (`keep=0.5`) keeps the object sharp and the background faint.

**Verdict on these data:** boxes are slightly better on single-object images. Soft masks name both objects in a
collage a little more often, but add more wrong labels. The objects here are well separated, so masks have
little to gain. Their expected advantage is overlapping or cluttered objects, where one box contains parts of another,
and that needs a real multi-object benchmark to test.

## Interpretation

1. **Object-level recognition is what makes multi-object images work.** It names both objects in 83% of collages; whole-image names both in none.
2. **On single-object benchmarks it costs about 1.6 points of accuracy and some unknown detection.** The detector sometimes
   splits or crops the subject, and the encoder loses context. These datasets are object-centric, so the whole-image
   embedding is already mostly the object.
3. **The detector must not be trusted blindly.** Parts and tiny fragments caused most failures. Area-aware consensus,
   area-weighted answers and a resolution floor are all needed; without them accuracy was 79.7%.
4. **Modularity is unchanged:** plugging a module in or out is still exact and instant, and whole domains unplug cleanly.

## Next steps
- Multi-scale crops (upsample small objects from the original image) instead of a hard 64 px floor
- A benchmark with real multi-object scenes (e.g. COCO or LVIS subsets) to replace the collages
- Sibling-aware leaf gates (class plug-out is still about 30%)
- CLIP-unfamiliar domains, to test whether leaves add knowledge
