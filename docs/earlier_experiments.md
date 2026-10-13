# Earlier experiments (1–7)

These ran before the project narrowed to the people domain and the MCP tool. Experiments 1–5 used
CLIP ViT-B/32 across five domains (animals, birds, vehicles, fruits, handheld objects); their
scripts are in `scripts/archive/` and their module directories (`modules_whole/`, `modules_objects/`,
`modules_masks/`) are no longer used. Experiments 6–7 are about the people domain and led to its
current design: the three-level tree and the switch to SigLIP SO400M.

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
