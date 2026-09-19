# Query understanding and compositional concepts (design, not built)

Status: **design note only.** Right now MOSAIC is being built as the routing part:
image → objects → the **basic element** each object is (`people/arm`, `handheld/phone`, …).
Query understanding and reasoning described here come later.

## Principle: basic elements, not "final boss" modules

A module holds one **basic element**: something with a stable look that can be recognised
on its own (hand, arm, t-shirt, phone, dog). We do **not** build modules for combinations
("red t-shirt", "man holding a phone", "dog on a sofa"). Those are answered by composing:

    basic element (from the registry)  +  simple checks (attributes, relations)

This keeps the registry small, and every module stays reusable. There's also no need to
collect data for every combination: no "red t-shirt" dataset, just `t-shirt` plus a colour check.

## Query understanding (future)

A query is parsed into an element, attributes and relations, then answered with
routing + cheap logic:

| Query | Element (registry) | Check (logic, no training) |
|---|---|---|
| red t-shirt | `clothing/tshirt` | colour of the mask pixels is red (hue histogram) |
| large dog on the left | `animals/dog` | mask area fraction; box centre in the left third |
| two cups | `handheld/cup` | count of accepted objects ≥ 2 |
| person holding a phone | `people/hand`, `handheld/phone` | hand and phone masks touch (`relate`) |
| something in the hand | `people/hand` | any object whose mask touches or overlaps the hand's |

Flow for "red t-shirt":
1. Parse the query → element `tshirt`, attribute `colour = red`.
2. Route to the element: find objects, keep those identified as `clothing/tshirt`
   (or search the region for `clothing/tshirt`).
3. Check each one: compute the dominant colour of its mask pixels, and keep it if it's red.
4. Answer with the boxes and masks that pass, plus the ones that failed the check (e.g. "a blue t-shirt at …").

Checks are plain functions over what the model already produces (mask pixels, box, area,
position, relations between masks). Adding a new attribute means adding a function, not data.

## Hierarchy of concepts

Terms come at different levels of abstraction, and the registry tree mirrors that:

    people/            (group: abstract)
      person           (whole)
      arm, leg, hand, face   (basic elements, part_of: people/person)
    clothing/          (group: abstract, future)
      tshirt, trousers, shoe …   (basic elements)

- **Abstract terms map to groups.** "clothing" means anything under `clothing/`, and
  "body part" means anything with `part_of: people/person`.
- **Labelling a clothed arm as `people/arm` is correct** at the basic-element level: the
  region *is* an arm. That it's covered by a sleeve is a second, overlapping element
  (`clothing/tshirt`) found separately. One region can carry several elements at
  different levels (arm + sleeve of a red t-shirt).
- Query understanding resolves abstract words to groups first ("clothing" → `clothing/*`),
  then routes to basic elements inside them.

## What this means for current work

- The router's job ends at **basic elements**. Accuracy targets are about getting the element
  right (arm vs leg, hand vs face), not about composite descriptions.
- Background suppression is still wanted for things that are no element at all (wall,
  sky, blur). Clothing on a limb isn't "background" for the limb.
- Not built yet: the query parser, attribute checks (colour, size, position, count), the
  relation logic on top of `relate`, and the `clothing/` domain.
