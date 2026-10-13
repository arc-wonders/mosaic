"""Box-annotated classes for people, body parts and things people hold.

Each class lists (source, category names). Instances are taken from the
datasets' own boxes, so no consensus selection is needed.
  humanparts  finedet/humanparts        (person / face / hand; MIT annotations), COCO-style xywh
  openimages  finedet/openimages        validation shards (CC-BY 4.0), xywh
  coco        detection-datasets/coco   train shards (COCO 2017, CC-BY 4.0 annotations), xyxy
"""

PART_DOMAINS = {
    "people": {
        "person": [("humanparts", ["person"])],
        "face": [("humanparts", ["face"])],
        "hand": [("humanparts", ["hand"])],
        "arm": [("openimages", ["Human arm"])],
        "leg": [("openimages", ["Human leg"])],
    },
    "handheld": {
        "phone": [("coco", ["cell phone"])],
        "bottle": [("coco", ["bottle"])],
        "cup": [("coco", ["cup"])],
        "knife": [("coco", ["knife"])],
        "ball": [("coco", ["sports ball"])],
        "book": [("coco", ["book"])],
    },
}

# part -> whole; stored in each leaf's manifest so a reasoning model sees the relation
PART_OF = {c: "people/person" for c in ["face", "hand", "arm", "leg"]}

PART_LABELS = [c for classes in PART_DOMAINS.values() for c in classes]
PART_DOMAIN_OF = {c: d for d, classes in PART_DOMAINS.items() for c in classes}
