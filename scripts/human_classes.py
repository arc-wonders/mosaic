"""The human identification model: people and body parts, all from Open Images.

Open Images annotates the person and every body part in the same photos, so a
detector proposal that matches no annotated box really is background.

Routing tree (3 levels):   people / {person, face_region/*, limbs/*}
Anatomy (part_of):         eye, nose, mouth -> face -> head -> person; hair, ear -> head
                           hand -> arm -> person; foot -> leg -> person
"""

HUMAN = {  # leaf path -> Open Images category names
    "people/person": ["Person", "Man", "Woman", "Boy", "Girl"],
    "people/face_region/head": ["Human head"],
    "people/face_region/face": ["Human face"],
    "people/face_region/hair": ["Human hair"],
    "people/face_region/eye": ["Human eye"],
    "people/face_region/nose": ["Human nose"],
    "people/face_region/mouth": ["Human mouth"],
    "people/face_region/ear": ["Human ear"],
    "people/limbs/arm": ["Human arm"],
    "people/limbs/hand": ["Human hand"],
    "people/limbs/leg": ["Human leg"],
    "people/limbs/foot": ["Human foot"],
}

PART_OF = {
    "people/face_region/head": "people/person",
    "people/face_region/face": "people/face_region/head",
    "people/face_region/hair": "people/face_region/head",
    "people/face_region/ear": "people/face_region/head",
    "people/face_region/eye": "people/face_region/face",
    "people/face_region/nose": "people/face_region/face",
    "people/face_region/mouth": "people/face_region/face",
    "people/limbs/arm": "people/person",
    "people/limbs/hand": "people/limbs/arm",
    "people/limbs/leg": "people/person",
    "people/limbs/foot": "people/limbs/leg",
}

LEAVES = list(HUMAN)
NAME = {leaf: leaf.rsplit("/", 1)[1] for leaf in LEAVES}          # short label, e.g. "eye"
OI_TO_LEAF = {oi: leaf for leaf, names in HUMAN.items() for oi in names}
