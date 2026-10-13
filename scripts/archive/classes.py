"""Class definitions shared by fetch_data.py and run_experiment.py.

Each class lists its image sources as (source, labels, count):
  cats_vs_dogs  microsoft/cats_vs_dogs, label 0=cat 1=dog
  imagenet      ImageNet-1k class index (real-world photos)
  fruits360     PedroSampaio/fruits-360 label indices (studio shots, white background)

Fruits mix ImageNet scenes with studio shots so that neither background style
becomes a shortcut for one class.
"""

DOMAINS = {
    "animals": {
        "cat": [("cats_vs_dogs", [0], 1000)],
        "dog": [("cats_vs_dogs", [1], 1000)],
        "elephant": [("imagenet", [386], 1000)],
        "zebra": [("imagenet", [340], 1000)],
    },
    "birds": {
        "eagle": [("imagenet", [22], 1000)],
        "owl": [("imagenet", [24], 1000)],
        "parrot": [("imagenet", [88], 1000)],
        "flamingo": [("imagenet", [130], 1000)],
        "penguin": [("imagenet", [145], 1000)],
    },
    "vehicles": {
        "sports car": [("imagenet", [817], 1000)],
        "pickup truck": [("imagenet", [717], 1000)],
        "school bus": [("imagenet", [779], 1000)],
        "scooter": [("imagenet", [670], 1000)],
        "airplane": [("imagenet", [404], 1000)],
    },
    "fruits": {
        "apple": [("imagenet", [948], 700), ("fruits360", [0, 1, 2, 3, 4, 5, 6, 7], 300)],
        "orange": [("imagenet", [950], 700), ("fruits360", [66], 300)],
        "banana": [("imagenet", [954], 700), ("fruits360", [11, 12, 13], 300)],
        "strawberry": [("imagenet", [949], 700), ("fruits360", [101], 300)],
        "pineapple": [("imagenet", [953], 700), ("fruits360", [86, 87], 300)],
    },
}

# Out-of-domain probes for abstention: lemon is near (fruit-like), pizza is far.
OOD = {
    "lemon": [("imagenet", [951], 200)],
    "pizza": [("imagenet", [963], 200)],
}

LABELS = [c for classes in DOMAINS.values() for c in classes]
DOMAIN_OF = {c: d for d, classes in DOMAINS.items() for c in classes}


def class_dir(root, cls):
    """data/raw/<domain>/<class> for known classes, data/raw/ood/<class> for probes."""
    return root / DOMAIN_OF.get(cls, "ood") / cls.replace(" ", "_")
