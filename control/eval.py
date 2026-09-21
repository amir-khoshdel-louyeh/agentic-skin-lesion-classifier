"""Ground-truth scoring harness (Phase 2, split discipline Phase 10).

Maps local ISIC2019 images to labels via the HAM10000 training CSV,
using the same 7 class names as control.schemas. `heldout_split()` is
the ONLY honest evaluation split: stratified, seeded, disjoint from the
fit pool, and excluding the 24 prompt demo images (which live inside the
training table, so measuring on them is optimistic).
"""

import csv
import random
from pathlib import Path

from control.paths import ROOT_DIR
from control.schemas import HAM10000_CLASSES

HELDOUT_SEED = 42
HELDOUT_PER_CLASS = 25

# Demo images (prompt.yaml) — excluded from every fit pool: they are the
# showcase set, and they sit inside the training table (leakage if scored).
PROMPT_STEMS = {
    "ISIC_0000000", "ISIC_0000001", "ISIC_0000002", "ISIC_0000003",
    "ISIC_0000013", "ISIC_0000074", "ISIC_0024345", "ISIC_0025302",
    "ISIC_0025509", "ISIC_0025513", "ISIC_0025758", "ISIC_0025767",
    "ISIC_0025781", "ISIC_0056166", "ISIC_0056176", "ISIC_0056302",
}

DX_TO_CLASS = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevi",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratoses",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesions",
}


def load_ground_truth(root: Path = ROOT_DIR) -> dict[str, str]:
    csv_path = root / "dataset" / "ISIC2019_full" / "ISIC_2019_Training_GroundTruth.csv"
    labels: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for code, name in DX_TO_CLASS.items():
                if row.get(code) == "1.0" or row.get(code) == "1":
                    labels[row["image"]] = name
    return labels


def score(
    predictions: dict[str, tuple[str, float]],
    ground_truth: dict[str, str],
) -> dict[str, object]:
    """predictions: image_stem -> (predicted_class, confidence)."""
    hits = 0
    total = 0
    conf_sum = 0.0
    per_class: dict[str, dict[str, int]] = {}
    for stem, truth in ground_truth.items():
        if stem not in predictions:
            continue
        pred, conf = predictions[stem]
        total += 1
        conf_sum += conf
        slot = per_class.setdefault(truth, {"n": 0, "hits": 0})
        slot["n"] += 1
        if pred == truth:
            hits += 1
            slot["hits"] += 1
    return {
        "n": total,
        "accuracy": round(hits / total, 4) if total else 0.0,
        "avg_confidence": round(conf_sum / total, 4) if total else 0.0,
        "per_class": {
            name: {
                "n": s["n"],
                "accuracy": round(s["hits"] / s["n"], 4) if s["n"] else 0.0,
            }
            for name, s in sorted(per_class.items())
        },
    }


def local_image_stems(root: Path = ROOT_DIR) -> list[str]:
    test = list((root / "dataset" / "ISIC2019_test").glob("*.jpg"))
    full = list((root / "dataset" / "ISIC2019_full").glob("*/*.jpg"))
    return sorted({p.stem for p in test + full})


def image_path_for_stem(stem: str, root: Path = ROOT_DIR) -> Path | None:
    """Resolve a stem to its file: test dir first, then full class folders."""
    direct = root / "dataset" / "ISIC2019_test" / f"{stem}.jpg"
    if direct.exists():
        return direct
    matches = sorted((root / "dataset" / "ISIC2019_full").glob(f"*/{stem}.jpg"))
    return matches[0] if matches else None


def heldout_split(
    seed: int = HELDOUT_SEED,
    per_class: int = HELDOUT_PER_CLASS,
    root: Path = ROOT_DIR,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratified (stem, label) split: (fit_pool, heldout).

    Deterministic in `seed`. Held-out takes `per_class` stems per class;
    prompt demo stems never enter the fit pool. SCC/UNK-only rows have no
    7-class label and are excluded from both (out-of-coverage, §10.3).
    """
    gt = load_ground_truth(root)
    by_class: dict[str, list[str]] = {}
    for stem, label in gt.items():
        if stem in PROMPT_STEMS:
            continue
        if image_path_for_stem(stem, root) is None:
            continue
        by_class.setdefault(label, []).append(stem)
    rng = random.Random(seed)
    heldout: list[tuple[str, str]] = []
    fit: list[tuple[str, str]] = []
    for label in sorted(by_class):
        stems = sorted(by_class[label])
        rng.shuffle(stems)
        take = min(per_class, len(stems))
        heldout += [(s, label) for s in stems[:take]]
        fit += [(s, label) for s in stems[take:]]
    rng.shuffle(heldout)
    rng.shuffle(fit)
    return fit, heldout
