"""Ground-truth scoring harness (Phase 2).

Maps the 24 local ISIC2019 images to labels via the HAM10000 training
CSV, using the same 7 class names as control.schemas.
"""

import csv
from pathlib import Path

from control.paths import ROOT_DIR
from control.schemas import HAM10000_CLASSES

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
