"""Evaluate pipeline predictions against the ISIC 2019 ground truth.

This script performs a conservative melanoma vs non-melanoma evaluation using
the pipeline's synthesized prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

from src.pipeline import OperationalPipeline
from src.ingest import ISICAdapter


def is_ground_truth_melanoma(val) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"mel", "melanoma"} or "melan" in s


def predict_melanoma_from_prediction(pred: str) -> bool:
    if not pred:
        return False
    p = pred.lower()
    if "melanoma" in p or "mel" in p:
        return True
    if "suspicious" in p:
        return True
    return False


def evaluate(
    dataset_root: str,
    row_start: int | None = None,
    row_end: int | None = None,
    call_model: bool = False,
):
    adapter = ISICAdapter(dataset_root)
    pipeline = OperationalPipeline(data_adapter=adapter)

    tp = fp = tn = fn = 0
    total = 0
    class_counts: dict[str, int] = defaultdict(int)

    for row_index in range(adapter.n_samples()):
        if row_start is not None and row_index < row_start:
            continue
        if row_end is not None and row_index >= row_end:
            break

        image = adapter.get_image(row_index)
        desc = pipeline.visual_feature_extraction(image, call_model=call_model)
        abcd = pipeline.check_abcd_criteria(desc)
        onc = pipeline.trigger_oncology_context(abcd, desc)
        refs = pipeline.literature_search(desc)
        report = pipeline.evidence_synthesis(desc, abcd, onc, refs)

        pred_mel = predict_melanoma_from_prediction(report.prediction)

        gt_val = adapter.get_label(row_index)
        gt_str = str(gt_val).lower() if gt_val is not None else ""
        class_counts[gt_str] += 1
        gt_mel = is_ground_truth_melanoma(gt_val)

        if gt_mel and pred_mel:
            tp += 1
        elif not gt_mel and pred_mel:
            fp += 1
        elif not gt_mel and not pred_mel:
            tn += 1
        elif gt_mel and not pred_mel:
            fn += 1

        total += 1

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_counts": dict(class_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pipeline predictions against the ISIC 2019 ground truth.")
    parser.add_argument("dataset_root", nargs="?", default="dataset", help="Path to the dataset root containing the ISIC 2019 files")
    parser.add_argument("--rows", type=int, nargs=2, metavar=("START", "END"), help="Row range to evaluate (START END, e.g. --rows 500 1000)")
    parser.add_argument("--call-model", dest="call_model", action="store_true", help="Call the vision LLM for richer descriptions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # If model calls are requested, validate prerequisites once before streaming
    if args.call_model:
        from models.vision_language import OllamaVisionLanguageInterface

        vl = OllamaVisionLanguageInterface()
        ok, msgs = vl.validate_prerequisites()
        if not ok:
            print("[prereq] vision model prerequisites not satisfied:")
            for m in msgs:
                print(" - ", m)
            raise SystemExit(1)
        print("[prereq] vision model ready")

    row_start = args.rows[0] if args.rows else None
    row_end = args.rows[1] if args.rows else None
    
    metrics = evaluate(args.dataset_root, row_start=row_start, row_end=row_end, call_model=args.call_model)

    print(json.dumps(metrics, indent=2))
    # Persian summary lines
    total = metrics.get("total", 0)
    tp = metrics.get("tp", 0)
    pct = (tp / total * 100) if total else 0.0
    print()
    print(f"از بین {total} نمونه عکس، سیستم موفق شد {tp} نمونه سرطانی را درست تشخیص دهد.")
    print(f"این معادل {pct:.1f}% از کل نمونه‌ها است.")


if __name__ == "__main__":
    main()
