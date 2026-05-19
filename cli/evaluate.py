"""Evaluate pipeline predictions against HMNIST/HAM10000 ground truth.

This script performs a conservative melanoma vs non-melanoma evaluation using
the pipeline's synthesized prediction. It streams the CSV to avoid large memory use.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterator
from collections import defaultdict

import pandas as pd

from src.pipeline import OperationalPipeline
from src.data_utils import reconstruct_image_from_row


HMNIST_PRESET_LABEL_MAP = {
    # mapping adapted to this CSV's encoding: label->dx
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "nv",
    5: "vasc",
    6: "mel",
}


def find_label_column(columns: Iterator[str]) -> str | None:
    for candidate in ("label", "dx", "diagnosis", "cell_type", "cell_type_idx"):
        if candidate in columns:
            return candidate
    return None


def is_ground_truth_melanoma(val, mapping=HMNIST_PRESET_LABEL_MAP) -> bool:
    if pd.isna(val):
        return False
    try:
        iv = int(val)
        return mapping.get(iv) == "mel"
    except Exception:
        s = str(val).lower()
        return "mel" in s or "melan" in s


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
    csv_path: str,
    row_start: int | None = None,
    row_end: int | None = None,
    call_model: bool = False,
    chunksize: int = 1024,
):
    pipeline = OperationalPipeline()

    tp = fp = tn = fn = 0
    total = 0
    row_index = 0
    class_counts: dict = defaultdict(int)

    reader = pd.read_csv(csv_path, chunksize=chunksize)
    label_col = None

    for chunk in reader:
        if label_col is None:
            label_col = find_label_column(chunk.columns)
        for idx, row in chunk.iterrows():
            # Skip rows before row_start
            if row_start is not None and row_index < row_start:
                row_index += 1
                continue
            
            # Stop after row_end
            if row_end is not None and row_index >= row_end:
                break

            image = reconstruct_image_from_row(row)
            desc = pipeline.visual_feature_extraction(image, call_model=call_model)
            abcd = pipeline.check_abcd_criteria(desc)
            onc = pipeline.trigger_oncology_context(abcd, desc)
            refs = pipeline.literature_search(desc)
            report = pipeline.evidence_synthesis(desc, abcd, onc, refs)

            pred = report.prediction
            pred_mel = predict_melanoma_from_prediction(pred)

            gt_val = row[label_col] if label_col is not None else None
            # normalize ground-truth label to a string (e.g. 'mel','nv',...)
            def _label_to_str(v):
                if pd.isna(v):
                    return ""
                try:
                    iv = int(v)
                    return HMNIST_PRESET_LABEL_MAP.get(iv, str(v)).lower()
                except Exception:
                    return str(v).lower()

            gt_str = _label_to_str(gt_val)
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
            row_index += 1

        if row_end is not None and row_index >= row_end:
            break

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
    parser = argparse.ArgumentParser(description="Evaluate pipeline predictions against HMNIST CSV ground truth.")
    parser.add_argument("csv", help="Path to hmnist CSV file")
    parser.add_argument("--rows", type=int, nargs=2, metavar=("START", "END"), help="Row range to evaluate (START END, e.g. --rows 500 1000)")
    parser.add_argument("--call-model", dest="call_model", action="store_true", help="Call the vision LLM for richer descriptions")
    parser.add_argument("--chunksize", type=int, default=1024, help="CSV chunksize for streaming")
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
    
    metrics = evaluate(args.csv, row_start=row_start, row_end=row_end, call_model=args.call_model, chunksize=args.chunksize)

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
