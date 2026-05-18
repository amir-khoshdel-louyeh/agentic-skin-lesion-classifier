"""Evaluate pipeline predictions against HMNIST/HAM10000 ground truth.

This script performs a conservative melanoma vs non-melanoma evaluation using
the pipeline's synthesized prediction. It streams the CSV to avoid large memory use.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterator

import pandas as pd

from src.pipeline import OperationalPipeline
from src.data_utils import reconstruct_image_from_row


HMNIST_PRESET_LABEL_MAP = {
    0: "nv",
    1: "mel",
    2: "bkl",
    3: "bcc",
    4: "akiec",
    5: "vasc",
    6: "df",
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
    max_rows: int | None = None,
    call_model: bool = False,
    chunksize: int = 1024,
):
    pipeline = OperationalPipeline()

    tp = fp = tn = fn = 0
    total = 0

    reader = pd.read_csv(csv_path, chunksize=chunksize)
    label_col = None

    for chunk in reader:
        if label_col is None:
            label_col = find_label_column(chunk.columns)
        for idx, row in chunk.iterrows():
            if max_rows is not None and total >= max_rows:
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

        if max_rows is not None and total >= max_rows:
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pipeline predictions against HMNIST CSV ground truth.")
    parser.add_argument("csv", help="Path to hm nist CSV file")
    parser.add_argument("--rows", type=int, help="Max rows to evaluate (streamed)")
    parser.add_argument("--call-model", dest="call_model", action="store_true", help="Call the vision LLM for richer descriptions")
    parser.add_argument("--chunksize", type=int, default=1024, help="CSV chunksize for streaming")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(args.csv, max_rows=args.rows, call_model=args.call_model, chunksize=args.chunksize)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
