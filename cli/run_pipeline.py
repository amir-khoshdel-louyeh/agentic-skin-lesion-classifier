"""Simple CLI runner for the OperationalPipeline."""

from __future__ import annotations

import argparse
import json
import logging

from src.pipeline import OperationalPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run operational pipeline on an HMNIST row index.")
    parser.add_argument("row_index", type=int, help="Row index in hmnist CSV to run pipeline on.")
    parser.add_argument("--no-model", dest="call_model", action="store_false", help="Do not call the vision LLM; use deterministic features only.")
    parser.add_argument("--verbose", action="store_true", help="Print short progress logs during pipeline run (implies --no-model False if not specified)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = OperationalPipeline()

    if args.call_model:
        ok, msgs = pipeline.vl.validate_prerequisites()
        if not ok:
            print("[prereq] vision model prerequisites not satisfied:")
            for m in msgs:
                print(" - ", m)
            raise SystemExit(1)
        if args.verbose:
            print("[prereq] vision model ready")

    report = pipeline.run(row_index=args.row_index, call_model=args.call_model, verbose=args.verbose)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
