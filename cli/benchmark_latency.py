"""Benchmark end-to-end pipeline latency across HMNIST samples."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import List

import pandas as pd

from src.pipeline import OperationalPipeline
from src.data_utils import load_hmnist_pixels


def compute_stats(samples: List[float]) -> dict:
    if not samples:
        return {}
    return {
        "count": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "p95": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline latency over HMNIST rows")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to benchmark")
    parser.add_argument("--start-index", type=int, default=0, help="Start row index")
    parser.add_argument("--call-model", action="store_true", help="Call vision LLM during benchmark")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before sampling")
    args = parser.parse_args()

    df = load_hmnist_pixels()
    n = len(df)
    indices = list(range(n))
    if args.shuffle:
        import random

        random.shuffle(indices)

    indices = indices[args.start_index : args.start_index + args.rows]

    pipeline = OperationalPipeline()

    wall_times: List[float] = []
    internal_times: List[float] = []

    for idx in indices:
        start = time.perf_counter()
        out = pipeline.run(row_index=idx, call_model=args.call_model)
        end = time.perf_counter()
        wall = end - start
        wall_times.append(wall)

        # internal timing included in pipeline output if available
        timing = out.get("timing", {})
        internal = timing.get("elapsed_seconds")
        if internal is not None:
            internal_times.append(float(internal))

    report = {
        "wall_clock_stats": compute_stats(wall_times),
        "internal_stats": compute_stats(internal_times),
        "samples": len(wall_times),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
