"""Benchmark the VLM high tier over local images (Phase 2).

VLM confidence is null by design (calibrated later via closed-set
protocol); scoring uses 0.0 placeholder for the average only.
Usage: .venv/bin/python reports/bench_vlm.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.eval import load_ground_truth, local_image_stems, score


def run_tool(image: Path) -> tuple[str, float, float]:
    start = time.time()
    proc = subprocess.run(
        [sys.executable, ROOT_DIR / "tools" / "drdiag_vlm.py",
         "--image", str(image)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - start
    payload = json.loads(proc.stdout)
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("message", "tool failed"))
    return (
        payload["disease_name"],
        float(payload.get("confidence_score") or 0.0),
        elapsed,
    )


def main() -> None:
    ground_truth = load_ground_truth(ROOT_DIR)
    predictions: dict[str, tuple[str, float]] = {}
    timings: list[float] = []
    failures: list[str] = []
    for stem in local_image_stems(ROOT_DIR):
        image = ROOT_DIR / "dataset" / "ISIC2019" / f"{stem}.jpg"
        try:
            pred, conf, elapsed = run_tool(image)
            predictions[stem] = (pred, conf)
            timings.append(elapsed)
            print(f"{stem}: {pred} {elapsed:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - benchmark must continue
            failures.append(f"{stem}: {exc}")
            print(f"{stem}: FAILED {exc}", flush=True)
    results = score(predictions, ground_truth)
    results["avg_seconds_per_image"] = (
        round(sum(timings) / len(timings), 2) if timings else 0.0
    )
    results["failures"] = failures
    lines = [
        "# Phase 2 Benchmark — drdiag-vlm (tier3_high)",
        "",
        f"- Images attempted: {len(local_image_stems(ROOT_DIR))}, "
        f"scored: {results['n']}",
        f"- Accuracy: {results['accuracy']}",
        "- Avg confidence: n/a (VLM confidence is null by design; "
        "calibrated via closed-set protocol)",
        f"- Avg seconds/image: {results['avg_seconds_per_image']}",
        "",
        "## Per-class accuracy",
        "",
    ]
    for name, slot in results["per_class"].items():
        lines.append(f"- {name}: {slot['accuracy']} (n={slot['n']})")
    lines += ["", "## Failures", ""]
    lines += [f"- {f}" for f in failures] or ["- none"]
    lines += [""]
    (ROOT_DIR / "reports" / "phase2_benchmark_vlm.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Wrote reports/phase2_benchmark_vlm.md", flush=True)


if __name__ == "__main__":
    main()
