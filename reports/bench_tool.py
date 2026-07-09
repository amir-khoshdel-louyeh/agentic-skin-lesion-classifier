"""Benchmark one diagnostic tool over local images (Phase 2).

Runs the tool CLI per image, scores against ground truth, writes a
Markdown report. Usage:
  .venv/bin/python reports/bench_tool.py
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

TOOL = ["tools/ham10000_cnn.py", "ham10000-cnn", "tier1_fast"]


def run_tool(image: Path) -> tuple[str, float, float]:
    start = time.time()
    proc = subprocess.run(
        [sys.executable, ROOT_DIR / TOOL[0], "--image", str(image)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start
    payload = json.loads(proc.stdout)
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("message", "tool failed"))
    return payload["disease_name"], float(payload["confidence_score"]), elapsed


def main() -> None:
    ground_truth = load_ground_truth(ROOT_DIR)
    predictions: dict[str, tuple[str, float]] = {}
    timings: list[float] = []
    failures: list[str] = []
    stems = local_image_stems(ROOT_DIR)
    for stem in stems:
        image = ROOT_DIR / "dataset" / "ISIC2019" / f"{stem}.jpg"
        try:
            pred, conf, elapsed = run_tool(image)
            predictions[stem] = (pred, conf)
            timings.append(elapsed)
            print(f"{stem}: {pred} ({conf}) {elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001 - benchmark must continue
            failures.append(f"{stem}: {exc}")
            print(f"{stem}: FAILED {exc}")
    results = score(predictions, ground_truth)
    results["tool"] = TOOL[1]
    results["tier"] = TOOL[2]
    results["avg_seconds_per_image"] = (
        round(sum(timings) / len(timings), 2) if timings else 0.0
    )
    results["failures"] = failures
    report = ROOT_DIR / "reports" / "phase2_benchmark.md"
    lines = [
        "# Phase 2 Benchmark — ham10000-cnn (tier1_fast)",
        "",
        f"- Images attempted: {len(stems)}, scored: {results['n']}",
        f"- Accuracy: {results['accuracy']}",
        f"- Avg confidence: {results['avg_confidence']}",
        f"- Avg seconds/image: {results['avg_seconds_per_image']}",
        "",
        "## Per-class accuracy",
        "",
    ]
    for name, slot in results["per_class"].items():
        lines.append(f"- {name}: {slot['accuracy']} (n={slot['n']})")
    lines += ["", "## Failures", ""]
    lines += [f"- {f}" for f in failures] or ["- none"]
    lines += [
        "",
        "## Role assignment",
        "Role decision for this tool is recorded after reviewing the table "
        "above against the 0.75 borderline policy (proposal.tmp Section 8).",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
