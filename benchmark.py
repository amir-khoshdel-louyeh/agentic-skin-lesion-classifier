"""Local benchmark harness (proposal.tmp §10).

Scores ready diagnostic tools against the labelled dataset using
`control.eval` (stratified held-out split, no training). This is the
missing entry point for the promised "benchmark the local images before
role assignment": without it `heldout_split`/`score` were never called.

Usage:
  python benchmark.py --dry-run                       # build split, run nothing
  python benchmark.py --split heldout --limit 35 \
      --tools ham10000-cnn,multimodal-fusion,ensemble-high
  python benchmark.py --split demo --tools ham10000-cnn

Running tools needs the model weights + GPU; `--dry-run` and `--split`
inspection work anywhere. Results print as a Markdown table and land in
`report/benchmark_<timestamp>.md` (report/ is git-ignored).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.eval import (heldout_split, image_path_for_stem,  # noqa: E402
                          load_ground_truth, local_image_stems, score)
from control.manifest import load_manifest  # noqa: E402
from control.paths import resolve  # noqa: E402


def build_split(kind: str, per_class: int, limit: int,
                root: Path = ROOT_DIR) -> tuple[list[tuple[str, str]],
                                                 dict[str, str]]:
    """Return (cases, ground_truth) for `kind`.

    - heldout: the honest evaluation split from control.eval.
    - fit:     the (non-held-out) fit pool — for reference only.
    - demo:    the tracked prompt images; scored but optimistic (they sit
               inside the training table), so use for smoke-checks only.
    """
    if kind == "demo":
        truth = load_ground_truth(root)
        cases = [(stem, truth[stem]) for stem in local_image_stems(root)
                 if stem in truth]
    else:
        fit, heldout = heldout_split(per_class=per_class, root=root)
        cases = heldout if kind == "heldout" else fit
    if limit > 0:
        cases = cases[:limit]
    return cases, dict(cases)


def predictions_for_tool(tool, stems: list[str], timeout: int = 300,
                         root: Path = ROOT_DIR) -> dict[str, tuple[str, float]]:
    """Run one diagnostic tool per image; keep only success + real numbers.

    Any crash, non-JSON, or missing class/confidence is skipped (never a
    guess) so a tool bug can't inflate or depress the score silently.
    """
    predictions: dict[str, tuple[str, float]] = {}
    for stem in stems:
        image = image_path_for_stem(stem, root)
        if image is None:
            continue
        cmd = [sys.executable, str(resolve(tool.command)), "--image",
               str(image)]
        if "metadata" in tool.requires:
            cmd += ["--metadata", "{}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=str(root))
            payload = json.loads(proc.stdout)
        except Exception:  # noqa: BLE001 - a failed tool is a skipped row
            continue
        if proc.returncode != 0 or payload.get("status") != "success":
            continue
        cls = payload.get("disease_name")
        conf = payload.get("confidence_score")
        if cls and conf is not None:
            predictions[stem] = (str(cls), float(conf))
    return predictions


def format_table(results: dict[str, dict]) -> str:
    lines = ["| tool | n | accuracy | avg confidence |",
             "|------|---|----------|----------------|"]
    for name, res in results.items():
        lines.append(
            f"| {name} | {res.get('n', 0)} | "
            f"{res.get('accuracy', 0.0):.4f} | "
            f"{res.get('avg_confidence', 0.0):.4f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ready diagnostic tools on the local dataset "
        "(no training; see proposal.tmp §10)."
    )
    parser.add_argument("--split", choices=("heldout", "fit", "demo"),
                        default="heldout",
                        help="Which cases to score (default: heldout).")
    parser.add_argument("--per-class", type=int, default=25,
                        help="Held-out images per class (default: 25).")
    parser.add_argument("--limit", type=int, default=50,
                        help="Cap the number of cases (0 = all).")
    parser.add_argument("--tools",
                        default="ham10000-cnn",
                        help="Comma-separated ready diagnostic tool names.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-inference timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the split and print counts; run no tools.")
    return parser


def resolve_tools(manifest, names: list[str]) -> list:
    """Ready diagnostic entries for `names`; SystemExit on anything else."""
    tools = []
    for name in names:
        try:
            entry = manifest.by_name(name)
        except KeyError:
            raise SystemExit(f"Unknown tool: {name!r}.")
        if entry.status != "ready" or entry.category != "diagnostic":
            raise SystemExit(
                f"{name!r} is not a ready diagnostic tool — not benchmarkable."
            )
        tools.append(entry)
    return tools


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(ROOT_DIR / "tools" / "manifest.yaml")
    names = [n.strip() for n in args.tools.split(",") if n.strip()]
    tools = resolve_tools(manifest, names)
    cases, ground_truth = build_split(args.split, args.per_class, args.limit)
    stems = [stem for stem, _ in cases]
    print(f"Split: {args.split} — {len(stems)} cases, "
          f"tools: {', '.join(names)}")
    if not stems:
        raise SystemExit("No cases matched; nothing to score.")
    if args.dry_run:
        per_class_counts: dict[str, int] = {}
        for _, label in cases:
            per_class_counts[label] = per_class_counts.get(label, 0) + 1
        for label, count in sorted(per_class_counts.items()):
            print(f"  {label}: {count}")
        print("dry-run: no tools executed.")
        return

    results: dict[str, dict] = {}
    for tool in tools:
        print(f"Running {tool.name} on {len(stems)} cases…")
        predictions = predictions_for_tool(tool, stems, timeout=args.timeout)
        results[tool.name] = score(predictions, ground_truth)
    table = format_table(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT_DIR / "report" / f"benchmark_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Local benchmark (research use only)\n\n"
        f"- Split: {args.split} ({len(stems)} cases)\n"
        f"- Threshold-free: raw tool accuracy, no training\n\n"
        f"{table}\n\n"
        "Research use only — not a medical diagnosis.\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
