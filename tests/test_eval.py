"""Evaluation harness checks (proposal.tmp §10, stdlib only).

Run: .venv/bin/python tests/test_eval.py
Exit non-zero on any failure.

Locks in: ground-truth loading, the honest stratified held-out split
(deterministic, disjoint, prompt images excluded), accuracy scoring, and
the benchmark entry point that finally calls them (tool resolution +
split building + table rendering) without needing GPU/weights.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

import benchmark  # noqa: E402
from control.eval import (  # noqa: E402
    PROMPT_STEMS,
    heldout_split,
    image_path_for_stem,
    load_ground_truth,
    prompt_stems,
    score,
)
from control.manifest import load_manifest  # noqa: E402
from control.schemas import HAM10000_CLASSES  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# 1. Ground truth: labels use the shared 7-class vocabulary.
truth = load_ground_truth()
check("ground-truth-nonempty", len(truth) > 1000, str(len(truth)))
check("ground-truth-classes",
      set(truth.values()) <= set(HAM10000_CLASSES),
      str(set(truth.values()) - set(HAM10000_CLASSES)))

# 2. Path resolution: known test stem resolves; junk does not.
known = image_path_for_stem("ISIC_0000000")
check("resolve-known-stem", known is not None and known.exists(),
      str(known))
check("resolve-unknown-stem", image_path_for_stem("NOPE_0000000") is None)

# 3. Held-out split: deterministic, disjoint, prompt images excluded.
fit1, held1 = heldout_split()
fit2, held2 = heldout_split()
check("split-deterministic", fit1 == fit2 and held1 == held2)
fit_stems = {s for s, _ in fit1}
held_stems = {s for s, _ in held1}
check("split-disjoint", not (fit_stems & held_stems))
check("split-excludes-prompts", not ({s for s, _ in fit1} & PROMPT_STEMS)
      and not ({s for s, _ in held1} & PROMPT_STEMS))
# The literal list went stale once (8 of 24 demo images were missing, so
# ISIC_0025504 leaked into held-out), so the real invariant is checked
# against prompt.yaml itself, not against the constant.
prompt_yaml_stems = {
    Path(rec["image_path"]).stem
    for rec in yaml.safe_load((ROOT / "prompt.yaml").read_text(encoding="utf-8"))
}
demo = prompt_stems()
check("prompt-stems-derives-from-prompt-yaml", prompt_yaml_stems <= demo,
      str(sorted(prompt_yaml_stems - demo)))
check("split-excludes-prompt-yaml-stems",
      not (fit_stems & demo) and not (held_stems & demo),
      str(sorted((fit_stems | held_stems) & demo)))
check("split-nonempty", len(held1) > 0 and len(fit1) > 0,
      f"held={len(held1)} fit={len(fit1)}")
per_class: dict[str, int] = {}
for _, label in held1:
    per_class[label] = per_class.get(label, 0) + 1
check("split-stratified", set(per_class) == set(HAM10000_CLASSES)
      and all(v <= 25 for v in per_class.values()), str(per_class))

# 4. Scoring: accuracy + per-class, missing stems ignored.
preds = {"a": ("Melanoma", 0.9), "b": ("Melanoma", 0.8),
         "c": ("Basal cell carcinoma", 0.7)}
gt = {"a": "Melanoma", "b": "Basal cell carcinoma", "c": "Basal cell carcinoma",
      "missing": "Melanoma"}
result = score(preds, gt)
check("score-counts-only-predicted", result["n"] == 3, str(result))
check("score-accuracy", result["accuracy"] == round(2 / 3, 4), str(result))
check("score-per-class",
      result["per_class"]["Basal cell carcinoma"]["accuracy"] == 0.5,
      str(result))

# 5. Benchmark entry point resolves only ready diagnostic tools.
manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
tools = benchmark.resolve_tools(manifest, ["ham10000-cnn", "multimodal-fusion"])
check("resolve-tools-ok", [t.name for t in tools] ==
      ["ham10000-cnn", "multimodal-fusion"])
for bad in ("quality-gate", "not-a-tool"):
    try:
        benchmark.resolve_tools(manifest, [bad])
        check(f"resolve-tools-rejects:{bad}", False, "no SystemExit")
    except SystemExit:
        check(f"resolve-tools-rejects:{bad}", True)

cases, gt_map = benchmark.build_split("heldout", per_class=5, limit=7)
check("benchmark-build-split", len(cases) == 7
      and gt_map == dict(cases), f"{len(cases)}")

# 6. Table renders each scored tool.
table = benchmark.format_table({"ham10000-cnn": score(preds, gt)})
check("benchmark-table", "| ham10000-cnn | 3 |" in table, table)

# 7. A failing tool yields no predictions (never a guessed row).
from control.manifest import ToolEntry  # noqa: E402

broken = ToolEntry(name="broken", command="tools/does_not_exist.py",
                   category="diagnostic", requires=["image"], modality="image")
check("benchmark-broken-tool-skipped",
      benchmark.predictions_for_tool(broken, ["ISIC_0000000"]) == {})

# 8. The calibration gate requires melanoma sensitivity, not just accuracy.
from tools.calibrate import gate  # noqa: E402

base = {"n": 175, "accuracy": 0.70, "ece": 0.05,
        "per_class": {"Melanoma": {"recall": 0.75}}}
check("gate-accepts-calibrated-and-sensitive",
      gate(base)["passed"] is True, str(gate(base)))
blind = {**base, "per_class": {"Melanoma": {"recall": 0.50}}}
check("gate-rejects-low-melanoma-recall",
      gate(blind)["passed"] is False
      and any("melanoma_recall" in reason
              for reason in gate(blind)["reasons"]),
      str(gate(blind)))

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)
