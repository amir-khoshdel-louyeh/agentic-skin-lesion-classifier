"""Local, LLM-free calibration of the diagnostic tool pool.

Runs every diagnostic tool through its real CLI on the honest held-out
split (control.eval), then reports accuracy, balanced accuracy, per-class
recall/precision, a 7x7 confusion matrix, ECE/MCE reliability bins, and
(where the tool exposes class probabilities) NLL/Brier.

Gate: a tool enters the production pool only if
  * it produced predictions on the full split,
  * accuracy >= MIN_ACCURACY (0.60),
  * ECE <= MAX_ECE (0.10), and
  * melanoma recall >= MIN_MELANOMA_RECALL (0.70).
The melanoma bar is separate from aggregate accuracy: a model can look
accurate while almost never flagging melanoma (the initial B4/ResNet
fits sat at 0.52-0.56). Failures are written with passed=false and
control.brain.diagnostic_pool drops them.

Temperature scaling (`--fit-temperature`) runs the same tools at T=1.0
on a disjoint calibration split, fits T by NLL grid search, and writes
`--temperature-out` (models/<name>/calibration.json for the new models).
No API calls and no agent sessions: tools only.

Usage:
  .venv/bin/python tools/calibrate.py --tools all \
      --per-class 30 --json-dir calibration --report calibration/SUMMARY.md
  .venv/bin/python tools/calibrate.py --tools isic-efficientnet-b4 \
      --split calibration --fit-temperature \
      --temperature-out models/isic-efficientnet-b4/calibration.json
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from control.calibration import (  # noqa: E402
    MAX_ECE,
    MIN_ACCURACY,
    MIN_MELANOMA_RECALL,
    fit_temperature,
)
from control.eval import (heldout_split, image_path_for_stem,  # noqa: E402
                          split_for_training)
from control.manifest import load_manifest  # noqa: E402
from control.paths import resolve  # noqa: E402
from control.schemas import HAM10000_CLASSES  # noqa: E402

CLASSES = list(HAM10000_CLASSES)
MIN_N = 150
N_BINS = 15


def load_metadata(root: Path = BASE_DIR) -> dict[str, dict]:
    """age/sex/localization per image stem (real metadata, not empty)."""
    path = (root / "dataset" / "ISIC2019_full" /
            "ISIC_2019_Training_Metadata.csv")
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                age = float(row.get("age_approx") or "")
            except ValueError:
                age = None
            out[row["image"]] = {
                "age": age,
                "sex": (row.get("sex") or "").lower() or None,
                "localization": (row.get("anatom_site_general") or "")
                .lower() or None,
            }
    return out


def cases_for_split(kind: str, per_class: int) -> list[tuple[str, str]]:
    if kind == "heldout":
        return heldout_split(per_class=per_class)[1]
    _, calibration, _, _ = split_for_training(
        heldout_per_class=max(per_class, 30), cal_per_class=per_class)
    return calibration


def run_tool(tool, cases, metadata, temperature, timeout,
             progress: bool = True) -> tuple[dict, list[str]]:
    """Run one tool per case. Returns (predictions, failures)."""
    predictions: dict[str, dict] = {}
    failures: list[str] = []
    total = len(cases)
    for i, (stem, _) in enumerate(cases, 1):
        image = image_path_for_stem(stem)
        if image is None:
            failures.append(f"{stem}: image missing")
            continue
        cmd = [sys.executable, str(resolve(tool.command)),
               "--image", str(image)]
        if "metadata" in tool.requires:
            cmd += ["--metadata", json.dumps(metadata.get(stem, {}))]
        if temperature is not None:
            cmd += ["--temperature", str(temperature)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=str(BASE_DIR))
            payload = json.loads(proc.stdout)
        except Exception as exc:  # noqa: BLE001 - a failure is a data point
            failures.append(f"{stem}: {str(exc)[:160]}")
            continue
        if proc.returncode != 0 or payload.get("status") != "success":
            failures.append(
                f"{stem}: exit={proc.returncode} "
                f"{str(payload.get('message'))[:120]}")
            continue
        cls = payload.get("disease_name")
        conf = payload.get("confidence_score")
        if cls not in CLASSES or conf is None:
            failures.append(f"{stem}: unusable verdict")
            continue
        probs = payload.get("class_probabilities")
        predictions[stem] = {
            "class": str(cls),
            "confidence": float(conf),
            "probs": ({k: float(v) for k, v in probs.items()}
                      if isinstance(probs, dict) else None),
            "temperature": payload.get("temperature"),
            "melanoma_bias": payload.get("melanoma_logit_bias"),
            "flags": payload.get("uncertainty_flags") or [],
        }
        if progress and i % 25 == 0:
            print(f"    {i}/{total} cases, {len(failures)} failed", flush=True)
    return predictions, failures


def _confusion(predictions: dict, truth: dict) -> dict:
    index = {name: i for i, name in enumerate(CLASSES)}
    rows = [[0] * len(CLASSES) for _ in CLASSES]
    for stem, item in predictions.items():
        if stem in truth:
            rows[index[truth[stem]]][index[item["class"]]] += 1
    return {"classes": CLASSES, "rows_truth": rows}


def _per_class(predictions: dict, truth: dict) -> dict:
    stats = {name: {"n": 0, "recall_hits": 0, "predicted": 0,
                    "precision_hits": 0} for name in CLASSES}
    for stem, item in predictions.items():
        if stem not in truth:
            continue
        gold, pred = truth[stem], item["class"]
        stats[gold]["n"] += 1
        stats[pred]["predicted"] += 1
        if gold == pred:
            stats[gold]["recall_hits"] += 1
            stats[pred]["precision_hits"] += 1
    out = {}
    for name, s in stats.items():
        recall = s["recall_hits"] / s["n"] if s["n"] else 0.0
        precision = (s["precision_hits"] / s["predicted"]
                     if s["predicted"] else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        out[name] = {"n": s["n"], "recall": round(recall, 4),
                     "precision": round(precision, 4),
                     "f1": round(f1, 4)}
    return out


def _reliability(predictions: dict, truth: dict) -> tuple[float, float, list]:
    bins = [[] for _ in range(N_BINS)]
    for stem, item in predictions.items():
        if stem not in truth:
            continue
        conf = item["confidence"]
        idx = min(int(conf * N_BINS), N_BINS - 1)
        bins[idx].append((conf, item["class"] == truth[stem]))
    n = sum(len(b) for b in bins)
    ece = 0.0
    mce = 0.0
    rows = []
    for i, bucket in enumerate(bins):
        lo, hi = i / N_BINS, (i + 1) / N_BINS
        if not bucket:
            rows.append({"bin": [round(lo, 4), round(hi, 4)], "n": 0,
                         "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += len(bucket) / n * abs(avg_conf - acc)
        mce = max(mce, abs(avg_conf - acc))
        rows.append({"bin": [round(lo, 4), round(hi, 4)], "n": len(bucket),
                     "avg_confidence": round(avg_conf, 4),
                     "accuracy": round(acc, 4)})
    return round(ece, 4), round(mce, 4), rows


def _nll_brier(predictions: dict, truth: dict) -> tuple:
    nll = brier = 0.0
    n = 0
    for stem, item in predictions.items():
        if stem not in truth or not item.get("probs"):
            continue
        probs = item["probs"]
        if truth[stem] not in probs:
            continue
        p_true = max(probs[truth[stem]], 1e-12)
        nll += -math.log(p_true)
        brier += sum((probs.get(c, 0.0) - (1.0 if c == truth[stem] else 0.0))
                     ** 2 for c in CLASSES)
        n += 1
    if not n:
        return None, None
    return round(nll / n, 4), round(brier / n, 4)


def metrics(predictions: dict, truth: dict) -> dict:
    scored = {s: v for s, v in predictions.items() if s in truth}
    n = len(scored)
    hits = sum(1 for s, v in scored.items() if v["class"] == truth[s])
    conf_sum = sum(v["confidence"] for v in scored.values())
    per_class = _per_class(scored, truth)
    present = [c for c in CLASSES if per_class[c]["n"] > 0]
    balanced = (sum(per_class[c]["recall"] for c in present) / len(present)
                if present else 0.0)
    ece, mce, bins = _reliability(scored, truth)
    nll, brier = _nll_brier(scored, truth)
    return {
        "n": n,
        "accuracy": round(hits / n, 4) if n else 0.0,
        "balanced_accuracy": round(balanced, 4),
        "mean_confidence": round(conf_sum / n, 4) if n else 0.0,
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "brier": brier,
        "per_class": per_class,
        "reliability_bins": bins,
        "confusion_matrix": _confusion(scored, truth),
    }


def gate(metrics_payload: dict) -> dict:
    reasons = []
    mel_recall = (metrics_payload.get("per_class", {})
                  .get("Melanoma", {}).get("recall", 0.0))
    if metrics_payload["n"] < MIN_N:
        reasons.append(f"n={metrics_payload['n']} < {MIN_N}")
    if metrics_payload["accuracy"] < MIN_ACCURACY:
        reasons.append(f"accuracy={metrics_payload['accuracy']:.4f} "
                       f"< {MIN_ACCURACY}")
    if metrics_payload["ece"] > MAX_ECE:
        reasons.append(f"ece={metrics_payload['ece']:.4f} > {MAX_ECE}")
    if mel_recall < MIN_MELANOMA_RECALL:
        reasons.append(f"melanoma_recall={mel_recall:.4f} "
                       f"< {MIN_MELANOMA_RECALL}")
    return {"passed": not reasons, "reasons": reasons,
            "min_accuracy": MIN_ACCURACY, "max_ece": MAX_ECE,
            "min_melanoma_recall": MIN_MELANOMA_RECALL,
            "min_n": MIN_N}


def summarize(results: dict[str, dict]) -> str:
    lines = ["# Local calibration report (research use only)", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}",
             "",
             f"Gate: n >= {MIN_N}, accuracy >= {MIN_ACCURACY}, "
             f"ECE <= {MAX_ECE}, melanoma recall >= {MIN_MELANOMA_RECALL}. "
             "Tools that fail stay out of the pool.", "",
             "| tool | n | accuracy | balanced acc | ECE | MEL recall | "
             "gate |",
             "|------|---|----------|--------------|-----|------------|------|"]
    for name, payload in results.items():
        m = payload["metrics"]
        ok = payload["gate"]["passed"]
        mel = m["per_class"]["Melanoma"]["recall"]
        lines.append(
            f"| {name} | {m['n']} | {m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['ece']:.4f} | "
            f"{mel:.4f} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "## Per-class recall", ""]
    for name, payload in results.items():
        m = payload["metrics"]
        detail = ", ".join(
            f"{c.split()[0].lower()} {m['per_class'][c]['recall']:.2f}"
            for c in CLASSES)
        lines.append(f"- **{name}** ({'PASS' if payload['gate']['passed'] else 'FAIL'}): {detail}")
    failed = {n: p["gate"]["reasons"] for n, p in results.items()
              if not p["gate"]["passed"]}
    lines += ["", "## Failures", ""]
    if failed:
        for name, reasons in failed.items():
            lines.append(f"- {name}: {'; '.join(reasons)}")
    else:
        lines.append("- none")
    lines += ["", "Full per-tool metrics (reliability bins, confusion "
              "matrix, per-class precision/F1) live in the JSON files next "
              "to this summary.", "", "Research use only — not a medical "
              "diagnosis.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate diagnostic tools locally (no LLM/API).")
    parser.add_argument("--tools", default="all",
                        help="Comma-separated names, or 'all'.")
    parser.add_argument("--split", choices=("heldout", "calibration"),
                        default="heldout")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-dir", default="calibration")
    parser.add_argument("--report", default="")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Scan temperature used before fitting.")
    parser.add_argument("--temperature-out", default="")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Tools to evaluate concurrently (VRAM-bound).")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild --report from existing JSON files.")
    args = parser.parse_args()

    if args.summarize_only:
        json_dir = BASE_DIR / args.json_dir
        results = {}
        for path in sorted(json_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            # Re-evaluate the gate against the stored evidence: the
            # criteria (e.g. melanoma recall) can tighten without
            # re-running a single inference pass.
            payload["gate"] = gate(payload["metrics"])
            path.write_text(json.dumps(payload, indent=2))
            results[payload.get("tool", path.stem)] = payload
        if not results:
            raise SystemExit(f"No calibration JSON in {json_dir}")
        if not args.report:
            raise SystemExit("--summarize-only needs --report")
        out = BASE_DIR / args.report
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summarize(results))
        print(summarize(results))
        print(f"Wrote {out}")
        return 0

    manifest = load_manifest(BASE_DIR / "tools" / "manifest.yaml")
    if args.tools == "all":
        tools = [t for t in manifest.tools
                 if t.category == "diagnostic" and t.status == "ready"]
    else:
        tools = [manifest.by_name(n.strip())
                 for n in args.tools.split(",") if n.strip()]
    cases = cases_for_split(args.split, args.per_class)
    truth = dict(cases)
    metadata = load_metadata()
    print(f"Split: {args.split} ({len(cases)} cases), "
          f"tools: {', '.join(t.name for t in tools)}")

    if args.fit_temperature:
        if not args.temperature_out:
            raise SystemExit("--fit-temperature needs --temperature-out")
        for tool in tools:
            print(f"Fitting temperature for {tool.name} ...", flush=True)
            raw, failures = run_tool(tool, cases, metadata,
                                     args.temperature, args.timeout)
            fit = fit_temperature(
                {stem: item["probs"] for stem, item in raw.items()
                 if item.get("probs")}, truth)
            if not fit.get("ok"):
                print(f"  skipped: {fit.get('reason')} "
                      f"({len(failures)} tool failures)")
                continue
            out = Path(args.temperature_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fit.update({"tool": tool.name, "split": args.split,
                        "per_class": args.per_class,
                        "scan_temperature": args.temperature})
            out.write_text(json.dumps(fit, indent=2))
            print(f"  T={fit['temperature']} "
                  f"NLL {fit['nll_before']} -> {fit['nll_after']} "
                  f"(n={fit['n']}); wrote {out}")
        return 0

    json_dir = BASE_DIR / args.json_dir
    json_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_tool(tool) -> dict:
        print(f"Running {tool.name} on {len(cases)} cases ...", flush=True)
        predictions, failures = run_tool(tool, cases, metadata, None,
                                         args.timeout)
        m = metrics(predictions, truth)
        temps = {v["temperature"] for v in predictions.values()
                 if v.get("temperature") is not None}
        biases = {v["melanoma_bias"] for v in predictions.values()
                  if v.get("melanoma_bias") is not None}
        payload = {
            "tool": tool.name,
            "command": tool.command,
            "split": args.split,
            "split_detail": {"per_class": args.per_class,
                             "n": len(cases),
                             "failures": len(failures)},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "temperature": sorted(temps)[0] if len(temps) == 1 else None,
            "melanoma_logit_bias": (sorted(biases)[0]
                                    if len(biases) == 1 else None),
            "metrics": m,
            "gate": gate(m),
            "failure_samples": failures[:5],
        }
        (json_dir / f"{tool.name}.json").write_text(
            json.dumps(payload, indent=2))
        mel = m["per_class"]["Melanoma"]["recall"]
        print(f"  {tool.name}: n={m['n']} acc={m['accuracy']:.4f} "
              f"bal={m['balanced_accuracy']:.4f} ece={m['ece']:.4f} "
              f"MEL={mel:.4f} "
              f"-> {'PASS' if payload['gate']['passed'] else 'FAIL'}"
              f" ({len(failures)} failures)", flush=True)
        return payload

    results: dict[str, dict] = {}
    if args.jobs > 1 and len(tools) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for tool, payload in zip(
                    tools, pool.map(evaluate_tool, tools)):
                results[tool.name] = payload
    else:
        for tool in tools:
            results[tool.name] = evaluate_tool(tool)

    if args.report:
        out = BASE_DIR / args.report
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summarize(results))
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
