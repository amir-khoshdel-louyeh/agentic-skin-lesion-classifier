"""Calibration registry: which diagnostic tools cleared the local gate.

`calibration/*.json` is written by `tools/calibrate.py` (local tools
only, no LLM/API). The diagnostic pool reads it so a tool that failed
calibration stays OUT of the pool even though its CLI still runs.

A missing or unreadable registry blocks nothing: a fresh clone without
calibration artifacts keeps working, and only measured failures gate.
"""

import json
import math
from pathlib import Path

from control.paths import ROOT_DIR
from control.schemas import HAM10000_CLASSES

CALIBRATION_DIRNAME = "calibration"
MIN_ACCURACY = 0.60
MAX_ECE = 0.10
# A tool that rarely predicts melanoma is unsafe for screening even when
# its aggregate accuracy looks fine (the B4/ResNet fits sat at 0.52-0.56).
# 0.90 is the screening floor: missing 1 in 10 melanomas is the most the
# pool may tolerate, and retraining is judged against it.
MIN_MELANOMA_RECALL = 0.90
TARGET_MELANOMA_RECALL = 0.90


def load_registry(root: Path = ROOT_DIR) -> dict[str, dict]:
    """tool name -> calibration payload (invalid files are ignored)."""
    directory = Path(root) / CALIBRATION_DIRNAME
    records: dict[str, dict] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("tool") or path.stem
        records[str(name)] = payload
    return records


def passed(record: dict) -> bool:
    gate = record.get("gate") or {}
    return bool(gate.get("passed"))


def blocked_tools(root: Path = ROOT_DIR) -> set[str]:
    """Names with a calibration record that did not pass the gate."""
    return {name for name, record in load_registry(root).items()
            if not passed(record)}


def fit_temperature(probs_by_stem: dict[str, dict],
                    truth: dict[str, str]) -> dict:
    """Fit T by NLL grid search on probabilities collected at T=1.0.

    `log(p)` equals the logits up to an additive constant, which softmax
    ignores, so scaling `log(p)` by 1/T is temperature scaling without
    needing raw logits from the tool.
    """
    usable = {stem: probs for stem, probs in probs_by_stem.items()
              if stem in truth and truth[stem] in probs}
    if len(usable) < 30:
        return {"ok": False, "reason": "not enough probability rows"}

    def nll(temperature: float) -> float:
        total = 0.0
        for stem, probs in usable.items():
            scaled = {c: math.log(max(p, 1e-12)) / temperature
                      for c, p in probs.items()}
            top = max(scaled.values())
            norm = top + math.log(sum(math.exp(v - top)
                                      for v in scaled.values()))
            total += -(scaled[truth[stem]] - norm)
        return total / len(usable)

    grid = [0.25 * (1.05 ** i) for i in range(90)]
    best_nll, best_t = min((nll(t), t) for t in grid)
    return {"ok": True, "temperature": round(best_t, 3),
            "n": len(usable), "nll_before": round(nll(1.0), 4),
            "nll_after": round(best_nll, 4)}


def fit_sensitive_calibration(probs_by_stem: dict[str, dict],
                              truth: dict[str, str],
                              target_recall: float = TARGET_MELANOMA_RECALL,
                              max_bias: float = 5.0,
                              step: float = 0.05) -> dict:
    """Fit a melanoma logit bias for sensitivity, then temperature.

    A constant added to the melanoma logit is a class-prior / decision-
    threshold adjustment: it raises the melanoma recall without retraining
    and preserves the temperature semantics. The bias is the SMALLEST one
    that reaches `target_recall` on the calibration split (max recall if
    unreachable); temperature is then refit by NLL on the adjusted logits
    so confidence stays calibrated after the shift.

    `log(p)` from a T=1.0 scan is the logit vector up to an additive
    constant, so no raw-logit export is needed.
    """
    classes = list(HAM10000_CLASSES)
    mel = classes.index("Melanoma")
    usable = {stem: probs for stem, probs in probs_by_stem.items()
              if stem in truth and truth[stem] in probs}
    if len(usable) < 30:
        return {"ok": False, "reason": "not enough probability rows"}
    rows = [([math.log(max(probs.get(c, 1e-12), 1e-12))
              for c in classes], classes.index(truth[stem]))
            for stem, probs in usable.items()]

    def predict(bias: float) -> tuple[float, float]:
        hits = mel_hits = mel_n = 0
        for logp, gold in rows:
            best, best_class = -math.inf, 0
            for i, value in enumerate(logp):
                score = value + (bias if i == mel else 0.0)
                if score > best:
                    best, best_class = score, i
            hits += int(best_class == gold)
            if gold == mel:
                mel_n += 1
                mel_hits += int(best_class == mel)
        recall = mel_hits / mel_n if mel_n else 0.0
        return recall, hits / len(rows)

    chosen_bias, chosen_recall, chosen_acc = max_bias, 0.0, 0.0
    bias = 0.0
    while bias <= max_bias + 1e-9:
        recall, acc = predict(bias)
        if recall >= target_recall:
            chosen_bias, chosen_recall, chosen_acc = bias, recall, acc
            break
        if recall > chosen_recall:
            chosen_bias, chosen_recall, chosen_acc = bias, recall, acc
        bias = round(bias + step, 4)

    def nll(temperature: float, bias_value: float) -> float:
        total = 0.0
        for logp, gold in rows:
            scaled = [value / temperature +
                      (bias_value / temperature if i == mel else 0.0)
                      for i, value in enumerate(logp)]
            top = max(scaled)
            norm = top + math.log(sum(math.exp(v - top) for v in scaled))
            total += -(scaled[gold] - norm)
        return total / len(rows)

    grid = [0.25 * (1.05 ** i) for i in range(90)]
    base_nll = nll(1.0, 0.0)
    best_nll, best_t = min((nll(t, chosen_bias), t) for t in grid)
    return {
        "ok": True,
        "melanoma_logit_bias": round(chosen_bias, 3),
        "target_melanoma_recall": target_recall,
        "calibration_melanoma_recall": round(chosen_recall, 4),
        "calibration_accuracy": round(chosen_acc, 4),
        "temperature": round(best_t, 3),
        "n": len(rows),
        "nll_before": round(base_nll, 4),
        "nll_after": round(best_nll, 4),
    }
