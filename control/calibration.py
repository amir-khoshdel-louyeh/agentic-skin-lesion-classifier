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

CALIBRATION_DIRNAME = "calibration"
MIN_ACCURACY = 0.60
MAX_ECE = 0.10


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
