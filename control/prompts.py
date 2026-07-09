"""Two-layer prompts: locked system + editable doctor (proposal.tmp Sec 6).

Doctor files may tune thresholds, but only within [0.6, 0.9] and only
toward caution; out-of-range values fail loading.
"""

import re
from pathlib import Path

from control.paths import ROOT_DIR

THRESHOLD_MIN = 0.6
THRESHOLD_MAX = 0.9

_THRESHOLD_RE = re.compile(
    r"^\s*(?:borderline_threshold|entropy_threshold|confidence_threshold)"
    r"\s*:\s*([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)

ROLES = ("orchestrator", "agent")


def load_role(role: str, prompts_dir: Path | None = None) -> str:
    if role not in ROLES:
        raise KeyError(f"Unknown role: {role}")
    base = prompts_dir or (ROOT_DIR / "prompts")
    system = (base / f"{role}.system.md").read_text(encoding="utf-8")
    doctor = (base / f"{role}.doctor.md").read_text(encoding="utf-8")
    validate_thresholds(doctor, role=role)
    return system.rstrip() + "\n\n" + doctor.strip() + "\n"


def validate_thresholds(doctor_text: str, role: str = "") -> dict[str, float]:
    found: dict[str, float] = {}
    for match in _THRESHOLD_RE.finditer(doctor_text):
        name = match.group(0).split(":")[0].strip()
        value = float(match.group(1))
        if not THRESHOLD_MIN <= value <= THRESHOLD_MAX:
            raise ValueError(
                f"[{role}] {name}={value} out of bounds "
                f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]."
            )
        found[name] = value
    return found
