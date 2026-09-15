"""Two-layer prompts: locked system + editable doctor (proposal.tmp Sec 6).

Physician Pilot (Phase 8) contract for `prompts/*.doctor.md`:

- Editable: role focus text, report tone, physician priorities, and the
  numeric thresholds below.
- Thresholds must stay within [0.6, 0.9] AND may only move toward caution
  (higher `borderline_threshold` / `confidence_threshold` flags MORE cases
  suspicious; lower `entropy_threshold` flags MORE cases borderline).
  Any lax (less-cautious) edit fails loading.
- Safety rules live in the LOCKED `*.system.md` layer. Doctor text that
  tries to override them ("ignore previous instructions", "confident
  benign", "skip receipts", ...) fails loading.
- Locked system files carry mandatory safety sentences; if they were
  edited to remove one, loading fails fast instead of running unsafe.
- Runtime code must NEVER hardcode thresholds: use
  `effective_thresholds(role)` so doctor edits actually take effect.
"""

import re
from pathlib import Path

from control.paths import ROOT_DIR

THRESHOLD_MIN = 0.6
THRESHOLD_MAX = 0.9

# Factory defaults = least-cautious allowed baseline. Doctor edits may only
# move toward caution from here, never back toward lax.
DEFAULTS: dict[str, float] = {
    "borderline_threshold": 0.75,
    "confidence_threshold": 0.75,
    "entropy_threshold": 0.9,
}

# Which direction counts as "more cautious" per key.
CAUTION_DIRECTION: dict[str, str] = {
    "borderline_threshold": "higher",  # higher bar => more suspicious flags
    "confidence_threshold": "higher",  # higher bar => more suspicious flags
    "entropy_threshold": "lower",  # lower bar => more borderline flags
}

_THRESHOLD_RE = re.compile(
    r"^\s*(?:borderline_threshold|entropy_threshold|confidence_threshold)"
    r"\s*:\s*([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)

# Override / jailbreak attempts in the editable layer. Matched
# case-insensitively; any hit fails loading with a clear error.
_FORBIDDEN_RE = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|system|locked)\s+(instructions|rules|prompts)"
    r"|disregard\s+(all\s+)?(previous|prior|system|locked)"
    r"|override\s+(the\s+)?(system|locked|safety)"
    r"|confident\s+benign"
    r"|never\s+(flag|escalate|refer)"
    r"|skip\s+(the\s+)?receipts?"
    r"|do\s+not\s+(flag|require)\s+(receipts?|borderline)"
    r"|jailbreak|prompt\s*injection",
    re.IGNORECASE,
)

ROLES = ("orchestrator", "agent")

# Safety sentences that MUST survive in the locked layer. If a system file
# was edited to drop one, loading fails fast.
LOCKED_SENTINELS: dict[str, tuple[str, ...]] = {
    "orchestrator": (
        "forces \"suspicious",
        "Research use only",
    ),
    "agent": (
        "Never fabricate tool output",
        "do not rely on me",
    ),
}


def _threshold_name(match: re.Match) -> str:
    return match.group(0).split(":")[0].strip()


def parse_thresholds(doctor_text: str) -> dict[str, float]:
    """Extract `name: value` threshold lines from doctor text (no checks)."""
    found: dict[str, float] = {}
    for match in _THRESHOLD_RE.finditer(doctor_text):
        found[_threshold_name(match)] = float(match.group(1))
    return found


def validate_thresholds(doctor_text: str, role: str = "") -> dict[str, float]:
    """Range check [0.6, 0.9] + only-more-cautious + no-override check.

    Returns the parsed doctor thresholds. Raises ValueError on any
    violation so the round fails fast instead of running unsafe/lax.
    """
    forbidden = _FORBIDDEN_RE.search(doctor_text)
    if forbidden:
        raise ValueError(
            f"[{role}] doctor file tries to override locked safety rules "
            f"({forbidden.group(0)!r}). Safety rules live in *.system.md "
            "and cannot be weakened from the doctor layer."
        )
    found = parse_thresholds(doctor_text)
    for name, value in found.items():
        if not THRESHOLD_MIN <= value <= THRESHOLD_MAX:
            raise ValueError(
                f"[{role}] {name}={value} out of bounds "
                f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]."
            )
        baseline = DEFAULTS[name]
        direction = CAUTION_DIRECTION[name]
        lax = value < baseline if direction == "higher" else value > baseline
        if lax:
            raise ValueError(
                f"[{role}] {name}={value} is LESS cautious than the "
                f"baseline {baseline}. Doctor edits may only move toward "
                f"caution ({direction} than {baseline})."
            )
    return found


def check_locked(role: str, prompts_dir: Path | None = None) -> None:
    """Fail if the locked system file lost a mandatory safety sentence."""
    base = prompts_dir or (ROOT_DIR / "prompts")
    system = (base / f"{role}.system.md").read_text(encoding="utf-8")
    for sentinel in LOCKED_SENTINELS[role]:
        if sentinel not in system:
            raise ValueError(
                f"[{role}] locked system file {role}.system.md lost its "
                f"safety rule ({sentinel!r}). Restore it before running."
            )


def load_role(role: str, prompts_dir: Path | None = None) -> str:
    if role not in ROLES:
        raise KeyError(f"Unknown role: {role}")
    base = prompts_dir or (ROOT_DIR / "prompts")
    check_locked(role, base)
    system = (base / f"{role}.system.md").read_text(encoding="utf-8")
    doctor = (base / f"{role}.doctor.md").read_text(encoding="utf-8")
    validate_thresholds(doctor, role=role)
    return system.rstrip() + "\n\n" + doctor.strip() + "\n"


def effective_thresholds(
    role: str, prompts_dir: Path | None = None
) -> dict[str, float]:
    """Merged runtime thresholds: factory defaults overridden by doctor file.

    This is the ONLY source runtime code may use for thresholds, so every
    physician edit takes effect after validation.
    """
    if role not in ROLES:
        raise KeyError(f"Unknown role: {role}")
    base = prompts_dir or (ROOT_DIR / "prompts")
    check_locked(role, base)
    doctor = (base / f"{role}.doctor.md").read_text(encoding="utf-8")
    return {**DEFAULTS, **validate_thresholds(doctor, role=role)}


def doctor_check(
    prompts_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Validate every role; return per-role status for CLI/GUI reporting.

    Never raises: each role maps to {"ok": bool, "effective": {...},
    "errors": [...]} so the physician sees ALL problems in one pass.
    """
    base = prompts_dir or (ROOT_DIR / "prompts")
    report: dict[str, dict[str, object]] = {}
    for role in ROLES:
        errors: list[str] = []
        effective: dict[str, float] = dict(DEFAULTS)
        try:
            check_locked(role, base)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        try:
            doctor = (base / f"{role}.doctor.md").read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"[{role}] cannot read doctor file: {exc}")
            doctor = ""
        if doctor or not errors:
            try:
                effective = {**DEFAULTS,
                             **validate_thresholds(doctor, role=role)}
            except ValueError as exc:
                errors.append(str(exc))
        report[role] = {
            "ok": not errors,
            "effective": effective,
            "errors": errors,
        }
    return report


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Physician customization check: validate "
        "prompts/*.doctor.md (bounds 0.6-0.9, only-more-cautious, "
        "no safety overrides) and print effective thresholds."
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit the check report as JSON.")
    args = parser.parse_args()
    report = doctor_check()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for role, info in report.items():
            status = "OK" if info["ok"] else "FAILED"
            print(f"[{role}] {status}")
            for name, value in info["effective"].items():
                print(f"  {name}: {value}")
            for err in info["errors"]:  # type: ignore[union-attr]
                print(f"  ! {err}")
    return 0 if all(info["ok"] for info in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
