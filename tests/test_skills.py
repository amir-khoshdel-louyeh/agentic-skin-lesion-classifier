"""Skill re-sync checks (plan.tmp T9, stdlib only).
Run: .venv/bin/python tests/test_skills.py
Exit non-zero on any failure.

Locks in §4/§6: exactly the six role skills
(quality/brain/diagnostic/critic/specialist/chat), legacy tier skills
gone, manifest stays the single source of WHAT exists (tools untouched),
installed copies in sync (forced reinstall).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.manifest import load_manifest  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


ROLES = ("quality", "brain", "diagnostic", "critic", "specialist", "chat")
MARKERS = {
    "quality": ("quality_gate", "preprocess", "abcde_analyzer",
                "never diagnoses"),
    "brain": ("hint only", "never sees images"),
    "diagnostic": ("ham10000_cnn", "NOT in the pool", "envelope",
                   "MUST start with", '"predicted_class": null',
                   "FIRST run at least one tool"),
    "critic": ("EXACTLY ONCE", "NO tools"),
    "specialist": ("advocate", "averag", "MUST start with",
                   "to null ALWAYS"),
    "chat": ("Not in the record", "re-open the round"),
}

# 1. Exactly the six role skills, each with matching frontmatter.
skills_dir = ROOT / "openclaw-skills"
present = sorted(p.name for p in skills_dir.iterdir() if p.is_dir())
check("six-role-skills", present == sorted(ROLES), str(present))
for role in ROLES:
    skill = skills_dir / role / "SKILL.md"
    check(f"skill-file-{role}", skill.exists())
    if not skill.exists():
        continue
    text = skill.read_text(encoding="utf-8")
    check(f"skill-name-{role}", f"name: {role}" in text)
    for marker in MARKERS[role]:
        check(f"skill-{role}-{marker[:16]}", marker in text)

# 2. Legacy tier skills gone (tools stay callable — only the skills go).
for legacy in ("skin-lesion-fast", "skin-lesion-mid", "skin-lesion-high"):
    check(f"tier-skill-gone-{legacy}",
          not (skills_dir / legacy).exists())

# 3. Manifest stays the single source: diagnostic tools untouched,
# legacy tools still callable, VLM still listed (excluded by pool code).
manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
names = {t.name for t in manifest.tools}
for tool in ("ham10000-cnn", "multimodal-fusion", "ensemble-high",
             "dermai-b0", "skin-lesion-fast", "skin-lesion-mid",
             "skin-lesion-high", "drdiag-vlm", "vlm-ollama",
             "quality-gate", "preprocess", "abcde-analyzer"):
    check(f"manifest-keeps-{tool}", tool in names)

# 4. Installed copies in sync (forced reinstall).
home_skills = Path.home() / ".openclaw" / "workspace" / "skills"
for role in ROLES:
    src = skills_dir / role / "SKILL.md"
    installed = home_skills / role / "SKILL.md"
    same = (installed.exists()
            and installed.read_text() == src.read_text())
    check(f"skill-sync-{role}", same, "reinstall with --force")
for legacy in ("skin-lesion-fast", "skin-lesion-mid", "skin-lesion-high"):
    check(f"installed-tier-gone-{legacy}",
          not (home_skills / legacy).exists())

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)
