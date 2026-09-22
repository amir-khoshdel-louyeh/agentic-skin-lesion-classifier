"""Contract + anti-drift checks (stdlib only, no pytest needed).

Usage: .venv/bin/python tests/test_contract.py [--live <tool-name>]
Static checks always run (manifest valid, skill copies in sync).
--live runs one ready tool on the first prompt image and validates keys.
Exit non-zero on any failure (CI-friendly).
"""
import difflib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = {
    "diagnostic": ["status", "tool", "model_tier", "model_executed",
                   "predicted_class_index", "disease_name",
                   "confidence_score"],
}
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


from control.manifest import load_manifest  # noqa: E402

manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
ready = [t for t in manifest.tools if t.status == "ready"]
check("manifest-loads", len(ready) >= 10, f"{len(ready)} ready")

home_skills = (Path.home() / ".openclaw" / "workspace" / "skills")
for tool in ready:
    skill = ROOT / "openclaw-skills" / tool.name / "SKILL.md"
    if not skill.exists():
        continue
    installed = home_skills / tool.name / "SKILL.md"
    same = installed.exists() and installed.read_text() == skill.read_text()
    check(f"skill-sync:{tool.name}", same, "reinstall with --force")

if len(sys.argv) == 3 and sys.argv[1] == "--live":
    import yaml  # noqa: E402

    tool = manifest.by_name(sys.argv[2])
    rec = yaml.safe_load((ROOT / "prompt.yaml").read_text())[0]
    cmd = [sys.executable, str(ROOT / tool.command), "--image",
           str(ROOT / rec["image_path"])]
    if "metadata" in tool.requires:
        cmd += ["--metadata", json.dumps(rec.get("metadata", {}))]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=str(ROOT))
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, TypeError):
        payload, proc.returncode = {}, 1
    check("live-exit-0", proc.returncode == 0, proc.stderr[-200:])
    check("live-status", payload.get("status") == "success",
          str(payload)[:200])
    for key in REQUIRED.get(tool.category, ["status"]):
        check(f"live-key:{key}", key in payload)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)
