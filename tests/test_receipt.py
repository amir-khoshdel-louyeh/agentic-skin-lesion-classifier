"""Receipt tolerance bands (stdlib only, runs one cheap tool 3x)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.manifest import load_manifest  # noqa: E402
from control.schemas import VerdictEnvelope  # noqa: E402
from orchestrator import verify_receipt  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
record = {"image_path": "dataset/ISIC2019_test/ISIC_0000000.jpg",
          "metadata": {}}
proc = subprocess.run(
    [sys.executable, "tools/skin_lesion_fast.py", "--image",
     record["image_path"]], capture_output=True, text=True, timeout=300,
    cwd=str(ROOT))
actual = json.loads(proc.stdout)
cls, conf = actual["disease_name"], actual["confidence_score"]


def envelope(confidence):
    return VerdictEnvelope(
        ran=True,
        command="python tools/skin_lesion_fast.py --image "
                "dataset/ISIC2019_test/ISIC_0000000.jpg",
        exit_code=0, predicted_class=cls, confidence=confidence,
        reasoning="t", uncertainty_flags=[])


exact = verify_receipt(envelope(conf), record, manifest)
check("exact-accepted", exact.ran and "control" not in exact.reasoning)

small = verify_receipt(envelope(round(conf + 0.03, 4)), record, manifest)
check("small-drift-corrected",
      small.ran and "corrected" in small.reasoning
      and "disagreement" in small.uncertainty_flags)

big = verify_receipt(envelope(round(min(conf + 0.2, 1.0), 4)), record,
                     manifest)
check("fabrication-rejected",
      not big.ran and "fabrication" in big.reasoning)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)
