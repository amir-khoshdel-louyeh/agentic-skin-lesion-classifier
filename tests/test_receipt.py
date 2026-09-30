"""Receipt single-writer checks (stdlib only, runs one cheap tool 4x).

Numbers always come from re-execution; agent drift only changes the
reliability tag (exact/rounded/fabricated/omitted), never the verdict.
"""
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
check("exact-keeps-tool-values",
      exact.ran and exact.predicted_class == cls
      and exact.confidence == conf and "reliability=exact" in exact.reasoning)

small = verify_receipt(envelope(round(conf + 0.03, 4)), record, manifest)
check("small-drift-flagged-not-rejected",
      small.ran and small.predicted_class == cls
      and small.confidence == conf
      and "reliability=rounded" in small.reasoning
      and "disagreement" in small.uncertainty_flags)

big = verify_receipt(envelope(round(min(conf + 0.2, 1.0), 4)), record,
                     manifest)
check("fabrication-yields-tool-values",
      big.ran and big.predicted_class == cls and big.confidence == conf
      and "reliability=fabricated" in big.reasoning
      and "disagreement" in big.uncertainty_flags)

omitted = verify_receipt(envelope(None), record, manifest)
check("omitted-numbers-filled",
      omitted.ran and omitted.predicted_class == cls
      and omitted.confidence == conf
      and "reliability=omitted" in omitted.reasoning)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)
