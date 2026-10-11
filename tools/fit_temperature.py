"""Fit sensitivity calibration for one fine-tuned ISIC2019 model.

Build script (not a manifest tool). Runs the deployed checkpoint on the
disjoint calibration split in one process, fits the smallest melanoma
logit bias that reaches the target melanoma recall, refits temperature by
NLL on the adjusted logits, and writes `models/<name>/calibration.json` —
the file `tools/isic_common.py` reads before serving a prediction.

Usage:
  .venv/bin/python tools/fit_temperature.py --model-dir models/isic-resnet50
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from control.calibration import fit_sensitive_calibration  # noqa: E402
from control.eval import image_path_for_stem, split_for_training  # noqa: E402
from tools.isic_common import probabilities  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on the calibration split.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--target-melanoma-recall", type=float, default=0.75)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    model_dir = (Path(args.model_dir) if Path(args.model_dir).is_absolute()
                 else BASE_DIR / args.model_dir)
    if not (model_dir / "best.pt").exists():
        raise SystemExit(f"No checkpoint at {model_dir / 'best.pt'}")

    _, calibration, _, _ = split_for_training(
        heldout_per_class=max(args.per_class, 30),
        cal_per_class=args.per_class)
    probs_by_stem: dict[str, dict[str, float]] = {}
    for i, (stem, _) in enumerate(calibration, 1):
        image = image_path_for_stem(stem)
        if image is None:
            continue
        probs_by_stem[stem] = probabilities(
            model_dir, str(image), temperature=1.0, melanoma_bias=0.0)
        if i % 50 == 0:
            print(f"  {i}/{len(calibration)}", flush=True)

    fit = fit_sensitive_calibration(
        probs_by_stem, dict(calibration),
        target_recall=args.target_melanoma_recall)
    if not fit.get("ok"):
        raise SystemExit(f"Fit failed: {fit.get('reason')}")
    fit.update({"tool": model_dir.name, "split": "calibration",
                "per_class": args.per_class,
                "generated_at": datetime.now(timezone.utc).isoformat()})
    out = Path(args.out) if args.out else model_dir / "calibration.json"
    out.write_text(json.dumps(fit, indent=2))
    print(f"bias={fit['melanoma_logit_bias']} T={fit['temperature']} "
          f"MEL recall {fit['calibration_melanoma_recall']} "
          f"acc {fit['calibration_accuracy']} "
          f"NLL {fit['nll_before']} -> {fit['nll_after']} "
          f"(n={fit['n']}); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
