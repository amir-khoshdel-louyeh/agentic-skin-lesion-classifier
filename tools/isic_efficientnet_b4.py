"""EfficientNet-B4 fine-tuned locally on ISIC2019 (7 ISIC classes).

Weights: models/isic-efficientnet-b4/best.pt, built by
tools/train_isic.py from ImageNet initialization on the train split of
control.eval.split_for_training. No API, no LLM. Uniform JSON contract;
temperature from models/isic-efficientnet-b4/calibration.json.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.isic_common import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(
        tool_name="isic-efficientnet-b4",
        model_dir_name="isic-efficientnet-b4",
        tier="tier2_mid",
        vram_gb=1.5,
        description="EfficientNet-B4 ISIC2019 7-class classifier.",
    ))
