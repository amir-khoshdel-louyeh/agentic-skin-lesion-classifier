"""ResNet-50 fine-tuned locally on ISIC2019 (7 ISIC classes).

Weights: models/isic-resnet50/best.pt, built by tools/train_isic.py.
Uniform JSON contract; temperature from models/isic-resnet50/calibration.json.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.isic_common import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(
        tool_name="isic-resnet50",
        model_dir_name="isic-resnet50",
        tier="tier2_mid",
        vram_gb=1.5,
        description="ResNet-50 ISIC2019 7-class classifier.",
    ))
