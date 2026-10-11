"""ViT-B/16 fine-tuned locally on ISIC2019 (7 ISIC classes).

Weights: models/isic-vit-b16/best.pt, built by tools/train_isic.py from
the torchvision ImageNet ViT-B/16. Uniform JSON contract; temperature
from models/isic-vit-b16/calibration.json.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.isic_common import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(
        tool_name="isic-vit",
        model_dir_name="isic-vit-b16",
        tier="tier3_high",
        vram_gb=2.0,
        description="ViT-B/16 ISIC2019 7-class classifier.",
    ))
