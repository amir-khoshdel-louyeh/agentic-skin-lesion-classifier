"""Local ISIC2019 fine-tuning for the diagnostic pool (no LLM, no API).

Build script, not a manifest tool: it produces `models/isic-<arch>/`
checkpoints that the thin wrappers (`tools/isic_*.py`) serve through the
uniform CLI contract. Trains on the 7 covered ISIC classes (MEL, NV, BCC,
AK, BKL, DF, VASC); SCC is out of coverage and excluded, matching
control.schemas.

Splits come from control.eval.split_for_training: held-out is never
touched, temperature fitting gets its own disjoint slice, and validation
drives checkpoint selection. ImageNet weights are the only pretraining.

Usage:
  .venv/bin/python tools/train_isic.py --arch efficientnet_b4 --epochs 4
  .venv/bin/python tools/train_isic.py --arch resnet50 --epochs 4
  .venv/bin/python tools/train_isic.py --arch vit_b_16 --epochs 4
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from control.eval import image_path_for_stem, split_for_training  # noqa: E402
from control.schemas import HAM10000_CLASSES  # noqa: E402

CLASSES = list(HAM10000_CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

ARCHS = {
    "efficientnet_b4": (models.efficientnet_b4, 32),
    "resnet50": (models.resnet50, 32),
    "vit_b_16": (models.vit_b_16, 24),
}

# Output directories match the manifest/tool names (kebab-case), not the
# torchvision architecture identifiers.
ARCH_DIRS = {
    "efficientnet_b4": "isic-efficientnet-b4",
    "resnet50": "isic-resnet50",
    "vit_b_16": "isic-vit-b16",
}


class SkinDataset(Dataset):
    def __init__(self, cases: list[tuple[str, str]], transform):
        self.cases = cases
        self.transform = transform

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int):
        stem, label = self.cases[i]
        image = ImageOps.exif_transpose(Image.open(image_path_for_stem(stem)))
        return self.transform(image.convert("RGB")), CLASS_TO_IDX[label]


def build_model(arch: str, num_classes: int = len(CLASSES)) -> nn.Module:
    factory, _ = ARCHS[arch]
    model = factory(weights="IMAGENET1K_V1")
    if arch == "efficientnet_b4":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                        num_classes)
    elif arch == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features,
                                     num_classes)
    return model


def transforms_for(size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.65, 1.0),
                                     ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    hits = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(images)
        loss_sum += float(criterion(logits.float(), labels)) * labels.size(0)
        hits += int((logits.argmax(1) == labels).sum())
        total += labels.size(0)
    return hits / max(total, 1), loss_sum / max(total, 1)


def class_weights(cases: list[tuple[str, str]]) -> torch.Tensor:
    """Inverse-sqrt frequency weights: lift DF/VASC without drowning NV."""
    counts = [0] * len(CLASSES)
    for _, label in cases:
        counts[CLASS_TO_IDX[label]] += 1
    total = sum(counts)
    weights = [math.sqrt(total / (len(CLASSES) * max(c, 1))) for c in counts]
    mean = sum(weights) / len(weights)
    return torch.tensor([w / mean for w in weights], dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune an ISIC2019 model.")
    parser.add_argument("--arch", choices=sorted(ARCHS), required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("CUDA not available; refusing to train on CPU.")
        return 1

    out_dir = Path(args.out or (BASE_DIR / "models" / ARCH_DIRS[args.arch]))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cases, cal_cases, val_cases, heldout = split_for_training()
    print(f"arch={args.arch} train={len(train_cases)} "
          f"cal={len(cal_cases)} val={len(val_cases)} "
          f"heldout={len(heldout)}")

    train_tf, eval_tf = transforms_for(args.size)
    loader_kwargs = dict(num_workers=args.workers, pin_memory=True,
                         persistent_workers=args.workers > 0)
    train_loader = DataLoader(SkinDataset(train_cases, train_tf),
                              batch_size=ARCHS[args.arch][1], shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(SkinDataset(val_cases, eval_tf),
                            batch_size=64, shuffle=False, **loader_kwargs)

    device = torch.device("cuda")
    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_cases).to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.05)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.15)

    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += float(loss) * labels.size(0)
            seen += labels.size(0)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} "
                      f"loss={running / seen:.4f}", flush=True)
        val_acc, val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - started
        history.append({"epoch": epoch, "train_loss": round(running / seen, 4),
                        "val_loss": round(val_loss, 4),
                        "val_accuracy": round(val_acc, 4),
                        "seconds": round(elapsed, 1)})
        print(f"epoch {epoch}: train_loss={running / seen:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({elapsed / 60:.1f} min)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "classes": CLASSES, "size": args.size},
                       out_dir / "best.pt")
    meta = {
        "arch": args.arch,
        "classes": CLASSES,
        "size": args.size,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_n": len(train_cases),
        "calibration_n": len(cal_cases),
        "val_n": len(val_cases),
        "heldout_n": len(heldout),
        "best_val_accuracy": round(best_acc, 4),
        "history": history,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'best.pt'} (val_acc={best_acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
