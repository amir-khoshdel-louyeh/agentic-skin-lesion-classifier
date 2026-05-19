"""Utilities for loading and preparing image-based skin lesion datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("dataset")
ISIC_IMAGE_ROOT = DATASET_ROOT / "ISIC_2019_Training_Input"
ISIC_GROUNDTRUTH_CSV = DATASET_ROOT / "ISIC_2019_Training_GroundTruth.csv"
ISIC_METADATA_CSV = DATASET_ROOT / "ISIC_2019_Training_Metadata.csv"
ISIC_CLASS_COLUMNS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

LEGACY_DATASET_ROOT = Path("dataset") / "HAM1000"
METADATA_CSV = LEGACY_DATASET_ROOT / "HAM10000_metadata.csv"
HMNIST_RGB_CSV = LEGACY_DATASET_ROOT / "hmnist_28_28_RGB.csv" / "hmnist_28_28_RGB.csv"


def confirm_dataset_files(
    metadata_csv: Path | str = METADATA_CSV,
    hmnist_csv: Path | str = HMNIST_RGB_CSV,
) -> dict[str, bool]:
    """Return availability status for required dataset files."""
    metadata_path = Path(metadata_csv)
    hmnist_path = Path(hmnist_csv)
    return {
        "metadata_csv_exists": metadata_path.exists(),
        "hmnist_rgb_csv_exists": hmnist_path.exists(),
    }


def confirm_isic_dataset_files(
    image_root: Path | str = ISIC_IMAGE_ROOT,
    groundtruth_csv: Path | str = ISIC_GROUNDTRUTH_CSV,
    metadata_csv: Path | str = ISIC_METADATA_CSV,
) -> dict[str, bool]:
    """Return availability status for the ISIC image dataset files."""
    image_root_path = Path(image_root)
    groundtruth_path = Path(groundtruth_csv)
    metadata_path = Path(metadata_csv)
    return {
        "image_root_exists": image_root_path.exists(),
        "groundtruth_csv_exists": groundtruth_path.exists(),
        "metadata_csv_exists": metadata_path.exists(),
    }


def load_metadata(metadata_csv: Path | str = METADATA_CSV) -> pd.DataFrame:
    """Load HAM10000 metadata table."""
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def load_hmnist_pixels(
    hmnist_csv: Path | str = HMNIST_RGB_CSV,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load HMNIST 28x28 RGB pixel table (2352 pixel values + label)."""
    hmnist_path = Path(hmnist_csv)
    if not hmnist_path.exists():
        raise FileNotFoundError(f"HMNIST RGB CSV not found: {hmnist_path}")
    return pd.read_csv(hmnist_path, nrows=nrows)


def load_isic_ground_truth(groundtruth_csv: Path | str = ISIC_GROUNDTRUTH_CSV) -> pd.DataFrame:
    """Load the ISIC one-hot label table."""
    groundtruth_path = Path(groundtruth_csv)
    if not groundtruth_path.exists():
        raise FileNotFoundError(f"ISIC ground-truth CSV not found: {groundtruth_path}")
    return pd.read_csv(groundtruth_path)


def load_isic_metadata(metadata_csv: Path | str = ISIC_METADATA_CSV) -> pd.DataFrame:
    """Load the ISIC metadata table."""
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"ISIC metadata CSV not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def load_isic_catalog(
    groundtruth_csv: Path | str = ISIC_GROUNDTRUTH_CSV,
    metadata_csv: Path | str = ISIC_METADATA_CSV,
) -> pd.DataFrame:
    """Load ISIC labels and merge metadata when available."""
    groundtruth = load_isic_ground_truth(groundtruth_csv=groundtruth_csv)

    metadata_path = Path(metadata_csv)
    if metadata_path.exists():
        metadata = load_isic_metadata(metadata_csv=metadata_path)
        if "image" in metadata.columns:
            return groundtruth.merge(metadata, on="image", how="left")

    return groundtruth


def build_isic_image_index(
    image_root: Path | str = ISIC_IMAGE_ROOT,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> dict[str, Path]:
    """Index image files by stem for fast lookup."""
    image_root_path = Path(image_root)
    if not image_root_path.exists():
        raise FileNotFoundError(f"ISIC image root not found: {image_root_path}")

    allowed = {ext.lower() for ext in extensions}
    index: dict[str, Path] = {}
    for image_path in image_root_path.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in allowed:
            continue
        stem = image_path.stem
        if stem in index and index[stem] != image_path:
            raise ValueError(f"Duplicate ISIC image stem found: {stem}")
        index[stem] = image_path
    return index


def resolve_isic_image_path(
    image_id: str | Path,
    image_root: Path | str = ISIC_IMAGE_ROOT,
    image_index: Mapping[str, Path] | None = None,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> Path:
    """Resolve an ISIC image stem or path to an on-disk image."""
    image_path = Path(image_id)
    if image_path.exists():
        return image_path

    stem = image_path.stem if image_path.suffix else str(image_id)
    if image_index and stem in image_index:
        return image_index[stem]

    image_root_path = Path(image_root)
    if not image_root_path.exists():
        raise FileNotFoundError(f"ISIC image root not found: {image_root_path}")

    for extension in extensions:
        candidate = image_root_path / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    allowed = {ext.lower() for ext in extensions}
    matches = [candidate for candidate in image_root_path.rglob(f"{stem}.*") if candidate.suffix.lower() in allowed]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve ISIC image '{image_id}' under {image_root_path}")


def load_isic_image(
    image_id: str | Path,
    image_root: Path | str = ISIC_IMAGE_ROOT,
    image_index: Mapping[str, Path] | None = None,
) -> np.ndarray:
    """Load an ISIC image as an RGB numpy array."""
    resolved_path = resolve_isic_image_path(image_id, image_root=image_root, image_index=image_index)
    with Image.open(resolved_path) as image:
        return np.asarray(image.convert("RGB"))


def get_isic_label_from_row(row: pd.Series | Mapping[str, Any]) -> str | None:
    """Derive the canonical ISIC class label from a metadata/ground-truth row."""
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    for candidate in ("label", "dx", "diagnosis", "class"):
        value = values.get(candidate)
        if value is not None and not pd.isna(value):
            return str(value).strip().lower()

    scores: dict[str, float] = {}
    for column in ISIC_CLASS_COLUMNS:
        if column not in values:
            continue
        value = values[column]
        if pd.isna(value):
            continue
        try:
            scores[column] = float(value)
        except (TypeError, ValueError):
            continue

    if not scores:
        return None

    return max(scores.items(), key=lambda item: item[1])[0].lower()


def get_pixel_columns(columns: Sequence[str]) -> list[str]:
    """Extract ordered pixel columns (pixel0000...pixel2351) from dataframe columns."""
    pixel_cols = [col for col in columns if col.startswith("pixel")]
    if not pixel_cols:
        raise ValueError("No pixel columns were found in the provided columns.")
    return sorted(pixel_cols)


def reconstruct_image_from_row(
    row: pd.Series | np.ndarray,
    pixel_columns: Sequence[str] | None = None,
    image_size: tuple[int, int] = (28, 28),
    channels: int = 3,
) -> np.ndarray:
    """Reconstruct a HxWxC image array from one HMNIST CSV row."""
    expected_values = image_size[0] * image_size[1] * channels

    if isinstance(row, pd.Series):
        if pixel_columns is None:
            pixel_columns = get_pixel_columns(row.index)
        pixel_values = row[list(pixel_columns)].to_numpy(dtype=np.uint8)
    else:
        pixel_values = np.asarray(row, dtype=np.uint8)

    if pixel_values.size != expected_values:
        raise ValueError(
            f"Expected {expected_values} pixel values, but got {pixel_values.size}."
        )

    return pixel_values.reshape(image_size[0], image_size[1], channels)


def preprocess_image(
    image: np.ndarray,
    normalize: bool = True,
    to_channels_first: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Preprocess image for model input with optional normalization and standardization."""
    if image.ndim != 3:
        raise ValueError("Expected image with shape (H, W, C).")

    processed = image.astype(np.float32)

    if normalize:
        processed /= 255.0

    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        if mean_arr.shape[-1] != processed.shape[-1] or std_arr.shape[-1] != processed.shape[-1]:
            raise ValueError("Mean and std must match the number of image channels.")
        processed = (processed - mean_arr) / std_arr

    if to_channels_first:
        processed = np.transpose(processed, (2, 0, 1))

    return processed


def load_and_prepare_sample(
    row_index: int = 0,
    hmnist_csv: Path | str = HMNIST_RGB_CSV,
    normalize: bool = True,
) -> tuple[np.ndarray, int | None]:
    """Load one HMNIST sample, reconstruct image, and return image + label."""
    df = load_hmnist_pixels(hmnist_csv=hmnist_csv)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index out of range: {row_index}")

    row = df.iloc[row_index]
    image = reconstruct_image_from_row(row)
    image = preprocess_image(image, normalize=normalize)

    label = int(row["label"]) if "label" in row else None
    return image, label


def load_and_prepare_isic_sample(
    row_index: int = 0,
    image_root: Path | str = ISIC_IMAGE_ROOT,
    groundtruth_csv: Path | str = ISIC_GROUNDTRUTH_CSV,
    metadata_csv: Path | str = ISIC_METADATA_CSV,
    normalize: bool = True,
) -> tuple[np.ndarray, str | None]:
    """Load one ISIC sample, resolve the image file, and return image + label."""
    catalog = load_isic_catalog(groundtruth_csv=groundtruth_csv, metadata_csv=metadata_csv)
    if row_index < 0 or row_index >= len(catalog):
        raise IndexError(f"row_index out of range: {row_index}")

    row = catalog.iloc[row_index]
    image = load_isic_image(row["image"], image_root=image_root)
    image = preprocess_image(image, normalize=normalize)
    label = get_isic_label_from_row(row)
    return image, label
