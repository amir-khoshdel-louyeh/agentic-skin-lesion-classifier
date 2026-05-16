"""Utilities for loading and preparing HAM10000/HMNIST tabular image data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DATASET_ROOT = Path("dataset") / "HAM1000"
METADATA_CSV = DATASET_ROOT / "HAM10000_metadata.csv"
HMNIST_RGB_CSV = DATASET_ROOT / "hmnist_28_28_RGB.csv" / "hmnist_28_28_RGB.csv"


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
