"""Data ingestion adapters to make the pipeline dataset-agnostic.

Provides a small `DatasetAdapter` API and an HMNIST adapter implementation.
Other adapters (ISIC, custom CSVs) can implement the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_utils import (
    build_isic_image_index,
    get_isic_label_from_row,
    load_hmnist_pixels,
    load_isic_catalog,
    load_isic_image,
    reconstruct_image_from_row,
    resolve_isic_image_path,
)


class DatasetAdapter(ABC):
    """Abstract dataset ingestion adapter.

    Implementations must provide `n_samples`, `get_image(index)` and optionally
    `get_label(index)` and `get_row(index)` for downstream evaluation and metadata.
    """

    @abstractmethod
    def n_samples(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_image(self, index: int) -> np.ndarray:
        raise NotImplementedError()

    def get_label(self, index: int) -> Any | None:
        return None

    def get_row(self, index: int) -> Mapping[str, Any] | None:
        return None


class HMNISTAdapter(DatasetAdapter):
    """Adapter for the provided `hmnist_28_28_RGB.csv` format."""

    def __init__(self, csv_path: str | Path | None = None):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def _ensure_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = load_hmnist_pixels(self.csv_path) if self.csv_path else load_hmnist_pixels()
        return self._df

    def n_samples(self) -> int:
        return len(self._ensure_df())

    def get_image(self, index: int) -> np.ndarray:
        df = self._ensure_df()
        if index < 0 or index >= len(df):
            raise IndexError("index out of range")
        return reconstruct_image_from_row(df.iloc[index])

    def get_label(self, index: int) -> Any | None:
        df = self._ensure_df()
        if "label" in df.columns:
            return df.iloc[index]["label"]
        return None

    def get_row(self, index: int) -> Mapping[str, Any] | None:
        df = self._ensure_df()
        return df.iloc[index].to_dict()


class ISICAdapter(DatasetAdapter):
    """Adapter for the image-based ISIC 2019 training dataset."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        groundtruth_csv: str | Path | None = None,
        metadata_csv: str | Path | None = None,
        image_root: str | Path | None = None,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root is not None else Path("dataset")
        self.groundtruth_csv = Path(groundtruth_csv) if groundtruth_csv is not None else self.dataset_root / "ISIC_2019_Training_GroundTruth.csv"
        self.metadata_csv = Path(metadata_csv) if metadata_csv is not None else self.dataset_root / "ISIC_2019_Training_Metadata.csv"
        self.image_root = Path(image_root) if image_root is not None else self.dataset_root / "ISIC_2019_Training_Input"
        self._df: pd.DataFrame | None = None
        self._image_index: dict[str, Path] | None = None

    def _ensure_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = load_isic_catalog(groundtruth_csv=self.groundtruth_csv, metadata_csv=self.metadata_csv)
        return self._df

    def _ensure_image_index(self) -> dict[str, Path]:
        if self._image_index is None:
            self._image_index = build_isic_image_index(self.image_root)
        return self._image_index

    def _ensure_index(self, index: int) -> pd.Series:
        df = self._ensure_df()
        if index < 0 or index >= len(df):
            raise IndexError("index out of range")
        return df.iloc[index]

    def n_samples(self) -> int:
        return len(self._ensure_df())

    def get_image(self, index: int) -> np.ndarray:
        row = self._ensure_index(index)
        return load_isic_image(row["image"], image_root=self.image_root, image_index=self._ensure_image_index())

    def get_label(self, index: int) -> Any | None:
        return get_isic_label_from_row(self._ensure_index(index))

    def get_row(self, index: int) -> Mapping[str, Any] | None:
        row = self._ensure_index(index).to_dict()
        image_name = row.get("image")
        if image_name is not None:
            try:
                row["image_path"] = str(
                    resolve_isic_image_path(
                        image_name,
                        image_root=self.image_root,
                        image_index=self._ensure_image_index(),
                    )
                )
            except Exception:
                pass
        return row


__all__ = ["DatasetAdapter", "HMNISTAdapter", "ISICAdapter"]
