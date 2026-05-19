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

from src.data_utils import load_hmnist_pixels, reconstruct_image_from_row


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


class ISICAdapter(HMNISTAdapter):
    """Placeholder for an ISIC adapter.

    Real implementation should convert ISIC image files and metadata into the
    same `get_image` / `get_label` interface.
    """

    def __init__(self, isic_root: str | Path):
        super().__init__()
        self.isic_root = Path(isic_root)

    # TODO: implement ISIC loading (images + metadata)


__all__ = ["DatasetAdapter", "HMNISTAdapter", "ISICAdapter"]
