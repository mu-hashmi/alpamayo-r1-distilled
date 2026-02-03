"""Abstract base class for benchmark datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class BenchmarkDataset(Dataset, ABC):
    """Abstract base class for motion forecasting benchmark datasets.

    All benchmark datasets should output data in a consistent format that
    can be consumed by the encoder abstraction layer.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary with benchmark-specific keys. Common keys include:
            - For raster input: "raster" (B, C, H, W)
            - For vector input: "agent_history", "map_polylines", etc.
            - For targets: "future_trajectory", "future_yaw"
        """
        pass

    @property
    @abstractmethod
    def input_type(self) -> str:
        """Return the input type: 'raster' or 'vector'."""
        pass

    @property
    @abstractmethod
    def num_modes(self) -> int:
        """Expected number of prediction modes for this benchmark."""
        pass

    @property
    @abstractmethod
    def prediction_horizon(self) -> int:
        """Number of future timesteps to predict."""
        pass

    @property
    @abstractmethod
    def prediction_hz(self) -> float:
        """Prediction frequency in Hz."""
        pass


def collate_benchmark_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for benchmark datasets.

    Handles variable-length sequences by padding with zeros and creating masks.
    """
    if not batch:
        return {}

    result = {}
    first = batch[0]

    for key in first.keys():
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            if all(v.shape == values[0].shape for v in values):
                result[key] = torch.stack(values)
            else:
                max_shape = [max(v.shape[i] for v in values) for i in range(values[0].dim())]
                padded = []
                masks = []
                for v in values:
                    pad_shape = list(max_shape)
                    padded_v = torch.zeros(*pad_shape, dtype=v.dtype)
                    slices = tuple(slice(0, s) for s in v.shape)
                    padded_v[slices] = v
                    padded.append(padded_v)

                    mask = torch.zeros(*pad_shape[:len(v.shape)], dtype=torch.bool)
                    mask[slices] = True
                    masks.append(mask)

                result[key] = torch.stack(padded)
                result[f"{key}_mask"] = torch.stack(masks)
        elif isinstance(values[0], str):
            result[key] = values
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values

    return result
