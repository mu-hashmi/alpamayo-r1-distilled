"""Abstract base class for scene encoders.

All encoders produce a common output format that can be consumed by the
trajectory decoder, regardless of input modality (camera, raster, vector).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    """Base configuration for all encoders."""

    output_channels: int = 512
    output_size: tuple[int, int] = (200, 200)


class AbstractEncoder(nn.Module, ABC):
    """Abstract base class for scene encoders.

    All encoders must produce a common output format:
    - Shape: (B, output_channels, H, W)
    - Semantics: BEV-like spatial features in ego-centric frame

    This allows the TrajectoryDecoder to work with any encoder type.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode inputs to spatial features.

        Args:
            inputs: Dictionary with encoder-specific keys:
                - camera encoder: {"images": (B, N_cam, C, H, W)}
                - raster encoder: {"raster_bev": (B, C, H, W)}
                - vector encoder: {"agent_history": (B, N_agents, T, D),
                                   "map_polylines": (B, N_lanes, P, 2)}

        Returns:
            Spatial features (B, output_channels, output_H, output_W)
        """
        pass

    @property
    @abstractmethod
    def input_type(self) -> str:
        """Return encoder input type: 'camera', 'raster', or 'vector'."""
        pass

    @property
    def output_channels(self) -> int:
        """Number of output feature channels."""
        return self.config.output_channels

    @property
    def output_size(self) -> tuple[int, int]:
        """Spatial size of output features (H, W)."""
        return self.config.output_size
