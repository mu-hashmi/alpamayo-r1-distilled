"""Encoder modules for different input modalities.

Supports:
- Camera: Multi-camera, multi-frame images → BEV features
- Raster: Rasterized BEV (nuScenes-style) → BEV features
- Vector: Agent tracks + map polylines (Argoverse 2-style) → BEV features
"""

from .base import AbstractEncoder, EncoderConfig
from .camera_encoder import (
    CameraBEVEncoder,
    CameraBEVEncoderDummy,
    CameraEncoderConfig,
)
from .raster_encoder import RasterEncoder, RasterEncoderConfig
from .vector_encoder import VectorEncoder, VectorEncoderConfig

__all__ = [
    "AbstractEncoder",
    "EncoderConfig",
    "CameraBEVEncoder",
    "CameraBEVEncoderDummy",
    "CameraEncoderConfig",
    "RasterEncoder",
    "RasterEncoderConfig",
    "VectorEncoder",
    "VectorEncoderConfig",
]
