"""Benchmark support for nuScenes and Argoverse 2 motion forecasting."""

from .output_adapters import (
    Argoverse2Adapter,
    NuScenesAdapter,
    OutputAdapter,
)

__all__ = [
    "OutputAdapter",
    "NuScenesAdapter",
    "Argoverse2Adapter",
]
