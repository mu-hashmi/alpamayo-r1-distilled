"""Datasets for training and evaluation."""

from .argoverse2_dataset import (
    Argoverse2MotionDataset,
    Argoverse2MotionDatasetDummy,
)
from .base import BenchmarkDataset, collate_benchmark_batch
from .nuscenes_dataset import (
    NuScenesMotionDataset,
    NuScenesMotionDatasetDummy,
)
from .physicalai import (
    DEFAULT_CAMERAS,
    DistillationDataset,
    DistillationDatasetLocal,
    DistillationDatasetOffline,
    create_dataloaders,
)

__all__ = [
    # PhysicalAI-AV
    "DEFAULT_CAMERAS",
    "DistillationDataset",
    "DistillationDatasetLocal",
    "DistillationDatasetOffline",
    "create_dataloaders",
    # Benchmarks
    "BenchmarkDataset",
    "collate_benchmark_batch",
    "NuScenesMotionDataset",
    "NuScenesMotionDatasetDummy",
    "Argoverse2MotionDataset",
    "Argoverse2MotionDatasetDummy",
]


def create_benchmark_dataset(
    benchmark: str,
    dataroot: str,
    split: str = "train",
    dummy: bool = False,
    **kwargs,
) -> BenchmarkDataset:
    """Factory function to create benchmark datasets.

    Args:
        benchmark: Benchmark name ('nuscenes' or 'argoverse2')
        dataroot: Path to dataset root
        split: Dataset split
        dummy: If True, return dummy dataset for testing
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        BenchmarkDataset instance
    """
    if benchmark == "nuscenes":
        if dummy:
            return NuScenesMotionDatasetDummy(**kwargs)
        return NuScenesMotionDataset(dataroot=dataroot, split=split, **kwargs)
    elif benchmark == "argoverse2":
        if dummy:
            return Argoverse2MotionDatasetDummy(**kwargs)
        return Argoverse2MotionDataset(dataroot=dataroot, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Expected 'nuscenes' or 'argoverse2'")
