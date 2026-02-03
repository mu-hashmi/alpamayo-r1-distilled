"""Alpamayo distillation package.

This package provides tools for distilling Alpamayo-R1's reasoning capabilities
into smaller student models.
"""

from .calibration import (
    BEVProjector,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)
from .datasets.physicalai import (
    DistillationDataset,
    DistillationDatasetLocal,
    DistillationDatasetOffline,
)
from .losses import (
    CombinedLoss,
    ContrastiveLoss,
    MetricsComputer,
    TrajectoryLoss,
    compute_ade,
    compute_fde,
)
from .models import BaselineStudent, ReasoningStudent

__all__ = [
    # Models
    "BaselineStudent",
    "ReasoningStudent",
    # Calibration
    "BEVProjector",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    # Dataset
    "DistillationDataset",
    "DistillationDatasetLocal",
    "DistillationDatasetOffline",
    # Losses
    "TrajectoryLoss",
    "ContrastiveLoss",
    "CombinedLoss",
    "MetricsComputer",
    "compute_ade",
    "compute_fde",
]
