"""nuScenes motion forecasting dataset.

Loads data from the nuScenes prediction challenge format and provides
rasterized BEV input for the RasterEncoder.

Requirements:
- nuScenes dataset (v1.0-trainval)
- nuscenes-devkit: pip install nuscenes-devkit

The dataset outputs:
- Rasterized BEV image with semantic channels (road, lanes, agents, etc.)
- Future trajectory in global coordinates
- Ego pose for coordinate transforms
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import BenchmarkDataset

logger = logging.getLogger(__name__)

NUSCENES_PREDICTION_HZ = 2.0
NUSCENES_PREDICTION_HORIZON = 12
NUSCENES_NUM_MODES = 5


def _try_import_nuscenes():
    """Try to import nuscenes-devkit."""
    try:
        from nuscenes import NuScenes
        from nuscenes.eval.prediction.splits import get_prediction_challenge_split
        from nuscenes.prediction import PredictHelper
        from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
        from nuscenes.prediction.input_representation.combinators import Rasterizer
        from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer

        return {
            "NuScenes": NuScenes,
            "get_prediction_challenge_split": get_prediction_challenge_split,
            "PredictHelper": PredictHelper,
            "Rasterizer": Rasterizer,
            "StaticLayerRasterizer": StaticLayerRasterizer,
            "AgentBoxesWithFadedHistory": AgentBoxesWithFadedHistory,
        }
    except ImportError:
        return None


class NuScenesMotionDataset(BenchmarkDataset):
    """nuScenes motion forecasting dataset with rasterized BEV input.

    Uses the official nuscenes-devkit to generate rasterized BEV images
    containing road geometry, lane markings, and agent history.
    """

    def __init__(
        self,
        dataroot: str | Path,
        split: str = "train",
        version: str = "v1.0-trainval",
        raster_resolution: float = 0.25,
        raster_size: tuple[int, int] = (200, 200),
        history_seconds: float = 2.0,
        future_seconds: float = 6.0,
    ):
        """Initialize nuScenes motion dataset.

        Args:
            dataroot: Path to nuScenes dataset root
            split: Dataset split ('train', 'train_val', 'val')
            version: nuScenes version string
            raster_resolution: Meters per pixel for rasterization
            raster_size: Output raster size (H, W)
            history_seconds: Seconds of agent history to include
            future_seconds: Seconds of future trajectory (should be 6.0 for benchmark)
        """
        self.dataroot = Path(dataroot)
        self.split = split
        self.version = version
        self.raster_resolution = raster_resolution
        self.raster_size = raster_size
        self.history_seconds = history_seconds
        self.future_seconds = future_seconds

        imports = _try_import_nuscenes()
        if imports is None:
            raise ImportError(
                "nuscenes-devkit is required for NuScenesMotionDataset. "
                "Install with: pip install nuscenes-devkit"
            )

        self._imports = imports

        logger.info(f"Loading nuScenes {version} from {dataroot}")
        self.nusc = imports["NuScenes"](version=version, dataroot=str(dataroot), verbose=False)
        self.helper = imports["PredictHelper"](self.nusc)

        self.samples = imports["get_prediction_challenge_split"](split, dataroot=str(dataroot))
        logger.info(f"Loaded {len(self.samples)} samples for split '{split}'")

        static_layer = imports["StaticLayerRasterizer"](
            self.helper,
            resolution=raster_resolution,
            meters_ahead=raster_size[0] * raster_resolution / 2,
            meters_behind=raster_size[0] * raster_resolution / 2,
            meters_left=raster_size[1] * raster_resolution / 2,
            meters_right=raster_size[1] * raster_resolution / 2,
        )

        agent_layer = imports["AgentBoxesWithFadedHistory"](
            self.helper,
            seconds_of_history=history_seconds,
            resolution=raster_resolution,
            meters_ahead=raster_size[0] * raster_resolution / 2,
            meters_behind=raster_size[0] * raster_resolution / 2,
            meters_left=raster_size[1] * raster_resolution / 2,
            meters_right=raster_size[1] * raster_resolution / 2,
        )

        self.rasterizer = imports["Rasterizer"](
            static_layer_rasterizer=static_layer,
            agent_rasterizer=agent_layer,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def input_type(self) -> str:
        return "raster"

    @property
    def num_modes(self) -> int:
        return NUSCENES_NUM_MODES

    @property
    def prediction_horizon(self) -> int:
        return NUSCENES_PREDICTION_HORIZON

    @property
    def prediction_hz(self) -> float:
        return NUSCENES_PREDICTION_HZ

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary with:
            - raster: (C, H, W) rasterized BEV image
            - future_trajectory: (12, 2) future xy positions in global frame
            - future_yaw: (12,) future yaw angles
            - ego_pose: (4, 4) ego-to-global transformation matrix
            - instance_token: str identifier
            - sample_token: str identifier
        """
        instance_token, sample_token = self.samples[idx].split("_")

        raster = self.rasterizer.make_input_representation(instance_token, sample_token)
        raster = torch.from_numpy(raster).permute(2, 0, 1).float() / 255.0

        future = self.helper.get_future_for_agent(
            instance_token,
            sample_token,
            seconds=self.future_seconds,
            in_agent_frame=False,
        )

        if len(future) < NUSCENES_PREDICTION_HORIZON:
            padding = np.zeros((NUSCENES_PREDICTION_HORIZON - len(future), 2))
            if len(future) > 0:
                padding[:] = future[-1]
            future = np.vstack([future, padding]) if len(future) > 0 else padding

        future_trajectory = torch.from_numpy(future[:NUSCENES_PREDICTION_HORIZON]).float()

        annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        current_yaw = np.arctan2(
            annotation["rotation"][2],
            annotation["rotation"][0],
        ) * 2

        future_yaw = torch.zeros(NUSCENES_PREDICTION_HORIZON)
        if len(future) >= 2:
            for i in range(min(len(future) - 1, NUSCENES_PREDICTION_HORIZON)):
                dx = future[i + 1, 0] - future[i, 0]
                dy = future[i + 1, 1] - future[i, 1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    future_yaw[i] = np.arctan2(dy, dx)
                else:
                    future_yaw[i] = future_yaw[i - 1] if i > 0 else current_yaw
            future_yaw[len(future) - 1 :] = future_yaw[len(future) - 2]

        sample = self.nusc.get("sample", sample_token)
        ego_pose_token = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"]
        ego_pose_record = self.nusc.get("ego_pose", ego_pose_token)

        ego_pose = np.eye(4, dtype=np.float32)
        ego_pose[:3, 3] = ego_pose_record["translation"]

        from pyquaternion import Quaternion

        q = Quaternion(ego_pose_record["rotation"])
        ego_pose[:3, :3] = q.rotation_matrix

        return {
            "raster": raster,
            "future_trajectory": future_trajectory,
            "future_yaw": future_yaw,
            "ego_pose": torch.from_numpy(ego_pose),
            "instance_token": instance_token,
            "sample_token": sample_token,
        }


class NuScenesMotionDatasetDummy(BenchmarkDataset):
    """Dummy nuScenes dataset for testing without actual data."""

    def __init__(
        self,
        num_samples: int = 100,
        raster_channels: int = 3,
        raster_size: tuple[int, int] = (200, 200),
    ):
        self.num_samples = num_samples
        self.raster_channels = raster_channels
        self.raster_size = raster_size

    def __len__(self) -> int:
        return self.num_samples

    @property
    def input_type(self) -> str:
        return "raster"

    @property
    def num_modes(self) -> int:
        return NUSCENES_NUM_MODES

    @property
    def prediction_horizon(self) -> int:
        return NUSCENES_PREDICTION_HORIZON

    @property
    def prediction_hz(self) -> float:
        return NUSCENES_PREDICTION_HZ

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raster = torch.randn(self.raster_channels, *self.raster_size)

        t = torch.linspace(0, 6.0, NUSCENES_PREDICTION_HORIZON)
        speed = 5.0 + torch.randn(1).item() * 2.0
        future_trajectory = torch.stack([speed * t, torch.zeros_like(t)], dim=-1)

        future_yaw = torch.zeros(NUSCENES_PREDICTION_HORIZON)

        ego_pose = torch.eye(4)
        ego_pose[0, 3] = idx * 10.0

        return {
            "raster": raster,
            "future_trajectory": future_trajectory,
            "future_yaw": future_yaw,
            "ego_pose": ego_pose,
            "instance_token": f"instance_{idx}",
            "sample_token": f"sample_{idx}",
        }
