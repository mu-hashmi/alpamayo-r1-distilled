"""Output adapters for benchmark-specific trajectory formats.

Transforms model output to benchmark requirements:
- nuScenes: 12 waypoints @ 2Hz, global coordinate frame
- Argoverse 2: 60 waypoints @ 10Hz, map coordinate frame
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class OutputAdapter(ABC):
    """Abstract base class for output format adapters."""

    @abstractmethod
    def adapt(
        self,
        trajectories: torch.Tensor,
        mode_probs: torch.Tensor | None = None,
        ego_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Adapt model output to benchmark format.

        Args:
            trajectories: Model output (B, K, T, 4) or (B, T, 4)
            mode_probs: Mode probabilities (B, K) if multi-modal
            ego_pose: Ego pose for coordinate transform (B, 4, 4) or (B, 3, 3)

        Returns:
            Adapted trajectories and mode probabilities
        """
        pass

    @property
    @abstractmethod
    def output_waypoints(self) -> int:
        """Number of output waypoints."""
        pass

    @property
    @abstractmethod
    def output_hz(self) -> float:
        """Output frequency in Hz."""
        pass


class NuScenesAdapter(OutputAdapter):
    """Adapter for nuScenes prediction benchmark.

    Requirements:
    - 12 waypoints @ 2Hz (6 second horizon)
    - Global coordinate frame
    - Up to 25 modes, typically K=5 or K=6
    """

    def __init__(
        self,
        model_hz: float = 10.0,
        model_waypoints: int = 64,
    ):
        self.model_hz = model_hz
        self.model_waypoints = model_waypoints
        self._output_waypoints = 12
        self._output_hz = 2.0

    @property
    def output_waypoints(self) -> int:
        return self._output_waypoints

    @property
    def output_hz(self) -> float:
        return self._output_hz

    def adapt(
        self,
        trajectories: torch.Tensor,
        mode_probs: torch.Tensor | None = None,
        ego_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Adapt to nuScenes format.

        Args:
            trajectories: (B, K, T, 4) or (B, T, 4) model output
            mode_probs: (B, K) mode probabilities
            ego_pose: (B, 4, 4) ego-to-global transform

        Returns:
            Resampled trajectories (B, K, 12, 4) or (B, 12, 4) in global frame
        """
        is_multimodal = trajectories.dim() == 4 and trajectories.shape[1] > 1

        if not is_multimodal and trajectories.dim() == 3:
            trajectories = trajectories.unsqueeze(1)

        B, K, T, D = trajectories.shape

        sample_rate = int(self.model_hz / self._output_hz)
        indices = torch.arange(0, T, sample_rate, device=trajectories.device)
        indices = indices[: self._output_waypoints]

        if len(indices) < self._output_waypoints:
            last_idx = indices[-1] if len(indices) > 0 else 0
            padding = torch.full(
                (self._output_waypoints - len(indices),),
                last_idx,
                device=trajectories.device,
            )
            indices = torch.cat([indices, padding])

        resampled = trajectories[:, :, indices, :]

        if ego_pose is not None:
            resampled = self._ego_to_global(resampled, ego_pose)

        if not is_multimodal:
            resampled = resampled.squeeze(1)

        return resampled, mode_probs

    def _ego_to_global(
        self,
        traj_ego: torch.Tensor,
        ego_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Transform from ego-centric to global coordinates.

        Args:
            traj_ego: (B, K, T, 4) trajectories in ego frame
            ego_pose: (B, 4, 4) ego-to-global transformation

        Returns:
            (B, K, T, 4) trajectories in global frame
        """
        B, K, T, _ = traj_ego.shape

        xy = traj_ego[..., :2]
        sin_yaw = traj_ego[..., 2]
        cos_yaw = traj_ego[..., 3]

        R = ego_pose[:, :2, :2]
        t = ego_pose[:, :2, 3]

        xy_flat = xy.reshape(B, -1, 2)
        xy_global = torch.einsum("bij,bnj->bni", R, xy_flat) + t.unsqueeze(1)
        xy_global = xy_global.reshape(B, K, T, 2)

        ego_yaw = torch.atan2(ego_pose[:, 1, 0], ego_pose[:, 0, 0])
        yaw = torch.atan2(sin_yaw, cos_yaw) + ego_yaw[:, None, None]

        return torch.cat(
            [xy_global, yaw.sin().unsqueeze(-1), yaw.cos().unsqueeze(-1)], dim=-1
        )


class Argoverse2Adapter(OutputAdapter):
    """Adapter for Argoverse 2 motion forecasting benchmark.

    Requirements:
    - 60 waypoints @ 10Hz (6 second horizon)
    - Map coordinate frame (similar to ego but may need rotation)
    - K=6 modes with probabilities
    """

    def __init__(
        self,
        model_hz: float = 10.0,
        model_waypoints: int = 64,
    ):
        self.model_hz = model_hz
        self.model_waypoints = model_waypoints
        self._output_waypoints = 60
        self._output_hz = 10.0

    @property
    def output_waypoints(self) -> int:
        return self._output_waypoints

    @property
    def output_hz(self) -> float:
        return self._output_hz

    def adapt(
        self,
        trajectories: torch.Tensor,
        mode_probs: torch.Tensor | None = None,
        ego_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Adapt to Argoverse 2 format.

        Args:
            trajectories: (B, K, T, 4) or (B, T, 4) model output (T >= 60)
            mode_probs: (B, K) mode probabilities
            ego_pose: (B, 4, 4) ego-to-map transform (optional)

        Returns:
            Truncated trajectories (B, K, 60, 4) or (B, 60, 4)
        """
        is_multimodal = trajectories.dim() == 4 and trajectories.shape[1] > 1

        if not is_multimodal and trajectories.dim() == 3:
            trajectories = trajectories.unsqueeze(1)

        truncated = trajectories[:, :, : self._output_waypoints, :]

        if ego_pose is not None:
            truncated = self._transform_to_map(truncated, ego_pose)

        if not is_multimodal:
            truncated = truncated.squeeze(1)

        return truncated, mode_probs

    def _transform_to_map(
        self,
        traj_ego: torch.Tensor,
        ego_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Transform from ego-centric to map coordinates.

        For AV2, this is typically just a rotation to align with map north.

        Args:
            traj_ego: (B, K, T, 4) trajectories in ego frame
            ego_pose: (B, 4, 4) ego-to-map transformation

        Returns:
            (B, K, T, 4) trajectories in map frame
        """
        B, K, T, _ = traj_ego.shape

        xy = traj_ego[..., :2]
        sin_yaw = traj_ego[..., 2]
        cos_yaw = traj_ego[..., 3]

        R = ego_pose[:, :2, :2]
        t = ego_pose[:, :2, 3]

        xy_flat = xy.reshape(B, -1, 2)
        xy_map = torch.einsum("bij,bnj->bni", R, xy_flat) + t.unsqueeze(1)
        xy_map = xy_map.reshape(B, K, T, 2)

        ego_yaw = torch.atan2(ego_pose[:, 1, 0], ego_pose[:, 0, 0])
        yaw = torch.atan2(sin_yaw, cos_yaw) + ego_yaw[:, None, None]

        return torch.cat(
            [xy_map, yaw.sin().unsqueeze(-1), yaw.cos().unsqueeze(-1)], dim=-1
        )


def create_adapter(benchmark: str | None) -> OutputAdapter | None:
    """Factory function to create output adapter.

    Args:
        benchmark: Benchmark name ("nuscenes", "argoverse2") or None

    Returns:
        OutputAdapter instance or None if benchmark is None
    """
    if benchmark is None:
        return None
    elif benchmark == "nuscenes":
        return NuScenesAdapter()
    elif benchmark == "argoverse2":
        return Argoverse2Adapter()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
