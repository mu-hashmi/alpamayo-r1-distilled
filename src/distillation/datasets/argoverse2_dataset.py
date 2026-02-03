"""Argoverse 2 motion forecasting dataset.

Loads data from the Argoverse 2 motion forecasting format and provides
vectorized input for the VectorEncoder.

Requirements:
- Argoverse 2 dataset
- av2: pip install av2

The dataset outputs:
- Agent history tracks (position, velocity, heading)
- HD map polylines (lane centerlines, boundaries)
- Future trajectory in map coordinates
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import BenchmarkDataset

logger = logging.getLogger(__name__)

ARGOVERSE2_PREDICTION_HZ = 10.0
ARGOVERSE2_PREDICTION_HORIZON = 60
ARGOVERSE2_NUM_MODES = 6
ARGOVERSE2_HISTORY_HZ = 10.0
ARGOVERSE2_HISTORY_LENGTH = 50


def _try_import_av2():
    """Try to import av2."""
    try:
        from av2.datasets.motion_forecasting import scenario_serialization
        from av2.datasets.motion_forecasting.data_schema import (
            ArgoverseScenario,
            ObjectType,
            TrackCategory,
        )
        from av2.map.map_api import ArgoverseStaticMap

        return {
            "scenario_serialization": scenario_serialization,
            "ArgoverseScenario": ArgoverseScenario,
            "ObjectType": ObjectType,
            "TrackCategory": TrackCategory,
            "ArgoverseStaticMap": ArgoverseStaticMap,
        }
    except ImportError:
        return None


class Argoverse2MotionDataset(BenchmarkDataset):
    """Argoverse 2 motion forecasting dataset with vectorized input.

    Provides agent tracks and HD map polylines in a format suitable
    for the VectorEncoder.
    """

    def __init__(
        self,
        dataroot: str | Path,
        split: str = "train",
        max_agents: int = 64,
        max_map_polylines: int = 256,
        max_points_per_polyline: int = 20,
        history_length: int = ARGOVERSE2_HISTORY_LENGTH,
        map_range: float = 100.0,
    ):
        """Initialize Argoverse 2 motion dataset.

        Args:
            dataroot: Path to Argoverse 2 dataset root
            split: Dataset split ('train', 'val', 'test')
            max_agents: Maximum number of agents to include
            max_map_polylines: Maximum number of map polylines
            max_points_per_polyline: Maximum points per polyline
            history_length: Number of history timesteps (50 @ 10Hz = 5s)
            map_range: Range in meters for map elements around focal agent
        """
        self.dataroot = Path(dataroot)
        self.split = split
        self.max_agents = max_agents
        self.max_map_polylines = max_map_polylines
        self.max_points_per_polyline = max_points_per_polyline
        self.history_length = history_length
        self.map_range = map_range

        imports = _try_import_av2()
        if imports is None:
            raise ImportError(
                "av2 is required for Argoverse2MotionDataset. "
                "Install with: pip install av2"
            )

        self._imports = imports

        split_dir = self.dataroot / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Fast path listing: list directories directly instead of globbing
        # This is much faster on network storage with 200K directories
        import os
        scenario_dirs = os.listdir(split_dir)
        self.scenario_paths = []
        for scenario_id in sorted(scenario_dirs):
            parquet_path = split_dir / scenario_id / f"scenario_{scenario_id}.parquet"
            self.scenario_paths.append(parquet_path)
        logger.info(f"Found {len(self.scenario_paths)} scenarios for split '{split}'")

    def __len__(self) -> int:
        return len(self.scenario_paths)

    @property
    def input_type(self) -> str:
        return "vector"

    @property
    def num_modes(self) -> int:
        return ARGOVERSE2_NUM_MODES

    @property
    def prediction_horizon(self) -> int:
        return ARGOVERSE2_PREDICTION_HORIZON

    @property
    def prediction_hz(self) -> float:
        return ARGOVERSE2_PREDICTION_HZ

    def _load_scenario(self, scenario_path: Path):
        """Load a scenario from parquet file."""
        scenario = self._imports["scenario_serialization"].load_argoverse_scenario_parquet(
            scenario_path
        )
        return scenario

    def _load_map(self, scenario_path: Path):
        """Load the map for a scenario."""
        map_path = scenario_path.parent / f"log_map_archive_{scenario_path.parent.name}.json"
        if map_path.exists():
            return self._imports["ArgoverseStaticMap"].from_json(map_path)
        return None

    def _get_agent_features(
        self,
        scenario,
        focal_track_id: str,
        current_timestep: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract agent history features.

        Returns:
            agent_history: (N_agents, T, 6) - x, y, vx, vy, heading, valid
            agent_mask: (N_agents, T) - valid timesteps
            agent_positions: (N_agents, 2) - current positions
            agent_types: (N_agents,) - agent type indices
        """
        ObjectType = self._imports["ObjectType"]

        type_to_idx = {
            ObjectType.VEHICLE: 0,
            ObjectType.PEDESTRIAN: 1,
            ObjectType.CYCLIST: 2,
            ObjectType.BUS: 3,
            ObjectType.MOTORCYCLIST: 4,
            ObjectType.UNKNOWN: 5,
        }

        focal_track = None
        for track in scenario.tracks:
            if track.track_id == focal_track_id:
                focal_track = track
                break

        if focal_track is None:
            raise ValueError(f"Focal track {focal_track_id} not found")

        focal_state = None
        for state in focal_track.object_states:
            if state.timestep == current_timestep:
                focal_state = state
                break

        if focal_state is None:
            raise ValueError(f"No state at timestep {current_timestep}")

        focal_pos = np.array([focal_state.position[0], focal_state.position[1]])
        focal_heading = focal_state.heading

        cos_h, sin_h = np.cos(-focal_heading), np.sin(-focal_heading)
        rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        agent_history = torch.zeros(self.max_agents, self.history_length, 6)
        agent_mask = torch.zeros(self.max_agents, self.history_length, dtype=torch.bool)
        agent_positions = torch.zeros(self.max_agents, 2)
        agent_types = torch.zeros(self.max_agents, dtype=torch.long)

        agent_idx = 0
        for track in scenario.tracks:
            if agent_idx >= self.max_agents:
                break

            track_states = {s.timestep: s for s in track.object_states}

            has_current = current_timestep in track_states
            if not has_current:
                continue

            current_state = track_states[current_timestep]
            pos = np.array([current_state.position[0], current_state.position[1]])
            rel_pos = rot_matrix @ (pos - focal_pos)

            if np.linalg.norm(rel_pos) > self.map_range:
                continue

            agent_positions[agent_idx] = torch.from_numpy(rel_pos).float()
            agent_types[agent_idx] = type_to_idx.get(track.object_type, 5)

            for t_offset in range(self.history_length):
                t = current_timestep - (self.history_length - 1 - t_offset)
                if t in track_states:
                    state = track_states[t]
                    pos = np.array([state.position[0], state.position[1]])
                    vel = np.array([state.velocity[0], state.velocity[1]])

                    rel_pos = rot_matrix @ (pos - focal_pos)
                    rel_vel = rot_matrix @ vel
                    rel_heading = state.heading - focal_heading

                    agent_history[agent_idx, t_offset, 0] = rel_pos[0]
                    agent_history[agent_idx, t_offset, 1] = rel_pos[1]
                    agent_history[agent_idx, t_offset, 2] = rel_vel[0]
                    agent_history[agent_idx, t_offset, 3] = rel_vel[1]
                    agent_history[agent_idx, t_offset, 4] = np.sin(rel_heading)
                    agent_history[agent_idx, t_offset, 5] = np.cos(rel_heading)
                    agent_mask[agent_idx, t_offset] = True

            agent_idx += 1

        return agent_history, agent_mask, agent_positions, agent_types

    def _get_map_features(
        self,
        static_map,
        focal_pos: np.ndarray,
        focal_heading: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract map polyline features.

        Returns:
            map_polylines: (N_polylines, N_points, 2) - xy coordinates
            map_mask: (N_polylines, N_points) - valid points
            map_positions: (N_polylines, 2) - polyline center positions
            map_types: (N_polylines,) - polyline type indices
        """
        map_polylines = torch.zeros(self.max_map_polylines, self.max_points_per_polyline, 2)
        map_mask = torch.zeros(
            self.max_map_polylines, self.max_points_per_polyline, dtype=torch.bool
        )
        map_positions = torch.zeros(self.max_map_polylines, 2)
        map_types = torch.zeros(self.max_map_polylines, dtype=torch.long)

        if static_map is None:
            return map_polylines, map_mask, map_positions, map_types

        cos_h, sin_h = np.cos(-focal_heading), np.sin(-focal_heading)
        rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        polyline_idx = 0

        for lane_segment in static_map.vector_lane_segments.values():
            if polyline_idx >= self.max_map_polylines:
                break

            centerline = lane_segment.polygon_boundary[:, :2]
            center = centerline.mean(axis=0)

            rel_center = rot_matrix @ (center - focal_pos)
            if np.linalg.norm(rel_center) > self.map_range:
                continue

            n_points = min(len(centerline), self.max_points_per_polyline)
            indices = np.linspace(0, len(centerline) - 1, n_points, dtype=int)

            for i, idx in enumerate(indices):
                pt = centerline[idx]
                rel_pt = rot_matrix @ (pt - focal_pos)
                map_polylines[polyline_idx, i] = torch.from_numpy(rel_pt).float()
                map_mask[polyline_idx, i] = True

            map_positions[polyline_idx] = torch.from_numpy(rel_center).float()
            map_types[polyline_idx] = 0

            polyline_idx += 1

        return map_polylines, map_mask, map_positions, map_types

    def _get_future_trajectory(
        self,
        scenario,
        focal_track_id: str,
        current_timestep: int,
        focal_pos: np.ndarray,
        focal_heading: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract future trajectory for the focal agent.

        Returns:
            future_trajectory: (60, 2) - xy positions
            future_yaw: (60,) - yaw angles
        """
        focal_track = None
        for track in scenario.tracks:
            if track.track_id == focal_track_id:
                focal_track = track
                break

        track_states = {s.timestep: s for s in focal_track.object_states}

        cos_h, sin_h = np.cos(-focal_heading), np.sin(-focal_heading)
        rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        future_trajectory = torch.zeros(ARGOVERSE2_PREDICTION_HORIZON, 2)
        future_yaw = torch.zeros(ARGOVERSE2_PREDICTION_HORIZON)

        last_valid_pos = None
        last_valid_yaw = 0.0

        for t_offset in range(ARGOVERSE2_PREDICTION_HORIZON):
            t = current_timestep + t_offset + 1
            if t in track_states:
                state = track_states[t]
                pos = np.array([state.position[0], state.position[1]])
                rel_pos = rot_matrix @ (pos - focal_pos)
                rel_yaw = state.heading - focal_heading

                future_trajectory[t_offset] = torch.from_numpy(rel_pos).float()
                future_yaw[t_offset] = rel_yaw
                last_valid_pos = rel_pos
                last_valid_yaw = rel_yaw
            elif last_valid_pos is not None:
                future_trajectory[t_offset] = torch.from_numpy(last_valid_pos).float()
                future_yaw[t_offset] = last_valid_yaw

        return future_trajectory, future_yaw

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary with:
            - agent_history: (N_agents, T, 6) agent tracks
            - agent_mask: (N_agents, T) valid timesteps
            - agent_positions: (N_agents, 2) current positions
            - agent_types: (N_agents,) type indices
            - map_polylines: (N_polylines, N_points, 2) map features
            - map_mask: (N_polylines, N_points) valid points
            - map_positions: (N_polylines, 2) polyline centers
            - map_types: (N_polylines,) type indices
            - future_trajectory: (60, 2) future xy positions
            - future_yaw: (60,) future yaw angles
            - scenario_id: str identifier
        """
        import random

        scenario_path = self.scenario_paths[idx]
        try:
            scenario = self._load_scenario(scenario_path)
        except FileNotFoundError:
            # Some scenarios may have incomplete downloads - retry with random index
            return self.__getitem__(random.randint(0, len(self) - 1))
        static_map = self._load_map(scenario_path)

        focal_track_id = scenario.focal_track_id
        current_timestep = 49

        focal_track = None
        for track in scenario.tracks:
            if track.track_id == focal_track_id:
                focal_track = track
                break

        focal_state = None
        for state in focal_track.object_states:
            if state.timestep == current_timestep:
                focal_state = state
                break

        focal_pos = np.array([focal_state.position[0], focal_state.position[1]])
        focal_heading = focal_state.heading

        agent_history, agent_mask, agent_positions, agent_types = self._get_agent_features(
            scenario, focal_track_id, current_timestep
        )

        map_polylines, map_mask, map_positions, map_types = self._get_map_features(
            static_map, focal_pos, focal_heading
        )

        future_trajectory, future_yaw = self._get_future_trajectory(
            scenario, focal_track_id, current_timestep, focal_pos, focal_heading
        )

        return {
            "agent_history": agent_history,
            "agent_mask": agent_mask,
            "agent_positions": agent_positions,
            "agent_types": agent_types,
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "map_positions": map_positions,
            "map_types": map_types,
            "future_trajectory": future_trajectory,
            "future_yaw": future_yaw,
            "scenario_id": scenario.scenario_id,
        }


class Argoverse2MotionDatasetDummy(BenchmarkDataset):
    """Dummy Argoverse 2 dataset for testing without actual data."""

    def __init__(
        self,
        num_samples: int = 100,
        max_agents: int = 64,
        max_map_polylines: int = 256,
        max_points_per_polyline: int = 20,
        history_length: int = ARGOVERSE2_HISTORY_LENGTH,
    ):
        self.num_samples = num_samples
        self.max_agents = max_agents
        self.max_map_polylines = max_map_polylines
        self.max_points_per_polyline = max_points_per_polyline
        self.history_length = history_length

    def __len__(self) -> int:
        return self.num_samples

    @property
    def input_type(self) -> str:
        return "vector"

    @property
    def num_modes(self) -> int:
        return ARGOVERSE2_NUM_MODES

    @property
    def prediction_horizon(self) -> int:
        return ARGOVERSE2_PREDICTION_HORIZON

    @property
    def prediction_hz(self) -> float:
        return ARGOVERSE2_PREDICTION_HZ

    def __getitem__(self, idx: int) -> dict[str, Any]:
        num_agents = min(10 + idx % 20, self.max_agents)
        num_polylines = min(50 + idx % 100, self.max_map_polylines)

        agent_history = torch.randn(self.max_agents, self.history_length, 6) * 10
        agent_mask = torch.zeros(self.max_agents, self.history_length, dtype=torch.bool)
        agent_mask[:num_agents, :] = True

        agent_positions = torch.randn(self.max_agents, 2) * 20
        agent_types = torch.randint(0, 6, (self.max_agents,))

        map_polylines = torch.randn(self.max_map_polylines, self.max_points_per_polyline, 2) * 30
        map_mask = torch.zeros(
            self.max_map_polylines, self.max_points_per_polyline, dtype=torch.bool
        )
        for i in range(num_polylines):
            n_pts = 5 + i % 15
            map_mask[i, :n_pts] = True

        map_positions = torch.randn(self.max_map_polylines, 2) * 30
        map_types = torch.randint(0, 4, (self.max_map_polylines,))

        t = torch.linspace(0, 6.0, ARGOVERSE2_PREDICTION_HORIZON)
        speed = 5.0 + torch.randn(1).item() * 2.0
        turn_rate = torch.randn(1).item() * 0.1
        future_trajectory = torch.stack(
            [
                speed * t,
                0.5 * turn_rate * t**2,
            ],
            dim=-1,
        )
        future_yaw = turn_rate * t

        return {
            "agent_history": agent_history,
            "agent_mask": agent_mask,
            "agent_positions": agent_positions,
            "agent_types": agent_types,
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "map_positions": map_positions,
            "map_types": map_types,
            "future_trajectory": future_trajectory,
            "future_yaw": future_yaw,
            "scenario_id": f"scenario_{idx}",
        }
