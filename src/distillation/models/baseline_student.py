"""Baseline Student: Trajectory-only supervision.

Architecture:
- ResNet-50/101 backbone with 12-channel input
- Ground-plane BEV projection
- DETR-style transformer decoder
- Outputs 64 waypoints × (x, y, sin_yaw, cos_yaw)
- ~485M parameters
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..calibration import CameraCalibration
from .config import ModelConfig, get_model_config
from .decoder import EgoHistoryEncoder, TrajectoryDecoder
from .encoder import BEVEncoder, BEVEncoderDummy

# Lazy imports for optional encoders
def _get_vector_encoder():
    from ..encoders.vector_encoder import VectorEncoder, VectorEncoderConfig
    return VectorEncoder, VectorEncoderConfig

def _get_raster_encoder():
    from ..encoders.raster_encoder import RasterEncoder, RasterEncoderConfig
    return RasterEncoder, RasterEncoderConfig


@dataclass
class BaselineStudentConfig:
    """Configuration for BaselineStudent model."""

    # Input type: "camera" (default), "vector" (AV2), "raster" (nuScenes)
    input_type: str = "camera"

    # Input (camera mode)
    num_cameras: int = 4
    num_frames: int = 4
    input_resolution: tuple[int, int] = (224, 224)

    # Encoder
    backbone: str = "resnet50"  # "resnet50", "resnet101", "resnet152"
    bev_size: tuple[int, int] = (200, 200)
    bev_resolution: float = 0.5  # meters per cell
    bev_channels: int = 256
    pretrained_backbone: bool = True

    # Decoder
    transformer_dim: int = 768
    transformer_heads: int = 12
    transformer_layers: int = 8
    transformer_ffn: int = 3072
    num_waypoints: int = 64
    output_dim: int = 4  # (x, y, sin_yaw, cos_yaw)
    dropout: float = 0.1
    gradient_checkpointing: bool = False

    # Ego history (for velocity context, used with unicycle mode)
    num_ego_history_steps: int = 16
    ego_history_dim: int = 6  # xyz (3) + rotation as euler/axis-angle (3)

    # Training mode flags
    use_unicycle: bool = True  # If False, output direct trajectory (direct_trajectory mode)
    use_ego_history: bool = True  # Always True (required for velocity context)

    # Unicycle kinematic model (only used when use_unicycle=True)
    unicycle_dt: float = 0.1
    accel_mean: float = 0.0
    accel_std: float = 1.0
    curvature_mean: float = 0.0
    curvature_std: float = 1.0
    accel_bounds: tuple[float, float] = (-9.8, 9.8)
    curvature_bounds: tuple[float, float] = (-0.2, 0.2)

    # Discrete actions (optional, for stop-gradient training)
    use_discrete_actions: bool = False
    num_action_bins: int = 256

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, **overrides) -> "BaselineStudentConfig":
        """Create BaselineStudentConfig from a ModelConfig.

        Args:
            model_config: ModelConfig with architecture parameters
            **overrides: Additional overrides (e.g., use_unicycle, use_ego_history)

        Returns:
            BaselineStudentConfig instance
        """
        return cls(
            num_cameras=model_config.num_cameras,
            num_frames=model_config.num_frames,
            input_resolution=model_config.input_resolution,
            backbone=model_config.backbone,
            bev_size=model_config.bev_size,
            bev_resolution=model_config.bev_resolution,
            bev_channels=model_config.bev_channels,
            transformer_dim=model_config.d_model,
            transformer_heads=model_config.n_heads,
            transformer_layers=model_config.n_layers,
            transformer_ffn=model_config.ffn_dim,
            num_waypoints=model_config.num_waypoints,
            output_dim=model_config.output_dim,
            dropout=model_config.dropout,
            gradient_checkpointing=model_config.gradient_checkpointing,
            num_ego_history_steps=model_config.num_ego_history_steps,
            ego_history_dim=model_config.ego_history_dim,
            use_unicycle=model_config.use_unicycle,
            use_ego_history=model_config.use_ego_history,
            unicycle_dt=model_config.unicycle_dt,
            accel_mean=model_config.accel_mean,
            accel_std=model_config.accel_std,
            curvature_mean=model_config.curvature_mean,
            curvature_std=model_config.curvature_std,
            accel_bounds=model_config.accel_bounds,
            curvature_bounds=model_config.curvature_bounds,
            **overrides,
        )


class BaselineStudent(nn.Module):
    """Baseline student model with trajectory-only supervision.

    Takes multi-camera, multi-frame images and outputs future trajectory.
    """

    def __init__(
        self,
        config: BaselineStudentConfig,
        calibrations: list[CameraCalibration] | None = None,
    ):
        """Initialize baseline student.

        Args:
            config: Model configuration
            calibrations: Camera calibrations for BEV projection.
                         If None, uses dummy encoder (for testing).
        """
        super().__init__()

        self.config = config

        # Create encoder based on input type
        if config.input_type == "vector":
            VectorEncoder, VectorEncoderConfig = _get_vector_encoder()
            self.encoder = VectorEncoder(VectorEncoderConfig(
                output_channels=config.bev_channels,
                output_size=config.bev_size,  # Already a tuple (H, W)
            ))
        elif config.input_type == "raster":
            RasterEncoder, RasterEncoderConfig = _get_raster_encoder()
            self.encoder = RasterEncoder(RasterEncoderConfig(
                output_channels=config.bev_channels,
                output_size=config.bev_size,  # Already a tuple (H, W)
                pretrained_backbone=config.pretrained_backbone,
            ))
        elif calibrations is not None:
            # Camera encoder with geometric BEV projection
            self.encoder = BEVEncoder(
                calibrations=calibrations,
                input_resolution=config.input_resolution,
                bev_size=config.bev_size,
                bev_resolution=config.bev_resolution,
                bev_channels=config.bev_channels,
                pretrained_backbone=config.pretrained_backbone,
                backbone_type=config.backbone,
            )
        else:
            # Dummy camera encoder for testing without calibration
            self.encoder = BEVEncoderDummy(
                num_cameras=config.num_cameras,
                input_resolution=config.input_resolution,
                bev_size=config.bev_size,
                bev_channels=config.bev_channels,
                pretrained_backbone=config.pretrained_backbone,
                backbone_type=config.backbone,
            )

        # Ego history encoder (only when use_ego_history=True)
        self.ego_history_encoder = None
        if config.use_ego_history:
            self.ego_history_encoder = EgoHistoryEncoder(
                num_history_steps=config.num_ego_history_steps,
                input_dim=config.ego_history_dim,
                hidden_dim=256,
                output_dim=config.transformer_dim,
            )

        # Decoder output dimension depends on mode
        # unicycle mode: output (accel, curvature) → 2 dims
        # direct mode: output (x, y, sin_yaw, cos_yaw) → 4 dims
        decoder_output_dim = 2 if config.use_unicycle else 4

        self.decoder = TrajectoryDecoder(
            bev_channels=config.bev_channels,
            bev_size=config.bev_size,
            transformer_dim=config.transformer_dim,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
            transformer_ffn=config.transformer_ffn,
            num_waypoints=config.num_waypoints,
            output_dim=decoder_output_dim,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
            use_discrete_actions=config.use_discrete_actions if config.use_unicycle else False,
            num_action_bins=config.num_action_bins,
            accel_bounds=config.accel_bounds,
            curv_bounds=config.curvature_bounds,
            accel_mean=config.accel_mean,
            accel_std=config.accel_std,
            curv_mean=config.curvature_mean,
            curv_std=config.curvature_std,
        )

        # Unicycle integrator (only when use_unicycle=True)
        self.unicycle = None
        if config.use_unicycle:
            from .unicycle import UnicycleIntegrator

            self.unicycle = UnicycleIntegrator(
                dt=config.unicycle_dt,
                n_waypoints=config.num_waypoints,
                accel_mean=config.accel_mean,
                accel_std=config.accel_std,
                curvature_mean=config.curvature_mean,
                curvature_std=config.curvature_std,
                accel_bounds=config.accel_bounds,
                curvature_bounds=config.curvature_bounds,
            )

    def forward(
        self,
        images: torch.Tensor | dict[str, torch.Tensor] | None = None,
        ego_history: torch.Tensor | None = None,
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        return_actions: bool = False,
        return_logits: bool = False,
        # For vector/raster encoders, pass inputs as dict
        encoder_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images (B, num_cameras, 12, H, W) for camera encoder
                   where 12 = 4 frames × 3 RGB channels. Can also pass dict for vector/raster.
            ego_history: Ego history (B, N, 6) for ego encoder (required if use_ego_history=True)
            ego_history_xyz: Raw positions (B, N, 3) for unicycle integration (required if use_unicycle=True)
            ego_history_rot: Raw rotations (B, N, 3, 3) for unicycle integration (required if use_unicycle=True)
            return_actions: If True, also return raw (accel, curvature) actions (only for unicycle mode)
            return_logits: If True and use_discrete_actions, return (trajectory, logits)
            encoder_inputs: Dict inputs for vector/raster encoders (alternative to images)

        Returns:
            If use_unicycle=False (direct_trajectory mode):
                Trajectory (B, num_waypoints, 4) with (x, y, sin_yaw, cos_yaw)
            If use_unicycle=True and use_discrete_actions=True and return_logits=True:
                Tuple of (trajectory, logits) where logits is (B, 64, 2, num_bins)
            If use_unicycle=True and return_actions=False:
                Trajectory (B, num_waypoints, 4) with (x, y, sin_yaw, cos_yaw)
            If use_unicycle=True and return_actions=True:
                Tuple of (trajectory, actions) where actions is (B, 64, 2)
        """
        # Encode to BEV based on input type
        if self.config.input_type in ("vector", "raster"):
            # Vector/raster encoders expect dict input
            inputs = encoder_inputs if encoder_inputs is not None else images
            if not isinstance(inputs, dict):
                raise ValueError(f"{self.config.input_type} encoder requires dict input")
            bev = self.encoder(inputs)
        else:
            # Camera encoder expects tensor input
            bev = self.encoder(images)

        # Encode ego history for velocity context (if enabled)
        ego_embedding = None
        if self.ego_history_encoder is not None and ego_history is not None:
            ego_embedding = self.ego_history_encoder(ego_history)

        # Direct trajectory mode: output trajectory directly (no unicycle)
        if not self.config.use_unicycle:
            trajectory = self.decoder(bev, ego_embedding=ego_embedding)
            return trajectory

        # Unicycle mode: integrate actions through unicycle dynamics
        # Discrete action mode with stop-gradient integration
        if self.config.use_discrete_actions:
            actions, logits = self.decoder(bev, ego_embedding=ego_embedding, return_logits=True)

            # Stop gradient - integration for monitoring only
            with torch.no_grad():
                xyz, rot = self.unicycle(actions, ego_history_xyz, ego_history_rot)

            # Explicit detach for clarity
            trajectory = torch.cat(
                [
                    xyz[..., :2],
                    rot[..., 1, 0:1],
                    rot[..., 0, 0:1],
                ],
                dim=-1,
            ).detach()

            if return_logits:
                return trajectory, logits
            return trajectory

        # Continuous action mode
        actions = self.decoder(bev, ego_embedding=ego_embedding)

        # Integrate via unicycle kinematic model
        xyz, rot = self.unicycle(actions, ego_history_xyz, ego_history_rot)

        # Convert to (x, y, sin_yaw, cos_yaw) format
        trajectory = torch.cat(
            [
                xyz[..., :2],
                rot[..., 1, 0:1],  # sin_yaw = R[1,0]
                rot[..., 0, 0:1],  # cos_yaw = R[0,0]
            ],
            dim=-1,
        )

        if return_actions:
            return trajectory, actions
        return trajectory

    def get_trajectory_xy(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Extract (x, y) positions from trajectory output.

        Args:
            trajectory: Model output (B, num_waypoints, 4)

        Returns:
            Positions (B, num_waypoints, 2)
        """
        return trajectory[:, :, :2]

    def get_yaw(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Extract yaw angles from trajectory output.

        Args:
            trajectory: Model output (B, num_waypoints, 4)

        Returns:
            Yaw angles in radians (B, num_waypoints)
        """
        sin_yaw = trajectory[:, :, 2]
        cos_yaw = trajectory[:, :, 3]
        return torch.atan2(sin_yaw, cos_yaw)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        if self.ego_history_encoder is not None:
            counts["ego_history_encoder"] = sum(p.numel() for p in self.ego_history_encoder.parameters())

        decoder = self.decoder
        counts["bev_downsample"] = sum(p.numel() for p in decoder.bev_downsample.parameters())
        counts["waypoint_queries"] = decoder.waypoint_queries.weight.numel()
        counts["temporal_pos_encoding"] = decoder.temporal_pos_encoding.weight.numel()
        counts["transformer_layers"] = sum(p.numel() for p in decoder.transformer_layers.parameters())
        counts["transformer_norm"] = sum(p.numel() for p in decoder.transformer_norm.parameters())
        counts["trajectory_head"] = sum(p.numel() for p in decoder.trajectory_head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig | str,
        calibrations: list[CameraCalibration] | None = None,
        pretrained_backbone: bool = True,
    ) -> "BaselineStudent":
        """Create BaselineStudent from a ModelConfig.

        Args:
            model_config: ModelConfig instance or name ('500m')
            calibrations: Camera calibrations (required for geometric BEV)
            pretrained_backbone: Use ImageNet pretrained weights

        Returns:
            BaselineStudent model
        """
        if isinstance(model_config, str):
            model_config = get_model_config(model_config)

        config = BaselineStudentConfig.from_model_config(
            model_config,
            pretrained_backbone=pretrained_backbone,
        )
        return cls(config=config, calibrations=calibrations)


def create_baseline_student(
    calibrations: list[CameraCalibration] | None = None,
    pretrained_backbone: bool = True,
    model_size: str | None = None,
    **config_overrides,
) -> BaselineStudent:
    """Factory function to create BaselineStudent.

    Args:
        calibrations: Camera calibrations (required for geometric BEV)
        pretrained_backbone: Use ImageNet pretrained weights
        model_size: Model size preset ('500m'). If provided, uses preset config.
        **config_overrides: Override default config values

    Returns:
        BaselineStudent model
    """
    if model_size is not None:
        return BaselineStudent.from_model_config(
            model_size,
            calibrations=calibrations,
            pretrained_backbone=pretrained_backbone,
        )

    config = BaselineStudentConfig(
        pretrained_backbone=pretrained_backbone,
        **config_overrides,
    )
    return BaselineStudent(config=config, calibrations=calibrations)
