"""Reasoning Student: Trajectory + contrastive reasoning supervision.

Architecture:
- Same as BaselineStudent
- Additional reasoning head that predicts teacher's CoC embedding
- Reasoning head discarded at inference (zero overhead)
- ~485M parameters
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..calibration import CameraCalibration
from .baseline_student import BaselineStudentConfig, _get_raster_encoder, _get_vector_encoder
from .config import ModelConfig, get_model_config
from .decoder import EgoHistoryEncoder, ReasoningHead, TrajectoryDecoder
from .encoder import BEVEncoder, BEVEncoderDummy


@dataclass
class ReasoningStudentConfig(BaselineStudentConfig):
    """Configuration for ReasoningStudent model."""

    # Reasoning head
    reasoning_hidden_dim: int = 1024
    reasoning_output_dim: int = 384  # Match sentence-transformers embedding

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, **overrides) -> "ReasoningStudentConfig":
        """Create ReasoningStudentConfig from a ModelConfig.

        Args:
            model_config: ModelConfig with architecture parameters
            **overrides: Additional overrides (e.g., use_unicycle, use_ego_history)

        Returns:
            ReasoningStudentConfig instance
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
            reasoning_hidden_dim=model_config.reasoning_hidden_dim,
            reasoning_output_dim=model_config.reasoning_output_dim,
            **overrides,
        )


class ReasoningStudent(nn.Module):
    """Reasoning student model with trajectory + reasoning supervision.

    Same architecture as BaselineStudent with an additional reasoning head
    that predicts the teacher's chain-of-thought embedding.
    """

    def __init__(
        self,
        config: ReasoningStudentConfig,
        calibrations: list[CameraCalibration] | None = None,
    ):
        """Initialize reasoning student.

        Args:
            config: Model configuration
            calibrations: Camera calibrations for BEV projection.
                         If None, uses dummy encoder (for testing).
        """
        super().__init__()

        self.config = config

        # Create encoder based on input type (same as baseline)
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

        # Reasoning head (additional for reasoning student)
        self.reasoning_head = ReasoningHead(
            input_dim=config.transformer_dim,
            hidden_dim=config.reasoning_hidden_dim,
            output_dim=config.reasoning_output_dim,
        )

    def forward(
        self,
        images: torch.Tensor | dict[str, torch.Tensor] | None = None,
        ego_history: torch.Tensor | None = None,
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        return_reasoning: bool = True,
        return_actions: bool = False,
        encoder_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images (B, num_cameras, 12, H, W) for camera encoder
            ego_history: Ego history (B, N, 6) for ego encoder (required if use_ego_history=True)
            ego_history_xyz: Raw positions (B, N, 3) for unicycle integration (required if use_unicycle=True)
            ego_history_rot: Raw rotations (B, N, 3, 3) for unicycle integration (required if use_unicycle=True)
            return_reasoning: Whether to compute reasoning embedding.
                             Set to False at inference for efficiency.
            return_actions: If True, also return raw (accel, curvature) actions (only for unicycle mode)
            encoder_inputs: Dict inputs for vector/raster encoders (alternative to images)

        Returns:
            If use_unicycle=False (direct_trajectory mode):
                Tuple of (trajectory, reasoning_embedding) or (trajectory, reasoning_embedding, None)
            If use_unicycle=True and return_actions=False:
                Tuple of (trajectory, reasoning_embedding)
            If use_unicycle=True and return_actions=True:
                Tuple of (trajectory, reasoning_embedding, actions)
        """
        # Encode to BEV based on input type
        if self.config.input_type in ("vector", "raster"):
            inputs = encoder_inputs if encoder_inputs is not None else images
            if not isinstance(inputs, dict):
                raise ValueError(f"{self.config.input_type} encoder requires dict input")
            bev = self.encoder(inputs)
        else:
            bev = self.encoder(images)

        # Encode ego history for velocity context (if enabled)
        ego_embedding = None
        if self.ego_history_encoder is not None and ego_history is not None:
            ego_embedding = self.ego_history_encoder(ego_history)

        # Direct trajectory mode: direct trajectory output (no unicycle)
        if not self.config.use_unicycle:
            if return_reasoning:
                decoder_output = self.decoder.get_decoder_output(bev, ego_embedding=ego_embedding)
                trajectory = self.decoder.trajectory_head(decoder_output)
                reasoning_emb = self.reasoning_head(decoder_output)
                if return_actions:
                    return trajectory, reasoning_emb, None
                return trajectory, reasoning_emb
            else:
                trajectory = self.decoder(bev, ego_embedding=ego_embedding)
                if return_actions:
                    return trajectory, None, None
                return trajectory, None

        # Unicycle mode: unicycle integration
        if return_reasoning:
            # Get decoder output for both trajectory and reasoning
            decoder_output = self.decoder.get_decoder_output(bev, ego_embedding=ego_embedding)

            # Get (accel, curvature) actions
            actions = self.decoder.trajectory_head(decoder_output)

            # Integrate via unicycle kinematic model
            xyz, rot = self.unicycle(actions, ego_history_xyz, ego_history_rot)
            trajectory = torch.cat(
                [
                    xyz[..., :2],
                    rot[..., 1, 0:1],
                    rot[..., 0, 0:1],
                ],
                dim=-1,
            )

            # Compute reasoning embedding
            reasoning_emb = self.reasoning_head(decoder_output)

            if return_actions:
                return trajectory, reasoning_emb, actions
            return trajectory, reasoning_emb
        else:
            # Inference mode: just compute trajectory
            actions = self.decoder(bev, ego_embedding=ego_embedding)
            xyz, rot = self.unicycle(actions, ego_history_xyz, ego_history_rot)
            trajectory = torch.cat(
                [
                    xyz[..., :2],
                    rot[..., 1, 0:1],
                    rot[..., 0, 0:1],
                ],
                dim=-1,
            )
            if return_actions:
                return trajectory, None, actions
            return trajectory, None

    def get_trajectory_xy(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Extract (x, y) positions from trajectory output."""
        return trajectory[:, :, :2]

    def get_yaw(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Extract yaw angles from trajectory output."""
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
        counts["reasoning_head"] = sum(p.numel() for p in self.reasoning_head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["inference_total"] = counts["total"] - counts["reasoning_head"]

        return counts

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig | str,
        calibrations: list[CameraCalibration] | None = None,
        pretrained_backbone: bool = True,
    ) -> "ReasoningStudent":
        """Create ReasoningStudent from a ModelConfig.

        Args:
            model_config: ModelConfig instance or name ('500m')
            calibrations: Camera calibrations (required for geometric BEV)
            pretrained_backbone: Use ImageNet pretrained weights

        Returns:
            ReasoningStudent model
        """
        if isinstance(model_config, str):
            model_config = get_model_config(model_config)

        config = ReasoningStudentConfig.from_model_config(
            model_config,
            pretrained_backbone=pretrained_backbone,
        )
        return cls(config=config, calibrations=calibrations)


def create_reasoning_student(
    calibrations: list[CameraCalibration] | None = None,
    pretrained_backbone: bool = True,
    model_size: str | None = None,
    **config_overrides,
) -> ReasoningStudent:
    """Factory function to create ReasoningStudent.

    Args:
        calibrations: Camera calibrations (required for geometric BEV)
        pretrained_backbone: Use ImageNet pretrained weights
        model_size: Model size preset ('500m'). If provided, uses preset config.
        **config_overrides: Override default config values

    Returns:
        ReasoningStudent model
    """
    if model_size is not None:
        return ReasoningStudent.from_model_config(
            model_size,
            calibrations=calibrations,
            pretrained_backbone=pretrained_backbone,
        )

    config = ReasoningStudentConfig(
        pretrained_backbone=pretrained_backbone,
        **config_overrides,
    )
    return ReasoningStudent(config=config, calibrations=calibrations)
