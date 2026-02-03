"""Model configuration for student models.

Defines ModelConfig dataclass and the 500M model preset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for student model architecture.

    Defines backbone, transformer dimensions, and training options.
    """

    name: str
    backbone: str  # "resnet50", "resnet101", "resnet152"
    d_model: int  # Transformer hidden dimension
    n_heads: int  # Number of attention heads
    n_layers: int  # Number of transformer decoder layers
    ffn_dim: int  # Feed-forward network dimension
    bev_channels: int  # BEV feature channels

    # Head dimensions
    traj_hidden_dim: int = 512
    reasoning_hidden_dim: int = 1024
    reasoning_output_dim: int = 384  # Match sentence-transformers embedding

    # Memory optimization
    gradient_checkpointing: bool = False

    # Input/output
    num_cameras: int = 4
    num_frames: int = 4
    input_resolution: tuple[int, int] = (224, 224)
    bev_size: tuple[int, int] = (200, 200)
    bev_resolution: float = 0.5  # meters per cell
    num_waypoints: int = 64
    output_dim: int = 4  # (x, y, sin_yaw, cos_yaw)
    dropout: float = 0.15

    # Ego history (for velocity context, used with unicycle mode)
    num_ego_history_steps: int = 16  # 16 past poses (1.6s at 10Hz, matching Alpamayo)
    ego_history_dim: int = 6  # xyz (3) + rotation angles (3)

    # Training mode flags
    use_unicycle: bool = True  # If False, output direct trajectory (direct_trajectory mode)
    use_ego_history: bool = True  # Always True (required for velocity context)

    # Unicycle kinematic model (stats computed from PhysicalAI-AV training set)
    unicycle_dt: float = 0.1
    accel_mean: float = -0.055
    accel_std: float = 0.989
    curvature_mean: float = 0.0
    curvature_std: float = 0.022  # Critical: actual curvature std is ~45x smaller than default
    accel_bounds: tuple[float, float] = (-9.8, 9.8)
    curvature_bounds: tuple[float, float] = (-0.3, 0.3)


# Preset model configurations
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "110m": ModelConfig(
        name="110m",
        backbone="resnet50",
        d_model=768,
        n_heads=12,
        n_layers=8,
        ffn_dim=3072,
        bev_channels=256,
        reasoning_hidden_dim=768,
        gradient_checkpointing=False,
    ),
    "250m": ModelConfig(
        name="250m",
        backbone="resnet50",
        d_model=1024,
        n_heads=16,
        n_layers=12,
        ffn_dim=4096,
        bev_channels=384,
        reasoning_hidden_dim=1024,
        gradient_checkpointing=False,
    ),
    "500m": ModelConfig(
        name="500m",
        backbone="resnet101",
        d_model=1280,
        n_heads=20,
        n_layers=16,
        ffn_dim=5120,
        bev_channels=512,
        gradient_checkpointing=True,
    ),
}


def get_model_config(name: str = "500m") -> ModelConfig:
    """Get model configuration by name.

    Args:
        name: Model size name ('500m')

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If name not in MODEL_CONFIGS
    """
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model config: {name}. Available: {available}")
    return MODEL_CONFIGS[name]
