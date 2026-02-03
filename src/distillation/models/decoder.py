"""DETR-style transformer decoder for trajectory prediction.

Architecture:
- BEV downsampler: 200×200 -> 40×40 with stride-5 conv
- 2D sinusoidal positional encoding for BEV features
- Learnable waypoint queries with temporal positional encoding
- Ego history encoder for velocity/motion context
- Configurable transformer decoder (16 layers for 500m model)
- Action head outputting (acceleration, curvature) for unicycle integration
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def positional_encoding_2d(height: int, width: int, dim: int, device: torch.device = None) -> torch.Tensor:
    """Generate 2D sinusoidal positional encoding.

    Args:
        height: Grid height (e.g., 40 for downsampled BEV)
        width: Grid width (e.g., 40)
        dim: Embedding dimension (e.g., 768), must be divisible by 4
        device: Device to create tensor on

    Returns:
        (1, height*width, dim) positional encoding
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D pos encoding, got {dim}")

    y_pos = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    x_pos = torch.arange(width, device=device).unsqueeze(0).expand(height, width)

    y_pos = y_pos.float() / height
    x_pos = x_pos.float() / width

    dim_quarter = dim // 4
    omega = torch.arange(dim_quarter, device=device).float() / dim_quarter
    omega = 1.0 / (10000**omega)

    y_emb = y_pos.flatten().unsqueeze(1) * omega.unsqueeze(0)
    x_emb = x_pos.flatten().unsqueeze(1) * omega.unsqueeze(0)

    pos_emb = torch.cat([torch.sin(y_emb), torch.cos(y_emb), torch.sin(x_emb), torch.cos(x_emb)], dim=-1)

    return pos_emb.unsqueeze(0)


class EgoHistoryEncoder(nn.Module):
    """Encodes ego history (past trajectory) into a context embedding.

    This provides the decoder with velocity/motion information that matches
    what the teacher model (Alpamayo) receives as ego_history input.
    """

    def __init__(
        self,
        num_history_steps: int = 16,
        input_dim: int = 6,  # xyz + rotation (simplified to euler or 3 values)
        hidden_dim: int = 256,
        output_dim: int = 768,  # Match transformer_dim
    ):
        """Initialize ego history encoder.

        Args:
            num_history_steps: Number of past timesteps (16 steps at 10Hz = 1.6s)
            input_dim: Dimension per timestep (xyz=3, or xyz+rot=6)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (should match transformer_dim)
        """
        super().__init__()

        self.num_history_steps = num_history_steps
        self.input_dim = input_dim

        # Flatten history and encode
        self.encoder = nn.Sequential(
            nn.Linear(num_history_steps * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ego_history: torch.Tensor) -> torch.Tensor:
        """Encode ego history.

        Args:
            ego_history: Past trajectory (B, num_history_steps, input_dim)
                        Typically (B, 16, 6) for 16 steps of xyz + rotation

        Returns:
            Ego embedding (B, output_dim)
        """
        B = ego_history.shape[0]
        # Flatten history
        flat = ego_history.reshape(B, -1)  # (B, num_history_steps * input_dim)
        return self.encoder(flat)  # (B, output_dim)


class TrajectoryDecoder(nn.Module):
    """DETR-style transformer decoder for trajectory prediction.

    Takes BEV features and outputs trajectory waypoints.
    """

    def __init__(
        self,
        bev_channels: int = 256,
        bev_size: tuple[int, int] = (200, 200),
        transformer_dim: int = 768,
        transformer_heads: int = 12,
        transformer_layers: int = 8,
        transformer_ffn: int = 3072,
        num_waypoints: int = 64,
        output_dim: int = 4,  # (x, y, sin_yaw, cos_yaw)
        dropout: float = 0.15,
        gradient_checkpointing: bool = False,
        use_discrete_actions: bool = False,
        num_action_bins: int = 256,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curv_bounds: tuple[float, float] = (-0.2, 0.2),
        accel_mean: float = -0.055,
        accel_std: float = 0.989,
        curv_mean: float = 0.0,
        curv_std: float = 0.022,
    ):
        """Initialize trajectory decoder.

        Args:
            bev_channels: Input BEV feature channels
            bev_size: (H, W) of input BEV grid
            transformer_dim: Transformer hidden dimension
            transformer_heads: Number of attention heads
            transformer_layers: Number of transformer layers
            transformer_ffn: Feed-forward network dimension
            num_waypoints: Number of output waypoints (64 for 6.4s @ 10Hz)
            output_dim: Output dimension per waypoint
            dropout: Dropout probability
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()

        self.bev_size = bev_size
        self.num_waypoints = num_waypoints
        self.transformer_dim = transformer_dim
        self.gradient_checkpointing = gradient_checkpointing

        # BEV downsampler: 200×200 -> 40×40
        # Stride 5 conv to reduce sequence length from 40000 to 1600
        self.bev_downsample = nn.Sequential(
            nn.Conv2d(bev_channels, transformer_dim, kernel_size=5, stride=5), nn.BatchNorm2d(transformer_dim), nn.ReLU()
        )

        # Downsampled BEV size
        self.down_h = bev_size[0] // 5
        self.down_w = bev_size[1] // 5

        # 2D positional encoding for BEV (computed once, cached)
        self.register_buffer("bev_pos_encoding", positional_encoding_2d(self.down_h, self.down_w, transformer_dim))

        # Learnable waypoint queries
        self.waypoint_queries = nn.Embedding(num_waypoints, transformer_dim)

        # Learnable temporal positional encoding for waypoints
        self.temporal_pos_encoding = nn.Embedding(num_waypoints, transformer_dim)

        # Initialize queries and temporal encodings
        nn.init.xavier_uniform_(self.waypoint_queries.weight)
        nn.init.xavier_uniform_(self.temporal_pos_encoding.weight)

        # Transformer decoder layers (stored individually for gradient checkpointing)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=False,  # (seq, batch, dim) format
                norm_first=True,  # Pre-LN for better training stability
            )
            for _ in range(transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(transformer_dim)

        # Trajectory output head (continuous actions)
        self.trajectory_head = nn.Sequential(
            nn.Linear(transformer_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, output_dim)
        )

        # Discrete action head (optional)
        self.use_discrete_actions = use_discrete_actions
        self.discrete_head = None
        if use_discrete_actions:
            self.discrete_head = DiscreteActionHead(
                input_dim=transformer_dim,
                num_bins=num_action_bins,
                dropout=dropout,
                accel_bounds=accel_bounds,
                curv_bounds=curv_bounds,
                accel_mean=accel_mean,
                accel_std=accel_std,
                curv_mean=curv_mean,
                curv_std=curv_std,
            )

    def forward(
        self,
        bev: torch.Tensor,
        ego_embedding: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode BEV features to trajectory or actions.

        Args:
            bev: BEV features (B, C, H, W) where H=W=200
            ego_embedding: Optional ego history embedding (B, transformer_dim)
                          If provided, adds motion context to queries
            return_logits: If True and use_discrete_actions, return (actions, logits)

        Returns:
            If use_discrete_actions=False:
                Trajectory/actions (B, num_waypoints, output_dim)
            If use_discrete_actions=True and return_logits=False:
                Normalized actions (B, 64, 2) for unicycle
            If use_discrete_actions=True and return_logits=True:
                Tuple of (normalized_actions, logits) where logits is (B, 64, 2, num_bins)
        """
        B = bev.shape[0]

        # Downsample BEV
        bev_down = self.bev_downsample(bev)  # (B, transformer_dim, 40, 40)

        # Flatten to sequence: (B, dim, H, W) -> (H*W, B, dim)
        bev_seq = bev_down.flatten(2).permute(2, 0, 1)  # (1600, B, 768)

        # Add 2D positional encoding
        bev_seq = bev_seq + self.bev_pos_encoding.permute(1, 0, 2)  # (1600, B, 768)

        # Prepare waypoint queries with temporal positional encoding
        queries = self.waypoint_queries.weight + self.temporal_pos_encoding.weight  # (64, 768)
        queries = queries.unsqueeze(1).expand(-1, B, -1)  # (64, B, 768)

        # Add ego history embedding to queries if provided
        # This injects velocity/motion context into all waypoint predictions
        if ego_embedding is not None:
            ego_embedding = ego_embedding.unsqueeze(0)  # (1, B, 768)
            queries = queries + ego_embedding  # Broadcast to all 64 queries

        # Run transformer decoder layers with optional gradient checkpointing
        output = queries
        for layer in self.transformer_layers:
            if self.gradient_checkpointing and self.training:
                output = checkpoint(
                    layer,
                    output,
                    bev_seq,
                    use_reentrant=False,
                )
            else:
                output = layer(output, bev_seq)

        output = self.transformer_norm(output)  # (64, B, 768)
        output = output.permute(1, 0, 2)  # (B, 64, 768)

        # Discrete action mode
        if self.use_discrete_actions and self.discrete_head is not None:
            logits = self.discrete_head(output)  # (B, 64, 2, num_bins)
            actions_normalized = self.discrete_head.logits_to_normalized_actions(logits)

            if return_logits:
                return actions_normalized, logits
            return actions_normalized

        # Continuous mode (original behavior)
        trajectory = self.trajectory_head(output)  # (B, 64, output_dim)
        if return_logits:
            return trajectory, None
        return trajectory

    def get_decoder_output(
        self, bev: torch.Tensor, ego_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get raw decoder output (before trajectory head).

        Used by ReasoningStudent to compute reasoning embedding.

        Args:
            bev: BEV features (B, C, H, W)
            ego_embedding: Optional ego history embedding (B, transformer_dim)

        Returns:
            Decoder output (B, num_waypoints, transformer_dim)
        """
        B = bev.shape[0]

        # Downsample BEV
        bev_down = self.bev_downsample(bev)
        bev_seq = bev_down.flatten(2).permute(2, 0, 1)
        bev_seq = bev_seq + self.bev_pos_encoding.permute(1, 0, 2)

        # Prepare queries
        queries = self.waypoint_queries.weight + self.temporal_pos_encoding.weight
        queries = queries.unsqueeze(1).expand(-1, B, -1)

        # Add ego history embedding if provided
        if ego_embedding is not None:
            ego_embedding = ego_embedding.unsqueeze(0)
            queries = queries + ego_embedding

        # Run transformer layers with optional gradient checkpointing
        output = queries
        for layer in self.transformer_layers:
            if self.gradient_checkpointing and self.training:
                output = checkpoint(
                    layer,
                    output,
                    bev_seq,
                    use_reentrant=False,
                )
            else:
                output = layer(output, bev_seq)

        output = self.transformer_norm(output)

        return output.permute(1, 0, 2)  # (B, 64, 768)


class DiscreteActionHead(nn.Module):
    """Predicts discrete action bins with centralized normalization.

    Outputs logits over action bins for cross-entropy training.
    Handles all conversions between bins, raw actions, and normalized actions.
    """

    def __init__(
        self,
        input_dim: int,
        num_bins: int = 256,
        dropout: float = 0.1,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curv_bounds: tuple[float, float] = (-0.2, 0.2),
        accel_mean: float = -0.055,
        accel_std: float = 0.989,
        curv_mean: float = 0.0,
        curv_std: float = 0.022,
    ):
        super().__init__()
        self.num_bins = num_bins

        # Store normalization params as buffers (move with model)
        self.register_buffer("accel_bounds", torch.tensor(accel_bounds))
        self.register_buffer("curv_bounds", torch.tensor(curv_bounds))
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curv_mean", torch.tensor(curv_mean))
        self.register_buffer("curv_std", torch.tensor(curv_std))

        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 64, 2, num_bins)."""
        logits = self.head(x)
        return logits.view(x.shape[0], x.shape[1], 2, self.num_bins)

    def logits_to_normalized_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Full pipeline: logits → bins → raw → normalized for unicycle."""
        bins = logits.argmax(dim=-1)  # (B, 64, 2)
        raw = self._bins_to_raw(bins)
        return self._raw_to_normalized(raw)

    def normalized_actions_to_bins(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """For creating training targets from teacher actions."""
        raw = self._normalized_to_raw(actions_norm)
        return self._raw_to_bins(raw)

    def _bins_to_raw(self, bins: torch.Tensor) -> torch.Tensor:
        """Convert bin indices to raw action values."""
        normalized = bins.float() / (self.num_bins - 1)  # [0, 1]
        accel = normalized[..., 0] * (self.accel_bounds[1] - self.accel_bounds[0]) + self.accel_bounds[0]
        curv = normalized[..., 1] * (self.curv_bounds[1] - self.curv_bounds[0]) + self.curv_bounds[0]
        return torch.stack([accel, curv], dim=-1)

    def _raw_to_bins(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw action values to bin indices."""
        accel_norm = (raw[..., 0] - self.accel_bounds[0]) / (self.accel_bounds[1] - self.accel_bounds[0])
        curv_norm = (raw[..., 1] - self.curv_bounds[0]) / (self.curv_bounds[1] - self.curv_bounds[0])
        accel_bins = (accel_norm * (self.num_bins - 1)).round().long().clamp(0, self.num_bins - 1)
        curv_bins = (curv_norm * (self.num_bins - 1)).round().long().clamp(0, self.num_bins - 1)
        return torch.stack([accel_bins, curv_bins], dim=-1)

    def _raw_to_normalized(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw to normalized (for unicycle input)."""
        accel_norm = (raw[..., 0] - self.accel_mean) / self.accel_std
        curv_norm = (raw[..., 1] - self.curv_mean) / self.curv_std
        return torch.stack([accel_norm, curv_norm], dim=-1)

    def _normalized_to_raw(self, norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized to raw (for bin conversion)."""
        accel_raw = norm[..., 0] * self.accel_std + self.accel_mean
        curv_raw = norm[..., 1] * self.curv_std + self.curv_mean
        return torch.stack([accel_raw, curv_raw], dim=-1)


class ReasoningHead(nn.Module):
    """Attention-pooled reasoning embedding prediction.

    Uses learned attention weights to pool over waypoints instead of simple mean pooling.
    This preserves temporal structure - different waypoints may contribute differently
    to reasoning (e.g., critical decision points vs. straight driving).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 384,
    ):
        super().__init__()

        # Attention for temporal pooling
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
        )

        # Projection head
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Compute reasoning embedding with attention pooling.

        Args:
            decoder_output: Decoder output (B, num_waypoints, dim)

        Returns:
            Reasoning embedding (B, output_dim)
        """
        attn_logits = self.attn(decoder_output)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (decoder_output * attn_weights).sum(dim=1)
        return self.head(pooled)
