"""Vector-based encoder for Argoverse 2-style input.

This encoder processes vectorized scene representations:
- Agent tracks: Position, velocity, heading sequences for all agents
- Map polylines: Lane centerlines, boundaries, crosswalks

Used for training/evaluation on Argoverse 2 motion forecasting benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import AbstractEncoder, EncoderConfig


@dataclass
class VectorEncoderConfig(EncoderConfig):
    """Configuration for vector-based encoder."""

    agent_dim: int = 6
    agent_history_len: int = 50
    map_point_dim: int = 2
    max_agents: int = 64
    max_map_polylines: int = 256
    max_points_per_polyline: int = 20
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1


class PolylineEncoder(nn.Module):
    """Encode a set of polylines (agent tracks or map lanes) to features.

    Uses MLP to embed each point, then max-pooling over points to get
    per-polyline features.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.point_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        polylines: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode polylines.

        Args:
            polylines: (B, N_polylines, N_points, input_dim)
            mask: (B, N_polylines, N_points) bool mask, True = valid

        Returns:
            Polyline features (B, N_polylines, d_model)
        """
        B, N, P, D = polylines.shape

        x = polylines.view(B * N, P, D)
        x = self.point_embed(x)

        if mask is not None:
            key_padding_mask = ~mask.view(B * N, P)
            # Identify fully masked polylines (no valid points)
            fully_masked = key_padding_mask.all(dim=-1)
        else:
            key_padding_mask = None
            fully_masked = None

        x_attn, _ = self.temporal_attention(
            x, x, x, key_padding_mask=key_padding_mask
        )

        # Replace NaN from fully-masked polylines with zeros
        if fully_masked is not None and fully_masked.any():
            x_attn = torch.where(
                fully_masked.unsqueeze(-1).unsqueeze(-1).expand_as(x_attn),
                torch.zeros_like(x_attn),
                x_attn,
            )

        x = self.norm(x + x_attn)

        if mask is not None:
            mask_expanded = mask.view(B * N, P, 1).float()
            x = x * mask_expanded
            x = x.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            x = x.max(dim=1)[0]

        x = x.view(B, N, -1)

        return x


class CrossAttentionFusion(nn.Module):
    """Fuse agent and map features via cross-attention."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        agent_features: torch.Tensor,
        map_features: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse agent and map features.

        Args:
            agent_features: (B, N_agents, d_model)
            map_features: (B, N_map, d_model)
            agent_mask: (B, N_agents) bool, True = valid
            map_mask: (B, N_map) bool, True = valid

        Returns:
            Fused features (B, N_agents + N_map, d_model)
        """
        x = torch.cat([agent_features, map_features], dim=1)

        if agent_mask is not None and map_mask is not None:
            mask = torch.cat([agent_mask, map_mask], dim=1)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return x


class VectorToBEVProjection(nn.Module):
    """Project vector features to BEV grid using spatial scatter.

    Agents are placed on the BEV grid based on their positions,
    and map features are averaged over the grid.
    """

    def __init__(
        self,
        d_model: int,
        output_size: tuple[int, int],
        output_channels: int,
        bev_range: tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()

        # Ensure output_size is a tuple of ints (not nested tuples)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple) and len(output_size) == 2:
            self.output_size = (int(output_size[0]), int(output_size[1]))
        else:
            self.output_size = (200, 200)  # fallback default
        self.bev_range = bev_range

        self.project = nn.Sequential(
            nn.Linear(d_model, output_channels),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project features to BEV.

        Args:
            features: (B, N, d_model) token features
            positions: (B, N, 2) xy positions for each token
            mask: (B, N) bool, True = valid

        Returns:
            BEV features (B, output_channels, H, W)
        """
        B, N, D = features.shape
        H, W = self.output_size
        x_min, x_max, y_min, y_max = self.bev_range

        features = self.project(features)
        C = features.shape[-1]

        # Compute grid indices for each token
        x_norm = (positions[..., 0] - x_min) / (x_max - x_min)
        y_norm = (positions[..., 1] - y_min) / (y_max - y_min)
        x_idx = (x_norm * (W - 1)).long().clamp(0, W - 1)
        y_idx = (y_norm * (H - 1)).long().clamp(0, H - 1)

        # Compute flat index: batch * H * W + y * W + x
        batch_idx = torch.arange(B, device=features.device).view(B, 1).expand(B, N)
        flat_idx = batch_idx * (H * W) + y_idx * W + x_idx  # (B, N)

        # Apply mask - set invalid tokens to index 0 (will be zeroed out)
        if mask is not None:
            valid = mask.to(features.dtype).unsqueeze(-1)  # (B, N, 1)
            features_masked = features * valid
            ones_masked = valid
        else:
            features_masked = features
            ones_masked = torch.ones(B, N, 1, device=features.device, dtype=features.dtype)

        # Flatten for scatter_add
        flat_idx_expanded = flat_idx.unsqueeze(-1).expand(-1, -1, C)  # (B, N, C)
        features_flat = features_masked.view(B * N, C)
        flat_idx_flat = flat_idx.view(B * N).unsqueeze(-1).expand(-1, C)

        # Scatter add features and counts
        bev_flat = torch.zeros(B * H * W, C, device=features.device, dtype=features.dtype)
        count_flat = torch.zeros(B * H * W, 1, device=features.device, dtype=features.dtype)

        bev_flat.scatter_add_(0, flat_idx_flat, features_flat)
        count_flat.scatter_add_(0, flat_idx.view(B * N, 1), ones_masked.view(B * N, 1))

        # Reshape and normalize
        bev = bev_flat.view(B, H, W, C)
        count = count_flat.view(B, H, W, 1)
        bev = bev / (count + 1e-8)
        bev = bev.permute(0, 3, 1, 2)  # (B, C, H, W)

        bev = self.refine(bev)

        return bev


class VectorEncoder(AbstractEncoder):
    """Encoder for vectorized scene representations (Argoverse 2-style).

    Processes agent tracks and map polylines, fuses them via cross-attention,
    and projects to a BEV feature map.
    """

    def __init__(self, config: VectorEncoderConfig | None = None):
        if config is None:
            config = VectorEncoderConfig()
        super().__init__(config)

        self._config = config

        self.agent_encoder = PolylineEncoder(
            input_dim=config.agent_dim,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        self.map_encoder = PolylineEncoder(
            input_dim=config.map_point_dim,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        self.fusion = CrossAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        self.to_bev = VectorToBEVProjection(
            d_model=config.d_model,
            output_size=config.output_size,
            output_channels=config.output_channels,
        )

        self.agent_type_embed = nn.Embedding(8, config.d_model)
        self.map_type_embed = nn.Embedding(8, config.d_model)

    @property
    def input_type(self) -> str:
        return "vector"

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode vectorized scene to BEV features.

        Args:
            inputs: Dictionary with:
                - "agent_history": (B, N_agents, T, agent_dim) agent tracks
                - "agent_mask": (B, N_agents, T) bool mask
                - "agent_positions": (B, N_agents, 2) current positions
                - "agent_types": (B, N_agents) int type indices
                - "map_polylines": (B, N_map, P, 2) map polylines
                - "map_mask": (B, N_map, P) bool mask
                - "map_positions": (B, N_map, 2) polyline centers
                - "map_types": (B, N_map) int type indices

        Returns:
            BEV features (B, output_channels, output_H, output_W)
        """
        agent_history = inputs["agent_history"]
        map_polylines = inputs["map_polylines"]

        agent_mask = inputs.get("agent_mask")
        map_mask = inputs.get("map_mask")
        agent_positions = inputs.get("agent_positions")
        map_positions = inputs.get("map_positions")
        agent_types = inputs.get("agent_types")
        map_types = inputs.get("map_types")

        N_agents = agent_history.shape[1]

        agent_features = self.agent_encoder(agent_history, agent_mask)

        if agent_types is not None:
            agent_features = agent_features + self.agent_type_embed(agent_types)

        map_features = self.map_encoder(map_polylines, map_mask)

        if map_types is not None:
            map_features = map_features + self.map_type_embed(map_types)

        if agent_mask is not None:
            agent_valid = agent_mask.any(dim=-1)
        else:
            agent_valid = None

        if map_mask is not None:
            map_valid = map_mask.any(dim=-1)
        else:
            map_valid = None

        fused = self.fusion(
            agent_features,
            map_features,
            agent_valid,
            map_valid,
        )

        agent_fused = fused[:, :N_agents]
        map_fused = fused[:, N_agents:]

        if agent_positions is None:
            agent_positions = agent_history[:, :, -1, :2]

        if map_positions is None:
            map_positions = map_polylines.mean(dim=2)[..., :2]

        all_features = torch.cat([agent_fused, map_fused], dim=1)
        all_positions = torch.cat([agent_positions, map_positions], dim=1)

        if agent_valid is not None and map_valid is not None:
            all_mask = torch.cat([agent_valid, map_valid], dim=1)
        else:
            all_mask = None

        bev = self.to_bev(all_features, all_positions, all_mask)

        return bev
