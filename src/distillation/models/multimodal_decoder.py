"""Multi-modal trajectory decoder for benchmark evaluation.

Extends TrajectoryDecoder to output K trajectory modes with probabilities,
as required by nuScenes and Argoverse 2 benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .decoder import TrajectoryDecoder, positional_encoding_2d


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal prediction."""

    num_modes: int = 6
    temperature: float = 1.0


class MultiModalTrajectoryDecoder(nn.Module):
    """DETR-style decoder with K trajectory modes.

    Uses K learnable mode embeddings to produce diverse trajectory predictions.
    Each mode shares the same transformer weights but starts with different
    query offsets, producing different trajectory hypotheses.

    The mode classifier predicts which mode is most likely given the scene.
    """

    def __init__(
        self,
        bev_channels: int = 512,
        bev_size: tuple[int, int] = (200, 200),
        transformer_dim: int = 768,
        transformer_heads: int = 12,
        transformer_layers: int = 8,
        transformer_ffn: int = 3072,
        num_waypoints: int = 64,
        output_dim: int = 4,
        dropout: float = 0.15,
        gradient_checkpointing: bool = False,
        num_modes: int = 6,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_modes = num_modes
        self.num_waypoints = num_waypoints
        self.transformer_dim = transformer_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gradient_checkpointing = gradient_checkpointing

        self.bev_downsample = nn.Sequential(
            nn.Conv2d(bev_channels, transformer_dim, kernel_size=5, stride=5),
            nn.BatchNorm2d(transformer_dim),
            nn.ReLU(),
        )

        self.down_h = bev_size[0] // 5
        self.down_w = bev_size[1] // 5

        self.register_buffer(
            "bev_pos_encoding",
            positional_encoding_2d(self.down_h, self.down_w, transformer_dim),
        )

        self.waypoint_queries = nn.Embedding(num_waypoints, transformer_dim)
        self.temporal_pos_encoding = nn.Embedding(num_waypoints, transformer_dim)
        self.mode_embeddings = nn.Embedding(num_modes, transformer_dim)

        nn.init.xavier_uniform_(self.waypoint_queries.weight)
        nn.init.xavier_uniform_(self.temporal_pos_encoding.weight)
        nn.init.xavier_uniform_(self.mode_embeddings.weight)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=False,
                norm_first=True,
            )
            for _ in range(transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(transformer_dim)

        self.trajectory_head = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

        self.mode_classifier = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        bev: torch.Tensor,
        ego_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode BEV features to K trajectory modes.

        Args:
            bev: BEV features (B, C, H, W)
            ego_embedding: Optional ego history embedding (B, transformer_dim)

        Returns:
            trajectories: (B, K, num_waypoints, output_dim)
            mode_probs: (B, K) probability per mode (sums to 1)
        """
        B = bev.shape[0]
        K = self.num_modes

        bev_down = self.bev_downsample(bev)
        bev_seq = bev_down.flatten(2).permute(2, 0, 1)
        bev_seq = bev_seq + self.bev_pos_encoding.permute(1, 0, 2)

        base_queries = self.waypoint_queries.weight + self.temporal_pos_encoding.weight

        all_trajectories = []
        all_mode_features = []

        for k in range(K):
            mode_emb = self.mode_embeddings.weight[k]

            queries = base_queries + mode_emb
            queries = queries.unsqueeze(1).expand(-1, B, -1)

            if ego_embedding is not None:
                ego_emb = ego_embedding.unsqueeze(0)
                queries = queries + ego_emb

            output = queries
            for layer in self.transformer_layers:
                if self.gradient_checkpointing and self.training:
                    output = checkpoint(layer, output, bev_seq, use_reentrant=False)
                else:
                    output = layer(output, bev_seq)

            output = self.transformer_norm(output)
            output = output.permute(1, 0, 2)

            trajectory = self.trajectory_head(output)
            all_trajectories.append(trajectory)

            mode_feature = output.mean(dim=1)
            all_mode_features.append(mode_feature)

        trajectories = torch.stack(all_trajectories, dim=1)
        mode_features = torch.stack(all_mode_features, dim=1)

        mode_logits = self.mode_classifier(mode_features).squeeze(-1)
        mode_probs = F.softmax(mode_logits / self.temperature, dim=-1)

        return trajectories, mode_probs

    def get_decoder_output(
        self,
        bev: torch.Tensor,
        ego_embedding: torch.Tensor | None = None,
        mode_idx: int = 0,
    ) -> torch.Tensor:
        """Get raw decoder output for a specific mode.

        Args:
            bev: BEV features (B, C, H, W)
            ego_embedding: Optional ego history embedding (B, transformer_dim)
            mode_idx: Which mode to return decoder output for

        Returns:
            Decoder output (B, num_waypoints, transformer_dim)
        """
        B = bev.shape[0]

        bev_down = self.bev_downsample(bev)
        bev_seq = bev_down.flatten(2).permute(2, 0, 1)
        bev_seq = bev_seq + self.bev_pos_encoding.permute(1, 0, 2)

        base_queries = self.waypoint_queries.weight + self.temporal_pos_encoding.weight
        mode_emb = self.mode_embeddings.weight[mode_idx]

        queries = base_queries + mode_emb
        queries = queries.unsqueeze(1).expand(-1, B, -1)

        if ego_embedding is not None:
            ego_emb = ego_embedding.unsqueeze(0)
            queries = queries + ego_emb

        output = queries
        for layer in self.transformer_layers:
            if self.gradient_checkpointing and self.training:
                output = checkpoint(layer, output, bev_seq, use_reentrant=False)
            else:
                output = layer(output, bev_seq)

        output = self.transformer_norm(output)

        return output.permute(1, 0, 2)


class MultiModalWrapper(nn.Module):
    """Wrapper to add multi-modal prediction to existing TrajectoryDecoder.

    This allows using a pre-trained single-mode decoder and adding multi-modal
    capability without retraining the base decoder.
    """

    def __init__(
        self,
        base_decoder: TrajectoryDecoder,
        num_modes: int = 6,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.base_decoder = base_decoder
        self.num_modes = num_modes
        self.temperature = temperature

        transformer_dim = base_decoder.transformer_dim

        self.mode_embeddings = nn.Embedding(num_modes, transformer_dim)
        nn.init.xavier_uniform_(self.mode_embeddings.weight)

        self.mode_classifier = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        bev: torch.Tensor,
        ego_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with K modes.

        Args:
            bev: BEV features (B, C, H, W)
            ego_embedding: Optional ego history embedding

        Returns:
            trajectories: (B, K, num_waypoints, output_dim)
            mode_probs: (B, K)
        """
        B = bev.shape[0]
        K = self.num_modes

        all_trajectories = []
        all_mode_features = []

        for k in range(K):
            mode_emb = self.mode_embeddings.weight[k]

            if ego_embedding is not None:
                modified_ego = ego_embedding + mode_emb
            else:
                modified_ego = mode_emb.unsqueeze(0).expand(B, -1)

            decoder_output = self.base_decoder.get_decoder_output(bev, modified_ego)

            trajectory = self.base_decoder.trajectory_head(decoder_output)
            all_trajectories.append(trajectory)

            mode_feature = decoder_output.mean(dim=1)
            all_mode_features.append(mode_feature)

        trajectories = torch.stack(all_trajectories, dim=1)
        mode_features = torch.stack(all_mode_features, dim=1)

        mode_logits = self.mode_classifier(mode_features).squeeze(-1)
        mode_probs = F.softmax(mode_logits / self.temperature, dim=-1)

        return trajectories, mode_probs
