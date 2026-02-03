"""Raster-based encoder for nuScenes-style BEV input.

This encoder processes rasterized BEV images containing:
- Agent history (ego + other agents as colored pixels with fading history)
- Map layers (drivable area, lanes, crosswalks, etc.)

Used for training/evaluation on nuScenes prediction benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)

from .base import AbstractEncoder, EncoderConfig


@dataclass
class RasterEncoderConfig(EncoderConfig):
    """Configuration for raster-based encoder."""

    input_channels: int = 9
    backbone_type: str = "resnet18"
    pretrained_backbone: bool = True


class RasterEncoder(AbstractEncoder):
    """Encoder for rasterized BEV images (nuScenes-style).

    Input: Rasterized BEV with agent history + map layers
    Output: BEV feature map compatible with TrajectoryDecoder

    The rasterized input typically has:
    - 3 channels for ego/agent history (RGB with fading colors for temporal info)
    - 6+ channels for map layers (drivable area, lanes, crosswalks, etc.)
    """

    def __init__(self, config: RasterEncoderConfig | None = None):
        if config is None:
            config = RasterEncoderConfig()
        super().__init__(config)

        self._config = config

        if config.backbone_type == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
            backbone = resnet18(weights=weights)
            backbone_out_channels = 512
        elif config.backbone_type == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
            backbone = resnet34(weights=weights)
            backbone_out_channels = 512
        else:
            raise ValueError(f"Unknown backbone type: {config.backbone_type}")

        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            config.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if config.pretrained_backbone:
            with torch.no_grad():
                self.conv1.weight.data[:, :3, :, :] = old_conv.weight.data
                if config.input_channels > 3:
                    self.conv1.weight.data[:, 3:, :, :] = (
                        old_conv.weight.data.mean(dim=1, keepdim=True).repeat(
                            1, config.input_channels - 3, 1, 1
                        )
                    )
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.adapter = nn.Sequential(
            nn.Conv2d(backbone_out_channels, config.output_channels, kernel_size=1),
            nn.BatchNorm2d(config.output_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(config.output_size),
        )

    @property
    def input_type(self) -> str:
        return "raster"

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode rasterized BEV to features.

        Args:
            inputs: Dictionary with "raster_bev" key containing
                    tensor of shape (B, input_channels, H, W)

        Returns:
            BEV features (B, output_channels, output_H, output_W)
        """
        x = inputs["raster_bev"]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adapter(x)

        return x
