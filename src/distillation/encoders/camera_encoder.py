"""Camera-based BEV encoder for multi-camera, multi-frame input.

This encoder processes raw camera images and projects them to BEV features.
Used for training on PhysicalAI-AV dataset and similar camera-based benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet50,
    resnet101,
    resnet152,
)

from ..calibration import BEVProjector, CameraCalibration
from .base import AbstractEncoder, EncoderConfig


@dataclass
class CameraEncoderConfig(EncoderConfig):
    """Configuration for camera-based BEV encoder."""

    backbone_type: str = "resnet50"
    pretrained_backbone: bool = True
    input_resolution: tuple[int, int] = (224, 224)
    bev_resolution: float = 0.5
    num_cameras: int = 4
    num_frames: int = 4


def create_backbone(backbone_type: str, pretrained: bool = True) -> nn.Module:
    """Create ResNet backbone for 12-channel input.

    Args:
        backbone_type: One of 'resnet50', 'resnet101', 'resnet152'
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        Modified ResNet with 12-channel first conv layer
    """
    if backbone_type == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        out_channels = 2048
    elif backbone_type == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet101(weights=weights)
        out_channels = 2048
    elif backbone_type == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet152(weights=weights)
        out_channels = 2048
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained:
        with torch.no_grad():
            model.conv1.weight.data = (
                old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, 12, 1, 1)
            )
    else:
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    model.out_channels = out_channels
    return model


class MultiFrameResNet(nn.Module):
    """ResNet backbone for multi-frame input (4 frames x 3 RGB = 12 channels)."""

    def __init__(self, backbone_type: str = "resnet50", pretrained: bool = True):
        super().__init__()

        backbone = create_backbone(backbone_type, pretrained)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = backbone.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CameraBEVEncoder(AbstractEncoder):
    """Camera-based BEV encoder with geometric projection.

    Takes multi-camera, multi-frame images and produces BEV feature map
    using ground-plane projection based on camera calibrations.
    """

    def __init__(
        self,
        calibrations: list[CameraCalibration],
        config: CameraEncoderConfig | None = None,
    ):
        if config is None:
            config = CameraEncoderConfig()
        super().__init__(config)

        self.num_cameras = len(calibrations)
        self._config = config

        self.backbone = MultiFrameResNet(
            backbone_type=config.backbone_type,
            pretrained=config.pretrained_backbone,
        )
        backbone_out_channels = self.backbone.out_channels

        self.feature_size = (
            config.input_resolution[0] // 32,
            config.input_resolution[1] // 32,
        )

        self.channel_reduce = nn.Conv2d(backbone_out_channels, 256, kernel_size=1)

        self.bev_projector = BEVProjector(
            calibrations=calibrations,
            feature_size=self.feature_size,
            bev_size=config.output_size,
            bev_resolution=config.bev_resolution,
            input_resolution=config.input_resolution,
        )

        self.bev_refine = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.output_channels),
            nn.ReLU(inplace=True),
        )

    @property
    def input_type(self) -> str:
        return "camera"

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multi-camera images to BEV.

        Args:
            inputs: Dictionary with "images" key containing
                    tensor of shape (B, num_cameras, 12, H, W)

        Returns:
            BEV features (B, output_channels, bev_H, bev_W)
        """
        images = inputs["images"]
        B, num_cams, C, H, W = images.shape

        x = images.view(B * num_cams, C, H, W)
        features = self.backbone(x)
        features = self.channel_reduce(features)

        feat_h, feat_w = features.shape[-2:]
        features = features.view(B, num_cams, 256, feat_h, feat_w)

        bev = self.bev_projector(features)
        bev = self.bev_refine(bev)

        return bev


class CameraBEVEncoderDummy(AbstractEncoder):
    """Dummy camera encoder for testing without calibration data.

    Uses simple pooling instead of geometric projection.
    """

    def __init__(
        self,
        config: CameraEncoderConfig | None = None,
    ):
        if config is None:
            config = CameraEncoderConfig()
        super().__init__(config)

        self._config = config

        self.backbone = MultiFrameResNet(
            backbone_type=config.backbone_type,
            pretrained=config.pretrained_backbone,
        )
        backbone_out_channels = self.backbone.out_channels

        self.channel_reduce = nn.Conv2d(backbone_out_channels, 256, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(256 * config.num_cameras, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(config.output_size),
            nn.Conv2d(512, config.output_channels, kernel_size=1),
            nn.BatchNorm2d(config.output_channels),
            nn.ReLU(inplace=True),
        )

    @property
    def input_type(self) -> str:
        return "camera"

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        images = inputs["images"]
        B, num_cams, C, H, W = images.shape

        x = images.view(B * num_cams, C, H, W)
        features = self.backbone(x)
        features = self.channel_reduce(features)

        feat_h, feat_w = features.shape[-2:]
        features = features.view(B, num_cams, 256, feat_h, feat_w)
        features = features.view(B, num_cams * 256, feat_h, feat_w)

        bev = self.fusion(features)

        return bev
