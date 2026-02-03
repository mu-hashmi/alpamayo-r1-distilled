"""Encoder module: ResNet-50 backbone with BEV projection.

Architecture:
- ResNet-50 with 12-channel input (4 frames × 3 RGB)
- Shared weights across all 4 cameras
- Ground-plane BEV projection using pre-computed indices
- BEV refinement with 3 conv layers
"""

from __future__ import annotations

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

    # Modify first conv for 12-channel input
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        12, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Initialize: average pretrained weights across input channels
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.data = (
                old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, 12, 1, 1)
            )
    else:
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # Store output channels as attribute
    model.out_channels = out_channels

    return model


class MultiFrameResNet(nn.Module):
    """ResNet backbone modified for multi-frame input.

    Takes 12-channel input (4 frames × 3 RGB) and outputs feature maps.
    Supports ResNet-50, ResNet-101, and ResNet-152.
    """

    def __init__(self, backbone_type: str = "resnet50", pretrained: bool = True):
        """Initialize multi-frame ResNet.

        Args:
            backbone_type: One of 'resnet50', 'resnet101', 'resnet152'
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        backbone = create_backbone(backbone_type, pretrained)

        # Copy layers from created backbone
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
        """Forward pass.

        Args:
            x: Input tensor (B, 12, H, W) - 4 frames × 3 RGB

        Returns:
            Feature map (B, out_channels, H/32, W/32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MultiFrameResNet50(nn.Module):
    """ResNet-50 modified for multi-frame input.

    Takes 12-channel input (4 frames × 3 RGB) and outputs 2048-dim features.
    First conv layer is randomly initialized for 12 channels, rest uses
    ImageNet pretrained weights.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(weights=None)

        # Replace first conv layer for 12-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            12,
            64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize with small random weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Copy rest of ResNet layers
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Output channels
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 12, H, W) - 4 frames × 3 RGB

        Returns:
            Feature map (B, 2048, H/32, W/32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BEVEncoder(nn.Module):
    """Complete BEV encoder: backbone + projection + refinement.

    Takes multi-camera, multi-frame images and produces BEV feature map.
    """

    def __init__(
        self,
        calibrations: list[CameraCalibration],
        input_resolution: tuple[int, int] = (224, 224),
        bev_size: tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,
        bev_channels: int = 256,
        pretrained_backbone: bool = True,
        backbone_type: str = "resnet50",
    ):
        """Initialize BEV encoder.

        Args:
            calibrations: Camera calibrations for BEV projection
            input_resolution: (H, W) of input images
            bev_size: (H, W) of BEV grid
            bev_resolution: Meters per BEV cell
            bev_channels: Output channels of BEV features
            pretrained_backbone: Use ImageNet pretrained weights for ResNet
            backbone_type: One of 'resnet50', 'resnet101', 'resnet152'
        """
        super().__init__()

        self.num_cameras = len(calibrations)
        self.bev_size = bev_size
        self.bev_channels = bev_channels

        # ResNet backbone (shared across cameras)
        self.backbone = MultiFrameResNet(backbone_type=backbone_type, pretrained=pretrained_backbone)
        backbone_out_channels = self.backbone.out_channels

        # Feature map size after ResNet (input/32)
        self.feature_size = (input_resolution[0] // 32, input_resolution[1] // 32)

        # Channel reduction: backbone_channels -> 256
        self.channel_reduce = nn.Conv2d(backbone_out_channels, 256, kernel_size=1)

        # BEV projector (pre-computed indices)
        self.bev_projector = BEVProjector(
            calibrations=calibrations,
            feature_size=self.feature_size,
            bev_size=bev_size,
            bev_resolution=bev_resolution,
            input_resolution=input_resolution,
        )

        # BEV refinement: 3 conv layers
        self.bev_refine = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, bev_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode multi-camera images to BEV.

        Args:
            images: Input images (B, num_cameras, 12, H, W)
                    where 12 = 4 frames × 3 RGB channels

        Returns:
            BEV features (B, bev_channels, bev_H, bev_W)
        """
        B, num_cams, C, H, W = images.shape

        # Reshape for shared backbone: (B * num_cams, 12, H, W)
        x = images.view(B * num_cams, C, H, W)

        # Extract features through ResNet
        features = self.backbone(x)  # (B * num_cams, 2048, H/32, W/32)

        # Reduce channels
        features = self.channel_reduce(features)  # (B * num_cams, 256, H/32, W/32)

        # Reshape back to (B, num_cams, 256, feat_H, feat_W)
        feat_h, feat_w = features.shape[-2:]
        features = features.view(B, num_cams, 256, feat_h, feat_w)

        # Project to BEV
        bev = self.bev_projector(features)  # (B, 256, bev_H, bev_W)

        # Refine BEV features
        bev = self.bev_refine(bev)  # (B, bev_channels, bev_H, bev_W)

        return bev


class BEVEncoderDummy(nn.Module):
    """Dummy BEV encoder for testing without calibration data.

    Uses simple pooling instead of geometric projection.
    """

    def __init__(
        self,
        num_cameras: int = 4,
        input_resolution: tuple[int, int] = (224, 224),
        bev_size: tuple[int, int] = (200, 200),
        bev_channels: int = 256,
        pretrained_backbone: bool = True,
        backbone_type: str = "resnet50",
    ):
        """Initialize dummy BEV encoder.

        Args:
            num_cameras: Number of cameras
            input_resolution: (H, W) of input images
            bev_size: (H, W) of BEV grid
            bev_channels: Output channels of BEV features
            pretrained_backbone: Use ImageNet pretrained weights for ResNet
            backbone_type: One of 'resnet50', 'resnet101', 'resnet152'
        """
        super().__init__()

        self.num_cameras = num_cameras
        self.bev_size = bev_size
        self.bev_channels = bev_channels

        # Configurable backbone
        self.backbone = MultiFrameResNet(
            backbone_type=backbone_type,
            pretrained=pretrained_backbone,
        )
        backbone_out_channels = self.backbone.out_channels

        # Channel reduction
        self.channel_reduce = nn.Conv2d(backbone_out_channels, 256, kernel_size=1)

        # Simple fusion: concat cameras then project to BEV size
        # This is NOT geometrically correct but allows testing without calibration
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * num_cameras, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(bev_size),
            nn.Conv2d(512, bev_channels, kernel_size=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to BEV (simplified, non-geometric).

        Args:
            images: Input images (B, num_cameras, 12, H, W)

        Returns:
            BEV features (B, bev_channels, bev_H, bev_W)
        """
        B, num_cams, C, H, W = images.shape

        # Process each camera through backbone
        x = images.view(B * num_cams, C, H, W)
        features = self.backbone(x)  # (B * num_cams, 2048, H/32, W/32)
        features = self.channel_reduce(features)  # (B * num_cams, 256, H/32, W/32)

        # Reshape and concat cameras
        feat_h, feat_w = features.shape[-2:]
        features = features.view(B, num_cams, 256, feat_h, feat_w)
        features = features.view(B, num_cams * 256, feat_h, feat_w)  # (B, 1024, H/32, W/32)

        # Fuse and project to BEV
        bev = self.fusion(features)  # (B, bev_channels, bev_H, bev_W)

        return bev
