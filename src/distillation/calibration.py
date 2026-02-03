"""F-Theta camera model and calibration utilities.

Implements the fisheye f-theta projection model used by PhysicalAI-AV cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation


@dataclass
class CameraIntrinsics:
    """F-theta fisheye camera intrinsics.

    The f-theta model maps 3D rays to 2D image coordinates:
        theta = arctan2(sqrt(x^2 + y^2), z)
        r = fw_poly_0 + fw_poly_1*theta + ... + fw_poly_4*theta^4
        u = cx + r * x / sqrt(x^2 + y^2)
        v = cy + r * y / sqrt(x^2 + y^2)

    And the inverse (backward projection):
        r = sqrt((u - cx)^2 + (v - cy)^2)
        theta = bw_poly_0 + bw_poly_1*r + ... + bw_poly_4*r^4
        ray_dir = [sin(theta) * (u-cx)/r, sin(theta) * (v-cy)/r, cos(theta)]
    """

    width: int
    height: int
    cx: float  # Principal point x
    cy: float  # Principal point y
    fw_poly: np.ndarray  # Forward polynomial coefficients (5,)
    bw_poly: np.ndarray  # Backward polynomial coefficients (5,)

    @classmethod
    def from_parquet_row(cls, row: dict) -> CameraIntrinsics:
        """Create from a row of camera_intrinsics.parquet."""
        fw_poly = np.array(
            [row["fw_poly_0"], row["fw_poly_1"], row["fw_poly_2"], row["fw_poly_3"], row["fw_poly_4"]]
        )
        bw_poly = np.array(
            [row["bw_poly_0"], row["bw_poly_1"], row["bw_poly_2"], row["bw_poly_3"], row["bw_poly_4"]]
        )
        return cls(
            width=int(row["width"]),
            height=int(row["height"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            fw_poly=fw_poly,
            bw_poly=bw_poly,
        )


@dataclass
class CameraExtrinsics:
    """Camera extrinsics (pose relative to vehicle frame).

    Coordinate frame: rear axle center, X forward, Y left, Z up.
    """

    rotation: np.ndarray  # (3, 3) rotation matrix
    translation: np.ndarray  # (3,) translation in meters

    @classmethod
    def from_parquet_row(cls, row: dict) -> CameraExtrinsics:
        """Create from a row of sensor_extrinsics.parquet."""
        # Quaternion (x, y, z, w) -> rotation matrix
        quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]])
        rotation = Rotation.from_quat(quat).as_matrix()
        translation = np.array([row["x"], row["y"], row["z"]])
        return cls(rotation=rotation, translation=translation)


@dataclass
class CameraCalibration:
    """Complete camera calibration (intrinsics + extrinsics)."""

    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


def backward_project_pixel(u: float, v: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project a pixel coordinate to a 3D ray direction in camera frame.

    Args:
        u: Pixel x coordinate
        v: Pixel y coordinate
        intrinsics: Camera intrinsics with f-theta model

    Returns:
        Unit ray direction (3,) in camera frame
    """
    dx = u - intrinsics.cx
    dy = v - intrinsics.cy
    r = np.sqrt(dx**2 + dy**2)

    if r < 1e-6:
        # Principal point - ray goes straight forward
        return np.array([0.0, 0.0, 1.0])

    # Apply backward polynomial: r -> theta
    r_powers = np.array([1, r, r**2, r**3, r**4])
    theta = np.dot(intrinsics.bw_poly, r_powers)

    # Convert to 3D ray
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    ray = np.array([sin_theta * dx / r, sin_theta * dy / r, cos_theta])
    return ray / np.linalg.norm(ray)


def compute_ground_plane_intersection(
    ray_camera: np.ndarray, extrinsics: CameraExtrinsics
) -> np.ndarray | None:
    """Compute where a camera ray intersects the ground plane (z=0).

    Args:
        ray_camera: Unit ray direction (3,) in camera frame
        extrinsics: Camera extrinsics (rotation, translation)

    Returns:
        (x, y) ground plane coordinates in vehicle frame, or None if no intersection
    """
    # Transform ray to vehicle frame
    ray_vehicle = extrinsics.rotation @ ray_camera

    # Camera position in vehicle frame
    cam_pos = extrinsics.translation

    # Find t where cam_pos + t * ray intersects z=0
    # cam_pos[2] + t * ray_vehicle[2] = 0
    # t = -cam_pos[2] / ray_vehicle[2]

    if abs(ray_vehicle[2]) < 1e-6:
        # Ray parallel to ground plane
        return None

    t = -cam_pos[2] / ray_vehicle[2]

    if t <= 0:
        # Intersection is behind camera
        return None

    # Ground intersection point
    intersection = cam_pos + t * ray_vehicle
    return intersection[:2]  # Return (x, y) only


def compute_bev_projection_indices(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    feature_size: tuple[int, int],
    bev_size: tuple[int, int],
    bev_resolution: float,
    input_resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute BEV projection indices for a camera.

    For each pixel in the feature map, compute which BEV cell it projects to.

    Args:
        intrinsics: Camera intrinsics
        extrinsics: Camera extrinsics
        feature_size: (H, W) of CNN feature map
        bev_size: (H, W) of BEV grid
        bev_resolution: Meters per BEV cell
        input_resolution: (H, W) of input image

    Returns:
        Tuple of:
        - valid_mask: (feat_H, feat_W) bool array - which pixels have valid projections
        - bev_y: (feat_H, feat_W) int array - BEV row index for each pixel
        - bev_x: (feat_H, feat_W) int array - BEV column index for each pixel
    """
    feat_h, feat_w = feature_size
    bev_h, bev_w = bev_size
    input_h, input_w = input_resolution

    # Scale factors from feature map to original image
    scale_h = input_h / feat_h
    scale_w = input_w / feat_w

    # BEV grid is centered at ego position
    # x forward (positive = ahead), y left (positive = left)
    # BEV coordinates: row 0 = far ahead, col 0 = far left
    bev_x_range = bev_w * bev_resolution / 2  # Half-width in meters
    bev_y_range = bev_h * bev_resolution / 2  # Half-depth in meters

    valid_mask = np.zeros((feat_h, feat_w), dtype=bool)
    bev_y_idx = np.zeros((feat_h, feat_w), dtype=np.int64)
    bev_x_idx = np.zeros((feat_h, feat_w), dtype=np.int64)

    for feat_v in range(feat_h):
        for feat_u in range(feat_w):
            # Map feature pixel to image pixel (center of receptive field)
            img_u = (feat_u + 0.5) * scale_w
            img_v = (feat_v + 0.5) * scale_h

            # Get ray direction
            ray = backward_project_pixel(img_u, img_v, intrinsics)

            # Find ground plane intersection
            ground_xy = compute_ground_plane_intersection(ray, extrinsics)

            if ground_xy is None:
                continue

            x, y = ground_xy

            # Check if within BEV range
            # BEV grid: x (forward) maps to row, y (left) maps to column
            # Row 0 = x_max (far ahead), Row bev_h-1 = x_min (behind)
            # Col 0 = y_max (far left), Col bev_w-1 = y_min (far right)
            if abs(x) > bev_y_range or abs(y) > bev_x_range:
                continue

            # Convert to BEV indices
            # x: positive = ahead, so row = bev_h/2 - x/resolution
            bev_row = int((bev_y_range - x) / bev_resolution)
            # y: positive = left, so col = bev_w/2 - y/resolution
            bev_col = int((bev_x_range - y) / bev_resolution)

            # Clamp to valid range
            bev_row = max(0, min(bev_h - 1, bev_row))
            bev_col = max(0, min(bev_w - 1, bev_col))

            valid_mask[feat_v, feat_u] = True
            bev_y_idx[feat_v, feat_u] = bev_row
            bev_x_idx[feat_v, feat_u] = bev_col

    return valid_mask, bev_y_idx, bev_x_idx


class BEVProjector(torch.nn.Module):
    """Projects camera features to BEV using pre-computed ground-plane projection.

    This module stores pre-computed projection indices for each camera and
    performs efficient feature splatting to BEV at runtime.
    """

    def __init__(
        self,
        calibrations: list[CameraCalibration],
        feature_size: tuple[int, int],
        bev_size: tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,
        input_resolution: tuple[int, int] = (224, 224),
    ):
        """Initialize BEV projector.

        Args:
            calibrations: List of camera calibrations
            feature_size: (H, W) of CNN feature map
            bev_size: (H, W) of BEV grid
            bev_resolution: Meters per BEV cell
            input_resolution: (H, W) of input image
        """
        super().__init__()

        self.num_cameras = len(calibrations)
        self.bev_size = bev_size
        self.feature_size = feature_size

        # Pre-compute projection indices for each camera
        valid_masks = []
        bev_y_indices = []
        bev_x_indices = []

        for calib in calibrations:
            valid_mask, bev_y, bev_x = compute_bev_projection_indices(
                intrinsics=calib.intrinsics,
                extrinsics=calib.extrinsics,
                feature_size=feature_size,
                bev_size=bev_size,
                bev_resolution=bev_resolution,
                input_resolution=input_resolution,
            )
            valid_masks.append(valid_mask)
            bev_y_indices.append(bev_y)
            bev_x_indices.append(bev_x)

        # Register as buffers (move to device with model)
        self.register_buffer("valid_masks", torch.tensor(np.stack(valid_masks)))  # (num_cams, H, W)
        self.register_buffer("bev_y_indices", torch.tensor(np.stack(bev_y_indices)))  # (num_cams, H, W)
        self.register_buffer("bev_x_indices", torch.tensor(np.stack(bev_x_indices)))  # (num_cams, H, W)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project camera features to BEV.

        Args:
            features: Camera features (B, num_cameras, C, feat_H, feat_W)

        Returns:
            BEV features (B, C, bev_H, bev_W)
        """
        B, num_cams, C, feat_h, feat_w = features.shape
        bev_h, bev_w = self.bev_size

        # Initialize BEV grid
        bev = features.new_zeros(B, C, bev_h, bev_w)

        # Splat features from each camera
        for cam_idx in range(num_cams):
            cam_features = features[:, cam_idx]  # (B, C, H, W)
            valid = self.valid_masks[cam_idx]  # (H, W)
            bev_y = self.bev_y_indices[cam_idx]  # (H, W)
            bev_x = self.bev_x_indices[cam_idx]  # (H, W)

            # Get valid feature positions
            valid_positions = valid.nonzero(as_tuple=True)  # (feat_y_coords, feat_x_coords)
            if len(valid_positions[0]) == 0:
                continue

            feat_y, feat_x = valid_positions
            target_y = bev_y[feat_y, feat_x]
            target_x = bev_x[feat_y, feat_x]

            # Extract valid features: (B, C, num_valid)
            valid_features = cam_features[:, :, feat_y, feat_x]

            # Scatter-add to BEV
            # Create flat target indices
            flat_target = target_y * bev_w + target_x  # (num_valid,)

            # Expand for batch and channel dimensions
            for b in range(B):
                for c in range(C):
                    bev[b, c].view(-1).scatter_add_(0, flat_target, valid_features[b, c])

        return bev


def load_calibrations_from_metadata(
    camera_names: list[str],
    clip_id: str | None,
    intrinsics_df,
    extrinsics_df,
) -> list[CameraCalibration]:
    """Load calibrations for specified cameras from metadata dataframes.

    Args:
        camera_names: List of camera names to load
        clip_id: Clip ID for per-clip calibrations, or None for static calibrations
        intrinsics_df: DataFrame from camera_intrinsics.parquet
        extrinsics_df: DataFrame from sensor_extrinsics.parquet

    Returns:
        List of CameraCalibration objects
    """
    calibrations = []

    for name in camera_names:
        # Look up intrinsics
        if clip_id is not None:
            intrinsics_row = intrinsics_df[
                (intrinsics_df["clip_id"] == clip_id) & (intrinsics_df["camera_name"] == name)
            ]
        else:
            intrinsics_row = None

        if intrinsics_row is None or len(intrinsics_row) == 0:
            # Fall back to static calibration (no clip_id filter)
            intrinsics_row = intrinsics_df[intrinsics_df["camera_name"] == name]

        if len(intrinsics_row) == 0:
            raise ValueError(f"No intrinsics found for camera {name}")

        intrinsics = CameraIntrinsics.from_parquet_row(intrinsics_row.iloc[0].to_dict())

        # Look up extrinsics
        extrinsics_row = extrinsics_df[extrinsics_df["sensor_name"] == name]
        if len(extrinsics_row) == 0:
            raise ValueError(f"No extrinsics found for camera {name}")

        extrinsics = CameraExtrinsics.from_parquet_row(extrinsics_row.iloc[0].to_dict())

        calibrations.append(CameraCalibration(name=name, intrinsics=intrinsics, extrinsics=extrinsics))

    return calibrations
