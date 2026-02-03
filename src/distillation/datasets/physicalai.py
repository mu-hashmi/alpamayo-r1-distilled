"""Dataset for student model training.

Loads:
- Teacher labels from npz files (trajectories, rotations, embeddings)
- Images from HuggingFace Hub or pre-extracted local frames
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default camera order (matching Alpamayo)
DEFAULT_CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

# Temporal parameters
NUM_HISTORY_FRAMES = 4  # 4 frames of context (matching Alpamayo's context_length)
FRAME_INTERVAL_US = 100_000  # 100ms = 10Hz


class DistillationDataset(Dataset):
    """Dataset for training student models.

    Loads pre-computed teacher labels and streams images from HuggingFace Hub.
    """

    def __init__(
        self,
        labels_path: str | Path,
        cameras: list[str] = None,
        resolution: tuple[int, int] = (224, 224),
        num_frames: int = NUM_HISTORY_FRAMES,
        transform: Any = None,
        cache_images: bool = False,
    ):
        """Initialize dataset.

        Args:
            labels_path: Path to teacher labels npz file
            cameras: List of camera names to use
            resolution: Output image resolution (H, W)
            num_frames: Number of history frames per camera
            transform: Additional image transforms (applied after resize)
            cache_images: Whether to cache decoded images (uses more RAM)
        """
        self.labels_path = Path(labels_path)
        self.cameras = cameras or DEFAULT_CAMERAS
        self.resolution = resolution
        self.num_frames = num_frames
        self.cache_images = cache_images
        self._image_cache: dict[tuple[str, int], torch.Tensor] = {}

        # Load teacher labels
        logger.info(f"Loading teacher labels from {self.labels_path}")
        data = np.load(self.labels_path, allow_pickle=True)

        self.clip_ids = data["clip_ids"].tolist()
        self.t0_us = data["t0_us"].astype(np.int64)
        self.trajectory_xyz = data["trajectory_xyz"].astype(np.float32)
        self.trajectory_rot = data["trajectory_rot"].astype(np.float32)
        self.coc_embeddings = data["coc_embeddings"].astype(np.float32)

        logger.info(f"Loaded {len(self.clip_ids)} samples")

        # Image transforms
        self.resize = transforms.Resize(resolution, antialias=True)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],  # ImageNet std
        )
        self.extra_transform = transform

        # Dataset interface (lazy init)
        self._avdi = None

    @property
    def avdi(self):
        """Lazy init of PhysicalAIAVDatasetInterface."""
        if self._avdi is None:
            import physical_ai_av

            self._avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        return self._avdi

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _load_frames(self, clip_id: str, t0_us: int) -> torch.Tensor:
        """Load and preprocess frames for all cameras.

        Args:
            clip_id: Clip identifier
            t0_us: Reference timestamp in microseconds

        Returns:
            Tensor (num_cameras, num_frames * 3, H, W)
        """
        # Check cache
        cache_key = (clip_id, t0_us)
        if self.cache_images and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        # Compute frame timestamps
        # Frames at t0 - 300ms, t0 - 200ms, t0 - 100ms, t0
        timestamps = np.array(
            [t0_us - (self.num_frames - 1 - i) * FRAME_INTERVAL_US for i in range(self.num_frames)]
        )

        all_camera_frames = []

        for camera in self.cameras:
            video_reader = self.avdi.get_clip_feature(clip_id, camera, maybe_stream=True)
            frames, actual_ts = video_reader.decode_images_from_timestamps(timestamps)
            video_reader.close()

            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            frames = self.resize(frames)
            frames = self.normalize(frames)
            frames = frames.view(-1, *frames.shape[2:])
            all_camera_frames.append(frames)

        result = torch.stack(all_camera_frames)

        if self.cache_images:
            self._image_cache[cache_key] = result

        return result

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
            - 'images': (num_cameras, 12, H, W) image tensor
            - 'trajectory_xyz': (64, 3) trajectory positions
            - 'trajectory_rot': (64, 3, 3) trajectory rotations
            - 'coc_embedding': (384,) teacher reasoning embedding
            - 'clip_id': clip identifier string
        """
        clip_id = self.clip_ids[idx]
        t0_us = int(self.t0_us[idx])

        images = self._load_frames(clip_id, t0_us)

        if self.extra_transform is not None:
            images = self.extra_transform(images)

        trajectory_xyz = torch.from_numpy(self.trajectory_xyz[idx])
        trajectory_rot = torch.from_numpy(self.trajectory_rot[idx])
        coc_embedding = torch.from_numpy(self.coc_embeddings[idx])

        return {
            "images": images,
            "trajectory_xyz": trajectory_xyz,
            "trajectory_rot": trajectory_rot,
            "coc_embedding": coc_embedding,
            "clip_id": clip_id,
        }


class DistillationDatasetOffline(Dataset):
    """Offline dataset for testing (pre-loaded images, no streaming).

    Uses dummy images for testing model architecture without HuggingFace access.
    """

    def __init__(
        self,
        labels_path: str | Path,
        cameras: list[str] = None,
        resolution: tuple[int, int] = (224, 224),
        num_frames: int = NUM_HISTORY_FRAMES,
        num_ego_history_steps: int = 16,
    ):
        """Initialize offline dataset."""
        self.labels_path = Path(labels_path)
        self.cameras = cameras or DEFAULT_CAMERAS
        self.resolution = resolution
        self.num_frames = num_frames
        self.num_cameras = len(self.cameras)
        self.num_ego_history_steps = num_ego_history_steps

        # Load teacher labels
        data = np.load(self.labels_path, allow_pickle=True)

        self.clip_ids = data["clip_ids"].tolist()
        self.t0_us = data["t0_us"].astype(np.int64)
        self.trajectory_xyz = data["trajectory_xyz"].astype(np.float32)
        self.trajectory_rot = data["trajectory_rot"].astype(np.float32)
        self.coc_embeddings = data["coc_embeddings"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample with dummy images."""
        clip_id = self.clip_ids[idx]

        # Dummy images (random noise, normalized)
        images = torch.randn(self.num_cameras, self.num_frames * 3, *self.resolution)

        # Teacher labels
        trajectory_xyz = torch.from_numpy(self.trajectory_xyz[idx])
        trajectory_rot = torch.from_numpy(self.trajectory_rot[idx])
        coc_embedding = torch.from_numpy(self.coc_embeddings[idx])

        # Dummy ego history for unicycle integration
        # Create positions going backward in time (vehicle moving forward at ~10 m/s)
        ego_xyz = torch.zeros(self.num_ego_history_steps, 3)
        for i in range(self.num_ego_history_steps):
            ego_xyz[i, 0] = -1.0 * (self.num_ego_history_steps - 1 - i)
        ego_rot = torch.eye(3).unsqueeze(0).expand(self.num_ego_history_steps, 3, 3).clone()

        return {
            "images": images,
            "trajectory_xyz": trajectory_xyz,
            "trajectory_rot": trajectory_rot,
            "coc_embedding": coc_embedding,
            "ego_history_xyz": ego_xyz,
            "ego_history_rot": ego_rot,
            "clip_id": clip_id,
        }


class DistillationDatasetLocal(Dataset):
    """Dataset loading from pre-extracted local frames.

    Use this for training after running scripts/extract_frames.py.
    Expects frames already resized to target resolution during extraction.
    """

    def __init__(
        self,
        labels_path: str | Path,
        frames_dir: str | Path,
        resolution: tuple[int, int] = (224, 224),
        num_frames: int = NUM_HISTORY_FRAMES,
        transform: Any = None,
        ego_history_path: str | Path | None = None,
    ):
        """Initialize local dataset.

        Args:
            labels_path: Path to teacher labels npz file
            frames_dir: Path to extracted frames directory (e.g., extracted_frames/train/)
            resolution: Expected image resolution (H, W) - frames should already be this size
            num_frames: Number of history frames per camera
            transform: Additional image transforms
            ego_history_path: Path to ego_history npz file (optional, for velocity context)
        """
        self.labels_path = Path(labels_path)
        self.frames_dir = Path(frames_dir)
        self.resolution = resolution
        self.num_frames = num_frames

        # Load teacher labels
        logger.info(f"Loading teacher labels from {self.labels_path}")
        data = np.load(self.labels_path, allow_pickle=True)

        self.clip_ids = data["clip_ids"].tolist()
        self.t0_us = data["t0_us"].astype(np.int64)
        self.trajectory_xyz = data["trajectory_xyz"].astype(np.float32)
        self.trajectory_rot = data["trajectory_rot"].astype(np.float32)
        self.coc_embeddings = data["coc_embeddings"].astype(np.float32)

        logger.info(f"Loaded {len(self.clip_ids)} samples")
        logger.info(f"Frames directory: {self.frames_dir}")

        # Load ego_history if provided
        self.ego_history_xyz = None
        self.ego_history_rot = None
        if ego_history_path is not None:
            ego_history_path = Path(ego_history_path)
            if ego_history_path.exists():
                logger.info(f"Loading ego_history from {ego_history_path}")
                ego_data = np.load(ego_history_path)
                self.ego_history_xyz = ego_data["ego_history_xyz"].astype(np.float32)
                self.ego_history_rot = ego_data["ego_history_rot"].astype(np.float32)
                logger.info(f"  ego_history_xyz: {self.ego_history_xyz.shape}")
            else:
                logger.warning(f"ego_history_path not found: {ego_history_path}")

        # Image normalization (no resize needed - frames pre-resized during extraction)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.extra_transform = transform

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample from local frames."""
        clip_id = self.clip_ids[idx]

        # Load pre-extracted frames
        frame_path = self.frames_dir / f"sample_{idx:05d}.npz"

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Frames not found: {frame_path}. Run scripts/extract_frames.py first."
            )

        frame_data = np.load(frame_path)
        frames = frame_data["frames"]  # (4, 4, H, W, 3) uint8

        # Convert to tensor and process
        # (num_cameras, num_frames, H, W, 3) -> (num_cameras, num_frames, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float() / 255.0

        processed_cameras = []
        for cam_idx in range(frames.shape[0]):
            cam_frames = frames[cam_idx]  # (num_frames, 3, H, W)
            cam_frames = self.normalize(cam_frames)
            # Flatten frames to channels: (num_frames, 3, H, W) -> (num_frames * 3, H, W)
            cam_frames = cam_frames.reshape(-1, *cam_frames.shape[2:])
            processed_cameras.append(cam_frames)

        images = torch.stack(processed_cameras)  # (num_cameras, 12, H, W)

        # Apply additional transforms
        if self.extra_transform is not None:
            images = self.extra_transform(images)

        # Teacher labels
        trajectory_xyz = torch.from_numpy(self.trajectory_xyz[idx])
        trajectory_rot = torch.from_numpy(self.trajectory_rot[idx])
        coc_embedding = torch.from_numpy(self.coc_embeddings[idx])

        result = {
            "images": images,
            "trajectory_xyz": trajectory_xyz,
            "trajectory_rot": trajectory_rot,
            "coc_embedding": coc_embedding,
            "clip_id": clip_id,
        }

        # Add ego_history if available
        if self.ego_history_xyz is not None:
            result["ego_history_xyz"] = torch.from_numpy(self.ego_history_xyz[idx])
            result["ego_history_rot"] = torch.from_numpy(self.ego_history_rot[idx])

        return result


def create_dataloaders(
    train_labels: str | Path,
    val_labels: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    resolution: tuple[int, int] = (224, 224),
    frames_dir: str | Path | None = None,
    offline: bool = False,
    ego_history_dir: str | Path | None = None,
) -> tuple:
    """Create train and validation dataloaders.

    Args:
        train_labels: Path to train labels npz
        val_labels: Path to val labels npz
        batch_size: Batch size
        num_workers: DataLoader workers
        resolution: Image resolution
        frames_dir: Path to extracted frames (e.g., 'extracted_frames/')
        offline: Use offline dataset (dummy images for testing)
        ego_history_dir: Path to ego_history labels (e.g., 'teacher_labels/')

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    # Determine ego_history paths
    train_ego_history = None
    val_ego_history = None
    if ego_history_dir is not None:
        ego_history_dir = Path(ego_history_dir)
        train_ego_history = ego_history_dir / "train_ego_history.npz"
        val_ego_history = ego_history_dir / "val_ego_history.npz"

    if offline:
        # Dummy images for testing
        train_dataset = DistillationDatasetOffline(labels_path=train_labels, resolution=resolution)
        val_dataset = DistillationDatasetOffline(labels_path=val_labels, resolution=resolution)
    elif frames_dir is not None:
        # Local pre-extracted frames (recommended for training)
        frames_dir = Path(frames_dir)
        train_dataset = DistillationDatasetLocal(
            labels_path=train_labels,
            frames_dir=frames_dir / "train",
            resolution=resolution,
            ego_history_path=train_ego_history,
        )
        val_dataset = DistillationDatasetLocal(
            labels_path=val_labels,
            frames_dir=frames_dir / "val",
            resolution=resolution,
            ego_history_path=val_ego_history,
        )
    else:
        # Stream from HuggingFace (slow, not recommended for training)
        logger.warning(
            "Streaming from HuggingFace - this will be slow! Consider using --frames-dir"
        )
        train_dataset = DistillationDataset(labels_path=train_labels, resolution=resolution)
        val_dataset = DistillationDataset(labels_path=val_labels, resolution=resolution)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For contrastive loss consistency
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader
