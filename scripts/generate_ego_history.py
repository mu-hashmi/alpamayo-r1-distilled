#!/usr/bin/env python3
"""Generate ego_history labels from PhysicalAI-AV dataset.

Extracts past vehicle poses (ego_history) for each sample in the teacher labels.
This provides velocity/motion context that matches what Alpamayo receives.

Usage:
    python scripts/generate_ego_history.py --split train
    python scripts/generate_ego_history.py --split val
    python scripts/generate_ego_history.py --split test

Output:
    teacher_labels/{split}_ego_history.npz containing:
    - ego_history_xyz: (N, num_history_steps, 3) - past positions in ego frame at t0
    - ego_history_rot: (N, num_history_steps, 3, 3) - past rotations in ego frame at t0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
REPO_ROOT = Path(__file__).parent.parent
LABELS_DIR = REPO_ROOT / "teacher_labels"

# Ego history parameters (matching Alpamayo)
NUM_HISTORY_STEPS = 16  # 16 past poses including t0 (1.6s at 10Hz)
HISTORY_INTERVAL_US = 100_000  # 100ms = 10Hz


def transform_to_ego_frame(
    positions: np.ndarray,
    rotations: np.ndarray,
    ref_position: np.ndarray,
    ref_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform poses to ego-centric frame at reference pose.

    Args:
        positions: (N, 3) world positions
        rotations: (N, 3, 3) world rotation matrices
        ref_position: (3,) reference position (ego origin)
        ref_rotation: (3, 3) reference rotation matrix

    Returns:
        ego_positions: (N, 3) positions in ego frame
        ego_rotations: (N, 3, 3) rotations in ego frame
    """
    # Inverse of reference rotation
    ref_rot_inv = ref_rotation.T

    # Transform positions: p_ego = R_ref^(-1) @ (p_world - t_ref)
    ego_positions = (positions - ref_position) @ ref_rot_inv.T

    # Transform rotations: R_ego = R_ref^(-1) @ R_world
    ego_rotations = np.einsum("ij,njk->nik", ref_rot_inv, rotations)

    return ego_positions, ego_rotations


def generate_ego_history_for_split(split: str) -> None:
    """Generate ego_history labels for a data split.

    Args:
        split: 'train', 'val', or 'test'
    """
    import physical_ai_av

    # Load teacher labels to get clip_ids and timestamps
    labels_path = LABELS_DIR / f"{split}_labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    logger.info(f"Loading labels from {labels_path}")
    data = np.load(labels_path, allow_pickle=True)
    clip_ids = data["clip_ids"].tolist()
    t0_us_list = data["t0_us"].astype(np.int64)

    num_samples = len(clip_ids)
    logger.info(f"Processing {num_samples} samples")

    # Initialize dataset interface (disable download confirmation for batch downloads)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        confirm_download_threshold_gb=float("inf")
    )

    # Pre-download all egomotion chunks upfront (MUCH faster than streaming)
    unique_clip_ids = list(set(clip_ids))
    logger.info(f"Pre-downloading egomotion for {len(unique_clip_ids)} unique clips...")
    avdi.download_clip_features(unique_clip_ids, features="egomotion", max_workers=8)
    logger.info("Download complete, processing samples...")

    # Output arrays
    all_ego_history_xyz = np.zeros((num_samples, NUM_HISTORY_STEPS, 3), dtype=np.float32)
    all_ego_history_rot = np.zeros((num_samples, NUM_HISTORY_STEPS, 3, 3), dtype=np.float32)

    # Track failures
    failed_samples = []
    current_clip_id = None
    egomotion_interp = None

    for idx in tqdm(range(num_samples), desc=f"Generating ego_history for {split}"):
        clip_id = clip_ids[idx]
        t0_us = int(t0_us_list[idx])

        try:
            # Load egomotion interpolator from cache (no streaming needed)
            if clip_id != current_clip_id:
                egomotion_interp = avdi.get_clip_feature(clip_id, "egomotion", maybe_stream=False)
                current_clip_id = clip_id

            # Compute history timestamps: t0 - 400ms, t0 - 300ms, t0 - 200ms, t0 - 100ms, t0
            history_timestamps = np.array(
                [t0_us - (NUM_HISTORY_STEPS - 1 - i) * HISTORY_INTERVAL_US for i in range(NUM_HISTORY_STEPS)]
            )

            # Query egomotion at each timestamp
            positions = []
            rotations = []

            for ts in history_timestamps:
                ego_state = egomotion_interp(ts)
                pose = ego_state.pose  # EgomotionState has .pose which is RigidTransform
                positions.append(pose.translation)
                rotations.append(pose.rotation.as_matrix())

            positions = np.array(positions)  # (5, 3)
            rotations = np.array(rotations)  # (5, 3, 3)

            # Transform to ego-centric frame at t0 (last timestamp)
            ref_position = positions[-1]
            ref_rotation = rotations[-1]

            ego_positions, ego_rotations = transform_to_ego_frame(
                positions, rotations, ref_position, ref_rotation
            )

            all_ego_history_xyz[idx] = ego_positions
            all_ego_history_rot[idx] = ego_rotations

        except Exception as e:
            logger.warning(f"Failed sample {idx} (clip {clip_id}): {e}")
            failed_samples.append((idx, clip_id, str(e)))
            # Leave as zeros for failed samples
            continue

    # Save output
    output_path = LABELS_DIR / f"{split}_ego_history.npz"
    np.savez_compressed(
        output_path,
        ego_history_xyz=all_ego_history_xyz,
        ego_history_rot=all_ego_history_rot,
    )

    logger.info(f"Saved ego_history to {output_path}")
    logger.info(f"  ego_history_xyz: {all_ego_history_xyz.shape}")
    logger.info(f"  ego_history_rot: {all_ego_history_rot.shape}")

    if failed_samples:
        logger.warning(f"Failed samples: {len(failed_samples)}")
        failed_path = LABELS_DIR / f"{split}_ego_history_failed.txt"
        with open(failed_path, "w") as f:
            for idx, clip_id, error in failed_samples:
                f.write(f"{idx}\t{clip_id}\t{error}\n")
        logger.warning(f"Failed samples saved to {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ego_history labels")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test", "all"],
        help="Data split to process",
    )
    args = parser.parse_args()

    if args.split == "all":
        for split in ["train", "val", "test"]:
            logger.info(f"\n=== Processing {split} split ===")
            generate_ego_history_for_split(split)
    else:
        generate_ego_history_for_split(args.split)


if __name__ == "__main__":
    main()
