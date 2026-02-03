#!/usr/bin/env python3
"""Extract training frames from HuggingFace to local storage.

Pre-extracts the specific frames needed for training based on clip IDs.
This eliminates network I/O during training.

Processes chunk-by-chunk to fit in limited disk space:
1. Download one chunk
2. Extract frames for all clips in that chunk
3. Delete the chunk to free space
4. Repeat

Can run in parallel with teacher label generation since both use:
- clip_ids from data_splits/clip_ids/
- fixed t0_us = 5_100_000 (5.1s into clip, from Alpamayo default)

Usage:
    # Extract all splits
    python scripts/extract_frames.py --split train
    python scripts/extract_frames.py --split val
    python scripts/extract_frames.py --split test

    # With sharding for parallel extraction
    python scripts/extract_frames.py --split train --shard 0/4
    python scripts/extract_frames.py --split train --shard 1/4
    ...
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
REPO_ROOT = Path(__file__).parent.parent
CLIP_IDS_DIR = REPO_ROOT / "data_splits" / "clip_ids"
OUTPUT_DIR = REPO_ROOT / "extracted_frames"

# Timestamp to sample (from Alpamayo default: 5.1s into clip)
# This allows 1.6s history and 6.4s future prediction
DEFAULT_T0_US = 5_100_000

# Frame extraction parameters (must match dataset.py)
NUM_HISTORY_FRAMES = 4
FRAME_INTERVAL_US = 100_000  # 100ms = 10Hz

# Cameras to extract
CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


def resize_frames(frames: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize frames to target resolution.

    Args:
        frames: (num_frames, H, W, 3) uint8
        resolution: (height, width) target size

    Returns:
        Resized frames (num_frames, new_H, new_W, 3) uint8
    """
    resized = []
    for frame in frames:
        resized.append(cv2.resize(frame, (resolution[1], resolution[0]), interpolation=cv2.INTER_AREA))
    return np.stack(resized)


def load_clip_ids(split: str) -> list[str]:
    """Load clip IDs for the given split from parquet file."""
    clip_ids_path = CLIP_IDS_DIR / f"{split}_clip_ids.parquet"
    if not clip_ids_path.exists():
        raise FileNotFoundError(f"Clip IDs not found: {clip_ids_path}")

    df = pd.read_parquet(clip_ids_path)
    clip_ids = df["clip_id"].tolist()
    logger.info(f"Loaded {len(clip_ids)} clip IDs for split '{split}'")
    return clip_ids


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def delete_chunk_cache(avdi, chunk_id: int) -> None:
    """Delete cached chunk files to free disk space."""
    cache_dir = get_hf_cache_dir() / "hub" / "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"

    # Find and delete chunk files
    for camera in CAMERAS:
        chunk_filename = avdi.features.get_chunk_feature_filename(chunk_id, camera)
        # The file is stored in blobs with a hash name, but we can find it via snapshots
        # Simplest approach: just log that we'd delete, actual deletion is tricky with HF cache
        pass

    # Alternative: delete entire cache periodically (aggressive but works)
    # For now, we'll rely on the fact that each chunk is ~2GB and we have 40GB
    # We can fit ~15-20 chunks, process them, then clear cache
    logger.debug(f"Would delete chunk {chunk_id} cache (not implemented - relying on disk space)")


def extract_frames_for_split(
    split: str,
    shard: str | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
    output_dir: Path = OUTPUT_DIR,
    start_chunk: int | None = None,
    resolution: tuple[int, int] | None = None,
    t0_us: int = DEFAULT_T0_US,
) -> None:
    """Extract frames for a data split, processing chunk-by-chunk.

    Args:
        split: 'train', 'val', or 'test'
        shard: Optional shard specification 'i/n' (e.g., '0/4')
        start_idx: Start sample index (inclusive)
        end_idx: End sample index (exclusive)
        output_dir: Output directory for extracted frames
        start_chunk: Start from this chunk index (for resuming)
        resolution: Optional (height, width) to resize frames. If None, keeps original.
        t0_us: Timestamp in microseconds to sample (default: 5.1s from Alpamayo)
    """
    import physical_ai_av

    # Load clip IDs from data_splits
    clip_ids = load_clip_ids(split)

    logger.info(f"Total samples: {len(clip_ids)}")
    logger.info(f"Using t0_us = {t0_us} ({t0_us / 1_000_000:.1f}s into clip)")

    # Initialize dataset interface
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Apply sharding or index range if specified
    shard_suffix = ""
    if shard:
        shard_idx, num_shards = map(int, shard.split("/"))
        shard_size = len(clip_ids) // num_shards
        start = shard_idx * shard_size
        end = len(clip_ids) if shard_idx == num_shards - 1 else (shard_idx + 1) * shard_size

        clip_ids = clip_ids[start:end]
        sample_indices = list(range(start, end))
        logger.info(f"Shard {shard_idx}/{num_shards}: samples {start}-{end} ({len(clip_ids)} samples)")
        shard_suffix = f"_shard{shard_idx}"
    elif start_idx is not None or end_idx is not None:
        start = start_idx or 0
        end = end_idx or len(clip_ids)
        clip_ids = clip_ids[start:end]
        sample_indices = list(range(start, end))
        logger.info(f"Index range {start}-{end}: {len(clip_ids)} samples")
        shard_suffix = f"_{start}_{end}"
    else:
        sample_indices = list(range(len(clip_ids)))
        logger.info(f"Processing all {len(clip_ids)} samples")

    # Create output directory
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Group clips by chunk
    clip_to_idx = {clip_id: idx for clip_id, idx in zip(clip_ids, sample_indices)}
    chunk_to_clips = {}
    for clip_id in clip_ids:
        chunk_id = avdi.clip_index.at[clip_id, "chunk"]
        if chunk_id not in chunk_to_clips:
            chunk_to_clips[chunk_id] = []
        chunk_to_clips[chunk_id].append(clip_id)

    chunks = sorted(chunk_to_clips.keys())
    logger.info(f"Need {len(chunks)} unique chunks: {chunks[:10]}{'...' if len(chunks) > 10 else ''}")

    # Apply start_chunk for resuming
    if start_chunk is not None:
        chunks = [c for c in chunks if c >= start_chunk]
        logger.info(f"Resuming from chunk {start_chunk}, {len(chunks)} chunks remaining")

    # Compute frame timestamps (same for all samples since t0_us is fixed)
    frame_timestamps = np.array(
        [t0_us - (NUM_HISTORY_FRAMES - 1 - j) * FRAME_INTERVAL_US for j in range(NUM_HISTORY_FRAMES)]
    )
    logger.info(f"Frame timestamps: {frame_timestamps} (relative to clip start)")

    # Track progress
    failed_samples = []
    total_extracted = 0

    # Process chunk by chunk
    for chunk_idx, chunk_id in enumerate(chunks):
        clips_in_chunk = chunk_to_clips[chunk_id]
        logger.info(f"[Chunk {chunk_idx + 1}/{len(chunks)}] Processing chunk {chunk_id} ({len(clips_in_chunk)} clips)")

        # Download this chunk
        logger.info(f"  Downloading chunk {chunk_id}...")
        avdi.download_chunk_features([int(chunk_id)], features=CAMERAS, max_workers=4)
        logger.info(f"  Download complete")

        # Extract frames for all clips in this chunk
        for clip_id in tqdm(clips_in_chunk, desc=f"Chunk {chunk_id}", leave=False):
            sample_idx = clip_to_idx[clip_id]
            sample_path = split_dir / f"sample_{sample_idx:05d}.npz"

            # Skip if already extracted
            if sample_path.exists():
                continue

            try:
                all_frames = []

                for camera in CAMERAS:
                    # Get video reader (reads from downloaded cache)
                    video_reader = avdi.get_clip_feature(clip_id, camera)

                    # Decode frames
                    frames, actual_ts = video_reader.decode_images_from_timestamps(frame_timestamps)
                    # frames: (num_frames, H, W, 3) uint8 RGB

                    video_reader.close()

                    # Resize if resolution specified
                    if resolution is not None:
                        frames = resize_frames(frames, resolution)

                    all_frames.append(frames)

                # Stack: (num_cameras, num_frames, H, W, 3)
                all_frames = np.stack(all_frames)

                # Save as compressed npz
                np.savez_compressed(
                    sample_path,
                    frames=all_frames,  # (4, 4, H, W, 3) uint8
                    clip_id=clip_id,
                    t0_us=t0_us,
                    timestamps=frame_timestamps,
                )
                total_extracted += 1

            except Exception as e:
                logger.warning(f"Failed sample {sample_idx} (clip {clip_id}): {e}")
                failed_samples.append((sample_idx, clip_id, str(e)))
                continue

        # Clear HF cache after processing chunk to free disk space
        logger.info(f"  Clearing cache for chunk {chunk_id}...")
        cache_dir = get_hf_cache_dir() / "hub" / "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
        if cache_dir.exists():
            # Remove blobs to free space (keeps metadata)
            blobs_dir = cache_dir / "blobs"
            if blobs_dir.exists():
                for f in blobs_dir.iterdir():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            logger.info(f"  Cache cleared")

        logger.info(f"[Chunk {chunk_idx + 1}/{len(chunks)}] Done. Extracted: {total_extracted}, Failed: {len(failed_samples)}")

    # Save failed samples list
    if failed_samples:
        failed_path = output_dir / f"{split}_failed{shard_suffix}.txt"
        with open(failed_path, "w") as f:
            for idx, clip_id, error in failed_samples:
                f.write(f"{idx}\t{clip_id}\t{error}\n")
        logger.warning(f"Failed samples saved to {failed_path}")

    logger.info(f"Extraction complete. Output: {split_dir}")
    logger.info(f"Total extracted: {total_extracted}, Failed: {len(failed_samples)}")


def main():
    parser = argparse.ArgumentParser(description="Extract training frames from HuggingFace")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Data split")
    parser.add_argument("--shard", type=str, default=None, help="Shard specification 'i/n' (e.g., '0/4')")
    parser.add_argument("--start-idx", type=int, default=None, help="Start sample index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (exclusive)")
    parser.add_argument("--start-chunk", type=int, default=None, help="Start from chunk index (for resuming)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Target resolution (height=width). Default 224 for training. Use 0 for original resolution.",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=DEFAULT_T0_US,
        help=f"Timestamp in microseconds to sample (default: {DEFAULT_T0_US} = 5.1s from Alpamayo)",
    )
    args = parser.parse_args()

    # Parse resolution
    resolution = (args.resolution, args.resolution) if args.resolution > 0 else None

    logger.info(f"=== Frame Extraction: {args.split} ===")
    if resolution:
        logger.info(f"Resizing to {resolution[0]}x{resolution[1]}")
    else:
        logger.info("Keeping original resolution")

    extract_frames_for_split(
        split=args.split,
        shard=args.shard,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        output_dir=Path(args.output_dir),
        start_chunk=args.start_chunk,
        resolution=resolution,
        t0_us=args.t0_us,
    )


if __name__ == "__main__":
    main()
