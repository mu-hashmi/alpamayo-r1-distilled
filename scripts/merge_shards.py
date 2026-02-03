#!/usr/bin/env python3
"""Merge sharded teacher label files into a single file.

Usage:
    uv run python scripts/merge_shards.py --split train --num-shards 2
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "teacher_labels"


def merge_shards(split: str, num_shards: int) -> None:
    """Merge sharded npz files into a single file."""
    all_data = {
        "clip_ids": [],
        "t0_us": [],
        "trajectory_xyz": [],
        "trajectory_rot": [],
        "coc_texts": [],
        "coc_embeddings": [],
    }

    for shard_idx in range(num_shards):
        shard_path = OUTPUT_DIR / f"{split}_labels_shard{shard_idx}.npz"
        if not shard_path.exists():
            logger.error(f"Shard file not found: {shard_path}")
            raise FileNotFoundError(shard_path)

        logger.info(f"Loading shard {shard_idx}: {shard_path}")
        data = np.load(shard_path, allow_pickle=True)

        all_data["clip_ids"].extend(data["clip_ids"].tolist())
        all_data["t0_us"].append(data["t0_us"])
        all_data["trajectory_xyz"].append(data["trajectory_xyz"])
        all_data["trajectory_rot"].append(data["trajectory_rot"])
        all_data["coc_texts"].extend(data["coc_texts"].tolist())
        all_data["coc_embeddings"].append(data["coc_embeddings"])

        logger.info(f"  - {len(data['clip_ids'])} samples")

    # Concatenate arrays
    merged = {
        "clip_ids": np.array(all_data["clip_ids"], dtype=object),
        "t0_us": np.concatenate(all_data["t0_us"]),
        "trajectory_xyz": np.concatenate(all_data["trajectory_xyz"]),
        "trajectory_rot": np.concatenate(all_data["trajectory_rot"]),
        "coc_texts": np.array(all_data["coc_texts"], dtype=object),
        "coc_embeddings": np.concatenate(all_data["coc_embeddings"]),
    }

    # Save merged file
    output_path = OUTPUT_DIR / f"{split}_labels.npz"
    np.savez_compressed(output_path, **merged)

    logger.info(f"Merged {len(merged['clip_ids'])} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge sharded teacher label files")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Data split to merge",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shards to merge",
    )
    args = parser.parse_args()

    merge_shards(args.split, args.num_shards)


if __name__ == "__main__":
    main()
