#!/usr/bin/env python3
"""Phase 2: Teacher Label Generation for Alpamayo Distillation.

Runs Alpamayo-R1 inference on curated clips and saves:
- Trajectories: 64 waypoints × (x, y, z) + rotation matrix
- CoC reasoning traces: Plain text strings
- Reasoning embeddings: 384-dim sentence-transformers embeddings

Usage:
    # Run on GPU instance (requires 24GB+ VRAM)
    python scripts/generate_teacher_labels.py --split train
    python scripts/generate_teacher_labels.py --split val
    python scripts/generate_teacher_labels.py --split test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add external repos to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "alpamayo" / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "physical_ai_av" / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration (defaults, can be overridden via CLI)
DEFAULT_DATA_SPLITS_DIR = REPO_ROOT / "data_splits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "teacher_labels"
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N clips

# Model parameters
MODEL_ID = "nvidia/Alpamayo-R1-10B"
DTYPE = torch.bfloat16

# Inference parameters
TOP_P = 0.98
TEMPERATURE = 0.6
MAX_GENERATION_LENGTH = 256

def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    import os
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def clear_hf_cache():
    """Clear HuggingFace dataset cache to free disk space."""
    cache_dir = get_hf_cache_dir() / "hub" / "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    if cache_dir.exists():
        blobs_dir = cache_dir / "blobs"
        if blobs_dir.exists():
            for f in blobs_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
        logger.info("  Cache cleared")


# Timestamps to sample per clip (in microseconds)
# Default: single sample at 5.1s into clip
# This allows 1.6s history and 6.4s future (total clip needs ~13s)
DEFAULT_T0_US = 5_100_000


def load_clip_ids(split: str, data_dir: Path) -> list[str]:
    """Load clip IDs for the given split."""
    clip_ids_path = data_dir / "clip_ids" / f"{split}_clip_ids.parquet"
    if not clip_ids_path.exists():
        raise FileNotFoundError(f"Clip IDs not found: {clip_ids_path}")

    df = pd.read_parquet(clip_ids_path)
    clip_ids = df["clip_id"].tolist()
    logger.info(f"Loaded {len(clip_ids)} clip IDs for split '{split}'")
    return clip_ids


def load_model():
    """Load Alpamayo-R1 model."""
    logger.info(f"Loading model: {MODEL_ID}")
    logger.info("This may take a few minutes for initial download (~22GB)...")

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=DTYPE)
    model = model.to("cuda")
    model.eval()

    logger.info("Model loaded successfully")
    return model


def load_embedding_model():
    """Load sentence-transformers model for CoC embedding."""
    logger.info("Loading sentence-transformers model...")

    from sentence_transformers import SentenceTransformer

    # all-MiniLM-L6-v2 produces 384-dim embeddings
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_model = embed_model.to("cuda")

    logger.info("Embedding model loaded")
    return embed_model


def run_inference(
    model,
    clip_id: str,
    t0_us: int,
    avdi,
) -> dict | None:
    """Run Alpamayo-R1 inference on a single clip.

    Returns:
        Dictionary with trajectory and reasoning data, or None if failed.
    """
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    try:
        # Load data (from downloaded cache, NOT streaming)
        data = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=t0_us,
            avdi=avdi,
            maybe_stream=False,
        )

        # Create message with images
        frames = data["image_frames"].flatten(0, 1)  # (N_cameras * num_frames, 3, H, W)
        messages = helper.create_message(frames)

        # Get processor
        processor = helper.get_processor(model.tokenizer)

        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Prepare model inputs
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")

        # Run inference
        with torch.autocast("cuda", dtype=DTYPE):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                num_traj_samples=1,
                max_generation_length=MAX_GENERATION_LENGTH,
                return_extra=True,
            )

        # Extract outputs
        # pred_xyz: (B=1, num_traj_sets=1, num_samples=1, num_steps=64, 3)
        # pred_rot: (B=1, num_traj_sets=1, num_samples=1, num_steps=64, 3, 3)
        trajectory_xyz = pred_xyz[0, 0, 0].cpu().numpy()  # (64, 3)
        trajectory_rot = pred_rot[0, 0, 0].cpu().numpy()  # (64, 3, 3)

        # Extract CoC text
        coc_text = extra.get("cot", [[[""]]]) if extra else [[[""]]]
        coc_text = coc_text[0][0][0] if coc_text else ""

        return {
            "clip_id": clip_id,
            "t0_us": t0_us,
            "trajectory_xyz": trajectory_xyz,
            "trajectory_rot": trajectory_rot,
            "coc_text": coc_text,
        }

    except Exception as e:
        logger.warning(f"Failed to process clip {clip_id}: {e}")
        return None


def embed_coc_text(embed_model, texts: list[str]) -> np.ndarray:
    """Embed CoC texts using sentence-transformers.

    Args:
        embed_model: SentenceTransformer model
        texts: List of CoC text strings

    Returns:
        Array of shape (N, 384) with embeddings
    """
    # Handle empty strings
    texts = [t if t else "No reasoning provided." for t in texts]

    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings


def save_results(results: list[dict], output_path: Path) -> None:
    """Save results to npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Organize data for saving
    clip_ids = [r["clip_id"] for r in results]
    t0_us_list = [r["t0_us"] for r in results]
    trajectories_xyz = np.stack([r["trajectory_xyz"] for r in results])  # (N, 64, 3)
    trajectories_rot = np.stack([r["trajectory_rot"] for r in results])  # (N, 64, 3, 3)
    coc_texts = [r["coc_text"] for r in results]
    coc_embeddings = np.stack([r["coc_embedding"] for r in results])  # (N, 384)

    np.savez_compressed(
        output_path,
        clip_ids=np.array(clip_ids, dtype=object),
        t0_us=np.array(t0_us_list, dtype=np.int64),
        trajectory_xyz=trajectories_xyz.astype(np.float32),
        trajectory_rot=trajectories_rot.astype(np.float32),
        coc_texts=np.array(coc_texts, dtype=object),
        coc_embeddings=coc_embeddings.astype(np.float32),
    )

    logger.info(f"Saved {len(results)} samples to {output_path}")


def load_checkpoint(checkpoint_path: Path) -> tuple[set[str], list[dict]]:
    """Load checkpoint if exists.

    Returns:
        Tuple of (processed_clip_ids, results)
    """
    if not checkpoint_path.exists():
        return set(), []

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    data = np.load(checkpoint_path, allow_pickle=True)

    processed_ids = set(data["clip_ids"].tolist())
    results = []

    for i in range(len(data["clip_ids"])):
        results.append(
            {
                "clip_id": data["clip_ids"][i],
                "t0_us": int(data["t0_us"][i]),
                "trajectory_xyz": data["trajectory_xyz"][i],
                "trajectory_rot": data["trajectory_rot"][i],
                "coc_text": data["coc_texts"][i],
                "coc_embedding": data["coc_embeddings"][i],
            }
        )

    logger.info(f"Loaded {len(results)} samples from checkpoint")
    return processed_ids, results


def main():
    parser = argparse.ArgumentParser(description="Generate teacher labels using Alpamayo-R1")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Data split to process",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=DEFAULT_T0_US,
        help=f"Timestamp in microseconds to sample (default: {DEFAULT_T0_US})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to process (for testing)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for clip processing (for multi-GPU parallelism)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for clip processing (for multi-GPU parallelism)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard specification as 'i/n' (e.g., '0/2' for first half, '1/2' for second half)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_SPLITS_DIR,
        help=f"Data splits directory (default: {DEFAULT_DATA_SPLITS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for labels (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Parse shard argument and determine output suffix
    shard_idx, num_shards = None, None
    if args.shard:
        shard_idx, num_shards = map(int, args.shard.split("/"))
        logger.info(f"=== Phase 2: Teacher Label Generation ({args.split}, shard {shard_idx}/{num_shards}) ===")
        shard_suffix = f"_shard{shard_idx}"
    elif args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or "end"
        logger.info(f"=== Phase 2: Teacher Label Generation ({args.split}, indices {start}-{end}) ===")
        shard_suffix = f"_{start}_{end}"
    else:
        logger.info(f"=== Phase 2: Teacher Label Generation ({args.split}) ===")
        shard_suffix = ""

    # Setup output paths
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_labels{shard_suffix}.npz"
    checkpoint_path = output_dir / f"{args.split}_checkpoint{shard_suffix}.npz"

    # Load checkpoint if resuming
    if args.resume:
        processed_ids, results = load_checkpoint(checkpoint_path)
    else:
        processed_ids, results = set(), []

    # Load clip IDs
    clip_ids = load_clip_ids(args.split, args.data_dir)

    # Apply sharding or index range
    if args.shard:
        # Split into num_shards equal parts, take shard_idx
        shard_size = len(clip_ids) // num_shards
        start = shard_idx * shard_size
        # Last shard gets any remainder
        end = len(clip_ids) if shard_idx == num_shards - 1 else (shard_idx + 1) * shard_size
        clip_ids = clip_ids[start:end]
        logger.info(f"Shard {shard_idx}/{num_shards}: clips {start}-{end} ({len(clip_ids)} clips)")
    elif args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(clip_ids)
        clip_ids = clip_ids[start:end]
        logger.info(f"Index range {start}-{end}: {len(clip_ids)} clips")

    if args.limit:
        clip_ids = clip_ids[: args.limit]
        logger.info(f"Limited to {args.limit} clips")

    # Filter out already processed clips
    remaining_clips = [c for c in clip_ids if c not in processed_ids]
    logger.info(f"Remaining clips to process: {len(remaining_clips)}")

    if not remaining_clips:
        logger.info("All clips already processed!")
        if results:
            save_results(results, output_path)
        return

    # Load models
    model = load_model()
    embed_model = load_embedding_model()

    # Initialize dataset interface
    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Features needed for Alpamayo inference
    REQUIRED_FEATURES = [
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "egomotion",
    ]

    # Group clips by chunk for chunk-by-chunk processing
    logger.info(f"Grouping {len(remaining_clips)} clips by chunk...")
    chunk_to_clips = {}
    for clip_id in remaining_clips:
        chunk_id = avdi.clip_index.at[clip_id, "chunk"]
        if chunk_id not in chunk_to_clips:
            chunk_to_clips[chunk_id] = []
        chunk_to_clips[chunk_id].append(clip_id)

    chunks = sorted(chunk_to_clips.keys())
    logger.info(f"Need {len(chunks)} unique chunks: {chunks[:5]}{'...' if len(chunks) > 5 else ''}")

    # Process clips chunk by chunk
    start_time = time.time()
    failed_clips = []
    pending_results = []
    total_processed = 0

    try:
        for chunk_idx, chunk_id in enumerate(chunks):
            clips_in_chunk = chunk_to_clips[chunk_id]
            logger.info(f"[Chunk {chunk_idx + 1}/{len(chunks)}] Processing chunk {chunk_id} ({len(clips_in_chunk)} clips)")

            # Download just this chunk
            logger.info(f"  Downloading chunk {chunk_id}...")
            avdi.download_chunk_features([int(chunk_id)], features=REQUIRED_FEATURES, max_workers=4)
            logger.info(f"  Download complete")

            # Process clips in this chunk
            for clip_id in tqdm(clips_in_chunk, desc=f"Chunk {chunk_id}", leave=False):
                # Run inference
                result = run_inference(model, clip_id, args.t0_us, avdi)

                if result is None:
                    failed_clips.append(clip_id)
                    continue

                pending_results.append(result)
                total_processed += 1

                # Batch embed CoC texts and save checkpoint periodically
                if len(pending_results) >= CHECKPOINT_INTERVAL:
                    # Embed CoC texts
                    coc_texts = [r["coc_text"] for r in pending_results]
                    embeddings = embed_coc_text(embed_model, coc_texts)
                    for j, emb in enumerate(embeddings):
                        pending_results[j]["coc_embedding"] = emb

                    results.extend(pending_results)
                    pending_results = []

                    # Save checkpoint
                    save_results(results, checkpoint_path)

                    elapsed = time.time() - start_time
                    clips_per_sec = total_processed / elapsed
                    eta = (len(remaining_clips) - total_processed) / clips_per_sec / 3600
                    logger.info(
                        f"Progress: {total_processed}/{len(remaining_clips)} | "
                        f"Speed: {clips_per_sec:.2f} clips/s | "
                        f"ETA: {eta:.1f}h"
                    )

            # Clear HF cache after processing chunk to free disk space
            logger.info(f"  Clearing cache for chunk {chunk_id}...")
            clear_hf_cache()

            logger.info(f"[Chunk {chunk_idx + 1}/{len(chunks)}] Done. Processed: {total_processed}, Failed: {len(failed_clips)}")

        # Process remaining pending results
        if pending_results:
            coc_texts = [r["coc_text"] for r in pending_results]
            embeddings = embed_coc_text(embed_model, coc_texts)
            for j, emb in enumerate(embeddings):
                pending_results[j]["coc_embedding"] = emb
            results.extend(pending_results)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving checkpoint...")
        # Save any pending results
        if pending_results:
            coc_texts = [r["coc_text"] for r in pending_results]
            embeddings = embed_coc_text(embed_model, coc_texts)
            for j, emb in enumerate(embeddings):
                pending_results[j]["coc_embedding"] = emb
            results.extend(pending_results)
        save_results(results, checkpoint_path)
        raise

    # Save final results
    save_results(results, output_path)

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"=== Phase 2 Complete ({args.split}) ===")
    logger.info(f"Total time: {elapsed / 3600:.2f}h")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {len(failed_clips)}")
    logger.info(f"Output: {output_path}")

    if failed_clips:
        failed_path = output_dir / f"{args.split}_failed_clips.json"
        with open(failed_path, "w") as f:
            json.dump(failed_clips, f)
        logger.info(f"Failed clips saved to: {failed_path}")


if __name__ == "__main__":
    main()
