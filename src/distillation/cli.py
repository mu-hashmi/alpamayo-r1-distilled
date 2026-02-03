"""Command-line argument parsing for training script."""

from __future__ import annotations

import argparse
from argparse import Namespace

DEFAULT_TRAIN_LABELS = "teacher_labels/train_labels.npz"
DEFAULT_VAL_LABELS = "teacher_labels/val_labels.npz"
DEFAULT_OUTPUT_DIR = "checkpoints"


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description="Train student model")

    # Model
    parser.add_argument(
        "--model", type=str, required=True, choices=["baseline", "reasoning"], help="Model type"
    )
    parser.add_argument("--model-size", type=str, default="500m", help="Model size config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data
    parser.add_argument(
        "--train-labels", type=str, default=DEFAULT_TRAIN_LABELS, help="Path to train labels"
    )
    parser.add_argument(
        "--val-labels", type=str, default=DEFAULT_VAL_LABELS, help="Path to val labels"
    )
    parser.add_argument("--frames-dir", type=str, default="extracted_frames")
    parser.add_argument("--offline", action="store_true", help="Use offline dataset (dummy images)")

    # Data sources
    parser.add_argument(
        "--ego-history-dir", type=str, default="teacher_labels", help="Path to ego_history npz"
    )

    # Benchmark evaluation (nuScenes / Argoverse 2)
    parser.add_argument(
        "--input-type",
        type=str,
        default="camera",
        choices=["camera", "raster", "vector"],
        help="Input modality: camera (default), raster (nuScenes BEV), vector (AV2 tracks+maps)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["nuscenes", "argoverse2"],
        help="Benchmark format for output adaptation and metrics",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=1,
        choices=[1, 6],
        help="Number of prediction modes (1=single-mode, 6=multi-modal)",
    )
    parser.add_argument(
        "--multimodal-loss",
        type=str,
        default="wta",
        choices=["wta", "nll"],
        help="Multi-modal loss type: wta (winner-takes-all) or nll (negative log-likelihood)",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=str,
        default=None,
        help="Path to nuScenes dataset root (required if --benchmark=nuscenes)",
    )
    parser.add_argument(
        "--argoverse2-root",
        type=str,
        default=None,
        help="Path to Argoverse 2 dataset root (required if --benchmark=argoverse2)",
    )

    # Training
    parser.add_argument("--max-epochs", type=int, default=150, help="Maximum number of epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size")
    parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="LR schedule: cosine decay or constant",
    )

    # EMA (Exponential Moving Average)
    parser.add_argument("--ema", action="store_true", help="Use EMA for model weights")
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay rate (default 0.999)"
    )

    # Loss weights
    parser.add_argument("--yaw-weight", type=float, default=0.3, help="Weight for heading loss")
    parser.add_argument(
        "--reasoning-weight", type=float, default=0.25, help="Weight for reasoning loss (lambda)"
    )
    parser.add_argument(
        "--reasoning-warmup-epochs", type=int, default=20, help="Epochs to warm up reasoning weight"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="Contrastive loss temperature"
    )

    # Action supervision (continuous)
    parser.add_argument(
        "--action-weight", type=float, default=0.0, help="Weight for action supervision"
    )
    parser.add_argument(
        "--action-smoothness", type=float, default=0.1, help="Smoothness penalty weight"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled)"
    )
    parser.add_argument(
        "--action-only", action="store_true", help="Train with action loss only (no trajectory)"
    )

    # Discrete action tokens (stop-gradient mode)
    parser.add_argument(
        "--discrete-actions",
        action="store_true",
        help="Use discrete action tokens with stop-gradient",
    )
    parser.add_argument(
        "--num-action-bins", type=int, default=256, help="Number of bins per action dimension"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="Label smoothing for discrete CE loss"
    )

    # Direct trajectory mode (direct trajectory, no unicycle integration)
    parser.add_argument(
        "--no-unicycle",
        action="store_true",
        help="Disable unicycle kinematic model (direct_trajectory mode: direct trajectory output)",
    )

    # Diagnostic options
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Freeze BatchNorm stats (backbone in eval mode during training)",
    )
    parser.add_argument(
        "--overfit-tiny",
        type=int,
        default=0,
        help="Use only N samples for train/val (for debugging overfitting)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Limit training samples (0=unlimited). Useful when frames partially extracted.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Limit validation samples (0=unlimited). Useful when frames partially extracted.",
    )
    parser.add_argument(
        "--log-action-stats",
        action="store_true",
        help="Log action statistics (accel/curvature ranges) during validation",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--save-final-only",
        action="store_true",
        help="Only save final best model (no intermediate checkpoints)",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for faster training"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    return parser


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = create_parser()
    return parser.parse_args()
