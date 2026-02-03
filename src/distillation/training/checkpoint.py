"""Checkpoint utilities for saving and loading model state."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def make_run_name(args: Namespace) -> str:
    """Generate descriptive run name from training config."""
    parts = [args.model, args.model_size]

    lr_exp = f"{args.lr:.0e}"
    if "e-" in lr_exp:
        lr_str = lr_exp.replace("e-0", "em").replace("e-", "em")
    else:
        lr_str = lr_exp.replace("e+0", "e").replace("e+", "e").replace("e0", "e")
    parts.append(f"lr{lr_str}")

    if args.ema:
        parts.append("ema")
    if args.freeze_bn:
        parts.append("frozenbn")
    if getattr(args, "discrete_actions", False):
        parts.append("discrete")
    if getattr(args, "no_unicycle", False):
        parts.append("direct")
    if getattr(args, "action_weight", 0) > 0:
        parts.append(f"actw{args.action_weight}")

    return "_".join(parts)


def make_checkpoint_name(args: Namespace, epoch: int | None = None, suffix: str = "best") -> str:
    """Generate descriptive checkpoint filename from training config."""
    parts = [args.model, args.model_size]

    lr_exp = f"{args.lr:.0e}"
    if "e-" in lr_exp:
        lr_str = lr_exp.replace("e-0", "em").replace("e-", "em")
    else:
        lr_str = lr_exp.replace("e+0", "e").replace("e+", "e").replace("e0", "e")
    parts.append(f"lr{lr_str}")

    if args.ema:
        parts.append("ema")
    if args.freeze_bn:
        parts.append("frozenbn")
    if getattr(args, "discrete_actions", False):
        parts.append("discrete")
    if getattr(args, "no_unicycle", False):
        parts.append("direct")
    if getattr(args, "action_weight", 0) > 0:
        parts.append(f"actw{args.action_weight}")

    if epoch is not None:
        parts.append(f"ep{epoch}")
    parts.append(suffix)

    return "_".join(parts) + ".pt"


def get_config_dict(args: Namespace) -> dict[str, Any]:
    """Extract config dict from args for checkpoint metadata."""
    return {
        "model": args.model,
        "model_size": args.model_size,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "ema": args.ema,
        "ema_decay": getattr(args, "ema_decay", 0.999),
        "freeze_bn": args.freeze_bn,
        "discrete_actions": getattr(args, "discrete_actions", False),
        "no_unicycle": getattr(args, "no_unicycle", False),
        "action_weight": getattr(args, "action_weight", 0),
        "grad_clip": getattr(args, "grad_clip", 0),
        "input_type": getattr(args, "input_type", "camera"),
        "benchmark": getattr(args, "benchmark", None),
        "num_modes": getattr(args, "num_modes", 1),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, Any],
    output_dir: Path,
    args: Namespace,
) -> Path:
    """Save training checkpoint with full config metadata.

    Returns:
        Path to saved checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": get_config_dict(args),
    }

    checkpoint_path = output_dir / make_checkpoint_name(args, epoch=epoch, suffix="checkpoint")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    latest_path = output_dir / make_checkpoint_name(args, suffix="latest")
    torch.save(checkpoint, latest_path)

    return checkpoint_path
