"""Weight schedules for training."""

from __future__ import annotations


def get_reasoning_weight(epoch: int, target_weight: float, warmup_epochs: int = 20) -> float:
    """Get reasoning weight with linear warmup.

    Linearly increases reasoning weight from 0 to target_weight over warmup_epochs.
    This prevents early divergence when reasoning loss dominates.

    Args:
        epoch: Current epoch (1-indexed)
        target_weight: Target reasoning weight
        warmup_epochs: Number of epochs to warm up over

    Returns:
        Current reasoning weight
    """
    if epoch <= warmup_epochs:
        return target_weight * (epoch / warmup_epochs)
    return target_weight


def get_action_weight(epoch: int, target_weight: float, total_epochs: int) -> float:
    """Get action weight with early emphasis and decay.

    Starts at 2x target_weight and decays to target_weight over first half of training.
    This provides strong action supervision early when gradients through integration are weak.

    Args:
        epoch: Current epoch (1-indexed)
        target_weight: Target action weight
        total_epochs: Total number of training epochs

    Returns:
        Current action weight
    """
    decay = max(0.5, 1.0 - epoch / (total_epochs * 0.5))
    return target_weight * (1.0 + decay)
