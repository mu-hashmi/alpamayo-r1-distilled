"""Training state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingState:
    """Encapsulates training state for early stopping and best model tracking."""

    best_val_loss: float = float("inf")
    best_metrics: dict[str, Any] = field(default_factory=dict)
    best_state_dict: dict[str, Any] | None = None
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    start_epoch: int = 1

    def update_best(
        self,
        val_loss: float,
        metrics: dict[str, Any],
        state_dict: dict[str, Any] | None = None,
    ) -> bool:
        """Update best model if val_loss improved.

        Args:
            val_loss: Validation loss for this epoch
            metrics: Full metrics dict for this epoch
            state_dict: Model state dict (optional, for save_final_only mode)

        Returns:
            True if this is a new best, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_metrics = metrics.copy()
            self.best_state_dict = state_dict
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_stop(self, patience: int) -> bool:
        """Check if early stopping should be triggered.

        Args:
            patience: Number of epochs without improvement before stopping

        Returns:
            True if should stop, False otherwise
        """
        return self.epochs_without_improvement >= patience
