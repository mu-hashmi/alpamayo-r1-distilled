"""Exponential Moving Average for model weights."""

from __future__ import annotations

import torch.nn as nn


class ExponentialMovingAverage:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights that is updated with exponential
    moving average after each training step. This smooths out weight oscillations
    and often produces better final models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, nn.Parameter] = {}
        self.backup: dict[str, nn.Parameter] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model: nn.Module) -> None:
        """Apply shadow weights to model (backup original weights first)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


# Alias for backwards compatibility
EMA = ExponentialMovingAverage
