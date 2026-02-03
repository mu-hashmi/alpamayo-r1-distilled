#!/usr/bin/env python3
"""Export trained model to HuggingFace format.

Usage:
    # Export checkpoint to HuggingFace-compatible format
    python -m src.distillation.export \
        --checkpoint checkpoints/reasoning_seed42_best.pt \
        --output-dir exported_models/alpamayo-r1-distilled-500m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .models import get_model_config
from .models.baseline_student import BaselineStudent
from .models.reasoning_student import ReasoningStudent


def export_model(
    checkpoint_path: str,
    output_dir: str,
    model_size: str = "500m",
    model_type: str | None = None,
    use_ego_history: bool | None = None,
) -> None:
    """Export checkpoint to HuggingFace-compatible format.

    Creates:
        - config.json: Model configuration
        - pytorch_model.bin: Model weights (PyTorch format)
        - model.safetensors: Model weights (safetensors format, if available)

    Args:
        checkpoint_path: Path to training checkpoint
        output_dir: Output directory for exported model
        model_size: Model size preset (500m)
        model_type: Override model type detection (baseline, reasoning)
        use_ego_history: Override ego history detection
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Determine model configuration
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        detected_type = checkpoint.get("model_type", "baseline")
        metrics = checkpoint.get("metrics", {})
    else:
        state_dict = checkpoint
        detected_type = "baseline"
        metrics = {}

    # Use detected or override values
    final_model_type = model_type or detected_type

    # Detect ego history from state dict
    has_ego_history = "decoder.ego_encoder.mlp.0.weight" in state_dict
    final_use_ego_history = use_ego_history if use_ego_history is not None else has_ego_history

    # Get model config
    model_config = get_model_config(model_size)

    print(f"Model type: {final_model_type}")
    print(f"Model size: {model_size}")
    print(f"Ego history: {final_use_ego_history}")
    print(f"Parameters: {model_config.d_model}d, {model_config.n_layers} layers, {model_config.backbone}")

    # Create model instance
    if final_model_type == "reasoning":
        model = ReasoningStudent.from_model_config(
            model_config,
            calibrations=None,
            use_ego_history=final_use_ego_history,
        )
    else:
        model = BaselineStudent.from_model_config(
            model_config,
            calibrations=None,
            use_ego_history=final_use_ego_history,
        )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    # Count parameters
    param_counts = model.count_parameters()
    total_params = param_counts.get("total", sum(p.numel() for p in model.parameters()))

    # Create config.json
    config = {
        "model_type": final_model_type,
        "model_size": model_size,
        "architecture": {
            "backbone": model_config.backbone,
            "d_model": model_config.d_model,
            "n_heads": model_config.n_heads,
            "n_layers": model_config.n_layers,
            "ffn_dim": model_config.ffn_dim,
            "bev_channels": model_config.bev_channels,
            "num_waypoints": 64,
            "use_ego_history": final_use_ego_history,
        },
        "input": {
            "num_cameras": 4,
            "num_frames": 4,
            "resolution": [224, 224],
            "channels": 12,  # 4 frames * 3 RGB
        },
        "output": {
            "num_waypoints": 64,
            "waypoint_dim": 4,  # x, y, sin_yaw, cos_yaw
            "frequency_hz": 10,
            "horizon_seconds": 6.4,
        },
        "training": {
            "metrics": metrics,
        },
        "num_parameters": total_params,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Save PyTorch format
    pytorch_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), pytorch_path)
    print(f"Saved PyTorch weights: {pytorch_path}")

    # Try to save safetensors format
    try:
        from safetensors.torch import save_file

        safetensors_path = output_dir / "model.safetensors"
        save_file(model.state_dict(), safetensors_path)
        print(f"Saved safetensors weights: {safetensors_path}")
    except ImportError:
        print("safetensors not installed, skipping .safetensors export")
        print("Install with: pip install safetensors")

    print(f"\nExport complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total parameters: {total_params:,}")


def main():
    parser = argparse.ArgumentParser(description="Export model to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model-size", type=str, default="500m", help="Model size")
    parser.add_argument("--model-type", type=str, choices=["baseline", "reasoning"], help="Override model type")
    parser.add_argument("--use-ego-history", action="store_true", help="Force ego history enabled")
    parser.add_argument("--no-ego-history", action="store_true", help="Force ego history disabled")
    args = parser.parse_args()

    use_ego_history = None
    if args.use_ego_history:
        use_ego_history = True
    elif args.no_ego_history:
        use_ego_history = False

    export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_size=args.model_size,
        model_type=args.model_type,
        use_ego_history=use_ego_history,
    )


if __name__ == "__main__":
    main()
