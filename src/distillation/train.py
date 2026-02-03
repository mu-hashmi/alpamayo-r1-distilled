"""Training script for student models.

Usage:
    # Train baseline student
    python -m src.distillation.train --model baseline --seed 42

    # Train reasoning student
    python -m src.distillation.train --model reasoning --seed 42 --reasoning-weight 0.25

    # Train with offline dataset (dummy images for testing)
    python -m src.distillation.train --model baseline --offline --epochs 2
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Subset

from .calibration import CameraCalibration, load_calibrations_from_metadata
from .cli import parse_args
from .datasets.physicalai import DEFAULT_CAMERAS, create_dataloaders
from .losses import ActionLoss, CombinedLoss, DiscreteActionLoss, TrajectoryLoss
from .models import BaselineStudent, ReasoningStudent, get_model_config
from .models.baseline_student import BaselineStudentConfig
from .models.reasoning_student import ReasoningStudentConfig
from .training import (
    ExponentialMovingAverage,
    TrainingState,
    get_action_weight,
    get_config_dict,
    get_reasoning_weight,
    make_checkpoint_name,
    make_run_name,
    save_checkpoint,
    train_epoch,
    train_epoch_benchmark,
    train_epoch_discrete,
    validate,
    validate_benchmark,
    validate_discrete,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_calibrations(camera_names: list[str] | None = None) -> list[CameraCalibration]:
    """Load camera calibrations from PhysicalAI-AV dataset."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    camera_names = camera_names or DEFAULT_CAMERAS

    logger.info("Loading camera calibrations from PhysicalAI-AV dataset...")

    intrinsics_path = hf_hub_download(
        repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
        filename="calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet",
        repo_type="dataset",
    )
    extrinsics_path = hf_hub_download(
        repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
        filename="calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet",
        repo_type="dataset",
    )

    intrinsics_df = pd.read_parquet(intrinsics_path)
    extrinsics_df = pd.read_parquet(extrinsics_path)

    intrinsics_df = intrinsics_df.reset_index()
    extrinsics_df = extrinsics_df.reset_index()

    calibrations = load_calibrations_from_metadata(
        camera_names=camera_names,
        clip_id=None,
        intrinsics_df=intrinsics_df,
        extrinsics_df=extrinsics_df,
    )

    logger.info(f"Loaded calibrations for {len(calibrations)} cameras:")
    for calib in calibrations:
        logger.info(f"  {calib.name}: {calib.intrinsics.width}x{calib.intrinsics.height}")

    return calibrations


def create_model(
    model_type: str,
    model_size: str,
    calibrations: list[CameraCalibration] | None,
    use_discrete_actions: bool = False,
    num_action_bins: int = 256,
    use_unicycle: bool = True,
    input_type: str = "camera",
) -> nn.Module:
    """Create student model."""
    model_config = get_model_config(model_size)

    model_config = replace(
        model_config,
        use_unicycle=use_unicycle,
        use_ego_history=True,
    )

    if model_type == "baseline":
        student_config = BaselineStudentConfig.from_model_config(
            model_config,
            use_discrete_actions=use_discrete_actions if use_unicycle else False,
            num_action_bins=num_action_bins,
            input_type=input_type,
        )
        model = BaselineStudent(config=student_config, calibrations=calibrations)
    elif model_type == "reasoning":
        student_config = ReasoningStudentConfig.from_model_config(
            model_config,
            input_type=input_type,
        )
        model = ReasoningStudent(config=student_config, calibrations=calibrations)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def setup_logging(log_dir: Path) -> None:
    """Setup file logging to log_dir."""
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logging.getLogger().addHandler(file_handler)


def freeze_batchnorm(model: nn.Module) -> None:
    """Freeze BatchNorm layers in model."""

    def freeze_bn(module: nn.Module) -> None:
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    model.apply(freeze_bn)

    original_train = model.train

    def train_with_frozen_bn(mode: bool = True) -> nn.Module:
        original_train(mode)
        if mode:
            model.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)
        return model

    model.train = train_with_frozen_bn


def write_eval_json(
    log_dir: Path, metrics: dict, args, epoch: int, elapsed: float, best_path: Path | None
) -> None:
    """Write eval.json for experiment tracking."""
    eval_metrics = {
        "ade": metrics.get("ade", 0),
        "fde": metrics.get("fde", 0),
        "loss": metrics.get("loss", 0),
        "ade_p50": metrics.get("ade_p50", 0),
        "ade_p90": metrics.get("ade_p90", 0),
        "ade_p99": metrics.get("ade_p99", 0),
        "ade_max": metrics.get("ade_max", 0),
        "ade_top10_mean": metrics.get("ade_top10_mean", 0),
        "ade_top50_mean": metrics.get("ade_top50_mean", 0),
        "mean_yaw_error": metrics.get("mean_yaw_error", 0),
        "final_yaw_error": metrics.get("final_yaw_error", 0),
        "target_final_dist_p50": metrics.get("target_final_dist_p50", 0),
        "target_final_dist_p99": metrics.get("target_final_dist_p99", 0),
        "target_final_dist_max": metrics.get("target_final_dist_max", 0),
    }

    if args.num_modes > 1:
        num_modes = args.num_modes
        eval_metrics[f"minADE_{num_modes}"] = metrics.get(f"minADE_{num_modes}", 0)
        eval_metrics[f"minFDE_{num_modes}"] = metrics.get(f"minFDE_{num_modes}", 0)
        eval_metrics["miss_rate_2m"] = metrics.get("miss_rate_2m", 0)
        eval_metrics["brier_minFDE"] = metrics.get("brier_minFDE", 0)

    eval_output = {
        "metrics": eval_metrics,
        "metadata": {
            "best_epoch": epoch,
            "total_epochs": epoch,
            "training_time_seconds": elapsed,
            "checkpoint_path": str(best_path) if best_path else None,
            "config": get_config_dict(args),
        },
    }
    eval_path = log_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    logger.info(f"Wrote eval.json: {eval_path}")


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = make_run_name(args)
    log_dir = Path("logs") / f"{timestamp}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir)
    logger.info(f"Log directory: {log_dir}")

    logger.info(f"=== Training {args.model} student ({args.model_size}, seed {args.seed}) ===")
    logger.info(f"Max epochs: {args.max_epochs}, Early stopping patience: {args.patience}")
    eff = args.batch_size * args.accumulation_steps
    logger.info(f"Batch size: {args.batch_size} x {args.accumulation_steps} (effective: {eff})")
    logger.info(f"Learning rate: {args.lr}")
    if args.model == "reasoning":
        logger.info(f"Reasoning weight (lambda): {args.reasoning_weight}")
        logger.info(f"Reasoning warmup epochs: {args.reasoning_warmup_epochs}")
        logger.info(f"Contrastive temperature: {args.temperature}")
    logger.info(f"Ego history dir: {args.ego_history_dir}")

    # Determine input type based on benchmark
    input_type = args.input_type
    if args.benchmark == "argoverse2":
        input_type = "vector"
        logger.info(f"BENCHMARK MODE: Argoverse 2 (vector input)")
    elif args.benchmark == "nuscenes":
        input_type = "raster"
        logger.info(f"BENCHMARK MODE: nuScenes (raster input)")

    calibrations = None
    if input_type == "camera":
        if args.offline:
            logger.info("BEV encoder: DUMMY (offline testing mode)")
        else:
            calibrations = load_calibrations()
            logger.info("BEV encoder: GEOMETRIC (ground-plane projection)")
    else:
        logger.info(f"Encoder: {input_type.upper()} (benchmark mode, no camera calibrations)")

    use_unicycle = not args.no_unicycle
    if not use_unicycle:
        logger.info("DIRECT TRAJECTORY MODE: Direct trajectory output (no unicycle integration)")
        if args.discrete_actions:
            logger.warning("--discrete-actions ignored in direct_trajectory mode")
        if args.action_weight > 0 or args.action_only:
            logger.warning("--action-weight/--action-only ignored in direct_trajectory mode")

    model = create_model(
        args.model,
        model_size=args.model_size,
        calibrations=calibrations,
        use_discrete_actions=args.discrete_actions,
        num_action_bins=args.num_action_bins,
        use_unicycle=use_unicycle,
        input_type=input_type,
    )
    model = model.to(device)

    if args.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    if args.freeze_bn:
        freeze_batchnorm(model)
        logger.info("BatchNorm layers frozen (eval mode, no gradient)")

    param_counts = model.count_parameters()
    logger.info("Model parameters:")
    for name, count in param_counts.items():
        logger.info(f"  {name}: {count:,}")

    # Create dataloaders based on benchmark or PhysicalAI-AV
    if args.benchmark:
        from .datasets import create_benchmark_dataset, collate_benchmark_batch

        benchmark_root = args.argoverse2_root if args.benchmark == "argoverse2" else args.nuscenes_root
        if benchmark_root is None:
            raise ValueError(f"--{args.benchmark.replace('2', '2-')}root is required for benchmark mode")

        logger.info(f"Loading {args.benchmark} dataset from {benchmark_root}")
        train_dataset = create_benchmark_dataset(
            args.benchmark, benchmark_root, split="train", dummy=args.offline
        )
        val_dataset = create_benchmark_dataset(
            args.benchmark, benchmark_root, split="val", dummy=args.offline
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_benchmark_batch,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_benchmark_batch,
            pin_memory=True,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            train_labels=args.train_labels,
            val_labels=args.val_labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resolution=(224, 224),
            frames_dir=None if args.offline else args.frames_dir,
            offline=args.offline,
            ego_history_dir=args.ego_history_dir,
        )

    if args.overfit_tiny > 0:
        n = args.overfit_tiny
        train_subset = Subset(train_loader.dataset, list(range(min(n, len(train_loader.dataset)))))
        val_subset = Subset(train_loader.dataset, list(range(min(n, len(train_loader.dataset)))))
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        logger.info(f"OVERFIT-TINY MODE: Using {n} samples for both train and val")

    if args.max_train_samples > 0 and args.overfit_tiny == 0:
        n = min(args.max_train_samples, len(train_loader.dataset))
        train_subset = Subset(train_loader.dataset, list(range(n)))
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        logger.info(f"Training limited to {n} samples (--max-train-samples)")

    if args.max_val_samples > 0 and args.overfit_tiny == 0:
        n = min(args.max_val_samples, len(val_loader.dataset))
        val_subset = Subset(val_loader.dataset, list(range(n)))
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        logger.info(f"Validation limited to {n} samples (--max-val-samples)")

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_schedule == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        logger.info("Using constant LR schedule")
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.max_epochs - args.warmup_epochs, eta_min=args.lr / 100
        )
        logger.info("Using cosine annealing LR schedule")

    if args.model == "reasoning":
        loss_fn = CombinedLoss(
            yaw_weight=args.yaw_weight,
            reasoning_weight=args.reasoning_weight,
            temperature=args.temperature,
        )
    else:
        loss_fn = TrajectoryLoss(yaw_weight=args.yaw_weight)

    action_loss_fn = None
    if args.action_weight > 0 or args.action_only:
        action_loss_fn = ActionLoss(smoothness_weight=args.action_smoothness)
        logger.info(
            f"Action supervision enabled: weight={args.action_weight}, "
            f"smoothness={args.action_smoothness}"
        )
        if args.action_only:
            logger.info("Action-only mode: training with action loss only")
        if args.grad_clip > 0:
            logger.info(f"Gradient clipping enabled: max_norm={args.grad_clip}")

    discrete_loss_fn = None
    if args.discrete_actions:
        discrete_loss_fn = DiscreteActionLoss(
            num_bins=args.num_action_bins, label_smoothing=args.label_smoothing
        )
        logger.info(
            f"Discrete action mode: {args.num_action_bins} bins, "
            f"label_smoothing={args.label_smoothing}"
        )
        if args.grad_clip > 0:
            logger.info(f"Gradient clipping enabled: max_norm={args.grad_clip}")

    scaler = GradScaler()

    ema = None
    if args.ema:
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        logger.info(f"EMA enabled with decay={args.ema_decay}")

    state = TrainingState()
    best_path = None

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state.start_epoch = checkpoint["epoch"] + 1
        if "metrics" in checkpoint and "loss" in checkpoint["metrics"]:
            state.best_val_loss = checkpoint["metrics"]["loss"]
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting at {state.start_epoch}")
        logger.info(f"Best val loss so far: {state.best_val_loss:.4f}")

    start_time = time.time()

    for epoch in range(state.start_epoch, args.max_epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.max_epochs} ===")

        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
            logger.info(f"Warmup LR: {warmup_lr:.2e}")
        else:
            scheduler.step()
            logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if args.model == "reasoning":
            current_lambda = get_reasoning_weight(
                epoch, args.reasoning_weight, args.reasoning_warmup_epochs
            )
            loss_fn.reasoning_weight = current_lambda
            logger.info(f"Reasoning lambda: {current_lambda:.4f}")

        # Benchmark mode (AV2 / nuScenes)
        if args.benchmark:
            train_losses = train_epoch_benchmark(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                accumulation_steps=args.accumulation_steps,
                grad_clip=args.grad_clip,
                ema=ema,
            )
            logger.info(
                f"Train - Loss: {train_losses['loss']:.4f}, "
                f"ADE: {train_losses['ade']:.3f}m, FDE: {train_losses['fde']:.3f}m"
            )
            logger.info(
                f"Train - Yaw: mean={train_losses['mean_yaw_error']:.4f}rad, "
                f"final={train_losses['final_yaw_error']:.4f}rad"
            )
        elif args.discrete_actions and use_unicycle:
            train_losses = train_epoch_discrete(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                discrete_loss_fn=discrete_loss_fn,
                scaler=scaler,
                device=device,
                accumulation_steps=args.accumulation_steps,
                grad_clip=args.grad_clip,
                ema=ema,
            )
            logger.info(
                f"Train - CE: {train_losses['loss']:.4f}, "
                f"Acc: {train_losses['exact_accuracy']:.1%}, "
                f"±1: {train_losses['within_1_bin']:.1%}, "
                f"±3: {train_losses['within_3_bins']:.1%}"
            )
            logger.info(f"Train - ADE: {train_losses['ade']:.3f}m, FDE: {train_losses['fde']:.3f}m")
        else:
            current_action_weight = 0.0
            if args.action_weight > 0 or args.action_only:
                current_action_weight = get_action_weight(
                    epoch, args.action_weight, args.max_epochs
                )
                logger.info(f"Action weight: {current_action_weight:.4f}")

            train_losses = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scaler=scaler,
                device=device,
                model_type=args.model,
                accumulation_steps=args.accumulation_steps,
                action_loss_fn=action_loss_fn,
                action_weight=current_action_weight,
                grad_clip=args.grad_clip,
                action_only=args.action_only,
                ema=ema,
                use_unicycle=use_unicycle,
            )

            logger.info(
                f"Train - Loss: {train_losses['loss']:.4f}, "
                f"ADE: {train_losses['ade']:.3f}m, FDE: {train_losses['fde']:.3f}m"
            )
            logger.info(
                f"Train - Yaw: mean={train_losses['mean_yaw_error']:.4f}rad, "
                f"final={train_losses['final_yaw_error']:.4f}rad"
            )
            if args.model == "reasoning":
                logger.info(f"Train - Reasoning Loss: {train_losses.get('reasoning_loss', 0):.4f}")
            if "action_loss" in train_losses:
                logger.info(
                    f"Train - Action Loss: {train_losses['action_loss']:.4f} "
                    f"(MSE: {train_losses['action_mse']:.4f}, "
                    f"Smooth: {train_losses['action_smooth']:.4f})"
                )

        if ema is not None:
            ema.apply(model)

        # Validation
        if args.benchmark:
            val_results = validate_benchmark(
                model=model,
                val_loader=val_loader,
                device=device,
            )
        elif args.discrete_actions and use_unicycle:
            val_results = validate_discrete(
                model=model,
                val_loader=val_loader,
                discrete_loss_fn=discrete_loss_fn,
                device=device,
            )
            logger.info(
                f"Val - CE: {val_results['loss']:.4f}, "
                f"Acc: {val_results['exact_accuracy']:.1%}, "
                f"±1: {val_results['within_1_bin']:.1%}"
            )
        else:
            val_results = validate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                model_type=args.model,
                use_unicycle=use_unicycle,
            )

        logger.info(
            f"Val - Loss: {val_results['loss']:.4f}, "
            f"ADE: {val_results['ade']:.3f}m, FDE: {val_results['fde']:.3f}m"
        )
        logger.info(
            f"Val - ADE percentiles: p50={val_results['ade_p50']:.2f}m, "
            f"p90={val_results['ade_p90']:.2f}m, p99={val_results['ade_p99']:.2f}m, "
            f"max={val_results['ade_max']:.2f}m"
        )
        logger.info(
            f"Val - ADE top-k: top10={val_results['ade_top10_mean']:.2f}m, "
            f"top50={val_results['ade_top50_mean']:.2f}m"
        )
        logger.info(
            f"Val - Target dist: p50={val_results['target_final_dist_p50']:.1f}m, "
            f"p99={val_results['target_final_dist_p99']:.1f}m, "
            f"max={val_results['target_final_dist_max']:.1f}m"
        )
        logger.info(
            f"Val - Yaw: mean={val_results['mean_yaw_error']:.4f}rad, "
            f"final={val_results['final_yaw_error']:.4f}rad"
        )

        model_state = None
        if args.save_final_only:
            model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        is_best = state.update_best(val_results["loss"], val_results, model_state)
        if is_best:
            state.best_epoch = epoch
            if args.save_final_only:
                logger.info(f"New best at epoch {epoch}: ADE={val_results['ade']:.3f}m")
            else:
                best_path = output_dir / make_checkpoint_name(args, epoch=epoch, suffix="best")
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_results,
                    "config": get_config_dict(args),
                }
                torch.save(best_checkpoint, best_path)
                logger.info(f"New best model saved: {best_path}")
        else:
            logger.info(
                f"No improvement for {state.epochs_without_improvement}/{args.patience} epochs"
            )

        if not args.save_final_only and epoch % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_results,
                output_dir=output_dir,
                args=args,
            )

        if ema is not None:
            ema.restore(model)

        if state.should_stop(args.patience):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    logger.info("\n=== Training Complete ===")
    logger.info(f"Total time: {elapsed / 3600:.2f}h")
    logger.info(f"Best val loss: {state.best_val_loss:.4f}")

    if args.save_final_only:
        if state.best_state_dict is not None:
            ckpt_name = make_checkpoint_name(args, epoch=state.best_epoch, suffix="best")
            best_path = output_dir / ckpt_name
            best_checkpoint = {
                "model_state_dict": state.best_state_dict,
                "epoch": state.best_epoch,
                "metrics": state.best_metrics,
                "config": get_config_dict(args),
            }
            torch.save(best_checkpoint, best_path)
            logger.info(f"Saved best model (epoch {state.best_epoch}): {best_path}")
            if state.best_metrics:
                logger.info(
                    f"Best metrics: ADE={state.best_metrics['ade']:.3f}m, "
                    f"FDE={state.best_metrics['fde']:.3f}m"
                )
        else:
            logger.warning("No model saved - no improvement during training")

    logger.info(f"Output directory: {output_dir}")

    if state.best_metrics:
        write_eval_json(log_dir, state.best_metrics, args, state.best_epoch, elapsed, best_path)


if __name__ == "__main__":
    main()
