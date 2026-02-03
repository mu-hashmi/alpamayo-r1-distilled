"""Training epoch loops."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ema import ExponentialMovingAverage
from ..losses import ActionLoss, DiscreteActionLoss, MetricsComputer


def prepare_ego_history(batch: dict, device: torch.device) -> torch.Tensor | None:
    """Prepare ego history tensor from batch.

    Combines ego_history_xyz and ego_history_rot into a single tensor.

    Args:
        batch: Batch from dataloader
        device: Device to move tensor to

    Returns:
        Ego history tensor (B, num_steps, 6) or None if not available
    """
    if "ego_history_xyz" not in batch:
        return None

    ego_xyz = batch["ego_history_xyz"].to(device)
    ego_rot = batch["ego_history_rot"].to(device)

    R = ego_rot
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    pitch = torch.asin(-R[..., 2, 0].clamp(-1, 1))
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])

    rot_angles = torch.stack([roll, pitch, yaw], dim=-1)
    ego_history = torch.cat([ego_xyz, rot_angles], dim=-1)

    return ego_history


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    model_type: str,
    accumulation_steps: int = 1,
    action_loss_fn: ActionLoss | None = None,
    action_weight: float = 0.0,
    grad_clip: float = 0.0,
    action_only: bool = False,
    ema: ExponentialMovingAverage | None = None,
    use_unicycle: bool = True,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Student model
        train_loader: Training dataloader
        optimizer: Optimizer
        loss_fn: Trajectory/combined loss function
        scaler: GradScaler for mixed precision
        device: Device
        model_type: 'baseline' or 'reasoning'
        accumulation_steps: Gradient accumulation steps
        action_loss_fn: ActionLoss instance (optional)
        action_weight: Weight for action supervision (0 = disabled)
        grad_clip: Max gradient norm for clipping (0 = disabled)
        action_only: Train with action loss only (skip trajectory integration)
        ema: EMA instance for weight averaging (optional)
        use_unicycle: If False, direct_trajectory mode (no action supervision)

    Returns:
        Dictionary of averaged losses
    """
    model.train()
    total_losses: dict[str, float] = {}
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(device)
        target_xyz = batch["trajectory_xyz"].to(device)
        target_rot = batch["trajectory_rot"].to(device)
        teacher_emb = batch["coc_embedding"].to(device)

        ego_history = prepare_ego_history(batch, device)
        ego_history_xyz = batch["ego_history_xyz"].to(device)
        ego_history_rot = batch["ego_history_rot"].to(device)

        with autocast(dtype=torch.bfloat16):
            use_action_supervision = use_unicycle and (action_weight > 0 or action_only)

            if model_type == "reasoning":
                if use_action_supervision:
                    pred_traj, student_emb, pred_actions = model(
                        images,
                        ego_history,
                        ego_history_xyz,
                        ego_history_rot,
                        return_reasoning=True,
                        return_actions=True,
                    )
                else:
                    pred_traj, student_emb = model(
                        images,
                        ego_history,
                        ego_history_xyz,
                        ego_history_rot,
                        return_reasoning=True,
                    )
                    pred_actions = None
            else:
                if use_action_supervision:
                    pred_traj, pred_actions = model(
                        images,
                        ego_history,
                        ego_history_xyz,
                        ego_history_rot,
                        return_actions=True,
                    )
                else:
                    pred_traj = model(images, ego_history, ego_history_xyz, ego_history_rot)
                    pred_actions = None
                student_emb = None

            if action_only:
                with torch.no_grad():
                    teacher_actions = model.unicycle.traj_to_action(
                        ego_history_xyz, ego_history_rot, target_xyz, target_rot
                    )
                action_loss_result = action_loss_fn(pred_actions, teacher_actions)
                loss_dict = {
                    "loss": action_loss_result["loss"],
                    "action_loss": action_loss_result["loss"],
                    "action_mse": action_loss_result["mse_loss"],
                    "action_smooth": action_loss_result["smoothness_loss"],
                }
            else:
                if model_type == "reasoning":
                    loss_dict = loss_fn(pred_traj, target_xyz, target_rot, student_emb, teacher_emb)
                else:
                    loss_dict = loss_fn(pred_traj, target_xyz, target_rot)

                if use_action_supervision and action_loss_fn is not None:
                    with torch.no_grad():
                        teacher_actions = model.unicycle.traj_to_action(
                            ego_history_xyz, ego_history_rot, target_xyz, target_rot
                        )
                    action_loss_result = action_loss_fn(pred_actions, teacher_actions)
                    loss_dict["action_loss"] = action_loss_result["loss"]
                    loss_dict["action_mse"] = action_loss_result["mse_loss"]
                    loss_dict["action_smooth"] = action_loss_result["smoothness_loss"]
                    loss_dict["loss"] = (
                        loss_dict["loss"] + action_weight * action_loss_result["loss"]
                    )

            loss = loss_dict["loss"] / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad()

        with torch.no_grad():
            batch_metrics = MetricsComputer.compute_metrics(
                pred_traj.float(), target_xyz, target_rot
            )
            loss_dict["ade"] = batch_metrics["ade"]
            loss_dict["fde"] = batch_metrics["fde"]
            loss_dict["mean_yaw_error"] = batch_metrics["mean_yaw_error"]
            loss_dict["final_yaw_error"] = batch_metrics["final_yaw_error"]

        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            if isinstance(value, torch.Tensor):
                total_losses[key] += value.item()
            else:
                total_losses[key] += value
        num_batches += 1

        pbar.set_postfix(
            {"loss": f"{loss_dict['loss'].item():.4f}", "ade": f"{batch_metrics['ade']:.2f}m"}
        )

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def train_epoch_discrete(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    discrete_loss_fn: DiscreteActionLoss,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    grad_clip: float = 0.0,
    ema: ExponentialMovingAverage | None = None,
) -> dict[str, float]:
    """Train for one epoch with discrete action tokens (stop-gradient mode).

    Args:
        model: Student model with use_discrete_actions=True
        train_loader: Training dataloader
        optimizer: Optimizer
        discrete_loss_fn: DiscreteActionLoss instance
        scaler: GradScaler for mixed precision
        device: Device
        accumulation_steps: Gradient accumulation steps
        grad_clip: Max gradient norm for clipping (0 = disabled)
        ema: EMA instance for weight averaging (optional)

    Returns:
        Dictionary of averaged losses and metrics
    """
    model.train()
    total_losses: dict[str, float] = {}
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training (discrete)")
    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(device)
        target_xyz = batch["trajectory_xyz"].to(device)
        target_rot = batch["trajectory_rot"].to(device)

        ego_history = prepare_ego_history(batch, device)
        ego_history_xyz = batch["ego_history_xyz"].to(device)
        ego_history_rot = batch["ego_history_rot"].to(device)

        with autocast(dtype=torch.bfloat16):
            pred_traj, logits = model(
                images,
                ego_history,
                ego_history_xyz,
                ego_history_rot,
                return_logits=True,
            )

            with torch.no_grad():
                teacher_actions_norm = model.unicycle.traj_to_action(
                    ego_history_xyz, ego_history_rot, target_xyz, target_rot
                )
                target_bins = model.decoder.discrete_head.normalized_actions_to_bins(
                    teacher_actions_norm
                )

            loss_dict = discrete_loss_fn(logits, target_bins)

            with torch.no_grad():
                traj_metrics = MetricsComputer.compute_metrics(pred_traj, target_xyz, target_rot)
                loss_dict["ade"] = torch.tensor(traj_metrics["ade"])
                loss_dict["fde"] = torch.tensor(traj_metrics["fde"])

            loss = loss_dict["loss"] / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad()

        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            val = value.item() if hasattr(value, "item") else value
            total_losses[key] += val
        num_batches += 1

        pbar.set_postfix(
            {
                "ce": f"{loss_dict['loss'].item():.3f}",
                "acc": f"{loss_dict['exact_accuracy'].item():.1%}",
            }
        )

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def train_epoch_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    grad_clip: float = 0.0,
    ema: ExponentialMovingAverage | None = None,
) -> dict[str, float]:
    """Train for one epoch on benchmark data (AV2 / nuScenes).

    Args:
        model: Student model with vector/raster encoder
        train_loader: Benchmark dataloader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Device
        accumulation_steps: Gradient accumulation steps
        grad_clip: Max gradient norm for clipping (0 = disabled)
        ema: EMA instance for weight averaging (optional)

    Returns:
        Dictionary of averaged losses
    """
    model.train()
    total_losses: dict[str, float] = {}
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training (benchmark)")
    for batch_idx, batch in enumerate(pbar):
        encoder_inputs = {}
        for key in ["agent_history", "agent_mask", "agent_positions", "agent_types",
                    "map_polylines", "map_mask", "map_positions", "map_types",
                    "raster_bev"]:
            if key in batch:
                encoder_inputs[key] = batch[key].to(device)

        target_xy = batch["future_trajectory"].to(device)
        target_yaw = batch["future_yaw"].to(device)

        with autocast(dtype=torch.bfloat16):
            result = model(encoder_inputs=encoder_inputs)
            pred_traj = result[0] if isinstance(result, tuple) else result

            pred_xy = pred_traj[:, :, :2]
            pred_sin_yaw = pred_traj[:, :, 2]
            pred_cos_yaw = pred_traj[:, :, 3]

            pred_len = pred_xy.shape[1]
            target_len = target_xy.shape[1]
            if pred_len != target_len:
                pred_xy = F.interpolate(
                    pred_xy.permute(0, 2, 1),
                    size=target_len,
                    mode="linear",
                    align_corners=True
                ).permute(0, 2, 1)
                pred_sin_yaw = F.interpolate(
                    pred_sin_yaw.unsqueeze(1),
                    size=target_len,
                    mode="linear",
                    align_corners=True
                ).squeeze(1)
                pred_cos_yaw = F.interpolate(
                    pred_cos_yaw.unsqueeze(1),
                    size=target_len,
                    mode="linear",
                    align_corners=True
                ).squeeze(1)

            pos_loss = F.mse_loss(pred_xy, target_xy)

            pred_yaw = torch.atan2(pred_sin_yaw, pred_cos_yaw)
            yaw_diff = pred_yaw - target_yaw
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            yaw_loss = (yaw_diff ** 2).mean()

            loss = pos_loss + 0.3 * yaw_loss

            loss_dict = {
                "loss": loss,
                "pos_loss": pos_loss,
                "yaw_loss": yaw_loss,
            }

            with torch.no_grad():
                ade = torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=-1)).mean()
                fde = torch.sqrt(((pred_xy[:, -1] - target_xy[:, -1]) ** 2).sum(dim=-1)).mean()

                loss_dict["ade"] = ade
                loss_dict["fde"] = fde
                loss_dict["mean_yaw_error"] = yaw_diff.abs().mean()
                loss_dict["final_yaw_error"] = yaw_diff[:, -1].abs().mean()

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad()

        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            if isinstance(value, torch.Tensor):
                total_losses[key] += value.item()
            else:
                total_losses[key] += value
        num_batches += 1

        pbar.set_postfix(
            {"loss": f"{loss_dict['loss'].item():.4f}", "ade": f"{loss_dict['ade'].item():.2f}m"}
        )

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses
