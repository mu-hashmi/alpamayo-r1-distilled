"""Validation functions."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import DiscreteActionLoss, MetricsComputer
from .loops import prepare_ego_history

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate_discrete(
    model: nn.Module,
    val_loader: DataLoader,
    discrete_loss_fn: DiscreteActionLoss,
    device: torch.device,
) -> dict[str, float]:
    """Validate model with discrete action tokens."""
    model.eval()
    total_losses: dict[str, float] = {}
    all_pred_traj = []
    all_target_xyz = []
    all_target_rot = []
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating (discrete)"):
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

            teacher_actions_norm = model.unicycle.traj_to_action(
                ego_history_xyz, ego_history_rot, target_xyz, target_rot
            )
            target_bins = model.decoder.discrete_head.normalized_actions_to_bins(
                teacher_actions_norm
            )

            loss_dict = discrete_loss_fn(logits, target_bins)

        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        num_batches += 1

        all_pred_traj.append(pred_traj.float().cpu())
        all_target_xyz.append(target_xyz.cpu())
        all_target_rot.append(target_rot.cpu())

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    all_pred_traj = torch.cat(all_pred_traj, dim=0)
    all_target_xyz = torch.cat(all_target_xyz, dim=0)
    all_target_rot = torch.cat(all_target_rot, dim=0)

    metrics = MetricsComputer.compute_metrics(all_pred_traj, all_target_xyz, all_target_rot)
    avg_losses.update(metrics)

    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    model_type: str,
    use_unicycle: bool = True,
) -> dict[str, float]:
    """Validate model with detailed metrics."""
    model.eval()
    total_losses: dict[str, float] = {}
    all_pred_traj = []
    all_target_xyz = []
    all_target_rot = []
    all_clip_ids = []
    all_sample_indices = []
    num_batches = 0
    sample_idx_offset = 0

    for batch in tqdm(val_loader, desc="Validating"):
        images = batch["images"].to(device)
        target_xyz = batch["trajectory_xyz"].to(device)
        target_rot = batch["trajectory_rot"].to(device)
        teacher_emb = batch["coc_embedding"].to(device)

        ego_history = prepare_ego_history(batch, device)
        ego_history_xyz = batch["ego_history_xyz"].to(device)
        ego_history_rot = batch["ego_history_rot"].to(device)

        with autocast(dtype=torch.bfloat16):
            if model_type == "reasoning":
                pred_traj, student_emb = model(
                    images,
                    ego_history,
                    ego_history_xyz,
                    ego_history_rot,
                    return_reasoning=True,
                )
                loss_dict = loss_fn(pred_traj, target_xyz, target_rot, student_emb, teacher_emb)
            else:
                pred_traj = model(images, ego_history, ego_history_xyz, ego_history_rot)
                loss_dict = loss_fn(pred_traj, target_xyz, target_rot)

        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        num_batches += 1

        all_pred_traj.append(pred_traj.float().cpu())
        all_target_xyz.append(target_xyz.cpu())
        all_target_rot.append(target_rot.cpu())
        batch_size = images.size(0)
        all_sample_indices.extend(range(sample_idx_offset, sample_idx_offset + batch_size))
        sample_idx_offset += batch_size
        if "clip_id" in batch:
            all_clip_ids.extend(batch["clip_id"])

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    all_pred_traj = torch.cat(all_pred_traj, dim=0)
    all_target_xyz = torch.cat(all_target_xyz, dim=0)
    all_target_rot = torch.cat(all_target_rot, dim=0)

    detailed_metrics = MetricsComputer.compute_metrics_detailed(
        all_pred_traj, all_target_xyz, all_target_rot
    )

    per_sample_ade = detailed_metrics.pop("per_sample_ade")
    detailed_metrics.pop("per_sample_fde")

    _, worst_indices = torch.sort(per_sample_ade, descending=True)
    worst_10_indices = worst_indices[:10].tolist()
    worst_50_indices = worst_indices[:50].tolist()
    worst_10_sample_idx = [all_sample_indices[i] for i in worst_10_indices]
    worst_10_ades = [per_sample_ade[i].item() for i in worst_10_indices]

    detailed_metrics["worst_10_indices"] = worst_10_sample_idx
    detailed_metrics["worst_50_indices"] = [all_sample_indices[i] for i in worst_50_indices]

    logger.info(f"Worst 10 sample indices: {worst_10_sample_idx}")
    logger.info(f"Worst 10 ADEs: {[f'{a:.1f}m' for a in worst_10_ades]}")
    if all_clip_ids and len(all_clip_ids) == len(per_sample_ade):
        worst_10_clips = [all_clip_ids[i] for i in worst_10_indices]
        logger.info(f"Worst 10 clip_ids: {worst_10_clips}")

    if detailed_metrics.get("has_nan_target"):
        logger.warning("NaN detected in target trajectories!")
    if detailed_metrics.get("has_inf_target"):
        logger.warning("Inf detected in target trajectories!")

    avg_losses.update(detailed_metrics)
    return avg_losses


@torch.no_grad()
def validate_benchmark(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate model on benchmark data (AV2 / nuScenes)."""
    model.eval()
    total_losses: dict[str, float] = {}
    all_pred_xy = []
    all_target_xy = []
    all_pred_yaw = []
    all_target_yaw = []
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating (benchmark)"):
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

            # Adapt prediction length to target length
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

            pred_yaw = torch.atan2(pred_sin_yaw, pred_cos_yaw)

            pos_loss = F.mse_loss(pred_xy, target_xy)
            yaw_diff = pred_yaw - target_yaw
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            yaw_loss = (yaw_diff ** 2).mean()
            loss = pos_loss + 0.3 * yaw_loss

            loss_dict = {
                "loss": loss.item(),
                "pos_loss": pos_loss.item(),
                "yaw_loss": yaw_loss.item(),
            }

        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value
        num_batches += 1

        all_pred_xy.append(pred_xy.float().cpu())
        all_target_xy.append(target_xy.cpu())
        all_pred_yaw.append(pred_yaw.float().cpu())
        all_target_yaw.append(target_yaw.cpu())

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    all_pred_xy = torch.cat(all_pred_xy, dim=0)
    all_target_xy = torch.cat(all_target_xy, dim=0)
    all_pred_yaw = torch.cat(all_pred_yaw, dim=0)
    all_target_yaw = torch.cat(all_target_yaw, dim=0)

    per_sample_ade = torch.sqrt(((all_pred_xy - all_target_xy) ** 2).sum(dim=-1)).mean(dim=1)
    per_sample_fde = torch.sqrt(((all_pred_xy[:, -1] - all_target_xy[:, -1]) ** 2).sum(dim=-1))

    avg_losses["ade"] = per_sample_ade.mean().item()
    avg_losses["fde"] = per_sample_fde.mean().item()

    avg_losses["ade_p50"] = torch.quantile(per_sample_ade, 0.5).item()
    avg_losses["ade_p90"] = torch.quantile(per_sample_ade, 0.9).item()
    avg_losses["ade_p99"] = torch.quantile(per_sample_ade, 0.99).item()
    avg_losses["ade_max"] = per_sample_ade.max().item()

    sorted_ade, _ = torch.sort(per_sample_ade, descending=True)
    avg_losses["ade_top10_mean"] = sorted_ade[:10].mean().item()
    avg_losses["ade_top50_mean"] = sorted_ade[:50].mean().item()

    all_yaw_diff = all_pred_yaw - all_target_yaw
    all_yaw_diff = torch.atan2(torch.sin(all_yaw_diff), torch.cos(all_yaw_diff))
    avg_losses["mean_yaw_error"] = all_yaw_diff.abs().mean().item()
    avg_losses["final_yaw_error"] = all_yaw_diff[:, -1].abs().mean().item()

    target_final_dist = torch.sqrt((all_target_xy[:, -1] ** 2).sum(dim=-1))
    avg_losses["target_final_dist_p50"] = torch.quantile(target_final_dist, 0.5).item()
    avg_losses["target_final_dist_p99"] = torch.quantile(target_final_dist, 0.99).item()
    avg_losses["target_final_dist_max"] = target_final_dist.max().item()

    return avg_losses
