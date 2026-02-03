"""Loss functions for student model training.

Losses:
- Trajectory loss: L2 on (x, y) + weighted L2 on (sin_yaw, cos_yaw)
- Contrastive reasoning loss: InfoNCE between student and teacher embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """Trajectory prediction loss.

    Combines:
    - Position loss: MSE on (x, y) coordinates
    - Heading loss: MSE on (sin_yaw, cos_yaw) predictions vs targets
    """

    def __init__(self, yaw_weight: float = 0.1):
        """Initialize trajectory loss.

        Args:
            yaw_weight: Weight for heading loss relative to position loss
        """
        super().__init__()
        self.yaw_weight = yaw_weight

    def forward(
        self,
        pred: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute trajectory loss.

        Args:
            pred: Predicted trajectory (B, 64, 4) with (x, y, sin_yaw, cos_yaw)
            target_xyz: Target positions (B, 64, 3) from teacher
            target_rot: Target rotation matrices (B, 64, 3, 3) from teacher

        Returns:
            Dictionary with:
            - 'loss': Total trajectory loss
            - 'pos_loss': Position loss (x, y)
            - 'yaw_loss': Heading loss (sin_yaw, cos_yaw)
        """
        # Position loss (x, y only - z is less critical for planning)
        pred_xy = pred[:, :, :2]  # (B, 64, 2)
        target_xy = target_xyz[:, :, :2]  # (B, 64, 2)
        pos_loss = F.mse_loss(pred_xy, target_xy)

        # Heading loss
        # Extract yaw from rotation matrix: yaw = atan2(R[1,0], R[0,0])
        target_sin_yaw = target_rot[:, :, 1, 0]  # (B, 64)
        target_cos_yaw = target_rot[:, :, 0, 0]  # (B, 64)

        pred_sin_yaw = pred[:, :, 2]  # (B, 64)
        pred_cos_yaw = pred[:, :, 3]  # (B, 64)

        yaw_loss = F.mse_loss(pred_sin_yaw, target_sin_yaw) + F.mse_loss(
            pred_cos_yaw, target_cos_yaw
        )

        # Combined loss
        total_loss = pos_loss + self.yaw_weight * yaw_loss

        return {"loss": total_loss, "pos_loss": pos_loss, "yaw_loss": yaw_loss}


class ContrastiveLoss(nn.Module):
    """Contrastive reasoning loss (InfoNCE).

    Pushes student embedding close to its matching teacher embedding
    and away from other teacher embeddings in the batch.
    """

    def __init__(self, temperature: float = 0.07):
        """Initialize contrastive loss.

        Args:
            temperature: Softmax temperature for InfoNCE
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            student_emb: Student reasoning embedding (B, D)
            teacher_emb: Teacher reasoning embedding (B, D)

        Returns:
            InfoNCE loss scalar
        """
        student_emb = F.normalize(student_emb, dim=-1)
        teacher_emb = F.normalize(teacher_emb, dim=-1)

        logits = student_emb @ teacher_emb.T / self.temperature

        labels = torch.arange(len(logits), device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class ActionLoss(nn.Module):
    """MSE loss on normalized (accel, curvature) action predictions.

    Includes optional smoothness penalty to prevent oscillatory actions.
    """

    def __init__(self, smoothness_weight: float = 0.1):
        """Initialize action loss.

        Args:
            smoothness_weight: Weight for temporal smoothness penalty (default 0.1)
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute action loss.

        Args:
            pred_actions: Predicted normalized actions (B, 64, 2)
            teacher_actions: Teacher normalized actions (B, 64, 2)

        Returns:
            Dictionary with:
            - 'loss': Total action loss
            - 'mse_loss': MSE component
            - 'smoothness_loss': Smoothness penalty component
        """
        mse_loss = F.mse_loss(pred_actions, teacher_actions)

        if self.smoothness_weight > 0:
            smoothness_loss = F.mse_loss(pred_actions[:, 1:], pred_actions[:, :-1])
        else:
            smoothness_loss = torch.tensor(0.0, device=pred_actions.device)

        total_loss = mse_loss + self.smoothness_weight * smoothness_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "smoothness_loss": smoothness_loss,
        }


class DiscreteActionLoss(nn.Module):
    """Cross-entropy loss on discretized action bins with detailed metrics.

    Used with DiscreteActionHead for stop-gradient training.
    """

    def __init__(self, num_bins: int = 256, label_smoothing: float = 0.1):
        """Initialize discrete action loss.

        Args:
            num_bins: Number of bins per action dimension
            label_smoothing: Label smoothing for cross-entropy (handles quantization noise)
        """
        super().__init__()
        self.num_bins = num_bins
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_bins: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute discrete action loss.

        Args:
            pred_logits: Predicted logits (B, 64, 2, num_bins)
            target_bins: Target bin indices (B, 64, 2) long

        Returns:
            Dictionary with:
            - 'loss': Cross-entropy loss
            - 'exact_accuracy': Exact bin match accuracy
            - 'within_1_bin': Accuracy within ±1 bin
            - 'within_3_bins': Accuracy within ±3 bins
            - 'accel_accuracy': Acceleration exact accuracy
            - 'curvature_accuracy': Curvature exact accuracy
        """
        B, T, C, num_bins = pred_logits.shape

        pred_flat = pred_logits.view(-1, num_bins)
        target_flat = target_bins.view(-1)

        loss = self.ce_loss(pred_flat, target_flat)

        with torch.no_grad():
            pred_bins = pred_logits.argmax(dim=-1)
            bin_diff = (pred_bins - target_bins).abs()

            exact_acc = (pred_bins == target_bins).float().mean()
            within_1 = (bin_diff <= 1).float().mean()
            within_3 = (bin_diff <= 3).float().mean()

            accel_exact = (pred_bins[..., 0] == target_bins[..., 0]).float().mean()
            curv_exact = (pred_bins[..., 1] == target_bins[..., 1]).float().mean()

        return {
            "loss": loss,
            "exact_accuracy": exact_acc,
            "within_1_bin": within_1,
            "within_3_bins": within_3,
            "accel_accuracy": accel_exact,
            "curvature_accuracy": curv_exact,
        }


class CombinedLoss(nn.Module):
    """Combined loss for reasoning student training.

    Total loss = trajectory_loss + lambda * reasoning_loss
    """

    def __init__(
        self,
        yaw_weight: float = 0.1,
        reasoning_weight: float = 0.25,
        temperature: float = 0.07,
    ):
        """Initialize combined loss.

        Args:
            yaw_weight: Weight for heading loss in trajectory loss
            reasoning_weight: Weight for contrastive reasoning loss (lambda)
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.trajectory_loss = TrajectoryLoss(yaw_weight=yaw_weight)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.reasoning_weight = reasoning_weight

    def forward(
        self,
        pred_trajectory: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rot: torch.Tensor,
        student_emb: torch.Tensor | None = None,
        teacher_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred_trajectory: Predicted trajectory (B, 64, 4)
            target_xyz: Target positions (B, 64, 3)
            target_rot: Target rotation matrices (B, 64, 3, 3)
            student_emb: Student reasoning embedding (B, D) - optional
            teacher_emb: Teacher reasoning embedding (B, D) - optional

        Returns:
            Dictionary with:
            - 'loss': Total combined loss
            - 'traj_loss': Trajectory loss component
            - 'pos_loss': Position loss
            - 'yaw_loss': Heading loss
            - 'reasoning_loss': Contrastive loss (0 if embeddings not provided)
        """
        # Trajectory loss
        traj_results = self.trajectory_loss(pred_trajectory, target_xyz, target_rot)

        # Reasoning loss
        if student_emb is not None and teacher_emb is not None:
            reasoning_loss = self.contrastive_loss(student_emb, teacher_emb)
        else:
            reasoning_loss = torch.tensor(0.0, device=pred_trajectory.device)

        # Combined loss
        total_loss = traj_results["loss"] + self.reasoning_weight * reasoning_loss

        return {
            "loss": total_loss,
            "traj_loss": traj_results["loss"],
            "pos_loss": traj_results["pos_loss"],
            "yaw_loss": traj_results["yaw_loss"],
            "reasoning_loss": reasoning_loss,
        }


def compute_ade(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Compute Average Displacement Error.

    Args:
        pred_xy: Predicted positions (B, T, 2) or (T, 2)
        target_xy: Target positions (B, T, 2) or (T, 2)

    Returns:
        ADE averaged over batch
    """
    displacement = torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=-1))
    ade = displacement.mean(dim=-1)
    return ade.mean()


def compute_fde(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Compute Final Displacement Error.

    Args:
        pred_xy: Predicted positions (B, T, 2) or (T, 2)
        target_xy: Target positions (B, T, 2) or (T, 2)

    Returns:
        FDE averaged over batch
    """
    pred_final = pred_xy[..., -1, :]
    target_final = target_xy[..., -1, :]
    fde = torch.sqrt(((pred_final - target_final) ** 2).sum(dim=-1))
    return fde.mean()


class MetricsComputer:
    """Compute evaluation metrics for trajectory prediction."""

    @staticmethod
    def compute_metrics(
        pred_trajectory: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            pred_trajectory: Predicted trajectory (B, 64, 4)
            target_xyz: Target positions (B, 64, 3)
            target_rot: Target rotation matrices (B, 64, 3, 3)

        Returns:
            Dictionary with ADE, FDE, and yaw error metrics
        """
        with torch.no_grad():
            pred_xy = pred_trajectory[:, :, :2]
            target_xy = target_xyz[:, :, :2]

            # Position metrics
            ade = compute_ade(pred_xy, target_xy).item()
            fde = compute_fde(pred_xy, target_xy).item()

            # Yaw metrics
            pred_yaw = torch.atan2(pred_trajectory[:, :, 2], pred_trajectory[:, :, 3])
            target_yaw = torch.atan2(target_rot[:, :, 1, 0], target_rot[:, :, 0, 0])

            # Angular error (handle wraparound)
            yaw_error = torch.abs(
                torch.atan2(torch.sin(pred_yaw - target_yaw), torch.cos(pred_yaw - target_yaw))
            )

            mean_yaw_error = yaw_error.mean().item()
            final_yaw_error = yaw_error[:, -1].mean().item()

        return {
            "ade": ade,
            "fde": fde,
            "mean_yaw_error": mean_yaw_error,
            "final_yaw_error": final_yaw_error,
        }

    @staticmethod
    def compute_metrics_detailed(
        pred_trajectory: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> dict[str, float | torch.Tensor]:
        """Compute detailed metrics including percentiles and per-sample values.

        Args:
            pred_trajectory: Predicted trajectory (B, 64, 4)
            target_xyz: Target positions (B, 64, 3)
            target_rot: Target rotation matrices (B, 64, 3, 3)

        Returns:
            Dictionary with:
            - ade, fde: Mean values
            - ade_p50/p90/p99/max, fde_p50/p90/p99/max: Percentiles
            - ade_top10_mean, ade_top50_mean: Mean of worst N samples
            - per_sample_ade, per_sample_fde: Raw per-sample values (for worst-sample logging)
            - target_final_dist_p50/p90/p99/max: Target displacement percentiles
            - has_nan_target, has_inf_target: Sanity check flags
        """
        with torch.no_grad():
            pred_xy = pred_trajectory[:, :, :2]
            target_xy = target_xyz[:, :, :2]

            # Per-sample ADE: mean displacement over time
            displacement = torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=-1))  # (B, T)
            per_sample_ade = displacement.mean(dim=-1)  # (B,)

            # Per-sample FDE: final displacement
            pred_final = pred_xy[:, -1, :]
            target_final = target_xy[:, -1, :]
            per_sample_fde = torch.sqrt(((pred_final - target_final) ** 2).sum(dim=-1))  # (B,)

            # Aggregate stats
            ade_mean = per_sample_ade.mean().item()
            fde_mean = per_sample_fde.mean().item()

            # ADE percentiles
            ade_p50 = torch.median(per_sample_ade).item()
            ade_p90 = torch.quantile(per_sample_ade, 0.9).item()
            ade_p99 = torch.quantile(per_sample_ade, 0.99).item()
            ade_max = per_sample_ade.max().item()

            # FDE percentiles
            fde_p50 = torch.median(per_sample_fde).item()
            fde_p90 = torch.quantile(per_sample_fde, 0.9).item()
            fde_p99 = torch.quantile(per_sample_fde, 0.99).item()
            fde_max = per_sample_fde.max().item()

            # Top-K means (more informative than just p99)
            B = per_sample_ade.shape[0]
            top10 = min(10, B)
            top50 = min(50, B)
            sorted_ade, _ = torch.sort(per_sample_ade, descending=True)
            ade_top10_mean = sorted_ade[:top10].mean().item()
            ade_top50_mean = sorted_ade[:top50].mean().item()

            # Target sanity checks (distance from origin at final timestep)
            target_final_dist = torch.sqrt((target_xy[:, -1, :] ** 2).sum(dim=-1))
            target_final_p50 = torch.median(target_final_dist).item()
            target_final_p90 = torch.quantile(target_final_dist, 0.9).item()
            target_final_p99 = torch.quantile(target_final_dist, 0.99).item()
            target_final_max = target_final_dist.max().item()

            has_nan_target = bool(torch.isnan(target_xyz).any().item())
            has_inf_target = bool(torch.isinf(target_xyz).any().item())

            # Yaw metrics (same as compute_metrics)
            pred_yaw = torch.atan2(pred_trajectory[:, :, 2], pred_trajectory[:, :, 3])
            target_yaw = torch.atan2(target_rot[:, :, 1, 0], target_rot[:, :, 0, 0])
            yaw_diff = pred_yaw - target_yaw
            yaw_error = torch.abs(torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)))
            mean_yaw_error = yaw_error.mean().item()
            final_yaw_error = yaw_error[:, -1].mean().item()

        return {
            # Standard metrics
            "ade": ade_mean,
            "fde": fde_mean,
            "mean_yaw_error": mean_yaw_error,
            "final_yaw_error": final_yaw_error,
            # ADE percentiles
            "ade_p50": ade_p50,
            "ade_p90": ade_p90,
            "ade_p99": ade_p99,
            "ade_max": ade_max,
            "ade_top10_mean": ade_top10_mean,
            "ade_top50_mean": ade_top50_mean,
            # FDE percentiles
            "fde_p50": fde_p50,
            "fde_p90": fde_p90,
            "fde_p99": fde_p99,
            "fde_max": fde_max,
            # Per-sample values (for worst-sample logging)
            "per_sample_ade": per_sample_ade,
            "per_sample_fde": per_sample_fde,
            # Target sanity checks
            "target_final_dist_p50": target_final_p50,
            "target_final_dist_p90": target_final_p90,
            "target_final_dist_p99": target_final_p99,
            "target_final_dist_max": target_final_max,
            "has_nan_target": has_nan_target,
            "has_inf_target": has_inf_target,
        }


class WinnerTakesAllLoss(nn.Module):
    """Winner-takes-all loss for multi-modal trajectory prediction.

    Only the best mode (closest to GT) receives gradient for trajectory loss.
    This encourages mode diversity by not penalizing "wrong" modes.

    Additionally includes mode classification loss to encourage the model
    to predict high probability for the best mode.
    """

    def __init__(
        self,
        yaw_weight: float = 0.1,
        mode_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.yaw_weight = yaw_weight
        self.mode_loss_weight = mode_loss_weight

    def forward(
        self,
        pred_trajectories: torch.Tensor,
        pred_probs: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute winner-takes-all loss.

        Args:
            pred_trajectories: Predicted trajectories (B, K, T, 4)
            pred_probs: Mode probabilities (B, K)
            target_xyz: Target positions (B, T, 3)
            target_rot: Target rotations (B, T, 3, 3)

        Returns:
            Dictionary with loss components and minADE metric
        """
        B, K, T, _ = pred_trajectories.shape

        pred_xy = pred_trajectories[..., :2]
        target_xy = target_xyz[:, :, :2].unsqueeze(1)

        per_mode_ade = ((pred_xy - target_xy) ** 2).sum(-1).sqrt().mean(-1)

        best_mode_idx = per_mode_ade.argmin(dim=1)

        batch_idx = torch.arange(B, device=pred_trajectories.device)
        best_pred = pred_trajectories[batch_idx, best_mode_idx]

        pos_loss = F.mse_loss(best_pred[..., :2], target_xyz[..., :2])

        target_sin = target_rot[:, :, 1, 0]
        target_cos = target_rot[:, :, 0, 0]
        yaw_loss = F.mse_loss(best_pred[..., 2], target_sin) + F.mse_loss(
            best_pred[..., 3], target_cos
        )

        mode_loss = F.cross_entropy(
            pred_probs.log().clamp(min=-100), best_mode_idx
        )

        traj_loss = pos_loss + self.yaw_weight * yaw_loss
        total = traj_loss + self.mode_loss_weight * mode_loss

        with torch.no_grad():
            min_ade = per_mode_ade.min(dim=1)[0].mean()
            min_fde = (
                (pred_xy[:, :, -1, :] - target_xy[:, :, -1, :])
                .pow(2)
                .sum(-1)
                .sqrt()
                .min(dim=1)[0]
                .mean()
            )

        return {
            "loss": total,
            "traj_loss": traj_loss,
            "pos_loss": pos_loss,
            "yaw_loss": yaw_loss,
            "mode_loss": mode_loss,
            "minADE": min_ade,
            "minFDE": min_fde,
        }


class MultiModalNLLLoss(nn.Module):
    """Negative log-likelihood loss for multi-modal trajectory prediction.

    Models the trajectory distribution as a mixture of Gaussians,
    where each mode is a Gaussian centered on the predicted trajectory.
    All modes contribute to the loss, weighted by predicted probability.

    This encourages well-calibrated probability estimates.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        yaw_weight: float = 0.1,
    ):
        super().__init__()
        self.sigma = sigma
        self.yaw_weight = yaw_weight

    def forward(
        self,
        pred_trajectories: torch.Tensor,
        pred_probs: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute mixture NLL loss.

        Args:
            pred_trajectories: Predicted trajectories (B, K, T, 4)
            pred_probs: Mode probabilities (B, K)
            target_xyz: Target positions (B, T, 3)
            target_rot: Target rotations (B, T, 3, 3)

        Returns:
            Dictionary with loss and metrics
        """
        B, K, T, _ = pred_trajectories.shape

        pred_xy = pred_trajectories[..., :2]
        target_xy = target_xyz[:, :, :2].unsqueeze(1)

        sq_dist = ((pred_xy - target_xy) ** 2).sum(-1)
        log_prob_pos = -0.5 * sq_dist.mean(-1) / (self.sigma**2)

        pred_sin = pred_trajectories[..., 2]
        pred_cos = pred_trajectories[..., 3]
        target_sin = target_rot[:, :, 1, 0].unsqueeze(1)
        target_cos = target_rot[:, :, 0, 0].unsqueeze(1)
        sq_yaw = (pred_sin - target_sin) ** 2 + (pred_cos - target_cos) ** 2
        log_prob_yaw = -0.5 * sq_yaw.mean(-1) / (self.sigma**2)

        log_prob = log_prob_pos + self.yaw_weight * log_prob_yaw

        log_mixture_prob = torch.logsumexp(log_prob + pred_probs.log(), dim=1)

        nll = -log_mixture_prob.mean()

        with torch.no_grad():
            per_mode_ade = ((pred_xy - target_xy) ** 2).sum(-1).sqrt().mean(-1)
            min_ade = per_mode_ade.min(dim=1)[0].mean()
            min_fde = (
                (pred_xy[:, :, -1, :] - target_xy[:, :, -1, :])
                .pow(2)
                .sum(-1)
                .sqrt()
                .min(dim=1)[0]
                .mean()
            )

        return {
            "loss": nll,
            "nll": nll,
            "minADE": min_ade,
            "minFDE": min_fde,
        }


class MultiModalMetrics:
    """Compute evaluation metrics for multi-modal trajectory prediction."""

    @staticmethod
    def compute_min_ade(
        pred_trajectories: torch.Tensor,
        target_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute minimum ADE across K modes.

        Args:
            pred_trajectories: (B, K, T, 2+) predicted trajectories
            target_xy: (B, T, 2) target positions

        Returns:
            Scalar minADE averaged over batch
        """
        pred_xy = pred_trajectories[..., :2]
        target = target_xy.unsqueeze(1)
        ade_per_mode = ((pred_xy - target) ** 2).sum(-1).sqrt().mean(-1)
        return ade_per_mode.min(dim=1)[0].mean()

    @staticmethod
    def compute_min_fde(
        pred_trajectories: torch.Tensor,
        target_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute minimum FDE across K modes."""
        pred_final = pred_trajectories[:, :, -1, :2]
        target_final = target_xy[:, -1, :].unsqueeze(1)
        fde_per_mode = ((pred_final - target_final) ** 2).sum(-1).sqrt()
        return fde_per_mode.min(dim=1)[0].mean()

    @staticmethod
    def compute_miss_rate(
        pred_trajectories: torch.Tensor,
        target_xy: torch.Tensor,
        threshold: float = 2.0,
    ) -> torch.Tensor:
        """Compute miss rate: fraction where no mode is within threshold of GT."""
        pred_final = pred_trajectories[:, :, -1, :2]
        target_final = target_xy[:, -1, :].unsqueeze(1)
        fde_per_mode = ((pred_final - target_final) ** 2).sum(-1).sqrt()
        min_fde = fde_per_mode.min(dim=1)[0]
        return (min_fde > threshold).float().mean()

    @staticmethod
    def compute_brier_min_fde(
        pred_trajectories: torch.Tensor,
        pred_probs: torch.Tensor,
        target_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Brier-minFDE: minFDE + (1 - p_best)^2.

        Penalizes low confidence on correct mode.
        """
        pred_final = pred_trajectories[:, :, -1, :2]
        target_final = target_xy[:, -1, :].unsqueeze(1)
        fde_per_mode = ((pred_final - target_final) ** 2).sum(-1).sqrt()
        best_mode = fde_per_mode.argmin(dim=1)

        batch_idx = torch.arange(len(best_mode), device=best_mode.device)
        min_fde = fde_per_mode[batch_idx, best_mode]
        best_prob = pred_probs[batch_idx, best_mode]

        brier = min_fde + (1 - best_prob) ** 2
        return brier.mean()

    @staticmethod
    def compute_all(
        pred_trajectories: torch.Tensor,
        pred_probs: torch.Tensor | None,
        target_xyz: torch.Tensor,
    ) -> dict[str, float]:
        """Compute all multi-modal metrics."""
        target_xy = target_xyz[..., :2]
        K = pred_trajectories.shape[1]

        metrics = {
            f"minADE_{K}": MultiModalMetrics.compute_min_ade(
                pred_trajectories, target_xy
            ).item(),
            f"minFDE_{K}": MultiModalMetrics.compute_min_fde(
                pred_trajectories, target_xy
            ).item(),
            "miss_rate_2m": MultiModalMetrics.compute_miss_rate(
                pred_trajectories, target_xy, 2.0
            ).item(),
        }

        if pred_probs is not None:
            metrics["brier_minFDE"] = MultiModalMetrics.compute_brier_min_fde(
                pred_trajectories, pred_probs, target_xy
            ).item()

        return metrics
