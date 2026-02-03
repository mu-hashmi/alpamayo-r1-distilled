"""Unicycle kinematic model for trajectory integration.

Self-contained implementation of the unicycle kinematic model, matching
Alpamayo's UnicycleAccelCurvatureActionSpace for exact mathematical compatibility.

Based on NVIDIA's Alpamayo-R1 implementation (Apache 2.0 License).
Original: https://github.com/nvidia/alpamayo
"""

from __future__ import annotations

import logging

import einops
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Geometry utilities
# =============================================================================


def so3_to_yaw(rot_mat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from SO3 rotation matrix (xyz euler order).

    Args:
        rot_mat: (..., 3, 3)

    Returns:
        yaw: (...)
    """
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def rotation_matrix_2d(angle: torch.Tensor) -> torch.Tensor:
    """Create 2D rotation matrices from angles.

    Args:
        angle: (...) angles in radians

    Returns:
        rot: (..., 2, 2) rotation matrices
    """
    return torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    """Convert 2D rotation matrix to 3D (flat xy plane).

    Args:
        rot: (..., 2, 2)

    Returns:
        rot_3d: (..., 3, 3)
    """
    rot = torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(rot.shape[:-2] + (1, 1)),
        ],
        dim=-2,
    )
    return rot


def round_2pi(x: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """Unwrap angles so consecutive differences are in (-pi, pi]."""
    d = torch.diff(phi, dim=-1)
    d = round_2pi(d)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


# =============================================================================
# Tikhonov-regularized least squares solvers
# =============================================================================


def _first_order_D(
    N: int, lead_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """First-order finite difference matrix."""
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def _second_order_D(
    N: int, lead_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Second-order finite difference matrix."""
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def _third_order_D(
    N: int, lead_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Third-order finite difference matrix."""
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def _construct_DTD(
    N: int,
    lead: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    w_smooth1: float | None,
    w_smooth2: float | None,
    w_smooth3: float | None,
    lam: float,
    dt: float,
) -> torch.Tensor:
    """Construct smoothing regularization matrix D^T D."""
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)

    if w_smooth1 is not None:
        lam_1 = lam / dt**2
        w1 = torch.full((*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device)
        D1 = _first_order_D(N, lead, device, dtype)
        DTD += lam_1 * einops.einsum(D1 * w1.unsqueeze(-1), D1, "... i j, ... i k -> ... j k")

    if w_smooth2 is not None:
        lam_2 = lam / dt**4
        w2 = torch.full((*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device)
        D2 = _second_order_D(N, lead, device, dtype)
        DTD += lam_2 * einops.einsum(D2 * w2.unsqueeze(-1), D2, "... i j, ... i k -> ... j k")

    if w_smooth3 is not None:
        lam_3 = lam / dt**6
        w3 = torch.full((*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device)
        D3 = _third_order_D(N, lead, device, dtype)
        DTD += lam_3 * einops.einsum(D3 * w3.unsqueeze(-1), D3, "... i j, ... i k -> ... j k")

    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def _solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_smooth1: float | None,
    w_smooth2: float | None,
    w_smooth3: float | None,
    lam: float,
    ridge: float,
    dt: float,
) -> torch.Tensor:
    """Solve constrained least squares with initial value fixed."""
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    w_data = torch.ones_like(x_target)
    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")

    DTD = _construct_DTD(N + 1, lead, device, dtype, w_smooth1, w_smooth2, w_smooth3, lam, dt)
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

    return torch.cat([x_init.unsqueeze(-1), x], dim=-1)


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def _solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None,
    w_smooth1: float | None,
    w_smooth2: float | None,
    w_smooth3: float | None,
    lam: float,
    ridge: float,
    dt: float,
) -> torch.Tensor:
    """Solve: min sum w_data_i (x_i * s_i - y_i)^2 + smoothing."""
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)

    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")

    DTD = _construct_DTD(N, lead, device, dtype, w_smooth1, w_smooth2, w_smooth3, lam, dt)

    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
            lhs = ATA + DTD + ridge_term
            if rhs.dtype != lhs.dtype:
                rhs = rhs.to(lhs.dtype)
            L = torch.linalg.cholesky(lhs)
        except RuntimeError:
            ridge *= 10
            logger.warning(f"Resolving singularity using ridge {ridge}")

    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


# =============================================================================
# Velocity estimation
# =============================================================================


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def _dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float,
    v_lambda: float,
    v_ridge: float,
) -> torch.Tensor:
    """Estimate velocity from position differences and heading (no v0 constraint)."""
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy

    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")

    DTD = _construct_DTD(N + 1, lead, device, dtype, None, None, 1.0, v_lambda, dt)
    ridge_term = v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(*lead, N + 1, N + 1)
    lhs = ATA + DTD + ridge_term

    L = torch.linalg.cholesky(lhs)
    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def _dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float,
    v_lambda: float,
    v_ridge: float,
) -> torch.Tensor:
    """Estimate velocity from position differences and heading (with v0 constraint)."""
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy

    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)

    DTD = _construct_DTD(N + 1, lead, device, dtype, None, None, 1.0, v_lambda, dt)
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)

    ridge_term = v_ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

    return torch.cat([v0.unsqueeze(-1), y], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def _theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float,
    theta_lambda: float,
    theta_ridge: float,
) -> torch.Tensor:
    """Smooth heading angles from rotation matrices."""
    theta = so3_to_yaw(traj_future_rot)
    theta = unwrap_angle(theta)
    theta_init = torch.zeros_like(theta[..., 0])
    return _solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )


# =============================================================================
# Main UnicycleIntegrator class
# =============================================================================


class UnicycleIntegrator(nn.Module):
    """Differentiable unicycle kinematic model.

    Integrates (acceleration, curvature) actions to (x, y, yaw) trajectory using:
    - Second-order heading terms: theta_t+1 = theta_t + kappa * v_t * dt + kappa * a * dt^2/2
    - Trapezoidal position integration: x_t+1 = x_t + (v_t*cos(theta_t) + v_t+1*cos(theta_t+1)) * dt/2

    Matches Alpamayo's UnicycleAccelCurvatureActionSpace for exact mathematical compatibility.
    """

    def __init__(
        self,
        dt: float = 0.1,
        n_waypoints: int = 64,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        super().__init__()

        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))

        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def forward(
        self,
        actions: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate actions to trajectory.

        Args:
            actions: Normalized actions (B, 64, 2) - (accel, curvature)
            ego_history_xyz: Past positions (B, N_hist, 3)
            ego_history_rot: Past rotations (B, N_hist, 3, 3)

        Returns:
            xyz: Future positions (B, 64, 3)
            rot: Future rotations (B, 64, 3, 3)
        """
        return self.action_to_traj(
            action=actions,
            traj_history_xyz=ego_history_xyz,
            traj_history_rot=ego_history_rot,
            t0_states=None,
        )

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform actions to trajectory (forward kinematics).

        Args:
            action: Normalized (accel, curvature) (..., T, 2)
            traj_history_xyz: Past positions (..., T_hist, 3)
            traj_history_rot: Past rotations (..., T_hist, 3, 3)
            t0_states: Optional initial state dict with 'v' key

        Returns:
            traj_future_xyz: Future positions (..., T, 3)
            traj_future_rot: Future rotations (..., T, 3, 3)
        """
        accel, kappa = action[..., 0], action[..., 1]

        # Denormalize
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        v0 = t0_states["v"]
        dt = self.dt

        # Velocity: v_t+1 = v_0 + cumsum(a * dt)
        dt_2_term = 0.5 * (dt**2)
        velocity = torch.cat(
            [
                v0.unsqueeze(-1),
                (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)),
            ],
            dim=-1,
        )  # (..., N+1)

        # Heading: theta_t+1 = theta_0 + cumsum(kappa * v * dt + kappa * a * dt^2/2)
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                (
                    initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
                ),
            ],
            dim=-1,
        )  # (..., N+1)

        # Position: trapezoidal integration
        half_dt = 0.5 * dt
        initial_x = torch.zeros_like(v0)
        initial_y = torch.zeros_like(v0)
        x = (
            initial_x.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt, dim=-1)
        )  # (..., N)
        y = (
            initial_y.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt, dim=-1)
        )  # (..., N)

        # Build output tensors
        batch_dim = traj_history_xyz.shape[:-2]
        traj_future_xyz = torch.zeros(
            *batch_dim,
            self.n_waypoints,
            3,
            device=traj_history_xyz.device,
            dtype=traj_history_xyz.dtype,
        )
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]  # Keep z from history

        traj_future_rot = rot_2d_to_3d(rotation_matrix_2d(theta[..., 1:]))

        return traj_future_xyz, traj_future_rot

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate initial velocity from ego history."""
        full_xy = traj_history_xyz[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = so3_to_yaw(traj_history_rot)
        theta = unwrap_angle(theta)

        v = _dxy_theta_to_v_without_v0(
            dxy=dxy, theta=theta, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge
        )
        v_t0 = v[..., -1]
        return {"v": v_t0}

    def estimate_v0(
        self,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate initial velocity from ego history."""
        t0_states = self.estimate_t0_states(ego_history_xyz, ego_history_rot)
        return t0_states["v"]

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        future_xyz: torch.Tensor,
        future_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Convert trajectory to normalized (accel, curvature) actions.

        Uses Tikhonov-regularized least-squares fitting for smooth actions.

        Args:
            ego_history_xyz: Past positions (B, N_hist, 3)
            ego_history_rot: Past rotations (B, N_hist, 3, 3)
            future_xyz: Future positions (B, 64, 3)
            future_rot: Future rotations (B, 64, 3, 3)

        Returns:
            actions: Normalized actions (B, 64, 2) - (accel, curvature)
        """
        if future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(
                f"future trajectory must have length {self.n_waypoints} "
                f"but got {future_xyz.shape[-2]}"
            )

        t0_states = self.estimate_t0_states(ego_history_xyz, ego_history_rot)

        # Concatenate last history position with future
        full_xy = torch.cat([ego_history_xyz[..., -1:, :], future_xyz], dim=-2)[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]

        theta = _theta_smooth(
            traj_future_rot=future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )

        v0 = t0_states["v"]
        v = _dxy_theta_to_v(
            dxy=dxy, theta=theta, v0=v0, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge
        )

        accel = self._v_to_a(v)
        kappa = self._theta_v_a_to_kappa(theta, v, accel)

        # Normalize
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std

        return torch.stack([accel, kappa], dim=-1)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        """Compute acceleration from velocity."""
        dv = (v[..., 1:] - v[..., :-1]) / self.dt
        return _solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            w_data=None,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute curvature from theta, velocity, and acceleration."""
        dtheta = theta[..., 1:] - theta[..., :-1]
        dt = self.dt
        s = dt * v[..., :-1] + (dt**2) / 2.0 * a

        return _solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=torch.ones_like(dtheta),
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )
