"""Wind field model: base wind + Ornstein-Uhlenbeck gusts + building wakes/deflection."""

from __future__ import annotations

import math

import numpy as np

from shared.schemas import Vec2
from sim.world import World


class Airflow:
    """Spatially varying 3D airflow with turbulent gusts and building wake cavities."""

    def __init__(
        self,
        world: World,
        base_wind_angle_deg: float = 15.0,
        base_wind_speed: float = 3.5,
        tau_w: float = 6.0,
        sigma_w: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.world = world
        self.tau_w = tau_w
        self.sigma_w = sigma_w
        self._rng = np.random.default_rng(seed)

        self.w0 = np.zeros(2, dtype=np.float64)
        self.w_turb = np.zeros(2, dtype=np.float64)
        self.set_wind(base_wind_angle_deg, base_wind_speed)

    def set_wind(self, angle_deg: float, speed: float) -> None:
        """Set base wind direction (degrees, 0 = +X, 90 = +Y) and speed (m/s)."""
        rad = math.radians(angle_deg)
        spd = max(0.0, float(speed))
        self.w0[0] = spd * math.cos(rad)
        self.w0[1] = spd * math.sin(rad)

    def reset(self, seed: int | None = None) -> None:
        """Reset turbulent gust state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.w_turb[:] = 0.0

    def step(self, dt: float) -> None:
        """Advance Ornstein-Uhlenbeck turbulent gust vector by dt seconds."""
        noise = self._rng.normal(0.0, 1.0, size=2)
        dw = -(1.0 / self.tau_w) * self.w_turb * dt + self.sigma_w * math.sqrt(dt) * noise
        self.w_turb += dw

    def current_wind_vec(self) -> Vec2:
        """Return spatially uniform ambient wind vector (wx, wy)."""
        w = self.w0 + self.w_turb
        return (float(w[0]), float(w[1]))

    def sample_wind(self, points_nx3: np.ndarray) -> np.ndarray:
        """Vectorized local 3D wind query at N points with building deflection & wakes."""
        pts = np.asarray(points_nx3, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]

        n = pts.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        w_amb = self.w0 + self.w_turb
        speed = float(np.hypot(w_amb[0], w_amb[1]))

        out = np.empty((n, 3), dtype=np.float64)
        out[:, 0] = w_amb[0]
        out[:, 1] = w_amb[1]
        out[:, 2] = 0.0

        if speed < 1e-3:
            return out[0] if single else out

        ux = w_amb[0] / speed
        uy = w_amb[1] / speed
        # Unit vector perpendicular to wind (left-hand normal)
        px_u = -uy
        py_u = ux

        for i in range(len(self.world.buildings)):
            cx, cy, _ = self.world._bldg_center[i]
            hx, hy, _ = self.world._bldg_half[i]
            height = float(self.world._bldg_max[i, 2])

            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            z = pts[:, 2]

            along = dx * ux + dy * uy
            cross = dx * px_u + dy * py_u
            abs_cross = np.abs(cross)

            half_along = hx * abs(ux) + hy * abs(uy)
            half_cross = hx * abs(uy) + hy * abs(ux)

            # 1) Upwind & side potential-flow deflection around/over building
            defl_mask = (
                (along >= -half_along - 6.0)
                & (along <= half_along + 2.0)
                & (abs_cross <= half_cross + 6.0)
                & (z <= height + 5.0)
            )
            if np.any(defl_mask):
                a_d = along[defl_mask]
                c_d = cross[defl_mask]
                z_d = z[defl_mask]

                dist_xy = np.hypot(
                    np.maximum(0.0, np.abs(a_d) - half_along),
                    np.maximum(0.0, np.abs(c_d) - half_cross),
                )
                dist_z = np.maximum(0.0, z_d - height)
                dist_3d = np.hypot(dist_xy, dist_z)
                strength = np.exp(-dist_3d / 3.5) * 0.45 * speed

                sign_cross = np.where(c_d >= 0.0, 1.0, -1.0)
                out[defl_mask, 0] += strength * sign_cross * px_u
                out[defl_mask, 1] += strength * sign_cross * py_u
                # Upwash over the roofline
                upwash = np.where(a_d < 0.0, 0.35 * strength, 0.0)
                out[defl_mask, 2] += upwash

            # 2) Downwind wake cavity (extends 2.5 * building_height downwind)
            wake_len = 2.5 * height
            wake_mask = (
                (along > half_along)
                & (along <= half_along + wake_len)
                & (abs_cross <= half_cross * 1.15)
                & (z <= height * 1.15)
            )
            if np.any(wake_mask):
                a_w = along[wake_mask] - half_along
                c_w = cross[wake_mask]
                # Smooth fade near wake boundary
                frac_along = a_w / wake_len
                # Damp wind speed to ~20% in core wake, recovering linearly near tail
                damp = 0.20 + 0.50 * (frac_along ** 2)
                damp = np.clip(damp, 0.20, 0.85)

                out[wake_mask, 0] *= damp
                out[wake_mask, 1] *= damp

                # Slight cross-axis recirculation / vortex swirl trapping gas
                swirl = -0.25 * speed * np.sin(math.pi * frac_along) * np.tanh(c_w / max(half_cross, 1.0))
                out[wake_mask, 0] += swirl * px_u
                out[wake_mask, 1] += swirl * py_u
                out[wake_mask, 2] -= 0.12 * speed * (1.0 - frac_along)

        return out[0] if single else out
