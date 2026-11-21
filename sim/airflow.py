"""High-fidelity 3D airflow: ABL vertical shear, OU gusts, 3D divergence-free eddies,
cylindrical potential flow + vortex shedding, and smooth building wake cavities.
"""

from __future__ import annotations

import math

import numpy as np

from shared.schemas import Vec2
from sim.world import World


class Airflow:
    """Spatially varying 3D airflow with ABL shear, 3D turbulence, and building aerodynamics."""

    # Atmospheric Boundary Layer (ABL) industrial power-law parameters
    Z_REF: float = 10.0
    ABL_ALPHA: float = 0.18
    N_EDDY_MODES: int = 6

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
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.w0 = np.zeros(2, dtype=np.float64)
        self.w_turb = np.zeros(2, dtype=np.float64)
        self._time: float = 0.0

        # Spatial 3D divergence-free Fourier eddy modes (advected via Taylor hypothesis)
        self._k_vecs = np.zeros((self.N_EDDY_MODES, 3), dtype=np.float64)
        self._e_vecs = np.zeros((self.N_EDDY_MODES, 3), dtype=np.float64)
        self._phases = np.zeros(self.N_EDDY_MODES, dtype=np.float64)
        self._eddy_weights = np.zeros(self.N_EDDY_MODES, dtype=np.float64)
        self._init_eddy_modes()

        self.set_wind(base_wind_angle_deg, base_wind_speed)

    def _init_eddy_modes(self) -> None:
        """Build K divergence-free 3D Kolmogorov-like spatial eddy modes."""
        # Wavelengths spanning 14m to 48m (building-scale to facility-scale eddies)
        wavelengths = np.geomspace(14.0, 48.0, self.N_EDDY_MODES)
        for m, lam in enumerate(wavelengths):
            k_mag = 2.0 * math.pi / float(lam)
            dir_vec = self._rng.normal(0.0, 1.0, size=3)
            dir_vec[2] *= 0.6  # Slightly flatter atmospheric eddies
            n_dir = float(np.linalg.norm(dir_vec))
            if n_dir < 1e-6:
                dir_vec = np.array([1.0, 0.0, 0.0])
            else:
                dir_vec /= n_dir
            self._k_vecs[m] = k_mag * dir_vec

            # Polarization vector strictly orthogonal to k_vec => div(u) == 0
            rand_v = self._rng.normal(0.0, 1.0, size=3)
            pol = np.cross(dir_vec, rand_v)
            n_pol = float(np.linalg.norm(pol))
            if n_pol < 1e-6:
                pol = np.array([0.0, 0.0, 1.0])
            else:
                pol /= n_pol
            # Damp vertical component slightly for realistic atmospheric anisotropy
            pol[2] *= 0.45
            self._e_vecs[m] = pol
            self._phases[m] = float(self._rng.uniform(0.0, 2.0 * math.pi))
            # Kolmogorov E(k) ~ k^(-5/3) amplitude scaling (~ lam^(5/6))
            self._eddy_weights[m] = (float(lam) / 28.0) ** 0.55

        # Normalize RMS eddy amplitude
        rms = float(np.sqrt(np.sum(0.5 * (self._eddy_weights ** 2))))
        if rms > 1e-6:
            self._eddy_weights /= rms

    def set_wind(self, angle_deg: float, speed: float) -> None:
        """Set base wind direction (degrees, 0 = +X, 90 = +Y) and speed (m/s)."""
        rad = math.radians(angle_deg)
        spd = max(0.0, float(speed))
        self.w0[0] = spd * math.cos(rad)
        self.w0[1] = spd * math.sin(rad)

    def reset(self, seed: int | None = None) -> None:
        """Reset turbulent gust state and eddy clock."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._init_eddy_modes()
        self.w_turb[:] = 0.0
        self._time = 0.0

    def step(self, dt: float) -> None:
        """Advance Ornstein-Uhlenbeck turbulent gust vector and eddy advection clock."""
        self._time += dt
        if self.sigma_w > 0.0:
            noise = self._rng.normal(0.0, 1.0, size=2)
            dw = -(1.0 / self.tau_w) * self.w_turb * dt + self.sigma_w * math.sqrt(dt) * noise
            self.w_turb += dw
        else:
            self.w_turb[:] = 0.0

    def current_wind_vec(self) -> Vec2:
        """Return spatially uniform reference-height ambient wind vector (wx, wy)."""
        w = self.w0 + self.w_turb
        return (float(w[0]), float(w[1]))

    def _abl_factor(self, z: np.ndarray) -> np.ndarray:
        """Atmospheric boundary layer power-law shear profile U(z) / U(z_ref)."""
        z_eff = np.maximum(z, 0.5)
        return np.clip(np.power(z_eff / self.Z_REF, self.ABL_ALPHA), 0.55, 1.35)

    def sample_turbulence_intensity(self, points_nx3: np.ndarray) -> np.ndarray:
        """Return local turbulent diffusivity multiplier (>= 1.0; elevated in building wakes)."""
        pts = np.asarray(points_nx3, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]

        n = pts.shape[0]
        mult = np.ones(n, dtype=np.float64)
        if n == 0:
            return mult

        w_amb = self.w0 + self.w_turb
        speed = float(np.hypot(w_amb[0], w_amb[1]))
        if speed < 1e-3:
            return mult[0] if single else mult

        ux = w_amb[0] / speed
        uy = w_amb[1] / speed
        px_u = -uy
        py_u = ux

        for i in range(len(self.world.buildings)):
            cx, cy, _ = self.world._bldg_center[i]
            hx, hy, _ = self.world._bldg_half[i]
            height = float(self.world._bldg_max[i, 2])
            is_porous = bool(self.world._bldg_is_porous[i])

            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            z = pts[:, 2]

            along = dx * ux + dy * uy
            abs_cross = np.abs(dx * px_u + dy * py_u)
            half_along = hx * abs(ux) + hy * abs(uy)
            half_cross = hx * abs(uy) + hy * abs(ux)
            wake_len = 2.5 * height

            in_wake = (
                (along > half_along)
                & (along <= half_along + wake_len)
                & (abs_cross <= half_cross * 1.25)
                & (z <= height * 1.25)
            )
            if np.any(in_wake):
                frac = (along[in_wake] - half_along) / wake_len
                boost = (0.7 if is_porous else 1.6) * np.sin(math.pi * np.clip(frac, 0.0, 1.0))
                mult[in_wake] = np.maximum(mult[in_wake], 1.0 + boost)

        return mult[0] if single else mult

    def sample_wind(self, points_nx3: np.ndarray) -> np.ndarray:
        """Vectorized local 3D wind query at N points with ABL shear, 3D eddies, and wakes."""
        wind, _ = self.sample_wind_and_turbulence(points_nx3, compute_turb=False)
        return wind

    def sample_wind_and_turbulence(
        self, points_nx3: np.ndarray, compute_turb: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-pass vectorized local 3D wind and optional wake turbulence multiplier."""
        pts = np.asarray(points_nx3, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]

        n = pts.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64), np.ones(0, dtype=np.float64)

        w_amb = self.w0 + self.w_turb
        speed = math.hypot(float(w_amb[0]), float(w_amb[1]))

        # 1) ABL vertical wind shear profile
        z_all = pts[:, 2]
        abl = self._abl_factor(z_all)

        out = np.empty((n, 3), dtype=np.float64)
        out[:, 0] = w_amb[0] * abl
        out[:, 1] = w_amb[1] * abl
        out[:, 2] = 0.0
        turb_mult = np.ones(n, dtype=np.float64) if compute_turb else np.empty(0, dtype=np.float64)

        # 2) Spatially-varying 3D divergence-free turbulent eddies (Taylor frozen-turbulence)
        if self.sigma_w > 0.0:
            adv_x = pts[:, 0] - self.w0[0] * self._time
            adv_y = pts[:, 1] - self.w0[1] * self._time
            phase = (
                adv_x[:, None] * self._k_vecs[None, :, 0]
                + adv_y[:, None] * self._k_vecs[None, :, 1]
                + z_all[:, None] * self._k_vecs[None, :, 2]
                + self._phases[None, :]
            )
            eddy_amp = 0.45 * self.sigma_w * np.clip(z_all * 0.25, 0.25, 1.0)
            sin_p = np.sin(phase) * self._eddy_weights[None, :]
            out += (sin_p @ self._e_vecs) * eddy_amp[:, None]

        if speed < 1e-3:
            if single:
                return out[0], (turb_mult[0] if compute_turb else turb_mult)
            return out, turb_mult

        ux = float(w_amb[0]) / speed
        uy = float(w_amb[1]) / speed
        px_u = -uy
        py_u = ux
        abs_ux = abs(ux)
        abs_uy = abs(uy)

        x_all = pts[:, 0]
        y_all = pts[:, 1]

        # 3) Building / Storage Tank / Pipe Rack aerodynamic perturbations
        for i in range(len(self.world.buildings)):
            cx = float(self.world._bldg_center[i, 0])
            cy = float(self.world._bldg_center[i, 1])
            hx = float(self.world._bldg_half[i, 0])
            hy = float(self.world._bldg_half[i, 1])
            height = float(self.world._bldg_max[i, 2])
            is_cyl = bool(self.world._bldg_is_cylinder[i])
            is_porous = bool(self.world._bldg_is_porous[i])

            dx = x_all - cx
            dy = y_all - cy

            along = dx * ux + dy * uy
            cross = dx * px_u + dy * py_u
            abs_cross = np.abs(cross)

            if is_cyl:
                r_cyl = float(self.world._bldg_cyl_radius[i])
                r2 = dx * dx + dy * dy
                near_cyl = (r2 <= (r_cyl + 12.0) ** 2) & (z_all <= height + 6.0)
                if near_cyl.any():
                    a_c = along[near_cyl]
                    c_c = cross[near_cyl]
                    z_c = z_all[near_cyl]
                    r_safe = np.maximum(np.sqrt(r2[near_cyl]), r_cyl * 1.02)

                    z_blend = 0.5 * (1.0 - np.tanh((z_c - height) / 2.2))
                    rr2 = (r_cyl / r_safe) ** 2
                    cos_t = a_c / r_safe
                    sin_t = c_c / r_safe
                    cos_2t = cos_t * cos_t - sin_t * sin_t
                    sin_2t = 2.0 * sin_t * cos_t

                    upwind_w = 0.5 * (1.0 - np.tanh((a_c - 0.3 * r_cyl) / (0.4 * r_cyl)))
                    u_loc = speed * abl[near_cyl] * z_blend * upwind_w * 0.75

                    du_along = -u_loc * rr2 * cos_2t
                    du_cross = -u_loc * rr2 * sin_2t

                    out[near_cyl, 0] += du_along * ux + du_cross * px_u
                    out[near_cyl, 1] += du_along * uy + du_cross * py_u
                    out[near_cyl, 2] += np.where(a_c < 0.0, 0.38 * u_loc * rr2, 0.0)

                half_along = r_cyl
                half_cross = r_cyl
            else:
                half_along = hx * abs_ux + hy * abs_uy
                half_cross = hx * abs_uy + hy * abs_ux

                defl_mask = (
                    (along >= -half_along - 7.0)
                    & (along <= half_along + 2.5)
                    & (abs_cross <= half_cross + 7.0)
                    & (z_all <= height + 5.5)
                )
                if defl_mask.any():
                    a_d = along[defl_mask]
                    c_d = cross[defl_mask]
                    z_d = z_all[defl_mask]

                    dist_xy = np.hypot(
                        np.maximum(0.0, np.abs(a_d) - half_along),
                        np.maximum(0.0, np.abs(c_d) - half_cross),
                    )
                    dist_z = np.maximum(0.0, z_d - height)
                    dist_3d = np.hypot(dist_xy, dist_z)

                    scale_factor = 0.22 if is_porous else 0.50
                    strength = np.exp(-dist_3d / 3.8) * scale_factor * speed * abl[defl_mask]

                    sign_cross = np.tanh(c_d / max(half_cross * 0.25, 0.8))
                    out[defl_mask, 0] += strength * sign_cross * px_u
                    out[defl_mask, 1] += strength * sign_cross * py_u
                    out[defl_mask, 2] += np.where(a_d < 0.0, 0.38 * strength, -0.08 * strength)

            # Downwind Wake Cavity (smoothly blended, extends 2.5 * height downwind)
            wake_len = 2.5 * height
            wake_mask = (
                (along > half_along - 0.5)
                & (along <= half_along + wake_len * 1.15)
                & (abs_cross <= half_cross * 1.35)
                & (z_all <= height * 1.30)
            )
            if wake_mask.any():
                a_w = np.maximum(0.0, along[wake_mask] - half_along)
                c_w = cross[wake_mask]
                z_w = z_all[wake_mask]

                frac_along = np.clip(a_w / wake_len, 0.0, 1.2)
                env_cross = np.exp(
                    -0.5
                    * (
                        np.maximum(0.0, np.abs(c_w) - half_cross * 0.75)
                        / max(half_cross * 0.35, 1.0)
                    )
                    ** 2
                )
                env_z = 0.5 * (1.0 - np.tanh((z_w - height * 1.05) / 1.6))
                env_tail = 0.5 * (1.0 - np.tanh((frac_along - 0.92) / 0.14))
                wake_env = env_cross * env_z * env_tail

                if is_porous:
                    core_damp = 0.65 + 0.35 * (frac_along * frac_along)
                else:
                    core_damp = 0.20 + 0.55 * (frac_along * frac_along)

                core_damp = np.clip(core_damp, 0.20, 1.0)
                eff_damp = 1.0 - wake_env * (1.0 - core_damp)

                out[wake_mask, 0] *= eff_damp
                out[wake_mask, 1] *= eff_damp

                if compute_turb:
                    boost = (0.7 if is_porous else 1.6) * wake_env * np.sin(
                        math.pi * np.clip(frac_along, 0.0, 1.0)
                    )
                    turb_mult[wake_mask] = np.maximum(turb_mult[wake_mask], 1.0 + boost)

                if not is_porous:
                    frac_c = np.clip(frac_along, 0.0, 1.0)
                    swirl = (
                        -0.28
                        * speed
                        * wake_env
                        * np.sin(math.pi * frac_c)
                        * np.tanh(c_w / max(half_cross, 1.0))
                    )
                    if is_cyl and self.sigma_w > 0.0:
                        strouhal_omega = 2.0 * math.pi * 0.20 * speed / max(2.0 * half_cross, 4.0)
                        swirl = swirl + (
                            0.18
                            * speed
                            * wake_env
                            * np.sin(strouhal_omega * self._time - 0.35 * a_w)
                        )

                    out[wake_mask, 0] += swirl * px_u
                    out[wake_mask, 1] += swirl * py_u
                    out[wake_mask, 2] -= 0.14 * speed * wake_env * (1.0 - frac_c)

        if single:
            return out[0], (turb_mult[0] if compute_turb else turb_mult)
        return out, turb_mult
