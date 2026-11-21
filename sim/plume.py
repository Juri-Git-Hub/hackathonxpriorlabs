"""Lagrangian 3D Gaussian puff dispersion model with buoyancy and ground reflection."""

from __future__ import annotations

import math

import numpy as np

from shared.schemas import LeakSource, PuffState, Vec3
from sim.airflow import Airflow
from sim.world import World


class Plume:
    """Vectorized ring-buffer Lagrangian Gaussian puff dispersion model."""

    MAX_PUFFS = 400

    def __init__(
        self,
        world: World,
        airflow: Airflow,
        spawn_interval: float = 0.25,
        d_turb: float = 0.25,
        seed: int = 101,
    ) -> None:
        self.world = world
        self.airflow = airflow
        self.spawn_interval = spawn_interval
        self.d_turb = d_turb
        self._rng = np.random.default_rng(seed)

        self.pos = np.zeros((self.MAX_PUFFS, 3), dtype=np.float64)
        self.age = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.mass = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.buoyancy_vz = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.active = np.zeros(self.MAX_PUFFS, dtype=bool)

        self._write_idx = 0
        self._spawn_accum = 0.0
        # Calibrated concentration scale so near-leak ~100-400 ppm, downwind ~5-80 ppm
        self._conc_scale = 2200.0

    def reset(self, seed: int | None = None) -> None:
        """Clear all active puffs."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.active[:] = False
        self._write_idx = 0
        self._spawn_accum = 0.0

    def _spawn_puff(self, leak: LeakSource) -> None:
        idx = self._write_idx
        self.pos[idx] = leak.pos
        self.age[idx] = 0.05
        self.mass[idx] = leak.emission_rate * self.spawn_interval
        self.buoyancy_vz[idx] = leak.buoyancy_vz
        self.active[idx] = True
        self._write_idx = (self._write_idx + 1) % self.MAX_PUFFS

    def step(self, dt: float, leaks: list[LeakSource]) -> None:
        """Spawn new puffs from active leaks and advect/disperse all active puffs."""
        self._spawn_accum += dt
        while self._spawn_accum >= self.spawn_interval:
            self._spawn_accum -= self.spawn_interval
            for leak in leaks:
                self._spawn_puff(leak)

        mask = self.active
        if not np.any(mask):
            return

        active_pos = self.pos[mask]
        wind = self.airflow.sample_wind(active_pos)

        n_active = active_pos.shape[0]
        diff_std = math.sqrt(2.0 * self.d_turb * dt)
        noise = self._rng.normal(0.0, diff_std, size=(n_active, 3))

        vel = wind.copy()
        vel[:, 2] += self.buoyancy_vz[mask]

        new_pos = active_pos + vel * dt + noise
        # Ground reflection / clamp above ground
        new_pos[:, 2] = np.maximum(0.3, new_pos[:, 2])
        # Keep puffs outside solid building interiors
        new_pos = self.world.push_outside_buildings(new_pos, margin=0.2)

        self.pos[mask] = new_pos
        self.age[mask] += dt

        # Deactivate puffs that drift far outside facility bounds or exceed max age
        out_of_bounds = (
            (np.abs(self.pos[:, 0]) > 85.0)
            | (np.abs(self.pos[:, 1]) > 85.0)
            | (self.pos[:, 2] > 45.0)
            | (self.age > 90.0)
        )
        self.active[out_of_bounds] = False

    def _sigmas(self, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma_xy = 0.8 + 0.35 * np.power(ages, 0.85)
        sigma_z = 0.5 + 0.20 * np.power(ages, 0.75)
        return sigma_xy, sigma_z

    def sample_concentration(self, pos: Vec3) -> float:
        """Sample gas concentration (ppm) at a single 3D point with ground reflection."""
        mask = self.active
        if not np.any(mask):
            return 0.0

        p_pos = self.pos[mask]
        p_age = self.age[mask]
        p_mass = self.mass[mask]

        sigma_xy, sigma_z = self._sigmas(p_age)

        dx = pos[0] - p_pos[:, 0]
        dy = pos[1] - p_pos[:, 1]
        dz_direct = pos[2] - p_pos[:, 2]
        dz_refl = pos[2] + p_pos[:, 2]

        r2_xy = (dx * dx + dy * dy) / (sigma_xy * sigma_xy)
        z2_dir = (dz_direct * dz_direct) / (sigma_z * sigma_z)
        z2_ref = (dz_refl * dz_refl) / (sigma_z * sigma_z)

        horiz = np.exp(-0.5 * np.minimum(r2_xy, 50.0))
        vert = np.exp(-0.5 * np.minimum(z2_dir, 50.0)) + np.exp(
            -0.5 * np.minimum(z2_ref, 50.0)
        )

        norm = (sigma_xy * sigma_xy) * sigma_z
        conc = np.sum(self._conc_scale * p_mass * horiz * vert / norm)
        return float(conc)

    def get_puff_states(self) -> list[PuffState]:
        """Serialize active puffs for frontend volumetric rendering (<= 400 puffs)."""
        mask = self.active
        if not np.any(mask):
            return []

        p_pos = self.pos[mask]
        p_age = self.age[mask]
        sigma_xy, sigma_z = self._sigmas(p_age)
        # Opacity fades smoothly as puff expands and ages
        opacity = np.clip(0.85 / (1.0 + 0.08 * p_age), 0.04, 0.85)

        states: list[PuffState] = []
        for i in range(p_pos.shape[0]):
            states.append(
                PuffState(
                    pos=(float(p_pos[i, 0]), float(p_pos[i, 1]), float(p_pos[i, 2])),
                    radius_xy=float(sigma_xy[i]),
                    radius_z=float(sigma_z[i]),
                    opacity=float(opacity[i]),
                )
            )
        return states
