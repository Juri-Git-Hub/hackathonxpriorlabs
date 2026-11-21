"""High-fidelity Lagrangian 3D Gaussian puff dispersion model.

Features:
- RK2 (midpoint) streamline advection around buildings and cylindrical tanks
- Wake-enhanced turbulent diffusivity & effective dispersion age
- Buoyancy plume rise with entrainment dilution
- Specular ground reflection + mixing-lid reflection
- Vectorized building-wall occlusion so gas cannot teleport through solid walls
"""

from __future__ import annotations

import math

import numpy as np

from shared.schemas import LeakSource, PuffState, Vec3
from sim.airflow import Airflow
from sim.world import World


class Plume:
    """Vectorized ring-buffer Lagrangian Gaussian puff dispersion model."""

    MAX_PUFFS = 400
    MIXING_LID_Z = 30.0

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
        self.eff_age = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.mass = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.buoyancy_vz = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.active = np.zeros(self.MAX_PUFFS, dtype=bool)

        self._write_idx = 0
        self._spawn_accum = 0.0
        # Calibrated concentration scale so near-leak ~100-400 ppm, downwind ~5-80 ppm
        self._conc_scale = 2200.0
        # Per-tick cache of active puff arrays & dispersion sigmas (avoids 4x redundant work/tick)
        self._cache_dirty = True
        self._c_pos = np.zeros((0, 3), dtype=np.float64)
        self._c_age = np.zeros(0, dtype=np.float64)
        self._c_mass = np.zeros(0, dtype=np.float64)
        self._c_sxy = np.zeros(0, dtype=np.float64)
        self._c_sz = np.zeros(0, dtype=np.float64)
        self._c_inv_sxy2 = np.zeros(0, dtype=np.float64)
        self._c_inv_sz2 = np.zeros(0, dtype=np.float64)
        self._c_inv_norm = np.zeros(0, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        """Clear all active puffs."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.active[:] = False
        self.age[:] = 0.0
        self.eff_age[:] = 0.0
        self._write_idx = 0
        self._spawn_accum = 0.0
        self._cache_dirty = True

    def _refresh_active_cache(self) -> None:
        mask = self.active
        if not mask.any():
            self._c_pos = np.zeros((0, 3), dtype=np.float64)
            self._cache_dirty = False
            return
        self._c_pos = self.pos[mask]
        self._c_age = self.age[mask]
        self._c_mass = self.mass[mask]
        sxy, sz = self._sigmas(self.eff_age[mask])
        self._c_sxy = sxy
        self._c_sz = sz
        sxy2 = sxy * sxy
        self._c_inv_sxy2 = 1.0 / sxy2
        self._c_inv_sz2 = 1.0 / (sz * sz)
        self._c_inv_norm = self._conc_scale * self._c_mass / (sxy2 * sz)
        self._cache_dirty = False

    def _spawn_puff(self, leak: LeakSource) -> None:
        idx = self._write_idx
        self.pos[idx] = leak.pos
        self.age[idx] = 0.05
        self.eff_age[idx] = 0.05
        self.mass[idx] = leak.emission_rate * self.spawn_interval
        self.buoyancy_vz[idx] = leak.buoyancy_vz
        self.active[idx] = True
        self._write_idx = (self._write_idx + 1) % self.MAX_PUFFS
        self._cache_dirty = True

    def step(self, dt: float, leaks: list[LeakSource]) -> None:
        """Spawn new puffs from active leaks and advect/disperse all active puffs (RK2)."""
        self._spawn_accum += dt
        while self._spawn_accum >= self.spawn_interval:
            self._spawn_accum -= self.spawn_interval
            for leak in leaks:
                self._spawn_puff(leak)

        mask = self.active
        if not mask.any():
            self._cache_dirty = True
            return

        active_pos = self.pos[mask]
        active_age = self.age[mask]
        # Buoyancy dilutes smoothly via ambient air entrainment as puff ages
        vz_buoy = self.buoyancy_vz[mask] / (1.0 + 0.035 * active_age)

        # RK2 (Midpoint) advection so puffs follow curved streamlines around tanks/buildings
        w1, turb_mult = self.airflow.sample_wind_and_turbulence(active_pos, compute_turb=True)
        w1[:, 2] += vz_buoy

        mid_pos = active_pos + 0.5 * w1 * dt
        np.maximum(0.2, mid_pos[:, 2], out=mid_pos[:, 2])
        mid_pos = self.world.push_outside_buildings(mid_pos, margin=0.2, skip_porous=True, copy=False)
        w2, _ = self.airflow.sample_wind_and_turbulence(mid_pos, compute_turb=False)
        w2[:, 2] += vz_buoy

        # Wake-enhanced turbulent eddy diffusivity
        n_active = active_pos.shape[0]
        diff_std = np.sqrt(2.0 * self.d_turb * turb_mult * dt)
        noise = self._rng.normal(0.0, 1.0, size=(n_active, 3))
        noise *= diff_std[:, None]
        noise[:, 2] *= 0.72  # Vertical eddy diffusivity is smaller than horizontal

        new_pos = active_pos + w2 * dt + noise

        # Specular ground reflection at z = 0.20 m
        below_ground = new_pos[:, 2] < 0.20
        if below_ground.any():
            new_pos[below_ground, 2] = np.maximum(0.20, 0.40 - new_pos[below_ground, 2])

        # Keep puffs outside solid buildings/tanks (porous pipe rack allows passage) & standalone pipes
        new_pos = self.world.push_outside_buildings(new_pos, margin=0.2, skip_porous=True, copy=False)
        new_pos = self.world.push_outside_pipes(new_pos, margin=0.05, copy=False)

        self.pos[mask] = new_pos
        self.age[mask] = active_age + dt
        # Effective dispersion age accelerates inside turbulent building wakes
        self.eff_age[mask] += dt * (0.7 + 0.3 * turb_mult)

        # Deactivate puffs that drift far outside facility bounds or exceed max age
        out_of_bounds = (
            (np.abs(self.pos[:, 0]) > 85.0)
            | (np.abs(self.pos[:, 1]) > 85.0)
            | (self.pos[:, 2] > 45.0)
            | (self.age > 90.0)
        )
        self.active[out_of_bounds] = False
        self._cache_dirty = True

    def _sigmas(self, eff_ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma_xy = 0.8 + 0.35 * np.power(eff_ages, 0.85)
        sigma_z = 0.5 + 0.20 * np.power(eff_ages, 0.75)
        return sigma_xy, sigma_z

    def sample_concentration(self, pos: Vec3) -> float:
        """Sample gas concentration (ppm) at a 3D point with ground/lid reflection & wall occlusion."""
        if self._cache_dirty:
            self._refresh_active_cache()

        p_pos = self._c_pos
        if p_pos.shape[0] == 0:
            return 0.0

        dx = pos[0] - p_pos[:, 0]
        dy = pos[1] - p_pos[:, 1]
        r2_xy = (dx * dx + dy * dy) * self._c_inv_sxy2

        # Exact early cull: since vert <= 3.0, horiz * vert > 1e-3 requires r2_xy < 2 * ln(3000) ~= 16.013
        near_xy = r2_xy < 16.1
        if not near_xy.any():
            return 0.0

        p_near = p_pos[near_xy]
        pz = p_near[:, 2]
        inv_sz2 = self._c_inv_sz2[near_xy]

        dz_direct = pos[2] - pz
        dz_ground = pos[2] + pz
        dz_lid = (2.0 * self.MIXING_LID_Z - pos[2]) - pz

        z2_dir = (dz_direct * dz_direct) * inv_sz2
        z2_gnd = (dz_ground * dz_ground) * inv_sz2
        z2_lid = (dz_lid * dz_lid) * inv_sz2

        horiz = np.exp(-0.5 * r2_xy[near_xy])
        vert = (
            np.exp(-0.5 * np.minimum(z2_dir, 50.0))
            + np.exp(-0.5 * np.minimum(z2_gnd, 50.0))
            + np.exp(-0.5 * np.minimum(z2_lid, 50.0))
        )

        # Building wall occlusion: only test puffs with non-negligible contribution
        raw_weights = horiz * vert
        active_contrib = raw_weights > 1e-3
        if not active_contrib.any():
            return 0.0

        trans = self.world.segment_transmission(p_near[active_contrib], pos)
        inv_norm = self._c_inv_norm[near_xy][active_contrib]
        conc = float(np.sum(inv_norm * raw_weights[active_contrib] * trans))
        return conc

    def get_puff_states(self) -> list[PuffState]:
        """Serialize active puffs for frontend volumetric rendering (<= 400 puffs)."""
        if self._cache_dirty:
            self._refresh_active_cache()

        p_pos = self._c_pos
        n = p_pos.shape[0]
        if n == 0:
            return []

        opacity = 0.85 / (1.0 + 0.08 * self._c_age)
        np.clip(opacity, 0.04, 0.85, out=opacity)

        xs = p_pos[:, 0].tolist()
        ys = p_pos[:, 1].tolist()
        zs = p_pos[:, 2].tolist()
        rxy = self._c_sxy.tolist()
        rz = self._c_sz.tolist()
        ops = opacity.tolist()

        # Fast-path PuffState instantiation (skips Pydantic model_construct per-field loop/dict.pop)
        new_obj = object.__new__
        set_attr = object.__setattr__
        fields_set = {"pos", "radius_xy", "radius_z", "opacity"}
        out: list[PuffState] = [None] * n  # type: ignore[list-item]
        for i in range(n):
            p = new_obj(PuffState)
            set_attr(
                p,
                "__dict__",
                {
                    "pos": (xs[i], ys[i], zs[i]),
                    "radius_xy": rxy[i],
                    "radius_z": rz[i],
                    "opacity": ops[i],
                },
            )
            set_attr(p, "__pydantic_fields_set__", fields_set)
            set_attr(p, "__pydantic_extra__", None)
            set_attr(p, "__pydantic_private__", None)
            out[i] = p
        return out
