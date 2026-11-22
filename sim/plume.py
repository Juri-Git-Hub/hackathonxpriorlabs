"""High-fidelity Lagrangian 3D Gaussian puff dispersion model.

Features:
- SI mass-to-ppmv ideal-gas concentration calibration (no arbitrary scale factor)
- Briggs (1973) urban/industrial Pasquill-Gifford stability classes (A-F)
- RK2 (midpoint) streamline advection around buildings and cylindrical tanks
- Wake-enhanced turbulent dispersion without double-counting resolved eddies
- Briggs R^(-1/3) entrainment-diluted buoyancy plume rise
- Specular ground reflection + configurable atmospheric mixing-lid reflection
- Vectorized building-wall occlusion so gas cannot teleport through solid walls
- Decoupled physics puff capacity (1200) and WebGL render cap (<= 400)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from shared.schemas import LeakSource, PuffState, Vec3
from sim.airflow import Airflow
from sim.world import World

StabilityClass = Literal["A", "B", "C", "D", "E", "F"]

# Briggs (1973) urban/industrial dispersion coefficients:
#   sigma_y(x) = a_y * x * (1 + b_y * x)^c_y
#   sigma_z(x) = a_z * x * (1 + b_z * x)^c_z
_BRIGGS_URBAN_COEFFS: dict[StabilityClass, tuple[float, float, float, float, float, float]] = {
    "A": (0.32, 0.0004, -0.5, 0.24, 0.0010, 0.5),
    "B": (0.32, 0.0004, -0.5, 0.24, 0.0010, 0.5),
    "C": (0.22, 0.0004, -0.5, 0.20, 0.0000, 0.0),
    "D": (0.16, 0.0004, -0.5, 0.14, 0.0003, -0.5),
    "E": (0.11, 0.0004, -0.5, 0.08, 0.0015, -0.5),
    "F": (0.11, 0.0004, -0.5, 0.08, 0.0015, -0.5),
}


class Plume:
    """Vectorized ring-buffer Lagrangian Gaussian puff dispersion model."""

    MAX_PUFFS = 1200
    MAX_RENDER_PUFFS = 400
    MAX_AGE_S = 180.0
    MIXING_LID_Z = 300.0

    # SI thermodynamic constants (default: methane CH4 at 20 C, 1 atm)
    R_UNIVERSAL = 8.314462618        # J / (mol * K)
    DEFAULT_MOLAR_MASS_KG = 0.01604  # kg / mol (CH4)
    DEFAULT_TEMP_K = 293.15          # K (20 C)
    DEFAULT_PRESSURE_PA = 101325.0   # Pa (1 atm)

    # Initial orifice/release puff radius and calm-wind diffusion floor
    SIGMA0_XY = 0.35                 # m
    SIGMA0_Z = 0.25                  # m
    K_CALM_XY = 0.05                 # m^2/s
    K_CALM_Z = 0.02                  # m^2/s

    def __init__(
        self,
        world: World,
        airflow: Airflow,
        spawn_interval: float = 0.25,
        d_turb: float = 0.02,
        seed: int = 101,
        stability_class: StabilityClass = "D",
        mixing_height_m: float = MIXING_LID_Z,
        molar_mass_kg: float = DEFAULT_MOLAR_MASS_KG,
        ambient_temp_k: float = DEFAULT_TEMP_K,
        ambient_pressure_pa: float = DEFAULT_PRESSURE_PA,
    ) -> None:
        if stability_class not in _BRIGGS_URBAN_COEFFS:
            raise ValueError(f"Unsupported stability class: {stability_class}")
        self.world = world
        self.airflow = airflow
        self.spawn_interval = spawn_interval
        self.d_turb = d_turb
        self.stability_class: StabilityClass = stability_class
        self.mixing_height_m = max(10.0, float(mixing_height_m))
        self.molar_mass_kg = float(molar_mass_kg)
        self.ambient_temp_k = float(ambient_temp_k)
        self.ambient_pressure_pa = float(ambient_pressure_pa)
        self._rng = np.random.default_rng(seed)

        # Ideal-gas mass concentration (kg/m^3) -> volume mixing ratio (ppmv)
        molar_volume_m3 = self.R_UNIVERSAL * self.ambient_temp_k / self.ambient_pressure_pa
        self.kg_m3_to_ppmv = 1.0e6 * molar_volume_m3 / self.molar_mass_kg
        self._gauss_norm = 1.0 / ((2.0 * math.pi) ** 1.5)
        self._ppm_gauss_norm = self.kg_m3_to_ppmv * self._gauss_norm

        self.pos = np.zeros((self.MAX_PUFFS, 3), dtype=np.float64)
        self.age = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.eff_age = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.eff_dist = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.wake_var = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.mass = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.buoyancy_vz = np.zeros(self.MAX_PUFFS, dtype=np.float64)
        self.active = np.zeros(self.MAX_PUFFS, dtype=bool)

        self._write_idx = 0
        self._spawn_accum = 0.0
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
        self.eff_dist[:] = 0.0
        self.wake_var[:] = 0.0
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
        sxy, sz = self._sigmas(self.eff_dist[mask], self.eff_age[mask], self.wake_var[mask])
        self._c_sxy = sxy
        self._c_sz = sz
        sxy2 = sxy * sxy
        self._c_inv_sxy2 = 1.0 / sxy2
        self._c_inv_sz2 = 1.0 / (sz * sz)
        self._c_inv_norm = self._ppm_gauss_norm * self._c_mass / (sxy2 * sz)
        self._cache_dirty = False

    def _spawn_puff(self, leak: LeakSource) -> None:
        idx = self._write_idx
        self.pos[idx] = leak.pos
        self.age[idx] = 0.0
        self.eff_age[idx] = 0.0
        self.eff_dist[idx] = 0.0
        self.wake_var[idx] = 0.0
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
        active_eff_age = self.eff_age[mask]
        active_eff_dist = self.eff_dist[mask]
        active_wake_var = self.wake_var[mask]

        # Briggs entrainment buoyancy: rise speed decays as R_eff^(-1/3) (~ t^(-1/3))
        sxy_cur, sz_cur = self._sigmas(active_eff_dist, active_eff_age, active_wake_var)
        r_eff = np.cbrt(sxy_cur * sxy_cur * sz_cur)
        vz_buoy = self.buoyancy_vz[mask] * np.cbrt(1.0 / np.maximum(r_eff, 1.0))

        # RK2 (Midpoint) advection so puffs follow curved streamlines around tanks/buildings
        w1, turb_mult = self.airflow.sample_wind_and_turbulence(active_pos, compute_turb=True)
        w1[:, 2] += vz_buoy

        mid_pos = active_pos + 0.5 * w1 * dt
        np.maximum(0.2, mid_pos[:, 2], out=mid_pos[:, 2])
        mid_pos = self.world.push_outside_buildings(mid_pos, margin=0.2, skip_porous=True, copy=False)
        w2, _ = self.airflow.sample_wind_and_turbulence(mid_pos, compute_turb=False)
        w2[:, 2] += vz_buoy

        # Wake-cavity turbulent mixing (zero in open air where turb_mult == 1.0)
        amb_wx, amb_wy = self.airflow.current_wind_vec()
        amb_speed = math.hypot(amb_wx, amb_wy)
        wake_excess = np.maximum(0.0, turb_mult - 1.0)
        k_wake = 0.30 * max(amb_speed, 0.5) * wake_excess

        # Sub-grid turbulent jitter (kept small in open air so resolved eddies + Briggs sigmas
        # do not double-count dispersion; enhanced inside turbulent building wakes)
        advect_delta = w2 * dt
        if self.d_turb > 0.0:
            n_active = active_pos.shape[0]
            d_eff = self.d_turb + 0.25 * k_wake
            diff_std = np.sqrt(2.0 * d_eff * dt)
            noise = self._rng.normal(0.0, 1.0, size=(n_active, 3))
            noise *= diff_std[:, None]
            noise[:, 2] *= 0.72  # Vertical eddy diffusivity is smaller than horizontal
            new_pos = active_pos + advect_delta + noise
        else:
            new_pos = active_pos + advect_delta

        # Specular ground reflection at z = 0.20 m
        below_ground = new_pos[:, 2] < 0.20
        if below_ground.any():
            new_pos[below_ground, 2] = np.maximum(0.20, 0.40 - new_pos[below_ground, 2])

        # Specular mixing-lid reflection when mixing height is shallow
        above_lid = new_pos[:, 2] > self.mixing_height_m
        if above_lid.any():
            new_pos[above_lid, 2] = np.maximum(
                0.20, 2.0 * self.mixing_height_m - new_pos[above_lid, 2]
            )

        # Keep puffs outside solid buildings/tanks (porous pipe rack allows passage) & standalone pipes
        new_pos = self.world.push_outside_buildings(new_pos, margin=0.2, skip_porous=True, copy=False)
        new_pos = self.world.push_outside_pipes(new_pos, margin=0.05, copy=False)

        self.pos[mask] = new_pos
        self.age[mask] = active_age + dt
        # Effective dispersion age, travel distance, and wake-cavity variance
        wake_scale = 0.7 + 0.3 * turb_mult
        self.eff_age[mask] = active_eff_age + dt * wake_scale
        step_dist = np.hypot(advect_delta[:, 0], advect_delta[:, 1])
        self.eff_dist[mask] = active_eff_dist + step_dist * wake_scale
        self.wake_var[mask] = active_wake_var + 2.0 * k_wake * dt

        # Deactivate puffs that drift far outside facility bounds or exceed max age
        z_cutoff = max(60.0, min(self.mixing_height_m + 5.0, 120.0))
        out_of_bounds = (
            (np.abs(self.pos[:, 0]) > 85.0)
            | (np.abs(self.pos[:, 1]) > 85.0)
            | (self.pos[:, 2] > z_cutoff)
            | (self.age > self.MAX_AGE_S)
        )
        self.active[out_of_bounds] = False
        self._cache_dirty = True

    def _sigmas(
        self,
        eff_dists: np.ndarray,
        eff_ages: np.ndarray,
        wake_vars: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Briggs (1973) urban/industrial Pasquill-Gifford sigmas + initial size, calm floor & wake mixing."""
        ay, by, cy, az, bz, cz = _BRIGGS_URBAN_COEFFS[self.stability_class]
        x = np.maximum(eff_dists, 0.0)
        sy_briggs = ay * x * np.power(1.0 + by * x, cy) if by > 0.0 else ay * x
        sz_briggs = az * x * np.power(1.0 + bz * x, cz) if bz > 0.0 else az * x
        t = np.maximum(eff_ages, 0.0)
        wvar = 0.0 if wake_vars is None else np.maximum(wake_vars, 0.0)
        sigma_xy = np.sqrt(
            self.SIGMA0_XY * self.SIGMA0_XY + sy_briggs * sy_briggs + 2.0 * self.K_CALM_XY * t + wvar
        )
        sigma_z = np.sqrt(
            self.SIGMA0_Z * self.SIGMA0_Z + sz_briggs * sz_briggs + 2.0 * self.K_CALM_Z * t + 0.6 * wvar
        )
        return sigma_xy, sigma_z

    def sample_mass_concentration(self, pos: Vec3) -> float:
        """Sample gas mass concentration (kg/m^3) at a 3D point."""
        return self.sample_concentration(pos) / self.kg_m3_to_ppmv

    def sample_concentration(self, pos: Vec3) -> float:
        """Sample gas concentration (ppmv) at a 3D point with ground/lid reflection & wall occlusion."""
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
        dz_lid = (2.0 * self.mixing_height_m - pos[2]) - pz

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

        if n > self.MAX_RENDER_PUFFS:
            idx = np.linspace(0, n - 1, self.MAX_RENDER_PUFFS, dtype=np.intp)
            p_pos = p_pos[idx]
            ages = self._c_age[idx]
            sxy = self._c_sxy[idx]
            sz = self._c_sz[idx]
            n = self.MAX_RENDER_PUFFS
        else:
            ages = self._c_age
            sxy = self._c_sxy
            sz = self._c_sz

        opacity = 0.85 / (1.0 + 0.08 * ages)
        np.clip(opacity, 0.04, 0.85, out=opacity)

        xs = p_pos[:, 0].tolist()
        ys = p_pos[:, 1].tolist()
        zs = p_pos[:, 2].tolist()
        rxy = sxy.tolist()
        rz = sz.tolist()
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
