"""Physics-informed, coordinate-free 3D feature & candidate engine for TabPFN (v4).

Shared identically between:
- Offline multi-world dataset generation (`scripts/generate_tabpfn_datasets.py`)
- Live Triple-TabPFN leak detection agent (`agents/reliable_tabpfn.py`)

Design highlights:
- `PRE_SOURCE_FEATURE_COLUMNS` (12 features): asset/pipe priors + clean-air falsification
  for TabPFN Model B (Phase 1 pre-contact search & clean-air exclusion).
- `POST_SOURCE_FEATURE_COLUMNS` (29 features): 100% PURE PHYSICS + GEOMETRY for TabPFN
  Model A (Phase 2 post-contact localization) — zero `dist_to_asset` / `min_obs_dist`
  shortcuts, calibrated steady-state Gaussian plume inversion against `sim/plume.py`.
- Causal desorption-lag deconvolution (`ObservationHistory`) distinguishing low-ppm
  washout tail (`is_washout -> is_clean`) from turbulent drops inside real gas plumes
  (`>= 10 ppm`, `is_transient`, never falsely marked as clean air).
- Non-truncating two-scale upwind corridor candidate generator preserving exact indices
  back to `fac.grid_np`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from shared.schemas import DroneObservation, Vec2, Vec3, WorldInitPayload

# TabPFN Model B: Pre-contact spatial prior & clean-air falsification (12 features)
PRE_SOURCE_FEATURE_COLUMNS: list[str] = [
    "nearest_asset_type",
    "dist_to_asset",
    "asset_age_years",
    "dist_to_pipe",
    "pipe_pressure_bar",
    "pipe_is_gas",
    "wall_distance",
    "in_wind_shadow",
    "z",
    "downwind_clear_score",
    "onsite_clear_score",
    "min_obs_dist",
]

# TabPFN Model A: Post-contact pure-physics 3D source inversion (21 features, v8)
POST_SOURCE_FEATURE_COLUMNS: list[str] = [
    # Global calibrated 3-buoyancy MAP plume fit (7)
    "plume_cos_best",
    "plume_r2_q1",
    "plume_r2_qhat",
    "plume_rmse_qhat",
    "hit_rmse_qhat",
    "abs_log_q_hat",
    "hit_explain_frac",
    # Local-peak-window MAP plume fit (multi-leak & wind-shift resilient) (2)
    "local_r2_qhat",
    "local_r2_gap_to_best",
    # Strictly monotonic within-snapshot comparative physics (6)
    "r2_gap_to_best",
    "r2_q1_gap_to_best",
    "rmse_ratio_to_best",
    "cos_gap_to_best",
    "explain_gap_to_best",
    "upwind_hit_ratio_to_best",
    # Bounded directional, peak geometry & balanced infrastructure tie-breaker (6)
    "unexplained_hit_log",
    "upwind_gas_mass_frac",
    "downwind_clear_core_frac",
    "peak_cross_norm",
    "peak_dz_buoy_min",
    "infra_prox",
]

# Backward-compatible alias
SOURCE_FEATURE_COLUMNS: list[str] = POST_SOURCE_FEATURE_COLUMNS

# TabPFN Model #2 / #3: Online + coordinate-masked prior Plume & Hazard Mapper (14 features)
PLUME_FEATURE_COLUMNS: list[str] = [
    "mission_id",
    "x",
    "y",
    "z",
    "wind_x",
    "wind_y",
    "nearest_asset_type",
    "dist_to_asset",
    "asset_age_years",
    "dist_to_pipe",
    "pipe_pressure_bar",
    "pipe_is_gas",
    "in_wind_shadow",
    "upwind_pipe_score",
]

_ASSET_TYPE_MAP: dict[str, str] = {
    "valve": "valve",
    "flange": "flange",
    "storage_tank": "storage_tank",
    "pipe_junction": "flange",
}

_BASELINE_PPM = 0.50
_TAU_RISE_S = 0.60
_TAU_DECAY_S = 1.80
_PLUME_AMP_SCALE = 5000.0
_Q_GRID = np.array([0.45, 0.70, 1.00, 1.40, 2.00], dtype=np.float64)


@dataclass(frozen=True)
class ProcessedObs:
    """Single causal, deconvolved observation sample."""

    sim_time: float
    pos: np.ndarray          # (3,)
    wind_est: np.ndarray     # (2,) causal ambient wind estimate at sim_time
    wind_dropped: bool
    ppm_raw: float
    ppm_inst: float          # Desorption-lag deconvolved instantaneous gas estimate
    log_raw: float
    log_inst: float
    is_clean: bool           # True when sample reliably indicates clean air
    is_washout: bool         # True during low-ppm sensor tail desorption
    in_shadow: bool
    is_transient: bool = False  # True during rapid drop at high ppm (>=10 ppm)



class ObservationHistory:
    """Maintains causal wind estimation and sensor-lag deconvolution over full flight."""

    def __init__(self) -> None:
        self.samples: list[ProcessedObs] = []
        self._raw_winds: list[tuple[float, float, float, bool]] = []  # (t, wx, wy, in_shadow)
        self._open_winds: list[tuple[float, float, float]] = []  # (t, wx, wy)
        self._wind_ema = np.array([3.0, 0.5], dtype=np.float64)
        self._has_wind = False
        self._prev_y: float = 0.0
        self._prev_t: float | None = None
        self._decay_streak: int = 0

    def clear(self) -> None:
        self.samples.clear()
        self._raw_winds.clear()
        self._open_winds.clear()
        self._wind_ema = np.array([3.0, 0.5], dtype=np.float64)
        self._has_wind = False
        self._prev_y = 0.0
        self._prev_t = None
        self._decay_streak = 0

    @property
    def current_wind(self) -> Vec2:
        return (float(self._wind_ema[0]), float(self._wind_ema[1]))

    @property
    def has_wind(self) -> bool:
        return self._has_wind

    def append(self, obs: DroneObservation) -> ProcessedObs:
        t = float(obs.sim_time)
        if self.samples and t <= self.samples[-1].sim_time + 1e-9:
            return self.samples[-1]

        dropped = obs.wind_measured[0] is None or obs.wind_measured[1] is None
        if not dropped:
            wx = float(obs.wind_measured[0])
            wy = float(obs.wind_measured[1])
            in_sh = bool(obs.in_wind_shadow)
            if not in_sh:
                if self._has_wind:
                    spd_new = math.hypot(wx, wy)
                    spd_old = float(np.linalg.norm(self._wind_ema))
                    if spd_new >= 1.0 and spd_old >= 1.0:
                        cos_shift = (wx * float(self._wind_ema[0]) + wy * float(self._wind_ema[1])) / (spd_new * spd_old)
                        if cos_shift < 0.64:
                            # Snap immediately on >=~50 deg wind shift so EMA does not lag at intermediate angles
                            self._open_winds.clear()
                            self._raw_winds.clear()
                            self._wind_ema = np.array([wx, wy], dtype=np.float64)
                self._open_winds.append((t, wx, wy))
                self._open_winds = [w for w in self._open_winds[-12:] if t - w[0] <= 35.0]
                w_med = np.median([(w[1], w[2]) for w in self._open_winds], axis=0)
                if not self._has_wind:
                    self._wind_ema = w_med
                    self._has_wind = True
                else:
                    self._wind_ema = 0.50 * w_med + 0.50 * self._wind_ema
            else:
                self._raw_winds.append((t, wx, wy, in_sh))
                self._open_winds = [w for w in self._open_winds[-12:] if t - w[0] <= 35.0]
                if not self._open_winds:
                    # Fallback only when no open-air wind reading was seen recently
                    recent = [w for w in self._raw_winds[-10:] if t - w[0] <= 5.0]
                    w_med = np.median([(w[1], w[2]) for w in recent], axis=0)
                    if not self._has_wind:
                        self._wind_ema = w_med
                        self._has_wind = True
                    else:
                        self._wind_ema = 0.15 * w_med + 0.85 * self._wind_ema

        ppm_raw = float(obs.ppm_measured)
        y_curr = max(0.0, ppm_raw - _BASELINE_PPM)
        is_washout = False
        is_transient = False

        if self._prev_t is None:
            ppm_inst = y_curr
            self._decay_streak = 0
        else:
            dt = max(0.1, min(3.0, t - self._prev_t))
            if y_curr >= self._prev_y:
                self._decay_streak = 0
                alpha = max(1.0 - math.exp(-dt / _TAU_RISE_S), 0.12)
                deconv = self._prev_y + (y_curr - self._prev_y) / alpha
                ppm_inst = max(0.0, min(5000.0, deconv))
            else:
                y_floor = self._prev_y * math.exp(-dt / _TAU_DECAY_S)
                if self._prev_y >= 2.0 and y_curr <= 1.07 * y_floor:
                    self._decay_streak += 1
                else:
                    self._decay_streak = 0
                alpha = max(1.0 - math.exp(-dt / _TAU_DECAY_S), 0.12)
                deconv = self._prev_y + (y_curr - self._prev_y) / alpha
                ppm_inst = max(0.0, min(5000.0, deconv))

                # Only low readings (<10 ppm, or <18 ppm on 2+ step pure decay) can be empty-air washout.
                # Inside high-ppm plumes (>=10 ppm), turbulent dips are real gas (transient downweighted).
                is_washout = (
                    (self._decay_streak >= 1 and y_curr < 10.0 and ppm_inst < 2.2)
                    or (self._decay_streak >= 2 and y_curr < 18.0 and ppm_inst < 2.5)
                )
                if not is_washout and y_curr >= 10.0 and y_curr <= 1.07 * y_floor and ppm_inst < 0.25 * y_curr:
                    is_transient = True
                    ppm_inst = max(ppm_inst, 0.35 * y_curr)

        self._prev_y = y_curr
        self._prev_t = t

        is_clean = (y_curr < 1.6) or is_washout
        eff_inst = 0.0 if is_clean else ppm_inst

        rec = ProcessedObs(
            sim_time=t,
            pos=np.array(obs.pos, dtype=np.float64),
            wind_est=self._wind_ema.copy(),
            wind_dropped=dropped,
            ppm_raw=ppm_raw,
            ppm_inst=eff_inst,
            log_raw=math.log1p(max(0.0, ppm_raw)),
            log_inst=math.log1p(max(0.0, eff_inst)),
            is_clean=is_clean,
            is_washout=is_washout,
            in_shadow=bool(obs.in_wind_shadow),
            is_transient=is_transient,
        )
        self.samples.append(rec)
        return rec


class FacilityContext:
    """Precomputed vectorized facility geometry from `WorldInitPayload`."""

    def __init__(self, world_init: WorldInitPayload) -> None:
        self.world_init = world_init
        self.bounds_min = np.asarray(world_init.bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(world_init.bounds_max, dtype=np.float64)
        self.grid_np = np.array(world_init.grid_coords, dtype=np.float64)
        if self.grid_np.ndim != 2 or self.grid_np.shape[0] == 0:
            self.grid_np = np.zeros((0, 3), dtype=np.float64)

        all_b = list(world_init.buildings)
        macro_b = [b for b in all_b if b.kind != "pipe_segment"]
        solid_b = [b for b in macro_b if b.kind != "pipe_rack"]

        def _pack_boxes(boxes: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            if not boxes:
                return (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=np.float64),
                )
            bmin = np.array([b.min_xyz for b in boxes], dtype=np.float64)
            bmax = np.array([b.max_xyz for b in boxes], dtype=np.float64)
            center = 0.5 * (bmin + bmax)
            half = 0.5 * (bmax - bmin)
            is_cyl = np.array([b.kind == "storage_tank" for b in boxes], dtype=bool)
            is_por = np.array([b.kind == "pipe_rack" for b in boxes], dtype=bool)
            cyl_r = np.minimum(half[:, 0], half[:, 1])
            return bmin, bmax, center, half, is_cyl, is_por, cyl_r

        (
            self.bldg_min,
            self.bldg_max,
            self.bldg_center,
            self.bldg_half,
            self.bldg_is_cyl,
            self.bldg_is_porous,
            self.bldg_cyl_r,
        ) = _pack_boxes(macro_b)
        self.bldg_transmission = np.where(self.bldg_is_porous, 0.85, 0.05)

        (
            self.solid_min,
            self.solid_max,
            self.solid_center,
            self.solid_half,
            self.solid_is_cyl,
            _,
            self.solid_cyl_r,
        ) = _pack_boxes(solid_b)

        if world_init.assets:
            self.asset_pos = np.array([a.pos for a in world_init.assets], dtype=np.float64)
            self.asset_age = np.array([a.age_years for a in world_init.assets], dtype=np.float64)
            self.asset_types = [_ASSET_TYPE_MAP.get(a.asset_type, "open_air") for a in world_init.assets]
        else:
            self.asset_pos = np.zeros((0, 3), dtype=np.float64)
            self.asset_age = np.zeros(0, dtype=np.float64)
            self.asset_types = []

        seg_a: list[Vec3] = []
        seg_b: list[Vec3] = []
        seg_p: list[float] = []
        seg_gas: list[float] = []
        pipe_sample_pts: list[np.ndarray] = []

        for pipe in (world_init.pipes or world_init.pipe_runs):
            is_gas = 1.0 if (pipe.material == "gas" or (pipe.service and "gas" in pipe.service.lower())) else 0.0
            p_bar = float(
                pipe.pressure_bar
                if pipe.pressure_bar is not None
                else (pipe.pressureBar if pipe.pressureBar is not None else 10.0)
            )
            for j in range(len(pipe.points) - 1):
                a, b = pipe.points[j], pipe.points[j + 1]
                dist = math.dist(a, b)
                if dist > 0.05:
                    seg_a.append(a)
                    seg_b.append(b)
                    seg_p.append(p_bar)
                    seg_gas.append(is_gas)
                    n_sub = max(2, int(math.ceil(dist / 4.5)) + 1)
                    ts = np.linspace(0.0, 1.0, n_sub)[:, None]
                    pa = np.asarray(a, dtype=np.float64)[None, :]
                    pb = np.asarray(b, dtype=np.float64)[None, :]
                    pipe_sample_pts.append(pa + ts * (pb - pa))

        if seg_a:
            self.pipe_seg_a = np.array(seg_a, dtype=np.float64)
            self.pipe_seg_ab = np.array(seg_b, dtype=np.float64) - self.pipe_seg_a
            self.pipe_seg_len2 = np.maximum(np.sum(self.pipe_seg_ab ** 2, axis=1), 1e-9)
            self.pipe_seg_pressure = np.array(seg_p, dtype=np.float64)
            self.pipe_seg_is_gas = np.array(seg_gas, dtype=np.float64)
            raw_pipe_pts = np.vstack(pipe_sample_pts)
            self.pipe_points = self._dedup_points(raw_pipe_pts, voxel_m=2.5)
        else:
            self.pipe_seg_a = np.zeros((0, 3), dtype=np.float64)
            self.pipe_seg_ab = np.zeros((0, 3), dtype=np.float64)
            self.pipe_seg_len2 = np.zeros(0, dtype=np.float64)
            self.pipe_seg_pressure = np.zeros(0, dtype=np.float64)
            self.pipe_seg_is_gas = np.zeros(0, dtype=np.float64)
            self.pipe_points = np.zeros((0, 3), dtype=np.float64)

        parts = []
        if self.grid_np.shape[0] > 0:
            parts.append(self.grid_np)
        if self.asset_pos.shape[0] > 0:
            parts.append(self.asset_pos)
        if self.pipe_points.shape[0] > 0:
            parts.append(self.pipe_points)
        self.base_src_coords = (
            self._dedup_preserving_prefix(parts, prefix_len=self.grid_np.shape[0], voxel_m=1.8)
            if parts
            else np.zeros((0, 3), dtype=np.float64)
        )
        self.grid_static = self.static_features(self.grid_np)
        self.base_src_static = self.static_features(self.base_src_coords)

    @staticmethod
    def _dedup_points(pts: np.ndarray, voxel_m: float = 1.5) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        keys = np.round(pts / voxel_m).astype(np.int64)
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(first_idx)]

    @staticmethod
    def _dedup_preserving_prefix(parts: list[np.ndarray], prefix_len: int, voxel_m: float = 1.5) -> np.ndarray:
        all_pts = np.vstack(parts)
        if all_pts.shape[0] <= prefix_len:
            return all_pts
        seen: set[tuple[int, int, int]] = set()
        keep = list(range(prefix_len))
        for i in range(prefix_len):
            k = tuple(np.round(all_pts[i] / voxel_m).astype(np.int64).tolist())
            seen.add(k)
        for i in range(prefix_len, all_pts.shape[0]):
            k = tuple(np.round(all_pts[i] / voxel_m).astype(np.int64).tolist())
            if k not in seen:
                seen.add(k)
                keep.append(i)
        return all_pts[keep]

    def inside_solid_mask(self, pts: np.ndarray, margin: float = -0.15) -> np.ndarray:
        """True if points lie strictly inside solid buildings or cylindrical tanks."""
        n = pts.shape[0]
        inside = np.zeros(n, dtype=bool)
        if n == 0 or self.solid_min.shape[0] == 0:
            return inside
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        for i in range(self.solid_min.shape[0]):
            z0 = float(self.solid_min[i, 2])
            z1 = float(self.solid_max[i, 2]) + margin
            in_z = (pz >= z0) & (pz <= z1)
            if not np.any(in_z):
                continue
            if self.solid_is_cyl[i]:
                r = float(self.solid_cyl_r[i]) + margin
                if r <= 0:
                    continue
                dx = px - float(self.solid_center[i, 0])
                dy = py - float(self.solid_center[i, 1])
                inside |= in_z & ((dx * dx + dy * dy) <= r * r)
            else:
                inside |= (
                    in_z
                    & (px >= self.solid_min[i, 0] - margin)
                    & (px <= self.solid_max[i, 0] + margin)
                    & (py >= self.solid_min[i, 1] - margin)
                    & (py <= self.solid_max[i, 1] + margin)
                )
        return inside

    def static_features(
        self, pts: np.ndarray
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(asset_types, dist_asset, asset_age, dist_pipe, pipe_pressure, pipe_is_gas, wall_dist)`."""
        n = pts.shape[0]
        if n == 0:
            z = np.zeros(0, dtype=np.float64)
            return [], z, z, z, z, z, z

        if self.asset_pos.shape[0] > 0:
            dist_a = np.linalg.norm(pts[:, None, :] - self.asset_pos[None, :, :], axis=2)
            nearest_idx = np.argmin(dist_a, axis=1)
            dist_to_asset = dist_a[np.arange(n), nearest_idx]
        else:
            nearest_idx = np.zeros(n, dtype=np.intp)
            dist_to_asset = np.full(n, 99.0, dtype=np.float64)

        if self.bldg_min.shape[0] > 0:
            clamped = np.clip(pts[:, None, :], self.bldg_min[None, :, :], self.bldg_max[None, :, :])
            wall_dist = np.min(np.linalg.norm(pts[:, None, :] - clamped, axis=2), axis=1)
        else:
            wall_dist = np.full(n, 99.0, dtype=np.float64)

        asset_types: list[str] = []
        asset_age = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if dist_to_asset[i] <= 12.0 and self.asset_pos.shape[0] > 0:
                idx = int(nearest_idx[i])
                asset_types.append(self.asset_types[idx])
                asset_age[i] = float(self.asset_age[idx])
            elif wall_dist[i] <= 5.0:
                asset_types.append("wall")
                asset_age[i] = 10.0
            else:
                asset_types.append("open_air")
                asset_age[i] = 0.0

        if self.pipe_seg_a.shape[0] > 0:
            ap = pts[:, None, :] - self.pipe_seg_a[None, :, :]
            t = np.clip(
                np.sum(ap * self.pipe_seg_ab[None, :, :], axis=2) / self.pipe_seg_len2[None, :],
                0.0,
                1.0,
            )
            closest = self.pipe_seg_a[None, :, :] + t[:, :, None] * self.pipe_seg_ab[None, :, :]
            dist_p = np.linalg.norm(pts[:, None, :] - closest, axis=2)
            best_seg = np.argmin(dist_p, axis=1)
            dist_to_pipe = dist_p[np.arange(n), best_seg]
            near_pipe = dist_to_pipe <= 12.0
            pipe_pressure = np.where(near_pipe, self.pipe_seg_pressure[best_seg], 0.0)
            pipe_is_gas = np.where(near_pipe, self.pipe_seg_is_gas[best_seg], 0.0)
        else:
            dist_to_pipe = np.full(n, 99.0, dtype=np.float64)
            pipe_pressure = np.zeros(n, dtype=np.float64)
            pipe_is_gas = np.zeros(n, dtype=np.float64)

        return asset_types, dist_to_asset, asset_age, dist_to_pipe, pipe_pressure, pipe_is_gas, wall_dist

    def wind_features(self, pts: np.ndarray, wind_vec: Vec2) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized `(in_wind_shadow, upwind_pipe_score)` for (N, 3) points."""
        n = pts.shape[0]
        if n == 0:
            z = np.zeros(0, dtype=np.float64)
            return z, z

        wx, wy = float(wind_vec[0]), float(wind_vec[1])
        speed = math.hypot(wx, wy)
        in_shadow = np.zeros(n, dtype=np.float64)
        upwind_score = np.zeros(n, dtype=np.float64)
        if speed < 1e-3:
            return in_shadow, upwind_score

        ux, uy = wx / speed, wy / speed
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

        for i in range(self.bldg_center.shape[0]):
            cx, cy, _ = self.bldg_center[i]
            hx, hy, _ = self.bldg_half[i]
            height = float(self.bldg_max[i, 2])
            if self.bldg_is_cyl[i]:
                r = float(self.bldg_cyl_r[i])
                half_along, half_cross = r, r
            else:
                half_along = hx * abs(ux) + hy * abs(uy)
                half_cross = hx * abs(uy) + hy * abs(ux)

            dx = px - cx
            dy = py - cy
            along = dx * ux + dy * uy
            cross = np.abs(-dx * uy + dy * ux)
            hit = (
                (pz <= height * 1.15)
                & (along > half_along)
                & (along <= half_along + 2.5 * height)
                & (cross <= half_cross * 1.1)
            )
            in_shadow = np.where(hit, 1.0, in_shadow)

        if self.asset_pos.shape[0] > 0:
            dx_a = px[:, None] - self.asset_pos[None, :, 0]
            dy_a = py[:, None] - self.asset_pos[None, :, 1]
            along = dx_a * ux + dy_a * uy
            cross = np.abs(-dx_a * uy + dy_a * ux)
            mask = (along > 0.0) & (along < 45.0)
            age_w = 0.5 + 0.5 * (self.asset_age[None, :] / 25.0)
            cone_sigma = 3.0 + 0.25 * np.maximum(along, 0.0)
            contrib = np.where(
                mask,
                age_w
                * np.exp(-0.5 * (cross / cone_sigma) ** 2)
                * np.exp(-np.maximum(along, 0.0) / 30.0),
                0.0,
            )
            upwind_score = np.clip(np.sum(contrib, axis=1) / 2.5, 0.0, 1.0)

        return in_shadow, upwind_score

    def segment_transmission_pairs(self, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        """Vectorized gas transmission in (0, 1] for K ray pairs `(p_from[k] -> p_to[k])`."""
        k = p_from.shape[0]
        trans = np.ones(k, dtype=np.float64)
        if k == 0 or self.bldg_min.shape[0] == 0:
            return trans

        # Reject rays whose segment bounds cannot intersect each building. The
        # precise intersection calculation below still uses every possible ray.
        lower = np.minimum(p_from, p_to) - 1e-9
        upper = np.maximum(p_from, p_to) + 1e-9
        for i in range(self.bldg_min.shape[0]):
            x0, y0, z0 = map(float, self.bldg_min[i])
            x1, y1, z1 = map(float, self.bldg_max[i])
            possible = ((lower[:, 0] <= x1) & (upper[:, 0] >= x0) &
                        (lower[:, 1] <= y1) & (upper[:, 1] >= y0) &
                        (lower[:, 2] <= z1) & (upper[:, 2] >= z0))
            indices = np.flatnonzero(possible)
            if not indices.size:
                continue
            px, py, pz = p_from[indices].T
            dx, dy, dz = (p_to[indices] - p_from[indices]).T
            par_x, par_y, par_z = np.abs(dx) < 1e-9, np.abs(dy) < 1e-9, np.abs(dz) < 1e-9
            inv_dx = np.where(par_x, 0.0, 1.0 / np.where(par_x, 1.0, dx))
            inv_dy = np.where(par_y, 0.0, 1.0 / np.where(par_y, 1.0, dy))
            inv_dz = np.where(par_z, 0.0, 1.0 / np.where(par_z, 1.0, dz))
            has_px, has_py, has_pz = par_x.any(), par_y.any(), par_z.any()
            bldg_trans = float(self.bldg_transmission[i])

            tz1 = (z0 - pz) * inv_dz
            tz2 = (z1 - pz) * inv_dz
            tz_enter = np.minimum(tz1, tz2)
            tz_exit = np.maximum(tz1, tz2)
            if has_pz:
                in_z = (pz >= z0) & (pz <= z1)
                tz_enter = np.where(par_z, np.where(in_z, -1e9, 1e9), tz_enter)
                tz_exit = np.where(par_z, np.where(in_z, 1e9, -1e9), tz_exit)

            if self.bldg_is_cyl[i]:
                cx, cy = float(self.bldg_center[i, 0]), float(self.bldg_center[i, 1])
                r2 = float(self.bldg_cyl_r[i]) ** 2
                ox = px - cx
                oy = py - cy
                a = dx * dx + dy * dy
                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - r2
                disc = b * b - 4.0 * a * c
                vert = a < 1e-9
                safe_a = np.where(vert, 1.0, a)
                sqrt_d = np.sqrt(np.maximum(disc, 0.0))
                inv_2a = 0.5 / safe_a
                tc_enter = np.where(
                    vert,
                    np.where(c <= 0.0, -1e9, 1e9),
                    np.where(disc >= 0.0, (-b - sqrt_d) * inv_2a, 1e9),
                )
                tc_exit = np.where(
                    vert,
                    np.where(c <= 0.0, 1e9, -1e9),
                    np.where(disc >= 0.0, (-b + sqrt_d) * inv_2a, -1e9),
                )
                t_enter = np.maximum(tc_enter, tz_enter)
                t_exit = np.minimum(tc_exit, tz_exit)
            else:
                tx1 = (x0 - px) * inv_dx
                tx2 = (x1 - px) * inv_dx
                tx_enter = np.minimum(tx1, tx2)
                tx_exit = np.maximum(tx1, tx2)
                if has_px:
                    in_x = (px >= x0) & (px <= x1)
                    tx_enter = np.where(par_x, np.where(in_x, -1e9, 1e9), tx_enter)
                    tx_exit = np.where(par_x, np.where(in_x, 1e9, -1e9), tx_exit)

                ty1 = (y0 - py) * inv_dy
                ty2 = (y1 - py) * inv_dy
                ty_enter = np.minimum(ty1, ty2)
                ty_exit = np.maximum(ty1, ty2)
                if has_py:
                    in_y = (py >= y0) & (py <= y1)
                    ty_enter = np.where(par_y, np.where(in_y, -1e9, 1e9), ty_enter)
                    ty_exit = np.where(par_y, np.where(in_y, 1e9, -1e9), ty_exit)

                t_enter = np.maximum(np.maximum(tx_enter, ty_enter), tz_enter)
                t_exit = np.minimum(np.minimum(tx_exit, ty_exit), tz_exit)

            hit = (t_exit >= t_enter) & (t_exit > 0.04) & (t_enter < 0.96)
            if hit.any():
                trans[indices[hit]] *= bldg_trans

        return trans


# ----------------------------------------------------------------------
# Candidate Generation (Non-Truncating Two-Scale Upwind Corridors)
# ----------------------------------------------------------------------

def _plausibility_grid_mask(
    fac: FacilityContext,
    anchors: Sequence[ProcessedObs],
    wind_now: Vec2,
    hits: Sequence[ProcessedObs],
) -> tuple[np.ndarray, np.ndarray]:
    """Select plausible `grid_np` points + return `(grid_subset, original_indices)`."""
    g = fac.grid_np
    ng = g.shape[0]
    if ng == 0:
        return g, np.zeros(0, dtype=np.intp)

    hit_arr = np.array([h.pos for h in hits], dtype=np.float64)
    d_hit = np.min(np.linalg.norm(g[:, None, :] - hit_arr[None, :, :], axis=2), axis=1)
    keep = d_hit <= 18.0

    for anc in anchors:
        wx, wy = float(anc.wind_est[0]), float(anc.wind_est[1])
        spd = math.hypot(wx, wy)
        if spd < 0.25:
            wx, wy = float(wind_now[0]), float(wind_now[1])
            spd = math.hypot(wx, wy)
        ux, uy = (wx / spd, wy / spd) if spd >= 0.2 else (1.0, 0.0)
        dx = anc.pos[0] - g[:, 0]
        dy = anc.pos[1] - g[:, 1]
        dz = np.abs(anc.pos[2] - g[:, 2])
        along = dx * ux + dy * uy
        cross = np.abs(-dx * uy + dy * ux)
        keep |= (along >= -6.0) & (along <= 58.0) & (cross <= 22.0) & (dz <= 16.0)

    idx = np.nonzero(keep)[0]
    if idx.size > 260:
        p0 = anchors[0]
        score = d_hit[idx] * 0.5 + 0.4 * np.linalg.norm(g[idx] - p0.pos[None, :], axis=1)
        idx = idx[np.argsort(score)[:260]]
        idx.sort()
    return g[idx], idx


def build_post_source_candidates(
    fac: FacilityContext,
    history: Sequence[ProcessedObs],
    wind_now: Vec2,
    prev_estimate: Vec3 | None = None,
    max_candidates: int = 950,
) -> tuple[np.ndarray, np.ndarray]:
    """Build multi-resolution 3D candidates for Model A and track `grid_np` indices.

    Returns:
        `(candidates_Mx3, prefix_grid_indices)` where the first `len(prefix_grid_indices)`
        rows correspond 1:1 to `fac.grid_np[prefix_grid_indices]`.
    """
    hits = [r for r in history if (not r.is_clean) or r.ppm_raw >= 3.5]
    if not hits:
        grid_idx = np.arange(fac.grid_np.shape[0], dtype=np.intp)
        return fac.base_src_coords, grid_idx

    ranked = sorted(hits, key=lambda r: (r.ppm_inst + 0.20 * r.ppm_raw), reverse=True)
    anchors: list[ProcessedObs] = [ranked[0]]
    for cand in ranked[1:]:
        if float(np.linalg.norm(cand.pos - anchors[0].pos)) >= 8.0:
            anchors.append(cand)
            if len(anchors) >= 2:
                break

    grid_sub, grid_idx = _plausibility_grid_mask(fac, anchors, wind_now, hits)
    base_parts: list[np.ndarray] = []
    if grid_sub.shape[0] > 0:
        base_parts.append(grid_sub)
    if fac.asset_pos.shape[0] > 0:
        base_parts.append(fac.asset_pos)
    if fac.pipe_points.shape[0] > 0:
        p0 = anchors[0]
        d_p0 = np.linalg.norm(fac.pipe_points - p0.pos[None, :], axis=1)
        if fac.pipe_points.shape[0] <= 90:
            base_parts.append(fac.pipe_points)
        else:
            base_parts.append(fac.pipe_points[np.argsort(d_p0)[:90]])

    base = (
        fac._dedup_preserving_prefix(base_parts, prefix_len=grid_sub.shape[0], voxel_m=1.6)
        if base_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    n_base = base.shape[0]

    lo_b = fac.bounds_min + np.array([0.8, 0.8, 0.8])
    hi_b = fac.bounds_max - np.array([0.8, 0.8, 0.8])
    extra_list: list[np.ndarray] = []

    p0 = anchors[0]
    for anc_idx, anc in enumerate(anchors):
        wx, wy = float(anc.wind_est[0]), float(anc.wind_est[1])
        spd = math.hypot(wx, wy)
        if spd < 0.25:
            wx, wy = float(wind_now[0]), float(wind_now[1])
            spd = math.hypot(wx, wy)
        ux, uy = (wx / spd, wy / spd) if spd >= 0.2 else (1.0, 0.0)
        cx, cy = -uy, ux

        if anc_idx == 0:
            # Fine upwind box [-18..+6m along]
            ag, cg, zg = np.meshgrid(
                np.arange(-18.0, 6.5, 2.5),
                np.arange(-9.0, 9.5, 2.5),
                np.arange(-6.0, 7.5, 2.5),
                indexing="ij",
            )
            corr_f = np.tile(anc.pos, (ag.size, 1))
            corr_f[:, 0] += ag.ravel() * ux + cg.ravel() * cx
            corr_f[:, 1] += ag.ravel() * uy + cg.ravel() * cy
            corr_f[:, 2] += zg.ravel()
            extra_list.append(corr_f)

            # Far upwind corridor [-46..-18m along]
            ag2, cg2, zg2 = np.meshgrid(
                np.arange(-46.0, -18.0, 4.0),
                np.arange(-14.0, 14.5, 4.0),
                np.arange(-8.0, 10.5, 3.5),
                indexing="ij",
            )
            corr_far = np.tile(anc.pos, (ag2.size, 1))
            corr_far[:, 0] += ag2.ravel() * ux + cg2.ravel() * cx
            corr_far[:, 1] += ag2.ravel() * uy + cg2.ravel() * cy
            corr_far[:, 2] += zg2.ravel()
            extra_list.append(corr_far)
        else:
            # Coarse corridor around secondary hit (supports multi-leak / wind shifts)
            ag, cg, zg = np.meshgrid(
                np.arange(-26.0, 6.0, 4.0),
                np.arange(-9.0, 9.5, 4.0),
                np.arange(-5.0, 6.5, 3.5),
                indexing="ij",
            )
            corr2 = np.tile(anc.pos, (ag.size, 1))
            corr2[:, 0] += ag.ravel() * ux + cg.ravel() * cx
            corr2[:, 1] += ag.ravel() * uy + cg.ravel() * cy
            corr2[:, 2] += zg.ravel()
            extra_list.append(corr2)

        # Dense 2m cube around anchor
        dx_c, dy_c, dz_c = np.meshgrid(
            np.arange(-6.0, 6.5, 2.0),
            np.arange(-6.0, 6.5, 2.0),
            np.arange(-5.0, 5.5, 2.0),
            indexing="ij",
        )
        extra_list.append(anc.pos[None, :] + np.c_[dx_c.ravel(), dy_c.ravel(), dz_c.ravel()])

    # 360-degree wake ring if primary peak is in shadow or close to a structure
    near_bldg = False
    if fac.bldg_min.shape[0] > 0:
        cl = np.clip(p0.pos[None, :], fac.bldg_min, fac.bldg_max)
        near_bldg = float(np.min(np.linalg.norm(p0.pos[None, :] - cl, axis=1))) <= 8.0
    if p0.in_shadow or near_bldg:
        angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        ring_pts = []
        for rad in (3.0, 6.0, 10.0):
            for dz in (-3.0, 0.0, 3.0):
                for ang in angles:
                    ring_pts.append([p0.pos[0] + rad * math.cos(ang), p0.pos[1] + rad * math.sin(ang), p0.pos[2] + dz])
        extra_list.append(np.asarray(ring_pts, dtype=np.float64))

    # 1.5m micro-refinement cube around previous estimate
    if prev_estimate is not None:
        est_arr = np.asarray(prev_estimate, dtype=np.float64)
        rx, ry, rz = np.meshgrid(
            np.arange(-4.5, 5.0, 1.5),
            np.arange(-4.5, 5.0, 1.5),
            np.arange(-3.0, 3.5, 1.5),
            indexing="ij",
        )
        extra_list.append(est_arr[None, :] + np.c_[rx.ravel(), ry.ravel(), rz.ravel()])

    extra = np.vstack(extra_list)
    extra = extra[np.all((extra >= lo_b) & (extra <= hi_b), axis=1)]
    if extra.shape[0] > 0:
        extra = extra[~fac.inside_solid_mask(extra, margin=-0.15)]

    merged = fac._dedup_preserving_prefix([base, extra], prefix_len=n_base, voxel_m=1.4)
    if merged.shape[0] > max_candidates:
        # Prioritize by upwind-plume alignment wrt p0 rather than raw proximity to p0!
        fine = merged[n_base:]
        wx0, wy0 = float(p0.wind_est[0]), float(p0.wind_est[1])
        spd0 = math.hypot(wx0, wy0)
        if spd0 < 0.25:
            wx0, wy0 = float(wind_now[0]), float(wind_now[1])
            spd0 = math.hypot(wx0, wy0)
        ux0, uy0 = (wx0 / spd0, wy0 / spd0) if spd0 >= 0.2 else (1.0, 0.0)
        dp_x = p0.pos[0] - fine[:, 0]
        dp_y = p0.pos[1] - fine[:, 1]
        al = dp_x * ux0 + dp_y * uy0
        cr = np.abs(-dp_x * uy0 + dp_y * ux0)
        dz = np.abs(p0.pos[2] - fine[:, 2])
        sigma_c = 4.0 + 0.22 * np.maximum(al, 0.0)
        plaus = np.exp(-0.5 * (cr / sigma_c) ** 2 - 0.5 * (dz / 6.5) ** 2) * np.where(
            (al >= -3.0) & (al <= 48.0), 1.0, np.exp(-0.5 * (np.minimum(al + 3.0, 48.0 - al) / 4.0) ** 2)
        )
        if prev_estimate is not None:
            est_arr = np.asarray(prev_estimate, dtype=np.float64)
            d_est = np.linalg.norm(fine - est_arr[None, :], axis=1)
            plaus += 0.35 * np.exp(-0.5 * (d_est / 4.5) ** 2)
        keep_n = max(0, max_candidates - n_base)
        keep_idx = np.argsort(plaus)[::-1][:keep_n]
        merged = np.vstack([merged[:n_base], fine[keep_idx]])

    return merged, grid_idx


def build_source_candidates(
    fac: FacilityContext,
    history: Sequence[ProcessedObs],
    wind_now: Vec2,
    prev_estimate: Vec3 | None = None,
    max_candidates: int = 950,
) -> np.ndarray:
    """Backward-compatible wrapper returning candidate coordinates only."""
    cands, _ = build_post_source_candidates(fac, history, wind_now, prev_estimate=prev_estimate, max_candidates=max_candidates)
    return cands


# ----------------------------------------------------------------------
# Feature Builders (Pre-Contact Model B + Post-Contact Model A)
# ----------------------------------------------------------------------

def _compute_plume_physics_tensors(
    fac: FacilityContext,
    coords: np.ndarray,
    history: Sequence[ProcessedObs],
    wind_now: Vec2,
) -> dict[str, Any]:
    """Shared vectorized steady-state plume & observation-comparison computation."""
    m = coords.shape[0]
    n = len(history)
    obs_t = np.array([r.sim_time for r in history], dtype=np.float64)
    obs_pos = np.array([r.pos for r in history], dtype=np.float64)
    obs_wind = np.array([r.wind_est for r in history], dtype=np.float64)
    obs_ppm_inst = np.array([r.ppm_inst for r in history], dtype=np.float64)
    obs_log_inst = np.array([r.log_inst for r in history], dtype=np.float64)
    obs_is_clean = np.array([r.is_clean for r in history], dtype=bool)
    obs_in_shadow = np.array([r.in_shadow for r in history], dtype=bool)
    obs_wind_drop = np.array([r.wind_dropped for r in history], dtype=bool)
    obs_is_transient = np.array([r.is_transient for r in history], dtype=bool)

    in_shadow, _ = fac.wind_features(coords, wind_now)

    dx = obs_pos[None, :, 0] - coords[:, None, 0]
    dy = obs_pos[None, :, 1] - coords[:, None, 1]
    dz = obs_pos[None, :, 2] - coords[:, None, 2]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    min_obs_dist = np.min(dist, axis=1)

    spd = np.hypot(obs_wind[:, 0], obs_wind[:, 1])
    valid_spd = np.maximum(spd, 0.25)
    ux = np.where(spd >= 0.2, obs_wind[:, 0] / valid_spd, 1.0)
    uy = np.where(spd >= 0.2, obs_wind[:, 1] / valid_spd, 0.0)

    along = dx * ux[None, :] + dy * uy[None, :]
    cross = np.abs(-dx * uy[None, :] + dy * ux[None, :])
    downwind = np.maximum(along, 0.0)

    u_eff = np.maximum(spd[None, :], 0.85)
    t_adv = downwind / u_eff
    in_wake_pair = (in_shadow[:, None] > 0.5) | obs_in_shadow[None, :]
    eff_age = t_adv * np.where(in_wake_pair, 1.40, 1.0) + 0.15
    sxy_phys = 0.8 + 0.35 * np.power(eff_age, 0.85)
    sz_phys = 0.5 + 0.20 * np.power(eff_age, 0.75)
    sigma_y = np.sqrt(sxy_phys ** 2 + 0.55 * eff_age + (0.14 * downwind) ** 2) + np.where(in_wake_pair, 2.4, 0.45)
    sigma_z = np.sqrt(sz_phys ** 2 + 0.30 * eff_age + (0.08 * downwind) ** 2) + np.where(in_wake_pair, 1.3, 0.35)
    sigma_up = np.where(in_wake_pair, 4.0, 1.35)

    horiz = np.exp(-0.5 * np.minimum((cross / sigma_y) ** 2, 35.0))
    horiz_core = np.exp(-0.5 * np.minimum((cross / (0.72 * sigma_y)) ** 2, 35.0))
    along_taper = np.where(
        along >= 0.0,
        1.0,
        np.exp(-0.5 * np.minimum((along / sigma_up) ** 2, 35.0)),
    )
    horiz_ok = (along >= -12.0) & (along <= 75.0) & ((cross / sigma_y) < 4.5)
    amp = _PLUME_AMP_SCALE / ((u_eff + 1.5) * sxy_phys * sz_phys * (1.0 + 0.025 * downwind))
    base_pref = amp * along_taper * horiz
    base_core = amp * along_taper * horiz_core

    def _conc_raw(b_vz: float) -> tuple[np.ndarray, np.ndarray]:
        z_plume = np.clip(coords[:, None, 2] + b_vz * t_adv / (1.0 + 0.035 * t_adv), 0.3, 29.5)
        dz_dir = obs_pos[None, :, 2] - z_plume
        dz_gnd = obs_pos[None, :, 2] + z_plume
        dz_lid = (60.0 - obs_pos[None, :, 2]) - z_plume
        vert = (
            np.exp(-0.5 * np.minimum((dz_dir / sigma_z) ** 2, 35.0))
            + np.exp(-0.5 * np.minimum((dz_gnd / sigma_z) ** 2, 35.0))
            + np.exp(-0.5 * np.minimum((dz_lid / sigma_z) ** 2, 35.0))
        )
        sz_c = 0.72 * sigma_z
        vert_core = (
            np.exp(-0.5 * np.minimum((dz_dir / sz_c) ** 2, 35.0))
            + np.exp(-0.5 * np.minimum((dz_gnd / sz_c) ** 2, 35.0))
            + np.exp(-0.5 * np.minimum((dz_lid / sz_c) ** 2, 35.0))
        )
        return np.where(horiz_ok, base_pref * vert, 0.0), np.where(horiz_ok, base_core * vert_core, 0.0)

    c_raw_b, cc_raw_b = _conc_raw(0.22)
    c_raw_n, cc_raw_n = _conc_raw(0.0)
    c_raw_h, cc_raw_h = _conc_raw(-0.22)
    c_raw_max = np.maximum(np.maximum(c_raw_b, c_raw_n), c_raw_h)

    trans = np.ones((m, n), dtype=np.float64)
    occ_mask = c_raw_max > 0.35
    if np.any(occ_mask) and fac.bldg_min.shape[0] > 0:
        mi, nj = np.nonzero(occ_mask)
        raw_tr = fac.segment_transmission_pairs(coords[mi], obs_pos[nj])
        # Roof-lifted 3D obstacle transmission: blocks tall walls in wake traps while clearing low roofs
        far_m = dist[mi, nj] > 10.0
        if np.any(far_m):
            mi_f, nj_f = mi[far_m], nj[far_m]
            p_from_lift = coords[mi_f].copy()
            p_from_lift[:, 2] = np.minimum(29.0, p_from_lift[:, 2] + 3.0)
            p_to_lift = obs_pos[nj_f].copy()
            p_to_lift[:, 2] = np.minimum(29.0, p_to_lift[:, 2] + 3.0)
            lift_tr = fac.segment_transmission_pairs(p_from_lift, p_to_lift)
            eff_tr = raw_tr.copy()
            eff_tr[far_m] = np.maximum(raw_tr[far_m], 0.55 * lift_tr)
            trans[mi, nj] = eff_tr
        else:
            trans[mi, nj] = raw_tr

    c_b = c_raw_b * trans
    c_n = c_raw_n * trans
    c_h = c_raw_h * trans
    c_max = c_raw_max * trans
    c_core_b = cc_raw_b * trans
    c_core_n = cc_raw_n * trans
    c_core_h = cc_raw_h * trans

    now_t = history[-1].sim_time
    age_w = np.exp(-(now_t - obs_t) / 110.0)
    wind_w = np.where(obs_wind_drop, 0.75, 1.0)
    trans_w = np.where(obs_is_transient, 0.25, 1.0)
    w_obs = age_w * wind_w * trans_w

    last_shift_t = -1e9
    if n > 2:
        prev_2s = np.clip(np.searchsorted(obs_t, obs_t - 2.0, side="left"), 0, n - 1)
        dot_1 = ux[1:] * ux[:-1] + uy[1:] * uy[:-1]
        dot_2s = ux[1:] * ux[prev_2s[1:]] + uy[1:] * uy[prev_2s[1:]]
        spd_ok = (spd[1:] >= 0.5) & (spd[:-1] >= 0.5)
        shift_mask = spd_ok & ((dot_1 < 0.64) | (dot_2s < 0.64))
        if np.any(shift_mask):
            shift_i = int(np.flatnonzero(shift_mask)[-1]) + 1
            last_shift_t = float(obs_t[shift_i])
            bent_mask = (obs_t >= last_shift_t) & (obs_t <= last_shift_t + 6.0)
            w_obs = np.where(bent_mask, w_obs * 0.25, w_obs)
            has_post_hits = bool(
                np.any((obs_t > last_shift_t + 5.0) & (~obs_is_clean) & (obs_log_inst > 0.45))
            )
            if has_post_hits:
                w_obs = np.where(obs_t < last_shift_t, w_obs * np.exp(-(now_t - obs_t) / 22.0), w_obs)

    clean_trust = np.ones((m, n), dtype=np.float64)
    obs_wind_stable = np.ones(n, dtype=np.float64)
    if n > 2:
        t_prev = obs_t[None, :] - np.clip(t_adv, 0.5, 10.0)
        prev_idx = np.clip(np.searchsorted(obs_t, t_prev, side="left"), 0, n - 1)
        cos_w = ux[prev_idx] * ux[None, :] + uy[prev_idx] * uy[None, :]
        clean_trust = np.where(along > 3.0, np.clip((cos_w - 0.50) / 0.35, 0.0, 1.0), 1.0)
        prev_6s = np.clip(np.searchsorted(obs_t, obs_t - 6.0, side="left"), 0, n - 1)
        obs_wind_stable = ux[prev_6s] * ux + uy[prev_6s] * uy

    return {
        "in_shadow": in_shadow,
        "obs_t": obs_t,
        "last_shift_t": last_shift_t,
        "obs_pos": obs_pos,
        "obs_wind": obs_wind,
        "obs_wind_stable": obs_wind_stable,
        "obs_ppm_inst": obs_ppm_inst,
        "obs_log_inst": obs_log_inst,
        "obs_is_clean": obs_is_clean,
        "obs_in_shadow": obs_in_shadow,
        "obs_is_transient": obs_is_transient,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "dist": dist,
        "min_obs_dist": min_obs_dist,
        "along": along,
        "cross": cross,
        "c_b": c_b,
        "c_n": c_n,
        "c_h": c_h,
        "c_max": c_max,
        "c_core_b": c_core_b,
        "c_core_n": c_core_n,
        "c_core_h": c_core_h,
        "trans": trans,
        "w_obs": w_obs,
        "clean_trust": clean_trust,
    }


def build_pre_source_feature_rows(
    fac: FacilityContext,
    coords: np.ndarray,
    history: Sequence[ProcessedObs],
    wind_now: Vec2,
    use_base_static: bool = False,
) -> list[list[Any]]:
    """Build 12 features for TabPFN Model B (Phase 1 pre-contact search & clean-air exclusion)."""
    m = coords.shape[0]
    if m == 0:
        return []

    if use_base_static and m == fac.base_src_coords.shape[0]:
        types, d_asset, age, d_pipe, press, is_gas, wall_d = fac.base_src_static
    else:
        types, d_asset, age, d_pipe, press, is_gas, wall_d = fac.static_features(coords)

    in_shadow, _ = fac.wind_features(coords, wind_now)
    zs = coords[:, 2]

    downwind_clear_score = np.zeros(m, dtype=np.float64)
    onsite_clear_score = np.zeros(m, dtype=np.float64)
    min_obs_dist = np.full(m, 60.0, dtype=np.float64)

    if history:
        ten = _compute_plume_physics_tensors(fac, coords, history, wind_now)
        min_obs_dist = ten["min_obs_dist"]
        obs_is_clean = ten["obs_is_clean"]
        if np.any(obs_is_clean):
            c_idx = np.nonzero(obs_is_clean)[0]
            w_c = ten["w_obs"][c_idx][None, :]
            d_c = ten["dist"][:, c_idx]
            near_clear = np.where(d_c < 5.5, w_c * np.exp(-0.5 * (d_c / 3.3) ** 2), 0.0)
            onsite_clear_score = np.log1p(np.sum(near_clear, axis=1))

            ct_c = ten["clean_trust"][:, c_idx]
            c_c = ten["c_max"][:, c_idx]
            al_c = ten["along"][:, c_idx]
            dw_weight = np.where(al_c > 0.5, w_c * ct_c * np.minimum(np.log1p(c_c) / 3.5, 1.0), 0.0)
            downwind_clear_score = np.log1p(np.sum(dw_weight, axis=1))

    return [
        [
            types[i],
            round(float(d_asset[i]), 3),
            round(float(age[i]), 2),
            round(float(d_pipe[i]), 3),
            round(float(press[i]), 2),
            float(is_gas[i]),
            round(float(wall_d[i]), 3),
            float(in_shadow[i]),
            round(float(zs[i]), 2),
            round(float(downwind_clear_score[i]), 4),
            round(float(onsite_clear_score[i]), 4),
            round(float(min_obs_dist[i]), 3),
        ]
        for i in range(m)
    ]


def build_post_source_feature_rows(
    fac: FacilityContext,
    coords: np.ndarray,
    history: Sequence[ProcessedObs],
    wind_now: Vec2,
    use_base_static: bool = False,
) -> list[list[Any]]:
    """Build 21 monotonic MAP plume-physics + balanced infra tie-breaker features for TabPFN Model A."""
    m = coords.shape[0]
    if m == 0:
        return []

    if use_base_static and m == fac.base_src_coords.shape[0]:
        d_asset = fac.base_src_static[1]
        d_pipe = fac.base_src_static[3]
    elif m >= fac.base_src_coords.shape[0] and fac.base_src_coords.shape[0] > 0 and np.array_equal(coords[:fac.base_src_coords.shape[0]], fac.base_src_coords):
        nb = fac.base_src_coords.shape[0]
        b_da, b_dp = fac.base_src_static[1], fac.base_src_static[3]
        if m > nb:
            ext_st = fac.static_features(coords[nb:])
            d_asset = np.concatenate([b_da, ext_st[1]])
            d_pipe = np.concatenate([b_dp, ext_st[3]])
        else:
            d_asset, d_pipe = b_da, b_dp
    else:
        st_f = fac.static_features(coords)
        d_asset, d_pipe = st_f[1], st_f[3]

    infra_prox = np.exp(-np.minimum(d_asset, d_pipe) / 3.5)

    plume_cos_best = np.zeros(m, dtype=np.float64)
    plume_r2_q1 = np.zeros(m, dtype=np.float64)
    plume_r2_qhat = np.zeros(m, dtype=np.float64)
    plume_rmse_qhat = np.zeros(m, dtype=np.float64)
    hit_rmse_qhat = np.zeros(m, dtype=np.float64)
    abs_log_q_hat = np.zeros(m, dtype=np.float64)
    hit_explain_frac = np.zeros(m, dtype=np.float64)

    local_r2_qhat = np.zeros(m, dtype=np.float64)
    local_r2_gap_to_best = np.zeros(m, dtype=np.float64)

    r2_gap_to_best = np.zeros(m, dtype=np.float64)
    r2_q1_gap_to_best = np.zeros(m, dtype=np.float64)
    rmse_ratio_to_best = np.ones(m, dtype=np.float64)
    cos_gap_to_best = np.zeros(m, dtype=np.float64)
    explain_gap_to_best = np.zeros(m, dtype=np.float64)
    upwind_hit_ratio_to_best = np.zeros(m, dtype=np.float64)

    unexplained_hit_log = np.zeros(m, dtype=np.float64)
    upwind_gas_mass_frac = np.zeros(m, dtype=np.float64)
    downwind_clear_core_frac = np.zeros(m, dtype=np.float64)
    peak_cross_norm = np.zeros(m, dtype=np.float64)
    peak_dz_buoy_min = np.zeros(m, dtype=np.float64)

    if history:
        ten = _compute_plume_physics_tensors(fac, coords, history, wind_now)
        in_shadow = ten["in_shadow"]
        obs_pos = ten["obs_pos"]
        obs_wind = ten["obs_wind"]
        obs_log_inst = ten["obs_log_inst"]
        obs_is_clean = ten["obs_is_clean"]
        along = ten["along"]
        c_b, c_n, c_h, c_max = ten["c_b"], ten["c_n"], ten["c_h"], ten["c_max"]
        c_core_b, c_core_n, c_core_h = ten["c_core_b"], ten["c_core_n"], ten["c_core_h"]
        trans = ten["trans"]
        w_obs = ten["w_obs"]
        clean_trust = ten["clean_trust"]

        # Gas-hit physics & 3-buoyancy x MAP Q fit (also selects best buoyancy for core clean check)
        c_core_best = c_core_n
        hit_mask = (~obs_is_clean) & (obs_log_inst > 0.45)
        h_idx = np.nonzero(hit_mask)[0]
        if h_idx.size > 0:
            y_hit = obs_log_inst[h_idx]
            w_h = w_obs[h_idx][None, :]
            c_h_max = c_max[:, h_idx]
            l_h_max = np.log1p(3.0 * c_h_max)

            total_hit_mass = max(float(np.sum(w_obs[h_idx] * y_hit)), 1e-6)
            hit_contrib = np.minimum(y_hit[None, :], l_h_max)
            upwind_hit_score = np.log1p(np.sum(w_h * hit_contrib, axis=1))
            hit_explain_frac = np.clip(np.sum(w_h * hit_contrib, axis=1) / total_hit_mass, 0.0, 1.0)
            unexplained_hit_log = np.max(np.where(c_h_max < 0.8, y_hit[None, :], 0.0), axis=1)

            non_trans_h = ~ten["obs_is_transient"][h_idx]
            hit_stable = ten["obs_wind_stable"][h_idx] > 0.75
            up_thr = np.where((in_shadow[:, None] > 0.5) | ten["obs_in_shadow"][None, h_idx], -6.0, -4.5)
            up_viol_mask = (
                (along[:, h_idx] < up_thr)
                & (trans[:, h_idx] > 0.45)
                & non_trans_h[None, :]
                & hit_stable[None, :]
            )
            up_mass = np.sum(np.where(up_viol_mask, w_h * y_hit[None, :], 0.0), axis=1)
            upwind_gas_mass_frac = np.clip(up_mass / total_hit_mass, 0.0, 1.0)

            y_all = np.where(obs_is_clean, 0.0, obs_log_inst)
            sum_w_h = max(float(np.sum(w_obs[h_idx])), 1e-6)
            sum_w_c = float(np.sum(w_obs[obs_is_clean]))
            c_scale = min(1.0, 1.5 * sum_w_h / max(sum_w_c, 1e-6))
            w_fit = np.where(obs_is_clean, w_obs * c_scale, w_obs * 3.5)
            sum_w_fit = max(float(np.sum(w_fit)), 1e-6)
            y_w = y_all * np.sqrt(w_fit)
            y_norm = max(float(np.linalg.norm(y_w)), 1e-6)
            y_energy = max(float(np.sum(w_fit * (y_all ** 2))), 1e-6)

            def _fit_buoy(c_mat: np.ndarray, c_core_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                l1 = np.where(obs_is_clean[None, :], clean_trust * np.log1p(c_core_mat), np.log1p(c_mat))
                lw = l1 * np.sqrt(w_fit)[None, :]
                lnorm = np.sqrt(np.sum(lw * lw, axis=1))
                cos_sim = np.where(lnorm > 1e-6, (lw @ y_w) / (np.maximum(lnorm, 1e-6) * y_norm), 0.0)

                best_sse = np.full(m, np.inf, dtype=np.float64)
                best_q = np.ones(m, dtype=np.float64)
                sse_q1 = np.full(m, y_energy, dtype=np.float64)
                for qv in _Q_GRID:
                    lq = (
                        l1
                        if abs(qv - 1.0) < 1e-9
                        else np.where(obs_is_clean[None, :], clean_trust * np.log1p(qv * c_core_mat), np.log1p(qv * c_mat))
                    )
                    # Huber-clipped squared error + MAP log(q)^2 prior avoids 2nd leak/extreme-Q distortions
                    sq_err = np.minimum((lq - y_all[None, :]) ** 2, 9.0)
                    sse = (sq_err @ w_fit) + 0.08 * y_energy * (math.log(float(qv)) ** 2)
                    if abs(qv - 1.0) < 1e-9:
                        sse_q1 = sse
                    imp = sse < best_sse
                    best_sse = np.where(imp, sse, best_sse)
                    best_q = np.where(imp, qv, best_q)

                r2_q1 = np.clip(1.0 - sse_q1 / y_energy, -1.5, 1.0)
                r2_qhat = np.clip(1.0 - best_sse / y_energy, -1.5, 1.0)
                rmse_qhat = np.sqrt(np.maximum(best_sse, 0.0) / sum_w_fit)
                return cos_sim, r2_q1, r2_qhat, rmse_qhat, best_q

            plume_cos_buoy, r2_1_b, r2_q_b, rmse_b, q_b = _fit_buoy(c_b, c_core_b)
            plume_cos_neut, r2_1_n, r2_q_n, rmse_n, q_n = _fit_buoy(c_n, c_core_n)
            plume_cos_heavy, r2_1_h, r2_q_h, rmse_h, q_h = _fit_buoy(c_h, c_core_h)

            r2_1_stack = np.stack([r2_1_b, r2_1_n, r2_1_h], axis=1)
            r2_q_stack = np.stack([r2_q_b, r2_q_n, r2_q_h], axis=1)
            rmse_stack = np.stack([rmse_b, rmse_n, rmse_h], axis=1)
            q_stack = np.stack([q_b, q_n, q_h], axis=1)
            best_b = np.argmax(r2_q_stack, axis=1)
            rows_idx = np.arange(m)
            plume_r2_q1 = r2_1_stack[rows_idx, best_b]
            plume_r2_qhat = r2_q_stack[rows_idx, best_b]
            plume_rmse_qhat = rmse_stack[rows_idx, best_b]
            q_best = q_stack[rows_idx, best_b]
            abs_log_q_hat = np.abs(np.log(np.maximum(q_best, 0.1)))

            c_sel = np.where(
                best_b[:, None] == 0,
                c_b,
                np.where(best_b[:, None] == 2, c_h, c_n),
            )
            c_core_best = np.where(
                best_b[:, None] == 0,
                c_core_b,
                np.where(best_b[:, None] == 2, c_core_h, c_core_n),
            )

            pred_hit = np.log1p(q_best[:, None] * c_sel[:, h_idx])
            hit_rmse_qhat = np.sqrt(np.maximum((((pred_hit - y_hit[None, :]) ** 2) @ w_obs[h_idx]) / sum_w_h, 0.0))

            # Strictly monotonic within-snapshot comparative physics features
            plume_cos_best = np.maximum(np.maximum(plume_cos_buoy, plume_cos_neut), plume_cos_heavy)
            ref_r2 = float(np.max(plume_r2_qhat))
            ref_r2_q1 = float(np.max(plume_r2_q1))
            ref_rmse = max(float(np.min(plume_rmse_qhat)), 0.15)
            ref_cos = float(np.max(plume_cos_best))
            ref_exp = float(np.max(hit_explain_frac))
            ref_uh = max(float(np.max(upwind_hit_score)), 0.5)

            r2_gap_to_best = np.clip(ref_r2 - plume_r2_qhat, 0.0, 2.0)
            r2_q1_gap_to_best = np.clip(ref_r2_q1 - plume_r2_q1, 0.0, 2.0)
            rmse_ratio_to_best = np.clip(plume_rmse_qhat / ref_rmse, 1.0, 8.0)
            cos_gap_to_best = np.clip(ref_cos - plume_cos_best, 0.0, 1.5)
            explain_gap_to_best = np.clip(ref_exp - hit_explain_frac, 0.0, 1.5)
            upwind_hit_ratio_to_best = np.clip(upwind_hit_score / ref_uh, 0.0, 1.0)

            # Peak-relative physics-normalized coordinates wrt highest (post-shift if available) hit
            obs_t = ten["obs_t"]
            last_shift_t = float(ten["last_shift_t"])
            post_shift_hits = np.nonzero(obs_t[h_idx] > last_shift_t + 5.0)[0]
            if post_shift_hits.size > 0:
                pk_idx = int(h_idx[int(post_shift_hits[int(np.argmax(y_hit[post_shift_hits]))])])
            else:
                pk_idx = int(h_idx[int(np.argmax(y_hit))])
            pk_pos = obs_pos[pk_idx]
            if obs_t[pk_idx] > last_shift_t + 4.0 and math.hypot(float(wind_now[0]), float(wind_now[1])) >= 0.2:
                pk_w = np.array([float(wind_now[0]), float(wind_now[1])], dtype=np.float64)
            else:
                pk_w = obs_wind[pk_idx]
            pk_spd = math.hypot(float(pk_w[0]), float(pk_w[1]))
            pk_ux, pk_uy = (float(pk_w[0]) / pk_spd, float(pk_w[1]) / pk_spd) if pk_spd >= 0.2 else (1.0, 0.0)
            dp_x = pk_pos[0] - coords[:, 0]
            dp_y = pk_pos[1] - coords[:, 1]
            peak_along_wind = np.clip(dp_x * pk_ux + dp_y * pk_uy, -25.0, 55.0)
            raw_peak_cross = np.abs(-dp_x * pk_uy + dp_y * pk_ux)
            peak_cross_norm = np.clip(raw_peak_cross / (3.0 + 0.28 * np.maximum(peak_along_wind, 0.0)), 0.0, 6.0)
            t_pk = np.maximum(peak_along_wind, 0.0) / max(pk_spd, 0.85)
            raw_peak_dz = pk_pos[2] - coords[:, 2]
            dz_scale = 2.0 + 0.15 * np.maximum(peak_along_wind, 0.0)
            buoy_shift = 0.25 * t_pk / (1.0 + 0.035 * t_pk)
            peak_dz_buoy_min = np.clip(
                np.minimum(
                    np.minimum(np.abs(raw_peak_dz - buoy_shift), np.abs(raw_peak_dz)),
                    np.abs(raw_peak_dz + buoy_shift),
                ) / dz_scale,
                0.0,
                6.0,
            )

            # Local-peak-window MAP plume fit (multi-leak & wind-shift resilient)
            loc_m = np.linalg.norm(obs_pos - pk_pos[None, :], axis=1) <= 32.0
            w_loc = np.where(loc_m, w_fit, 0.0)
            ye_loc = float(np.sum(w_loc * (y_all ** 2)))
            if ye_loc < 1e-6:
                local_r2_qhat = plume_r2_qhat.copy()
            else:
                local_r2_qhat = np.full(m, -1.5, dtype=np.float64)
                for cm, ccm in ((c_b, c_core_b), (c_n, c_core_n), (c_h, c_core_h)):
                    for qv in _Q_GRID:
                        lq = np.where(obs_is_clean[None, :], clean_trust * np.log1p(qv * ccm), np.log1p(qv * cm))
                        sse_l = (((lq - y_all[None, :]) ** 2) @ w_loc) + 0.08 * ye_loc * (math.log(float(qv)) ** 2)
                        local_r2_qhat = np.maximum(local_r2_qhat, np.clip(1.0 - sse_l / ye_loc, -1.5, 1.0))
            local_r2_gap_to_best = np.clip(float(np.max(local_r2_qhat)) - local_r2_qhat, 0.0, 2.0)

        # Bounded time-invariant clean-air falsification fraction
        if np.any(obs_is_clean):
            c_idx = np.nonzero(obs_is_clean)[0]
            w_c = w_obs[c_idx][None, :]
            ct_c = clean_trust[:, c_idx]
            al_c = along[:, c_idx]

            clean_core_mass = np.sum(
                w_c * ct_c * np.where(al_c > 1.5, np.log1p(c_core_best[:, c_idx]), 0.0),
                axis=1,
            )
            hit_core_mass = (
                np.sum(w_obs[h_idx][None, :] * np.log1p(c_max[:, h_idx]), axis=1)
                if h_idx.size > 0
                else np.zeros(m, dtype=np.float64)
            )
            downwind_clear_core_frac = clean_core_mass / (clean_core_mass + hit_core_mass + 1.5)

    return [
        [
            round(float(plume_cos_best[i]), 4),
            round(float(plume_r2_q1[i]), 4),
            round(float(plume_r2_qhat[i]), 4),
            round(float(plume_rmse_qhat[i]), 4),
            round(float(hit_rmse_qhat[i]), 4),
            round(float(abs_log_q_hat[i]), 4),
            round(float(hit_explain_frac[i]), 4),
            round(float(local_r2_qhat[i]), 4),
            round(float(local_r2_gap_to_best[i]), 4),
            round(float(r2_gap_to_best[i]), 4),
            round(float(r2_q1_gap_to_best[i]), 4),
            round(float(rmse_ratio_to_best[i]), 4),
            round(float(cos_gap_to_best[i]), 4),
            round(float(explain_gap_to_best[i]), 4),
            round(float(upwind_hit_ratio_to_best[i]), 4),
            round(float(unexplained_hit_log[i]), 4),
            round(float(upwind_gas_mass_frac[i]), 4),
            round(float(downwind_clear_core_frac[i]), 4),
            round(float(peak_cross_norm[i]), 4),
            round(float(peak_dz_buoy_min[i]), 4),
            round(float(infra_prox[i]), 4),
        ]
        for i in range(m)
    ]


def build_source_feature_rows(
    fac: FacilityContext,
    coords: np.ndarray,
    history: Sequence[ProcessedObs],
    wind_now: Vec2,
    use_base_static: bool = False,
) -> list[list[Any]]:
    """Backward-compatible alias pointing to `build_post_source_feature_rows`."""
    return build_post_source_feature_rows(fac, coords, history, wind_now, use_base_static=use_base_static)


def source_proximity_target(dist_m: float, scale_m: float = 5.0) -> float:
    """Smooth continuous source-proximity target in (0, 1] for TabPFN Model A & B."""
    return float(math.exp(-max(0.0, dist_m) / scale_m))
