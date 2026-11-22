"""Generate multi-world, multi-regime offline datasets for the Triple-TabPFN agent (v4).

Strict train/test separation:
- Trains ONLY on procedural `generated` layouts (`layout_seed = 1000..1419`) and
  NEVER touches the 5 named evaluation worlds (`refinery`, `pipe_loop`,
  `tank_terminal`, `compressor_yard`, `cooling_works`) or held-out seeds.
- Uses the exact shared `FacilityContext`, `ObservationHistory`,
  `build_post_source_candidates`, `build_post_source_feature_rows`, and
  `build_pre_source_feature_rows` used at live inference.

Outputs:
- `data/tabpfn_source_post_v4.json.gz` (~4,096 post-contact pure-physics rows for Model A)
- `data/tabpfn_source_pre_v4.json.gz`  (~1,600 pre-contact spatial-prior rows for Model B)
- `data/tabpfn_plume_prior.json`       (~160 coordinate-masked rows for TabPFN Plume Mapper)

Usage:
    python -m scripts.generate_tabpfn_datasets --episodes 240 --workers 10
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import math
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from agents.base import BaseAgent
from agents.gradient import GradientAgent
from agents.lawnmower import LawnmowerAgent
from agents.tabpfn_features import (
    PLUME_FEATURE_COLUMNS,
    POST_SOURCE_FEATURE_COLUMNS,
    PRE_SOURCE_FEATURE_COLUMNS,
    FacilityContext,
    ObservationHistory,
    build_post_source_candidates,
    build_post_source_feature_rows,
    build_pre_source_feature_rows,
    source_proximity_target,
)
from agents.tabpfn_nav import FastPlanner3D, safe_goal_3d
from shared.schemas import DroneObservation, JudgeCommand, LeakSource, Vec3, WorldInitPayload
from sim.engine import SimEngine


class _TrainingCollector(BaseAgent):
    """Source-blind mobile collector mixing converge, linger, gradient, downwind sweep, and lawnmower."""

    def __init__(self, mode: str, seed: int, converge_waypoints: list[Vec3] | None = None) -> None:
        super().__init__("collector")
        self.mode = mode  # "converge", "linger", "gradient", "mixed", "lawnmower"
        self._rng = np.random.default_rng(seed)
        self._grad = GradientAgent("collector")
        self._lawn = LawnmowerAgent("collector")
        self.obs_history = ObservationHistory()
        self._fac: FacilityContext | None = None
        self._planner: FastPlanner3D | None = None
        self._route: list[Vec3] = []
        self._goal: Vec3 | None = None
        self._goal_time: float = -100.0
        self._visited_pos: list[np.ndarray] = []
        self._linger_anchor: np.ndarray | None = None
        self._converge_wps: list[Vec3] = list(converge_waypoints or [])
        self._conv_idx: int = 0

    def reset(self, world_init: WorldInitPayload) -> None:
        self._grad.reset(world_init)
        self._lawn.reset(world_init)
        self.obs_history.clear()
        self._fac = FacilityContext(world_init)
        self._planner = FastPlanner3D(world_init)
        self._route.clear()
        self._goal = None
        self._goal_time = -100.0
        self._visited_pos.clear()
        self._linger_anchor = None
        self._conv_idx = 0

    def set_converge_waypoints(self, wps: list[Vec3]) -> None:
        self._converge_wps = list(wps)
        self._conv_idx = 0

    def _pick_goal(self, obs: DroneObservation) -> Vec3:
        assert self._fac is not None and self._planner is not None
        if self.mode == "converge" and self._converge_wps:
            pt = self._converge_wps[self._conv_idx % len(self._converge_wps)]
            self._conv_idx += 1
            return safe_goal_3d(self._planner, pt)

        wx, wy = self.obs_history.current_wind
        spd = math.hypot(wx, wy)
        ux, uy = (wx / spd, wy / spd) if spd > 0.2 else (1.0, 0.0)
        cx, cy = -uy, ux

        if self.mode == "linger":
            if self._linger_anchor is None and obs.ppm_measured >= 5.0:
                self._linger_anchor = np.array(obs.pos, dtype=np.float64)
            if self._linger_anchor is not None:
                # Oscillate crosswind & slightly further downwind around first plume hit (downwind trap realism)
                dn_off = float(self._rng.uniform(0.0, 12.0))
                cr_off = float(self._rng.uniform(-9.0, 9.0))
                dz_off = float(self._rng.uniform(-2.5, 3.0))
                pt = self._linger_anchor + np.array([dn_off * ux + cr_off * cx, dn_off * uy + cr_off * cy, dz_off])
                return safe_goal_3d(self._planner, pt)

        if self.mode == "lawnmower":
            return safe_goal_3d(self._planner, self._lawn.update(obs))
        if self.mode == "gradient" or (obs.ppm_measured >= 4.0 and self._rng.random() < 0.72):
            return safe_goal_3d(self._planner, self._grad.update(obs))

        grid = self._fac.grid_np
        if grid.shape[0] == 0:
            return obs.pos
        if self._fac.asset_pos.shape[0] > 0 and self._rng.random() < 0.55:
            a_pts = self._fac.asset_pos
            ap = a_pts[int(self._rng.integers(a_pts.shape[0]))]
            dist_dn = float(self._rng.uniform(4.0, 22.0))
            pt = np.array([ap[0] + ux * dist_dn, ap[1] + uy * dist_dn, ap[2] + float(self._rng.uniform(-2.0, 5.0))])
            return safe_goal_3d(self._planner, pt)

        if self._visited_pos:
            vis = np.asarray(self._visited_pos, dtype=np.float64)
            d_vis = np.min(np.linalg.norm(grid[:, None, :] - vis[None, :, :], axis=2), axis=1)
            idx = int(np.argmax(d_vis + self._rng.uniform(0.0, 6.0, size=grid.shape[0])))
        else:
            idx = int(self._rng.integers(grid.shape[0]))
        return safe_goal_3d(self._planner, grid[idx])

    def update(self, obs: DroneObservation) -> Vec3:
        self.obs_history.append(obs)
        self._visited_pos.append(np.array(obs.pos, dtype=np.float64))
        assert self._planner is not None

        while len(self._route) > 1 and math.dist(obs.pos, self._route[0]) < 2.2:
            self._route.pop(0)

        need_new = (
            not self._route
            or self._goal is None
            or math.dist(obs.pos, self._goal) < 3.0
            or (obs.sim_time - self._goal_time) >= 6.0
        )
        if need_new:
            g = self._pick_goal(obs)
            path = self._planner.route(obs.pos, g)
            if path and len(path) >= 2:
                self._route = path[1:]
                self._goal = g
            else:
                self._route = [g]
                self._goal = g
            self._goal_time = obs.sim_time

        return self._route[0] if self._route else obs.pos


def _build_generated_engine(layout_seed: int, sim_seed: int, collector: BaseAgent) -> SimEngine:
    for attempt in range(8):
        try:
            return SimEngine(
                agents={"collector": collector},
                seed=sim_seed,
                layout_seed=layout_seed + attempt * 1000,
                map_id="generated",
            )
        except ValueError:
            continue
    return SimEngine(agents={"collector": collector}, seed=sim_seed, layout_seed=42, map_id="refinery")


def _sample_leak_pos(
    rng: np.random.Generator,
    engine: SimEngine,
    fac: FacilityContext,
    world_init: WorldInitPayload,
    ep_idx: int,
    asset_frac: float,
    pipe_frac: float,
) -> Vec3:
    u = float(rng.random())
    if u < asset_frac and world_init.assets:
        ages = np.array([max(2.0, a.age_years) for a in world_init.assets], dtype=np.float64)
        asset = world_init.assets[int(rng.choice(len(world_init.assets), p=ages / np.sum(ages)))]
        return asset.pos
    if u < asset_frac + pipe_frac and fac.pipe_points.shape[0] > 0:
        pt = fac.pipe_points[int(rng.integers(fac.pipe_points.shape[0]))]
        z_val = float(rng.uniform(10.0, 22.0)) if rng.random() < 0.50 else float(pt[2])
        cand = np.array([float(pt[0]), float(pt[1]), z_val], dtype=np.float64)
        if not engine.world.is_inside_building(tuple(map(float, cand)), margin=-0.05):
            return (float(cand[0]), float(cand[1]), float(cand[2]))
        return (float(pt[0]), float(pt[1]), float(pt[2]))

    # Off-asset free 3D point (half elevated z=10..22m, half low/mid near walls or open air)
    for _ in range(12):
        g_pt = fac.grid_np[int(rng.integers(fac.grid_np.shape[0]))]
        jitter = rng.uniform(-4.5, 4.5, size=3)
        cand = np.clip(g_pt + jitter, fac.bounds_min + [5.0, 5.0, 2.5], fac.bounds_max - [5.0, 5.0, 6.0])
        if rng.random() < 0.50:
            cand[2] = float(rng.uniform(10.0, 22.0))
        else:
            cand[2] = float(rng.uniform(2.5, 9.5))
        if not engine.world.is_inside_building(tuple(map(float, cand)), margin=-0.05):
            return (float(cand[0]), float(cand[1]), float(cand[2]))
    g_pt = fac.grid_np[int(rng.integers(fac.grid_np.shape[0]))]
    return (float(g_pt[0]), float(g_pt[1]), float(g_pt[2]))


def _emission_rate(rng: np.random.Generator, low: float, high: float, weak: bool) -> float:
    fraction = float(rng.random())
    if weak:
        # Cover low-rate SI releases alongside the stronger explicit scenarios.
        return math.exp(math.log(.008) + fraction * math.log(.18 / .008))
    return low + fraction * (high - low)


def _simulate_post_episode(ep_idx: int, seed: int, calibrated_rates: bool = False, calibrated_physics: bool = False) -> tuple[list[list[Any]], list[float], int, int]:
    """Generate post-contact training samples for TabPFN Model A."""
    rng = np.random.default_rng(seed)
    u_mode = float(rng.random())
    if u_mode < 0.20:
        mode = "converge"
    elif u_mode < 0.45:
        mode = "linger"
    elif u_mode < 0.70:
        mode = "gradient"
    elif u_mode < 0.90:
        mode = "mixed"
    else:
        mode = "lawnmower"

    collector = _TrainingCollector(mode=mode, seed=seed + 1000)
    engine = _build_generated_engine(layout_seed=1000 + ep_idx, sim_seed=seed, collector=collector)
    world_init = engine.get_world_init()
    fac = FacilityContext(world_init)

    wind_angle = float(rng.uniform(0.0, 360.0))
    wind_speed = float(rng.uniform(0.6, 8.0))
    dropout_on = float(rng.random()) < 0.35
    wind_shift = float(rng.random()) < 0.30
    shift_time = float(rng.uniform(12.0, 30.0))
    shift_delta = float(rng.choice([-135.0, -90.0, 90.0, 135.0, 180.0]))
    buoyancy = float(rng.choice([-0.28, -0.20, -0.10, 0.0, 0.12, 0.25, 0.35]))
    n_leaks = 2 if float(rng.random()) < 0.12 else 1

    leaks: list[LeakSource] = []
    ux_w, uy_w = math.cos(math.radians(wind_angle)), math.sin(math.radians(wind_angle))
    for lk_i in range(n_leaks):
        if lk_i == 0 and float(rng.random()) < 0.20 and fac.bldg_min.shape[0] > 0:
            # Explicit building/tank lee-wake shadow leak (matches wake_trap physics)
            bi = int(rng.integers(fac.bldg_min.shape[0]))
            b_ctr = 0.5 * (fac.bldg_min[bi] + fac.bldg_max[bi])
            b_half = 0.5 * (fac.bldg_max[bi] - fac.bldg_min[bi])
            reach = float(abs(b_half[0] * ux_w) + abs(b_half[1] * uy_w) + rng.uniform(2.0, 5.5))
            lee_z = float(np.clip(rng.uniform(2.5, max(3.5, float(fac.bldg_max[bi, 2]) - 1.5)), 2.5, 22.0))
            cand_w = np.array([b_ctr[0] + ux_w * reach, b_ctr[1] + uy_w * reach, lee_z], dtype=np.float64)
            cand_w[:2] += np.array([-uy_w, ux_w]) * float(rng.uniform(-0.6, 0.6)) * float(max(b_half[0], b_half[1]))
            cand_w = np.clip(cand_w, fac.bounds_min + [4.0, 4.0, 2.5], fac.bounds_max - [4.0, 4.0, 5.0])
            if not engine.world.is_inside_building(tuple(map(float, cand_w)), margin=-0.05):
                pos = (float(cand_w[0]), float(cand_w[1]), float(cand_w[2]))
            else:
                pos = _sample_leak_pos(rng, engine, fac, world_init, ep_idx, asset_frac=0.30, pipe_frac=0.15)
        else:
            pos = _sample_leak_pos(rng, engine, fac, world_init, ep_idx, asset_frac=0.30, pipe_frac=0.15)
        emission_rate = _emission_rate(rng, .45, 2.2, calibrated_rates and (ep_idx + lk_i) % 2 == 0)
        leaks.append(
            LeakSource(
                id=f"post_{ep_idx:03d}_{lk_i}",
                pos=pos,
                emission_rate=emission_rate,
                buoyancy_vz=buoyancy,
            )
        )

    engine.airflow.set_wind(wind_angle, wind_speed)
    engine.sensor_dropout_active = dropout_on
    engine.reset_race(leaks=leaks)

    sensor = engine.sensors["collector"]
    sensor.tau_rise = float(rng.uniform(0.45, 0.85))
    sensor.tau_decay = float(rng.uniform(1.4, 2.3))

    if mode == "converge":
        lk0 = np.array(leaks[0].pos, dtype=np.float64)
        ang_rad = math.radians(wind_angle)
        ux_c, uy_c = math.cos(ang_rad), math.sin(ang_rad)
        cx_c, cy_c = -uy_c, ux_c
        w_eff = (wind_angle + shift_delta) if wind_shift else wind_angle
        ux_2, uy_2 = math.cos(math.radians(w_eff)), math.sin(math.radians(w_eff))
        up_d1 = float(rng.uniform(5.0, 9.0))
        up_d2 = float(rng.uniform(5.0, 9.0))
        cr_d = float(rng.uniform(2.5, 4.5))
        conv_wps: list[Vec3] = [
            (
                float(lk0[0] + rng.uniform(-1.5, 1.5)),
                float(lk0[1] + rng.uniform(-1.5, 1.5)),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
            (
                float(lk0[0] + ux_c * rng.uniform(2.0, 6.0)),
                float(lk0[1] + uy_c * rng.uniform(2.0, 6.0)),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
            (
                float(lk0[0] - ux_c * up_d1),
                float(lk0[1] - uy_c * up_d1),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
            (
                float(lk0[0] + cx_c * cr_d),
                float(lk0[1] + cy_c * cr_d),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
            (
                float(lk0[0] + ux_2 * rng.uniform(2.0, 6.0)),
                float(lk0[1] + uy_2 * rng.uniform(2.0, 6.0)),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
            (
                float(lk0[0] - ux_2 * up_d2),
                float(lk0[1] - uy_2 * up_d2),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
            (
                float(lk0[0] - cx_c * cr_d),
                float(lk0[1] - cy_c * cr_d),
                float(np.clip(lk0[2] + rng.uniform(-2.5, 2.5), 2.0, 26.0)),
            ),
        ]
        collector.set_converge_waypoints(conv_wps)

    # Start downwind of leak in converge/linger or 50% of search episodes
    if fac.grid_np.shape[0] > 0:
        if mode in ("converge", "linger") or rng.random() < 0.50:
            lk0 = np.array(leaks[0].pos, dtype=np.float64)
            ang_rad = math.radians(wind_angle)
            dist_st = (
                float(rng.uniform(6.0, 20.0))
                if mode == "converge"
                else (float(rng.uniform(12.0, 34.0)) if mode == "linger" else float(rng.uniform(6.0, 30.0)))
            )
            dn = lk0 + np.array([math.cos(ang_rad), math.sin(ang_rad), 0.0]) * dist_st
            dn[0] += float(rng.uniform(-4.0, 4.0))
            dn[1] += float(rng.uniform(-4.0, 4.0))
            dn[2] = float(np.clip(lk0[2] + rng.uniform(-2.5, 4.0), 2.5, 24.0))
            st_pt = safe_goal_3d(FastPlanner3D(world_init), dn)
            engine.drones["collector"].reset(start_pos=st_pt)
        elif rng.random() < 0.50:
            g_pt = fac.grid_np[int(rng.integers(fac.grid_np.shape[0]))]
            engine.drones["collector"].reset(start_pos=(float(g_pt[0]), float(g_pt[1]), float(g_pt[2])))

    snapshot_times = [7.0, 16.0, 28.0, 45.0, 68.0, 88.0]
    next_snap_i = 0
    shifted = False
    prev_est: Vec3 | None = None

    rows_out: list[list[Any]] = []
    y_out: list[float] = []
    snaps_with_contact = 0
    snaps_covered = 0

    leak_arr = np.array([lk.pos for lk in leaks], dtype=np.float64)
    col_he = POST_SOURCE_FEATURE_COLUMNS.index("hit_explain_frac")
    col_ur = POST_SOURCE_FEATURE_COLUMNS.index("upwind_hit_ratio_to_best")
    col_cos = POST_SOURCE_FEATURE_COLUMNS.index("plume_cos_best")

    for _ in range(int(92.0 / engine.DT)):
        if wind_shift and not shifted and engine.sim_time >= shift_time:
            shifted = True
            engine.apply_command(
                JudgeCommand(
                    action="set_wind",
                    wind_angle_deg=(wind_angle + shift_delta) % 360.0,
                    wind_speed=float(rng.uniform(1.5, 6.5)),
                )
            )
        engine.step(include_puffs=False)

        if next_snap_i < len(snapshot_times) and engine.sim_time >= snapshot_times[next_snap_i]:
            next_snap_i += 1
            samples = collector.obs_history.samples
            if not samples:
                continue
            has_contact = any((not r.is_clean and r.ppm_inst >= 2.5) or r.ppm_raw >= 3.5 for r in samples[-50:])
            if not has_contact:
                continue
            snaps_with_contact += 1

            wind_now = collector.obs_history.current_wind
            cands, _ = build_post_source_candidates(fac, samples, wind_now, prev_estimate=prev_est, max_candidates=950)
            feat_rows = build_post_source_feature_rows(fac, cands, samples, wind_now)
            hits = [r for r in samples if (not r.is_clean and r.ppm_inst >= 2.0) or r.ppm_raw >= 3.5]

            # In multi-leak episodes, only label leak(s) with observable gas contact
            if len(leaks) > 1 and hits:
                hit_pts = np.array([h.pos for h in hits], dtype=np.float64)
                active_leaks: list[np.ndarray] = []
                lk_scores: list[float] = []
                for lk_p in leak_arr:
                    ci = int(np.argmin(np.linalg.norm(cands - lk_p[None, :], axis=1)))
                    he_lk = float(feat_rows[ci][col_he])
                    ur_lk = float(feat_rows[ci][col_ur])
                    d_hit_lk = float(np.min(np.linalg.norm(hit_pts - lk_p[None, :], axis=1)))
                    lk_scores.append(he_lk + 0.5 * ur_lk)
                    if he_lk >= 0.25 or ur_lk >= 0.55 or d_hit_lk <= 12.0:
                        active_leaks.append(lk_p)
                if not active_leaks:
                    active_leaks = [leak_arr[int(np.argmax(lk_scores))]]
                eff_leaks = np.asarray(active_leaks, dtype=np.float64)
            else:
                eff_leaks = leak_arr

            dists = np.min(np.linalg.norm(cands[:, None, :] - eff_leaks[None, :, :], axis=2), axis=1)

            best_pos = int(np.argmin(dists))
            cos_best = float(feat_rows[best_pos][col_cos])
            if dists[best_pos] > 3.8:
                continue
            if float(feat_rows[best_pos][col_ur]) < 0.25 and float(feat_rows[best_pos][col_he]) < 0.15 and cos_best < 0.25:
                continue
            snaps_covered += 1

            chosen: list[int] = [best_pos]
            prev_est = (float(cands[best_pos, 0]), float(cands[best_pos, 1]), float(cands[best_pos, 2]))
            all_idx = np.arange(len(dists))

            # Pure ground-truth distance-stratified sampling (ZERO Berkson feature conditioning)
            near_pool = np.flatnonzero((dists > 1.5) & (dists <= 5.0) & (all_idx != best_pos))
            if near_pool.size > 0:
                ni = int(rng.choice(near_pool))
                if ni not in chosen:
                    chosen.append(ni)

            mid_pool = np.flatnonzero((dists > 6.0) & (dists <= 18.0))
            if mid_pool.size > 0:
                for mi in rng.choice(mid_pool, size=min(2, mid_pool.size), replace=False):
                    if int(mi) not in chosen:
                        chosen.append(int(mi))

            # Ground-truth downwind-of-leak candidate under current ambient wind
            w_spd_now = math.hypot(float(wind_now[0]), float(wind_now[1]))
            ux_n, uy_n = (float(wind_now[0]) / w_spd_now, float(wind_now[1]) / w_spd_now) if w_spd_now >= 0.2 else (1.0, 0.0)
            nearest_lk = (
                eff_leaks[0]
                if eff_leaks.shape[0] == 1
                else eff_leaks[int(np.argmin(np.linalg.norm(eff_leaks - cands[best_pos][None, :], axis=1)))]
            )
            dlk = cands - nearest_lk[None, :]
            along_lk = dlk[:, 0] * ux_n + dlk[:, 1] * uy_n
            cross_lk = np.abs(-dlk[:, 0] * uy_n + dlk[:, 1] * ux_n)
            dn_raw = np.flatnonzero((dists > 6.0) & (along_lk > 3.0) & (cross_lk < 12.0))
            dn_pool = [int(i) for i in dn_raw if int(i) not in chosen]
            if dn_pool:
                chosen.append(int(rng.choice(dn_pool)))

            far_raw = np.flatnonzero(dists > 18.0)
            far_pool = [int(i) for i in far_raw if int(i) not in chosen]
            if not far_pool:
                far_pool = [int(i) for i in np.flatnonzero(dists > 6.0) if int(i) not in chosen]
            if far_pool:
                chosen.append(int(rng.choice(far_pool)))

            for idx in chosen:
                rows_out.append(feat_rows[idx])
                y_out.append(round(source_proximity_target(float(dists[idx]), scale_m=5.0), 5))

    return rows_out, y_out, snaps_with_contact, snaps_covered


def _simulate_pre_episode(ep_idx: int, seed: int, calibrated_rates: bool = False) -> tuple[list[list[Any]], list[float]]:
    """Generate pre-contact training samples for TabPFN Model B."""
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["mixed", "lawnmower", "mixed"]))
    collector = _TrainingCollector(mode=mode, seed=seed + 2000)
    engine = _build_generated_engine(layout_seed=1300 + ep_idx, sim_seed=seed, collector=collector)
    world_init = engine.get_world_init()
    fac = FacilityContext(world_init)

    wind_angle = float(rng.uniform(0.0, 360.0))
    wind_speed = float(rng.uniform(0.8, 7.5))
    dropout_on = float(rng.random()) < 0.30
    buoyancy = float(rng.choice([-0.22, 0.0, 0.25]))
    has_leak = float(rng.random()) >= 0.15

    leaks: list[LeakSource] = []
    if has_leak:
        pos = _sample_leak_pos(rng, engine, fac, world_init, ep_idx, asset_frac=0.45, pipe_frac=0.20)
        leaks.append(LeakSource(id=f"pre_{ep_idx:03d}_0", pos=pos, emission_rate=_emission_rate(rng, .6, 1.8, calibrated_rates and ep_idx % 2 == 0), buoyancy_vz=buoyancy))

    engine.airflow.set_wind(wind_angle, wind_speed)
    engine.sensor_dropout_active = dropout_on
    engine.reset_race(leaks=leaks)

    if fac.grid_np.shape[0] > 0 and rng.random() < 0.40:
        g_pt = fac.grid_np[int(rng.integers(fac.grid_np.shape[0]))]
        engine.drones["collector"].reset(start_pos=(float(g_pt[0]), float(g_pt[1]), float(g_pt[2])))

    snapshot_times = [4.0, 12.0, 22.0, 34.0]
    next_snap_i = 0
    rows_out: list[list[Any]] = []
    y_out: list[float] = []
    leak_arr = np.array([lk.pos for lk in leaks], dtype=np.float64) if leaks else np.zeros((0, 3), dtype=np.float64)

    col_da = PRE_SOURCE_FEATURE_COLUMNS.index("dist_to_asset")
    col_dwc = PRE_SOURCE_FEATURE_COLUMNS.index("downwind_clear_score")

    for _ in range(int(36.0 / engine.DT)):
        engine.step(include_puffs=False)
        samples = collector.obs_history.samples
        if not samples:
            continue
        # Stop collecting once contact occurs
        if any((not r.is_clean and r.ppm_inst >= 2.5) or r.ppm_raw >= 3.5 for r in samples):
            break

        if next_snap_i < len(snapshot_times) and engine.sim_time >= snapshot_times[next_snap_i]:
            next_snap_i += 1
            wind_now = collector.obs_history.current_wind
            coords = fac.base_src_coords
            feat_rows = build_pre_source_feature_rows(fac, coords, samples, wind_now, use_base_static=True)
            chosen: list[int] = []

            if leaks:
                dists = np.linalg.norm(coords - leak_arr[0][None, :], axis=1)
                best_pos = int(np.argmin(dists))
                chosen.append(best_pos)
                non_leak = np.flatnonzero(dists > 7.5)
            else:
                dists = np.full(coords.shape[0], 99.0, dtype=np.float64)
                non_leak = np.arange(coords.shape[0])

            if non_leak.size > 0:
                # 2 non-leaking assets (preferring ones that also show clear downwind evidence)
                non_assets = [int(i) for i in non_leak if float(feat_rows[int(i)][col_da]) < 1.0]
                if non_assets:
                    # Prefer cleared non-assets if present, else mix
                    cleared_assets = [i for i in non_assets if float(feat_rows[i][col_dwc]) > 0.25]
                    if cleared_assets:
                        pick_c = rng.choice(cleared_assets, size=min(2, len(cleared_assets)), replace=False)
                        chosen.extend(int(i) for i in pick_c)
                    rem_a = [i for i in non_assets if i not in chosen]
                    if rem_a and len(chosen) < 4:
                        chosen.extend(int(i) for i in rng.choice(rem_a, size=min(4 - len(chosen), len(rem_a)), replace=False))

                # 2 general non-asset / elevated grid points
                rem_other = [int(i) for i in non_leak if int(i) not in chosen]
                if rem_other:
                    chosen.extend(int(i) for i in rng.choice(rem_other, size=min(2, len(rem_other)), replace=False))

            for idx in chosen:
                rows_out.append(feat_rows[idx])
                y_out.append(round(source_proximity_target(float(dists[idx]), scale_m=5.5), 5) if leaks else 0.0)

    return rows_out, y_out


def generate_post_dataset(episodes: int = 240, seed: int = 60000, workers: int = 8, max_rows: int = 4096, *, calibrated_rates: bool = False, calibrated_physics: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    all_x: list[list[Any]] = []
    all_y: list[float] = []
    tot_c = 0
    tot_cov = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_simulate_post_episode, ep, seed + ep * 41, calibrated_rates, calibrated_physics): ep for ep in range(episodes)}
        results: dict[int, tuple[list[list[Any]], list[float], int, int]] = {}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    for ep in range(episodes):
        xs, ys, nc, ncov = results[ep]
        all_x.extend(xs)
        all_y.extend(ys)
        tot_c += nc
        tot_cov += ncov

    if len(all_x) > max_rows:
        rng = np.random.default_rng(seed + 999)
        idx = rng.choice(len(all_x), size=max_rows, replace=False)
        idx.sort()
        all_x = [all_x[int(i)] for i in idx]
        all_y = [all_y[int(i)] for i in idx]

    return {
        "version": "dual-tabpfn-post-v8-calibrated" if calibrated_rates else "dual-tabpfn-post-v8",
        "feature_physics": "si" if calibrated_physics else "legacy",
        "episodes": episodes,
        "seed": seed,
        "rows": len(all_x),
        "candidate_coverage_rate": round(tot_cov / max(1, tot_c), 4),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "columns": POST_SOURCE_FEATURE_COLUMNS,
        "X_train": all_x,
        "y_train": all_y,
    }


def generate_pre_dataset(episodes: int = 120, seed: int = 70000, workers: int = 8, max_rows: int = 1600, *, calibrated_rates: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    all_x: list[list[Any]] = []
    all_y: list[float] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_simulate_pre_episode, ep, seed + ep * 43, calibrated_rates): ep for ep in range(episodes)}
        results: dict[int, tuple[list[list[Any]], list[float]]] = {}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    for ep in range(episodes):
        xs, ys = results[ep]
        all_x.extend(xs)
        all_y.extend(ys)

    if len(all_x) > max_rows:
        rng = np.random.default_rng(seed + 888)
        idx = rng.choice(len(all_x), size=max_rows, replace=False)
        idx.sort()
        all_x = [all_x[int(i)] for i in idx]
        all_y = [all_y[int(i)] for i in idx]

    return {
        "version": "dual-tabpfn-pre-v4",
        "episodes": episodes,
        "seed": seed,
        "rows": len(all_x),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "columns": PRE_SOURCE_FEATURE_COLUMNS,
        "X_train": all_x,
        "y_train": all_y,
    }


def generate_plume_prior(n_missions: int = 8, seed: int = 23456) -> dict[str, Any]:
    """Generate compact coordinate-masked offline prior for online Plume Mapper."""
    rng = np.random.default_rng(seed)
    rows: list[list[Any]] = []
    ys: list[float] = []

    class _Idle(BaseAgent):
        def reset(self, w: WorldInitPayload) -> None:
            pass

        def update(self, o: DroneObservation) -> Vec3:
            return o.pos

    for m in range(n_missions):
        eng = SimEngine(agents={"idle": _Idle("idle")}, seed=seed + m * 31, layout_seed=1250 + m, map_id="generated")
        world_init = eng.get_world_init()
        fac = FacilityContext(world_init)
        g_types, g_da, g_age, g_dp, g_press, g_gas, _ = fac.grid_static
        asset = world_init.assets[(m * 2 + 1) % len(world_init.assets)]
        eng.airflow.set_wind(m * (360.0 / n_missions), float(rng.uniform(2.0, 6.5)))
        eng.reset_race(
            leaks=[LeakSource(id=f"hist_{m:02d}", pos=asset.pos, emission_rate=1.0, buoyancy_vz=float(rng.choice([-0.22, 0.05, 0.22])))]
        )
        for _ in range(int(15.0 / eng.DT)):
            eng.step()

        w_amb = eng.airflow.current_wind_vec()
        pts = fac.grid_np
        true_ppm = np.array([eng.plume.sample_concentration(tuple(p)) for p in pts], dtype=np.float64)
        hot_idx = np.nonzero(true_ppm >= 5.0)[0]
        cold_idx = np.nonzero(true_ppm < 2.0)[0]
        chosen: list[int] = []
        if hot_idx.size > 0:
            chosen.extend(int(i) for i in rng.choice(hot_idx, size=min(8, hot_idx.size), replace=False))
        if cold_idx.size > 0:
            chosen.extend(int(i) for i in rng.choice(cold_idx, size=min(12, cold_idx.size), replace=False))

        in_sh, up_sc = fac.wind_features(pts, w_amb)
        for i in chosen:
            rows.append([
                f"hist_{m:02d}", None, None, round(float(pts[i, 2]), 2),
                round(float(w_amb[0]), 3), round(float(w_amb[1]), 3),
                g_types[i], round(float(g_da[i]), 3), round(float(g_age[i]), 2),
                round(float(g_dp[i]), 3), round(float(g_press[i]), 2), float(g_gas[i]),
                float(in_sh[i]), round(float(up_sc[i]), 4),
            ])
            ys.append(round(math.log1p(max(0.0, float(true_ppm[i]) + 0.5)), 5))

    return {
        "version": "plume-prior-generated-v4",
        "missions": n_missions,
        "rows": len(rows),
        "columns": list(PLUME_FEATURE_COLUMNS),
        "X_train": rows,
        "y_train": ys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v5 multi-world TabPFN datasets")
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--pre-episodes", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibrated-physics", action="store_true", help="Compute post-contact features using SI plume dispersion.")
    parser.add_argument("--post-only", action="store_true")
    parser.add_argument("--calibrated-rates", action="store_true", help="Include weaker SI leak rates in the post-contact context.")
    parser.add_argument("--post-out", type=Path, default=Path("data/tabpfn_source_post_warm30_v1.json.gz"))
    parser.add_argument("--pre-out", type=Path, default=Path("data/tabpfn_source_pre_v4.json.gz"))
    parser.add_argument("--plume-out", type=Path, default=Path("data/tabpfn_plume_prior.json"))
    args = parser.parse_args()
    if args.calibrated_physics:
        parser.error("--calibrated-physics is unavailable with the restored tagged features")

    post_payload = generate_post_dataset(
        episodes=args.episodes,
        workers=args.workers,
        calibrated_rates=args.calibrated_rates,
        calibrated_physics=args.calibrated_physics,
    )
    args.post_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.post_out, "wt", encoding="utf-8") as fp:
        json.dump(post_payload, fp, separators=(",", ":"))
    yp = np.asarray(post_payload["y_train"], dtype=np.float64)
    print(
        f"Saved {post_payload['rows']} POST rows ({post_payload['episodes']} eps) to {args.post_out} "
        f"in {post_payload['elapsed_seconds']}s | cov={post_payload['candidate_coverage_rate']*100:.1f}% "
        f"pos(y>0.5)={int(np.sum(yp > 0.5))} mean_y={float(np.mean(yp)):.3f}",
        flush=True,
    )

    if not args.post_only:
        pre_payload = generate_pre_dataset(episodes=args.pre_episodes, workers=args.workers, calibrated_rates=args.calibrated_rates)
        args.pre_out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.pre_out, "wt", encoding="utf-8") as fp:
            json.dump(pre_payload, fp, separators=(",", ":"))
        ypr = np.asarray(pre_payload["y_train"], dtype=np.float64)
        print(
            f"Saved {pre_payload['rows']} PRE rows ({pre_payload['episodes']} eps) to {args.pre_out} "
            f"in {pre_payload['elapsed_seconds']}s | pos(y>0.5)={int(np.sum(ypr > 0.5))} mean_y={float(np.mean(ypr)):.3f}",
            flush=True,
        )

        plume_payload = generate_plume_prior()
        args.plume_out.write_text(json.dumps(plume_payload, separators=(",", ":")), encoding="utf-8")
        print(f"Saved {plume_payload['rows']} plume prior rows to {args.plume_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
