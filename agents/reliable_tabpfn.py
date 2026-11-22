"""Triple-TabPFN active 3D spatial gas-leak search and localization agent (v4).

Uses three specialized TabPFN configurations hosted on the RTX 4090 server:
1. Model A (`reliable_source_post_v4`, KV-cached): Pure-physics 3D source proximity
   inversion (29 features; zero asset/pipe or min_obs_dist shortcuts).
2. Model B (`reliable_source_pre_v4`, KV-cached): Pre-contact spatial prior &
   clean-air falsification (12 features) driving full-3D downwind catchment.
3. Model C (online + coordinate-masked prior, uncached): 3D plume & hazard
   field predictor (`q10`, `q50`, `q90` in `log1p(ppm)`).
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Optional

import numpy as np

from agents.base import AsyncInferenceAgent
from agents.search_sector import sector_mask
from agents.tabpfn_features import (
    PLUME_FEATURE_COLUMNS,
    POST_SOURCE_FEATURE_COLUMNS,
    PRE_SOURCE_FEATURE_COLUMNS,
    FacilityContext,
    ObservationHistory,
    ProcessedObs,
    build_post_source_candidates,
    build_post_source_feature_rows,
    build_pre_source_feature_rows,
)
from agents.tabpfn_nav import FastPlanner3D, safe_goal_3d
from agents.tabpfn_remote import RemoteTabPFNRegressor
from shared.schemas import (
    BeliefUpdatePayload,
    DroneObservation,
    TabPFNMapPayload,
    Vec2,
    Vec3,
    WorldInitPayload,
)

logger = logging.getLogger("reliable_tabpfn")

DEFAULT_SOURCE_POST_DATA = Path(__file__).resolve().parents[1] / "data" / "tabpfn_source_post_warm30_v1.json.gz"
DEFAULT_SOURCE_PRE_DATA = Path(__file__).resolve().parents[1] / "data" / "tabpfn_source_pre_v4.json.gz"
DEFAULT_PLUME_DATA = Path(__file__).resolve().parents[1] / "data" / "tabpfn_plume_prior.json"


@dataclass(frozen=True)
class _ColumnSpec:
    columns: list[str]


class ReliableTabPFNAgent(AsyncInferenceAgent):
    """Triple-TabPFN agent driving 3D search, localization, and hazard forecasting on the 4090."""

    SOURCE_INTERVAL_SEARCH_S: float = 2.0
    SOURCE_INTERVAL_TRACK_S: float = 1.5
    HAZARD_INTERVAL_S: float = 5.0
    MAX_PLUME_ROWS: int = 90
    GAS_CONTACT_INST_PPM: float = 2.5
    GAS_CONTACT_RAW_PPM: float = 3.5
    LOCK_HOLD_PPM: float = 15.0
    POST_CACHE_KEY: str = "reliable_source_post_warm30_v1"
    PRE_CACHE_KEY: str = "reliable_source_pre_v4"

    def __init__(
        self,
        drone_id: str = "tabpfn",
        api_url: str | None = None,
        n_estimators: int = 8,
        sync_mode: bool = False,
        apply_rpc_latency: bool = False,
        source_post_data_path: str | Path | None = None,
        source_pre_data_path: str | Path | None = None,
        plume_data_path: str | Path | None = None,
        adaptive_wind: bool = True,
        search_sector: int = 0,
        sector_count: int = 1,
    ) -> None:
        super().__init__(drone_id=drone_id, default_target=(-20.0, -15.0, 5.0))
        self.api_url = api_url
        self.n_estimators = n_estimators
        self.sync_mode = sync_mode
        self.apply_rpc_latency = apply_rpc_latency
        self.adaptive_wind = adaptive_wind
        self.search_sector = search_sector
        self.sector_count = sector_count
        self.source_post_data_path = Path(source_post_data_path) if source_post_data_path else DEFAULT_SOURCE_POST_DATA
        self.source_pre_data_path = Path(source_pre_data_path) if source_pre_data_path else DEFAULT_SOURCE_PRE_DATA
        self.plume_data_path = Path(plume_data_path) if plume_data_path else DEFAULT_PLUME_DATA

        self._data_lock = threading.Lock()
        self._remote: Optional[RemoteTabPFNRegressor] = None
        self._online_remote: Optional[RemoteTabPFNRegressor] = None

        # Cached offline datasets (loaded once)
        self._post_cols: list[str] = list(POST_SOURCE_FEATURE_COLUMNS)
        self._post_x: list[list[Any]] = []
        self._post_y: list[float] = []
        self._post_hash: str | None = None
        self._post_loaded: bool = False

        self._pre_cols: list[str] = list(PRE_SOURCE_FEATURE_COLUMNS)
        self._pre_x: list[list[Any]] = []
        self._pre_y: list[float] = []
        self._pre_hash: str | None = None
        self._pre_loaded: bool = False

        self._plume_cols: list[str] = list(PLUME_FEATURE_COLUMNS)
        self._plume_x: list[list[Any]] = []
        self._plume_y: list[float] = []
        self._plume_loaded: bool = False

        # Episode state
        self._world_init: Optional[WorldInitPayload] = None
        self._fac: Optional[FacilityContext] = None
        self._planner: Optional[FastPlanner3D] = None
        self._obs_hist = ObservationHistory()
        self._search_waypoints = np.zeros((0, 3), dtype=np.float64)
        self._search_grid_map = np.zeros(0, dtype=np.intp)

        self.history: list[DroneObservation] = []
        self.route: list[Vec3] = []
        self.planned_route: list[Vec3] = []
        self.goal: Optional[np.ndarray] = None
        self.goal_at: float = -100.0
        self.goal_timeout: float = 15.0
        self.goal_wind: Optional[np.ndarray] = None
        self.waypoint: Optional[Vec3] = None

        # TabPFN predictions & diagnostics (read by server/app.py BackendStatusPayload)
        self.models: list[_ColumnSpec] = [
            _ColumnSpec(list(POST_SOURCE_FEATURE_COLUMNS)),
            _ColumnSpec(list(PRE_SOURCE_FEATURE_COLUMNS)),
            _ColumnSpec(list(PLUME_FEATURE_COLUMNS)),
        ]
        self.error: Optional[str] = None
        self.latency: float = 0.0
        self.exposure: float = 0.0
        self.hazard: Optional[np.ndarray] = None
        self.source_scores: Optional[np.ndarray] = None
        self._tabpfn_search_map: Optional[TabPFNMapPayload] = None
        self._tabpfn_source_map: Optional[TabPFNMapPayload] = None
        self.search_phase: str = "search"
        self.completed_probes: int = 0
        self.source_prediction_calls: int = 0
        self.hazard_prediction_calls: int = 0
        self.prediction_decisions: int = 0
        self.decision_basis: str = "tabpfn_map_search"
        self.wind_replans: int = 0
        self.plan_state: str = "preparing"
        self.plan_revision: int = 0
        self.search_candidate_count: int = 0
        self.prediction_grid: Optional[np.ndarray] = None

        self._last_source_infer_t: float = -100.0
        self._last_hazard_infer_t: float = -100.0
        self._last_hazard_obs_t: float = 0.0
        self._est_source: Optional[Vec3] = None
        self._last_src_cands: Optional[np.ndarray] = None
        self._last_src_score: Optional[np.ndarray] = None
        self._last_src_unc: Optional[np.ndarray] = None
        self._last_plume_q50: Optional[np.ndarray] = None
        self._last_plume_unc: Optional[np.ndarray] = None
        self._had_contact_at_last_infer: bool = False

        self._visited_search: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._stalled_goals: list[tuple[float, np.ndarray]] = []
        self._near_est_since: Optional[float] = None
        self._near_est_pos: Optional[np.ndarray] = None
        self._pending_sync_result: Optional[tuple[float, Vec3, BeliefUpdatePayload, list[Vec3], str]] = None

    # ------------------------------------------------------------------
    # Dataset loaders & 4090 RPC
    # ------------------------------------------------------------------

    def _get_remote(self) -> RemoteTabPFNRegressor:
        if self._remote is None:
            self._remote = RemoteTabPFNRegressor(
                api_url=self.api_url,
                n_estimators=self.n_estimators,
            )
        return self._remote

    def _get_online_remote(self) -> RemoteTabPFNRegressor:
        if self._online_remote is None:
            self._online_remote = RemoteTabPFNRegressor(
                api_url=self.api_url,
                n_estimators=4,
            )
        return self._online_remote

    @staticmethod
    def _load_dataset(path: Path, expected_cols: list[str]) -> tuple[list[str], list[list[Any]], list[float], str]:
        if not path.exists():
            raise FileNotFoundError(
                f"TabPFN dataset not found at {path}. "
                "Run `.venv/bin/python -m scripts.generate_tabpfn_datasets` first."
            )
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fp:
                data = json.load(fp)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
        cols = list(data["columns"])
        if cols != expected_cols:
            raise ValueError(f"Dataset schema mismatch for {path}: {len(cols)} != {len(expected_cols)}")
        x_tr = data["X_train"]
        y_tr = [float(v) for v in data["y_train"]]
        return cols, x_tr, y_tr, ""

    def _ensure_post_data(self) -> None:
        if self._post_loaded:
            return
        cols, x_tr, y_tr, _ = self._load_dataset(self.source_post_data_path, POST_SOURCE_FEATURE_COLUMNS)
        sig = json.dumps([cols, self.n_estimators, x_tr, y_tr], separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._post_cols = cols
        self._post_x = x_tr
        self._post_y = y_tr
        self._post_hash = hashlib.sha256(sig).hexdigest()
        if self.source_post_data_path.resolve() != DEFAULT_SOURCE_POST_DATA.resolve():
            self.POST_CACHE_KEY = f"reliable_source_post_{self._post_hash[:12]}"
        self._post_loaded = True

    def _ensure_pre_data(self) -> None:
        if self._pre_loaded:
            return
        cols, x_tr, y_tr, _ = self._load_dataset(self.source_pre_data_path, PRE_SOURCE_FEATURE_COLUMNS)
        sig = json.dumps([cols, self.n_estimators, x_tr, y_tr], separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._pre_cols = cols
        self._pre_x = x_tr
        self._pre_y = y_tr
        self._pre_hash = hashlib.sha256(sig).hexdigest()
        if self.source_pre_data_path.resolve() != DEFAULT_SOURCE_PRE_DATA.resolve():
            self.PRE_CACHE_KEY = f"reliable_source_pre_{self._pre_hash[:12]}"
        self._pre_loaded = True

    def _ensure_plume_data(self) -> bool:
        if self._plume_loaded:
            return len(self._plume_x) > 0
        self._plume_loaded = True
        if not self.plume_data_path.exists():
            return False
        data = json.loads(self.plume_data_path.read_text(encoding="utf-8"))
        if list(data.get("columns", [])) != PLUME_FEATURE_COLUMNS:
            return False
        self._plume_cols = list(data["columns"])
        self._plume_x = data["X_train"]
        self._plume_y = [float(v) for v in data["y_train"]]
        return len(self._plume_x) > 0

    def warmup_remote(self) -> None:
        """Pre-load & KV-cache both post-contact and pre-contact TabPFN models on the 4090."""
        self._ensure_post_data()
        self._ensure_pre_data()
        self._ensure_plume_data()
        remote = self._get_remote()

        dummy_pre = [[0.0 if c != "nearest_asset_type" else "open_air" for c in self._pre_cols]]
        remote.predict_quantiles(
            self._pre_cols, self._pre_x, self._pre_y, dummy_pre,
            quantiles=(0.1, 0.5, 0.9), cache_key=self.PRE_CACHE_KEY, data_hash=self._pre_hash,
        )
        dummy_post = [[0.0 for _ in self._post_cols]]
        remote.predict_mean_and_quantiles(
            self._post_cols, self._post_x, self._post_y, dummy_post,
            quantiles=(0.1, 0.5, 0.9), cache_key=self.POST_CACHE_KEY, data_hash=self._post_hash,
        )

    # ------------------------------------------------------------------
    # Reset & Precomputation
    # ------------------------------------------------------------------

    def _on_reset(self, world_init: WorldInitPayload) -> None:
        fac = FacilityContext(world_init)
        planner = FastPlanner3D(world_init)

        raw_wps: list[np.ndarray] = []
        if fac.grid_np.shape[0] > 0:
            raw_wps.append(fac.grid_np[planner.free_batch(fac.grid_np)])

        extra_pts: list[Vec3] = []
        for a in world_init.assets:
            extra_pts.append(safe_goal_3d(planner, a.pos))
            extra_pts.append(safe_goal_3d(planner, (a.pos[0], a.pos[1], min(26.0, a.pos[2] + 3.0))))
            if a.pos[2] >= 9.0:
                for ox, oy in ((8.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (0.0, -8.0)):
                    extra_pts.append(safe_goal_3d(planner, (a.pos[0] + ox, a.pos[1] + oy, a.pos[2])))
                    extra_pts.append(safe_goal_3d(planner, (a.pos[0] + ox, a.pos[1] + oy, min(26.0, a.pos[2] + 2.5))))
        for p in fac.pipe_points:
            extra_pts.append(safe_goal_3d(planner, p))
        for b in world_init.buildings:
            if b.kind == "pipe_segment":
                continue
            xm = 0.5 * (b.min_xyz[0] + b.max_xyz[0])
            ym = 0.5 * (b.min_xyz[1] + b.max_xyz[1])
            zm = max(3.0, min(8.0, 0.5 * b.max_xyz[2]))
            for pt in (
                (b.min_xyz[0] - 3.5, ym, zm),
                (b.max_xyz[0] + 3.5, ym, zm),
                (xm, b.min_xyz[1] - 3.5, zm),
                (xm, b.max_xyz[1] + 3.5, zm),
            ):
                extra_pts.append(safe_goal_3d(planner, pt))
        if extra_pts:
            raw_wps.append(np.asarray(extra_pts, dtype=np.float64))

        merged = np.vstack(raw_wps) if raw_wps else planner.points.copy()
        merged = fac._dedup_points(merged, voxel_m=3.2)
        merged = merged[planner.free_batch(merged)]
        if merged.shape[0] == 0:
            merged = planner.points.copy()

        if self.sector_count > 1:
            merged = merged[sector_mask(merged, world_init, self.search_sector, self.sector_count)]
            if len(merged) == 0:
                merged = planner.points[
                    sector_mask(planner.points, world_init, self.search_sector, self.sector_count)
                ]

        grid_map = (
            np.argmin(np.linalg.norm(merged[:, None, :] - fac.grid_np[None, :, :], axis=2), axis=1)
            if fac.grid_np.shape[0] > 0
            else np.zeros(merged.shape[0], dtype=np.intp)
        )

        with self._data_lock:
            self._world_init = world_init
            self._fac = fac
            self._planner = planner
            self._obs_hist.clear()
            self._search_waypoints = merged
            self._search_grid_map = grid_map
            self.prediction_grid = fac.grid_np.copy()
            self.search_candidate_count = int(fac.grid_np.shape[0])

            self.history.clear()
            self.route.clear()
            self.planned_route.clear()
            self.goal = None
            self.goal_at = -100.0
            self.goal_timeout = 15.0
            self.goal_wind = None
            self.waypoint = None

            self.error = None
            self.latency = 0.0
            self.exposure = 0.0
            self.hazard = None
            self.source_scores = None
            self._tabpfn_search_map = None
            self._tabpfn_source_map = None
            self.search_phase = "search"
            self.completed_probes = 0
            self.source_prediction_calls = 0
            self.hazard_prediction_calls = 0
            self.prediction_decisions = 0
            self.decision_basis = "tabpfn_map_search"
            self.wind_replans = 0
            self.plan_state = "preparing"
            self.plan_revision = 0

            self._last_source_infer_t = -100.0
            self._last_hazard_infer_t = -100.0
            self._last_hazard_obs_t = 0.0
            self._est_source = None
            self._last_src_cands = None
            self._last_src_score = None
            self._last_src_unc = None
            self._last_plume_q50 = None
            self._last_plume_unc = None
            self._had_contact_at_last_infer = False
            self._visited_search.clear()
            self._stalled_goals.clear()
            self._near_est_since = None
            self._near_est_pos = None
            self._pending_sync_result = None

    # ------------------------------------------------------------------
    # Observation ingestion & 20Hz-safe target stepping
    # ------------------------------------------------------------------

    def _has_recent_contact(self, window_s: float = 45.0) -> bool:
        samples = self._obs_hist.samples
        if not samples:
            return False
        now_t = samples[-1].sim_time
        previous_hit_time = None
        for r in reversed(samples):
            dt = now_t - r.sim_time
            if dt > 55.0:
                break
            if r.ppm_raw >= 15.0:
                return True
            if dt <= window_s and r.ppm_raw >= 6.0 and (
                (not r.is_clean and r.ppm_inst >= self.GAS_CONTACT_INST_PPM)
                or r.ppm_raw >= self.GAS_CONTACT_RAW_PPM
            ):
                if previous_hit_time is not None and previous_hit_time - r.sim_time <= 1.0:
                    return True
                previous_hit_time = r.sim_time
        return False

    def _record_observation(self, obs: DroneObservation) -> Optional[Vec3]:
        with self._data_lock:
            if self._fac is None or self._planner is None:
                return (0.0, 0.0, 6.0)
            if self.history and obs.sim_time <= self.history[-1].sim_time + 1e-9:
                return self.route[0] if self.route else obs.pos

            elapsed = max(0.0, obs.sim_time - self.history[-1].sim_time) if self.history else 0.0
            self.history.append(obs)
            self.exposure += float(obs.ppm_measured) * elapsed
            rec = self._obs_hist.append(obs)
            self._visited_search.append((rec.sim_time, rec.pos.copy(), rec.wind_est.copy()))

            if self.source_prediction_calls == 0 and self.goal is None:
                wx, wy = self._obs_hist.current_wind
                spd = math.hypot(wx, wy)
                ux, uy = (wx / spd, wy / spd) if spd >= 0.2 else (1.0, 0.0)
                if self._fac.asset_pos.shape[0] > 0:
                    idx = int(np.argmax(self._fac.asset_age))
                    ap = self._fac.asset_pos[idx]
                    probe = safe_goal_3d(self._planner, (ap[0] + ux * 10.0, ap[1] + uy * 10.0, max(3.5, ap[2])))
                else:
                    probe = safe_goal_3d(self._planner, (0.0, 0.0, 6.0))
                probes = [probe]
                if self.sector_count > 1:
                    closest = np.argsort(np.linalg.norm(self._search_waypoints - probe, axis=1))
                    probes = [tuple(self._search_waypoints[idx]) for idx in closest[:24]]
                for probe in probes:
                    path = self._planner.route(obs.pos, probe)
                    if len(path) >= 2:
                        break
                else:
                    return None if self.sector_count > 1 else obs.pos
                self.goal = np.array(probe, dtype=np.float64)
                self.goal_at = obs.sim_time
                self.goal_timeout = 18.0
                self.goal_wind = np.array([wx, wy], dtype=np.float64)
                self.route = path[1:]
                self.waypoint = probe
                self.planned_route = [obs.pos, *self.route]
                return self.route[0]

            return None

    def _follow_route(self, obs: DroneObservation) -> Vec3:
        planner = self._planner
        while (len(self.route) > 1 and math.dist(obs.pos, self.route[0]) < 2.2
               and planner.visible(obs.pos, self.route[1])):
            self.route.pop(0)
        if self.route and planner.free(np.asarray(obs.pos)) and not planner.visible(obs.pos, self.route[0]):
            self.route.clear()
        self.planned_route = [obs.pos, *self.route]
        return self.route[0] if self.route else obs.pos

    def update(self, obs: DroneObservation) -> Vec3:
        """Non-blocking in live/async mode; deterministic optional latency in sync_mode."""
        if not self.sync_mode:
            super().update(obs)
            with self._data_lock:
                return self._follow_route(obs)

        self._record_observation(obs)
        with self._data_lock:
            if self._pending_sync_result is not None and obs.sim_time + 1e-9 >= self._pending_sync_result[0]:
                _, target, belief, route, basis = self._pending_sync_result
                self._pending_sync_result = None
                self.route = list(route)
                self.planned_route = [obs.pos, *self.route]
                self.decision_basis = basis
                self._latest_belief = belief

        if self._should_infer(obs):
            t_start = time.perf_counter()
            target, belief = self._run_inference(obs)
            wall_infer = time.perf_counter() - t_start
            if belief is not None:
                if self.apply_rpc_latency and wall_infer > 0.0:
                    with self._data_lock:
                        self._pending_sync_result = (
                            obs.sim_time + wall_infer,
                            target,
                            belief,
                            list(self.route),
                            self.decision_basis,
                        )
                else:
                    with self._lock:
                        self._latest_belief = belief
                    with self._data_lock:
                        self._current_target = target

        with self._data_lock:
            return self._follow_route(obs)

    def _should_infer(self, obs: DroneObservation) -> bool:
        with self._data_lock:
            if self._fac is None:
                return False
            has_contact = self._has_recent_contact()
            dt_src = obs.sim_time - self._last_source_infer_t
            interval = self.SOURCE_INTERVAL_TRACK_S if has_contact else self.SOURCE_INTERVAL_SEARCH_S
            if self.source_prediction_calls == 0:
                return True
            if has_contact and not self._had_contact_at_last_infer:
                return True
            if dt_src >= interval - 1e-9:
                return True
            if dt_src >= 0.5 - 1e-9 and self.goal is not None and math.dist(obs.pos, tuple(self.goal)) < 3.6:
                return True
            return False

    # ------------------------------------------------------------------
    # Triple-TabPFN Inference & Active 3D Search Policy
    # ------------------------------------------------------------------

    def _build_plume_train_rows(
        self, fac: FacilityContext, samples: list[ProcessedObs], wind_now: Vec2
    ) -> tuple[list[list[Any]], list[float]]:
        w_cur = np.asarray(wind_now, dtype=np.float64)
        w_cur_n = max(float(np.linalg.norm(w_cur)), 1e-6)
        regime = [
            r
            for r in samples
            if (float(np.dot(r.wind_est, w_cur)) / max(float(np.linalg.norm(r.wind_est)) * w_cur_n, 1e-6)) > 0.65
            or r.ppm_inst >= 10.0
        ]
        pool = regime if len(regime) >= 4 else samples
        if len(pool) > self.MAX_PLUME_ROWS:
            hits = sorted([r for r in pool if r.ppm_inst >= 3.0], key=lambda r: r.ppm_inst, reverse=True)[:25]
            hit_times = {r.sim_time for r in hits}
            rest = [r for r in pool if r.sim_time not in hit_times]
            step = max(1, len(rest) // max(1, self.MAX_PLUME_ROWS - len(hits)))
            pool = sorted(hits + rest[::step][-(self.MAX_PLUME_ROWS - len(hits)):], key=lambda r: r.sim_time)

        pts = np.array([r.pos for r in pool], dtype=np.float64)
        types, d_a, age, d_p, press, is_gas, _ = fac.static_features(pts)
        in_sh, up_sc = fac.wind_features(pts, wind_now)

        live_x: list[list[Any]] = []
        live_y: list[float] = []
        for i, r in enumerate(pool):
            wx = None if r.wind_dropped else float(r.wind_est[0])
            wy = None if r.wind_dropped else float(r.wind_est[1])
            live_x.append([
                "LIVE", float(r.pos[0]), float(r.pos[1]), float(r.pos[2]), wx, wy,
                types[i], float(d_a[i]), float(age[i]), float(d_p[i]), float(press[i]),
                float(is_gas[i]), float(in_sh[i]), float(up_sc[i]),
            ])
            live_y.append(float(r.log_inst))

        if self._ensure_plume_data():
            return self._plume_x + live_x, self._plume_y + live_y
        return live_x, live_y

    def _build_plume_query_rows(self, fac: FacilityContext, wind_now: Vec2) -> list[list[Any]]:
        g = fac.grid_np
        g_types, g_da, g_age, g_dp, g_press, g_gas, _ = fac.grid_static
        in_sh, up_sc = fac.wind_features(g, wind_now)
        wx, wy = float(wind_now[0]), float(wind_now[1])
        return [
            [
                "LIVE", float(g[i, 0]), float(g[i, 1]), float(g[i, 2]), wx, wy,
                g_types[i], float(g_da[i]), float(g_age[i]), float(g_dp[i]), float(g_press[i]),
                float(g_gas[i]), float(in_sh[i]), float(up_sc[i]),
            ]
            for i in range(g.shape[0])
        ]

    @staticmethod
    def _hazard_prob_from_quantiles(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> np.ndarray:
        thr = math.log1p(10.0)
        q10 = np.asarray(q10, dtype=np.float64)
        q50 = np.maximum(q10 + 1e-4, np.asarray(q50, dtype=np.float64))
        q90 = np.maximum(q50 + 1e-4, np.asarray(q90, dtype=np.float64))
        probs = np.zeros_like(q50)
        b10 = thr <= q10
        b10_50 = (thr > q10) & (thr <= q50)
        b50_90 = (thr > q50) & (thr <= q90)
        a90 = thr > q90
        probs[b10] = np.clip(0.90 + 0.08 * (q10[b10] - thr) / np.maximum(q10[b10], 0.5), 0.90, 0.98)
        probs[b10_50] = 0.50 + 0.40 * (q50[b10_50] - thr) / (q50[b10_50] - q10[b10_50])
        probs[b50_90] = 0.10 + 0.40 * (q90[b50_90] - thr) / (q90[b50_90] - q50[b50_90])
        probs[a90] = np.clip(0.10 * np.exp(-(thr - q90[a90]) / 0.8), 0.005, 0.10)
        return np.clip(probs, 0.0, 1.0)

    def _compute_downwind_catchment(
        self,
        fac: FacilityContext,
        wps: np.ndarray,
        src_coords: np.ndarray,
        pre_risk: np.ndarray,
        dw_clear: np.ndarray,
        os_clear: np.ndarray,
        wind_now: Vec2,
    ) -> np.ndarray:
        """Project ALL unverified 3D source risk downwind onto reachable flight waypoints (no top-k=80 blind spot)."""
        if wps.shape[0] == 0 or src_coords.shape[0] == 0:
            return np.zeros(wps.shape[0], dtype=np.float64)

        # Downweight verified-clean hypotheses dynamically; keep top 380 for fast Phase 1 CPU
        s_w_all = np.maximum(0.0, pre_risk) * np.exp(-0.85 * dw_clear - 1.25 * os_clear)
        if src_coords.shape[0] > 380:
            keep = np.argpartition(s_w_all, -380)[-380:]
            src_pts = src_coords[keep]
            s_w = s_w_all[keep]
        else:
            src_pts = src_coords
            s_w = s_w_all
        wx, wy = float(wind_now[0]), float(wind_now[1])
        spd = math.hypot(wx, wy)
        dx = wps[:, None, 0] - src_pts[None, :, 0]
        dy = wps[:, None, 1] - src_pts[None, :, 1]
        dz = wps[:, None, 2] - src_pts[None, :, 2]

        if spd < 0.8:
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            kern = np.where(dist <= 16.0, np.exp(-0.5 * (dist / 6.5) ** 2), 0.0)
        else:
            ux, uy = wx / spd, wy / spd
            along = dx * ux + dy * uy
            cross = np.abs(-dx * uy + dy * ux)
            valid = (along >= 1.0) & (along <= 28.0) & (dz >= -5.5) & (dz <= 7.0)
            sigma_y = 2.6 + 0.24 * np.maximum(along, 0.0)
            kern = np.where(
                valid,
                np.exp(-0.5 * (cross / sigma_y) ** 2 - 0.5 * (dz / 4.2) ** 2 - np.maximum(along, 0.0) / 24.0),
                0.0,
            )

        # Occlusion check on non-negligible links
        occ = kern > 0.14
        if np.any(occ) and fac.bldg_min.shape[0] > 0:
            wi, sj = np.nonzero(occ)
            if wi.size > 4000:
                order = np.argsort(kern[wi, sj])[::-1][:4000]
                wi, sj = wi[order], sj[order]
            trans = fac.segment_transmission_pairs(src_pts[sj], wps[wi])
            kern[wi, sj] *= trans

        catch = kern @ s_w
        c_max = float(np.max(catch))
        return catch / c_max if c_max > 1e-9 else catch

    def _select_phase1_target(
        self,
        obs: DroneObservation,
        fac: FacilityContext,
        planner: FastPlanner3D,
        src_coords: np.ndarray,
        src_score: np.ndarray,
        src_unc: np.ndarray,
        dw_clear: np.ndarray,
        os_clear: np.ndarray,
        plume_q50: np.ndarray,
        plume_unc: np.ndarray,
        wind_now: Vec2,
    ) -> tuple[Vec3, list[Vec3], bool]:
        """Choose next collision-free 3D waypoint via full-3D TabPFN catchment + Z-aware novelty."""
        drone_pos = np.array(obs.pos, dtype=np.float64)
        wps = self._search_waypoints
        if len(wps) == 0:
            return obs.pos, [], False
        w_cur = np.array(wind_now, dtype=np.float64)
        w_cur_n = max(float(np.linalg.norm(w_cur)), 1e-6)

        continuing_search = self.sector_count == 1 or not self._had_contact_at_last_infer
        if continuing_search and self.goal is not None and self.search_phase == "search" and self.route:
            d_goal = float(np.linalg.norm(drone_pos - self.goal))
            wind_changed = False
            if self.adaptive_wind and self.goal_wind is not None:
                gw_n = float(np.linalg.norm(self.goal_wind))
                if gw_n >= 0.4 and w_cur_n >= 0.4:
                    wind_changed = float(np.dot(w_cur, self.goal_wind) / (w_cur_n * gw_n)) < 0.70
            if d_goal >= 3.8 and (obs.sim_time - self.goal_at) <= self.goal_timeout and not wind_changed:
                return (float(self.goal[0]), float(self.goal[1]), float(self.goal[2])), list(self.route), False
            if wind_changed:
                self.wind_replans += 1

        pre_risk = src_score + 0.85 * src_unc
        catchment = self._compute_downwind_catchment(fac, wps, src_coords, pre_risk, dw_clear, os_clear, wind_now)

        plume_ucb = plume_q50 + 1.2 * plume_unc
        p_max = float(np.max(plume_ucb)) if plume_ucb.size else 1.0
        norm_plume = plume_ucb / max(p_max, 1e-6)
        wp_plume = norm_plume[self._search_grid_map] if norm_plume.size else np.zeros(wps.shape[0])

        compat_pos = [
            vp
            for vt, vp, vw in self._visited_search
            if (obs.sim_time - vt <= 85.0)
            and (
                float(np.dot(vw, w_cur)) / max(float(np.linalg.norm(vw)) * w_cur_n, 1e-6) > 0.60
                or float(np.linalg.norm(vp - drone_pos)) < 6.0
            )
        ]
        if compat_pos:
            vis_arr = np.asarray(compat_pos, dtype=np.float64)
            # Z-weighted metric so visiting z=4m does not mark z=14m elevated volume as covered!
            d_xy = wps[:, None, :2] - vis_arr[None, :, :2]
            d_z = (wps[:, None, 2] - vis_arr[None, :, 2]) * 1.6
            min_vis = np.min(np.sqrt(np.sum(d_xy * d_xy, axis=2) + d_z * d_z), axis=1)
            novelty = 1.0 - np.exp(-min_vis / 7.5)
            revisit_mask = np.where(min_vis < 6.5, -10.0, 0.0)
        else:
            novelty = np.ones(wps.shape[0], dtype=np.float64)
            revisit_mask = np.zeros(wps.shape[0], dtype=np.float64)

        d_xy = np.hypot(wps[:, 0] - drone_pos[0], wps[:, 1] - drone_pos[1])
        d_z = np.abs(wps[:, 2] - drone_pos[2])
        travel_s = np.hypot(d_xy, 1.3 * d_z) / 7.5

        utility = (
            (2.6 * catchment + 0.9 * wp_plume + 0.35) * (0.20 + 0.80 * novelty) / (1.0 + 0.045 * travel_s)
            + revisit_mask
        )
        for idx in np.argsort(utility)[::-1][:24]:
            cand = wps[int(idx)]
            g = (float(cand[0]), float(cand[1]), float(cand[2]))
            path = planner.route(obs.pos, g)
            if len(path) >= 2:
                return g, path[1:], True

        if self.sector_count > 1:
            return obs.pos, [], False
        fallback = safe_goal_3d(planner, (0.0, 0.0, 6.0))
        path = planner.route(obs.pos, fallback)
        return (fallback, path[1:], True) if len(path) >= 2 else (obs.pos, [], False)

    def _select_phase2_target(
        self,
        obs: DroneObservation,
        fac: FacilityContext,
        planner: FastPlanner3D,
        samples: list[ProcessedObs],
        src_coords: np.ndarray,
        src_score: np.ndarray,
        src_unc: np.ndarray,
        est_src: Vec3,
        plume_q50: np.ndarray,
        wind_now: Vec2,
    ) -> tuple[Vec3, list[Vec3], str]:
        """TabPFN-driven convergence, lock hold, and Model-C-guided 3D micro-lock box."""
        drone_pos = np.array(obs.pos, dtype=np.float64)
        est_np = np.array(est_src, dtype=np.float64)
        goal_est = safe_goal_3d(planner, est_np)
        goal_est_np = np.array(goal_est, dtype=np.float64)
        dist_drone_est = float(np.linalg.norm(drone_pos - est_np))
        dist_drone_goal = float(np.linalg.norm(drone_pos - goal_est_np))

        self._stalled_goals = [(t, p) for t, p in self._stalled_goals if obs.sim_time - t <= 45.0]

        if dist_drone_goal <= 3.5 or dist_drone_est <= 4.5:
            if self._near_est_pos is None or float(np.linalg.norm(self._near_est_pos - est_np)) > 3.0:
                self._near_est_since = obs.sim_time
                self._near_est_pos = est_np.copy()
        else:
            self._near_est_since = None
            self._near_est_pos = None

        hover_s = (obs.sim_time - self._near_est_since) if self._near_est_since is not None else 0.0
        last_hit_t = max(
            (r.sim_time for r in samples if (not r.is_clean and r.ppm_inst >= self.GAS_CONTACT_INST_PPM) or r.ppm_raw >= 4.0),
            default=obs.sim_time,
        )
        time_since_hit = obs.sim_time - last_hit_t

        # 1. CRITICAL LOCK HOLD: stay steady inside gas core when near TabPFN estimate with >=15 ppm
        if (dist_drone_est <= 4.5 or dist_drone_goal <= 3.0) and obs.ppm_measured >= self.LOCK_HOLD_PPM and hover_s < 4.5:
            if dist_drone_est <= 4.2 or dist_drone_goal <= 3.0:
                hold_pt = safe_goal_3d(planner, obs.pos)
                if math.dist(hold_pt, obs.pos) <= 1.5:
                    path = planner.route(obs.pos, hold_pt)
                    if path:
                        return hold_pt, path[1:], "tabpfn_source_prediction"
            path = planner.route(obs.pos, goal_est)
            if path:
                return goal_est, path[1:], "tabpfn_source_prediction"

        wx, wy = float(wind_now[0]), float(wind_now[1])
        spd = math.hypot(wx, wy)
        ux, uy = (wx / spd, wy / spd) if spd >= 0.2 else (1.0, 0.0)

        # 1a. CORE ANCHOR: when flying directly through a very high-ppm core (>=120 ppm),
        # do not sprint far upwind to a distant first-contact estimate; probe locally
        # just upwind so TabPFN quickly brackets the leak without losing contact.
        if obs.ppm_measured >= 120.0 and dist_drone_est > 4.5:
            local_upwind = safe_goal_3d(
                planner,
                (drone_pos[0] - ux * 2.5, drone_pos[1] - uy * 2.5, drone_pos[2]),
            )
            path = planner.route(obs.pos, local_upwind)
            if path:
                return local_upwind, (path[1:] if len(path) >= 2 else [local_upwind]), "tabpfn_source_prediction"

        # 1b. Model-C-guided 3D micro-lock box BEFORE marking an estimate stalled
        near_clean = (dist_drone_goal <= 3.5 or dist_drone_est <= 4.5) and obs.ppm_measured < self.LOCK_HOLD_PPM
        if near_clean and hover_s >= 0.8:
            recent_10s = np.array([r.pos for r in samples if obs.sim_time - r.sim_time <= 10.0], dtype=np.float64)
            near_hot = [
                r
                for r in samples
                if (obs.sim_time - r.sim_time <= 45.0)
                and r.ppm_raw >= 25.0
                and float(np.linalg.norm(r.pos - est_np)) <= 5.5
                and float(np.linalg.norm(r.pos - drone_pos)) >= 1.8
                and (recent_10s.shape[0] == 0 or float(np.min(np.linalg.norm(recent_10s - r.pos[None, :], axis=1))) >= 2.2)
                and not any(float(np.linalg.norm(r.pos - sp)) < 2.2 for _, sp in self._stalled_goals)
            ]
            if near_hot:
                best_hit = max(near_hot, key=lambda r: r.ppm_raw)
                probe_hit = safe_goal_3d(planner, best_hit.pos)
                path = planner.route(obs.pos, probe_hit)
                if len(path) >= 2:
                    return probe_hit, path[1:], "tabpfn_source_prediction"

            recent_20s = np.array([r.pos for r in samples if obs.sim_time - r.sim_time <= 20.0], dtype=np.float64)
            raw_offsets = [
                (est_np[0] + ux * 1.8, est_np[1] + uy * 1.8, min(28.0, est_np[2] + 0.6)),
                (est_np[0] + ux * 2.8, est_np[1] + uy * 2.8, est_np[2]),
                (est_np[0] + ux * 2.5, est_np[1] + uy * 2.5, min(28.0, est_np[2] + 2.0)),
                (est_np[0] + ux * 2.5, est_np[1] + uy * 2.5, max(1.5, est_np[2] - 2.0)),
                (est_np[0] - uy * 2.5, est_np[1] + ux * 2.5, est_np[2]),
            ]
            valid_offsets: list[tuple[int, Vec3]] = []
            for oi, raw_pt in enumerate(raw_offsets):
                cand_pt = safe_goal_3d(planner, raw_pt)
                if math.dist(cand_pt, obs.pos) < 1.8:
                    continue
                if recent_20s.shape[0] > 0 and float(np.min(np.linalg.norm(recent_20s - np.asarray(cand_pt)[None, :], axis=1))) < 2.2:
                    continue
                if any(math.dist(cand_pt, tuple(sp)) < 2.2 for _, sp in self._stalled_goals):
                    continue
                valid_offsets.append((oi, cand_pt))

            if valid_offsets:
                if fac.grid_np.shape[0] > 0 and plume_q50.size == fac.grid_np.shape[0]:
                    off_arr = np.array([pt for _, pt in valid_offsets], dtype=np.float64)
                    g_idx = np.argmin(np.linalg.norm(off_arr[:, None, :] - fac.grid_np[None, :, :], axis=2), axis=1)
                    mc_vals = plume_q50[g_idx]
                    if float(np.ptp(mc_vals)) > 1e-4:
                        ranked_offsets = [
                            valid_offsets[k][1]
                            for k in sorted(range(len(valid_offsets)), key=lambda idx: (-float(mc_vals[idx]), valid_offsets[idx][0]))
                        ]
                    else:
                        ranked_offsets = [pt for _, pt in valid_offsets]
                else:
                    ranked_offsets = [pt for _, pt in valid_offsets]

                for probe_off in ranked_offsets:
                    path = planner.route(obs.pos, probe_off)
                    if len(path) >= 2:
                        return probe_off, path[1:], "tabpfn_source_prediction"

        # 2. Detect stall (>4.5s with >=15ppm without lock, OR >2.2s in clean air after local micro-lock checks)
        is_stalled_here = (
            (dist_drone_goal <= 3.5 or dist_drone_est <= 4.5)
            and ((obs.ppm_measured >= self.LOCK_HOLD_PPM and hover_s >= 4.5) or (obs.ppm_measured < self.LOCK_HOLD_PPM and hover_s >= 2.2))
        )
        if is_stalled_here:
            self._stalled_goals.append((obs.sim_time, goal_est_np.copy()))
            self._near_est_since = None
            self._near_est_pos = None

        goal_is_stalled = any(float(np.linalg.norm(goal_est_np - sp)) < 3.5 for _, sp in self._stalled_goals)

        # 3. Direct approach to TabPFN estimate when not stalled & recent gas contact
        if not goal_is_stalled and time_since_hit <= max(18.0, dist_drone_est / 2.2 + 8.0):
            path = planner.route(obs.pos, goal_est)
            if path:
                return goal_est, (path[1:] if len(path) >= 2 else [goal_est]), "tabpfn_source_prediction"

        # 4. Direction-aware disambiguation when stalled!
        recent_samples = [r for r in samples if obs.sim_time - r.sim_time <= 50.0]
        if goal_is_stalled and recent_samples:
            if obs.ppm_measured >= self.LOCK_HOLD_PPM:
                # Stalled IN gas without lock -> stepwise upwind core-seeking: 4.5m -> 9.5m -> crosswind
                rec_pos = np.array([r.pos for r in recent_samples], dtype=np.float64)
                dx_r = est_np[0] - rec_pos[:, 0]
                dy_r = est_np[1] - rec_pos[:, 1]
                al_r = dx_r * ux + dy_r * uy
                cr_r = np.abs(-dx_r * uy + dy_r * ux)
                dz_r = np.abs(est_np[2] - rec_pos[:, 2])

                up_steps = [
                    (
                        bool(np.any((al_r >= 2.5) & (al_r <= 6.5) & (cr_r <= 3.5) & (dz_r <= 3.5))),
                        (est_np[0] - ux * 4.5, est_np[1] - uy * 4.5, est_np[2]),
                        2.5,
                    ),
                    (
                        bool(np.any((al_r >= 6.5) & (al_r <= 13.5) & (cr_r <= 4.5) & (dz_r <= 4.0))),
                        (est_np[0] - ux * 9.5, est_np[1] - uy * 9.5, est_np[2]),
                        3.5,
                    ),
                    (
                        False,
                        (est_np[0] - ux * 4.5 - uy * 3.5, est_np[1] - uy * 4.5 + ux * 3.5, est_np[2]),
                        2.5,
                    ),
                    (
                        False,
                        (est_np[0] - ux * 4.5 + uy * 3.5, est_np[1] - uy * 4.5 - ux * 3.5, est_np[2]),
                        2.5,
                    ),
                ]
                for already_checked, raw_up, min_step_d in up_steps:
                    if already_checked:
                        continue
                    probe_up = safe_goal_3d(planner, raw_up)
                    if (
                        math.dist(probe_up, obs.pos) >= min_step_d
                        and float(np.min(np.linalg.norm(rec_pos - np.asarray(probe_up)[None, :], axis=1))) >= 2.2
                        and not any(math.dist(probe_up, tuple(sp)) < 3.0 for _, sp in self._stalled_goals)
                    ):
                        path = planner.route(obs.pos, probe_up)
                        if len(path) >= 2:
                            return probe_up, path[1:], "tabpfn_source_prediction"
            else:
                rec_pos = np.array([r.pos for r in recent_samples], dtype=np.float64)
                dx_d = rec_pos[:, 0] - est_np[0]
                dy_d = rec_pos[:, 1] - est_np[1]
                al_d = dx_d * ux + dy_d * uy
                cr_d = np.abs(-dx_d * uy + dy_d * ux)
                dz_d = np.abs(rec_pos[:, 2] - est_np[2])
                dw_checked = bool(np.any((al_d >= 2.0) & (al_d <= 5.0) & (cr_d <= 2.5) & (dz_d <= 2.5)))
                if not dw_checked:
                    probe_dw = safe_goal_3d(planner, (est_np[0] + ux * 3.5, est_np[1] + uy * 3.5, est_np[2]))
                    if (
                        math.dist(probe_dw, obs.pos) >= 2.0
                        and not any(math.dist(probe_dw, tuple(sp)) < 2.8 for _, sp in self._stalled_goals)
                    ):
                        path = planner.route(obs.pos, probe_dw)
                        if len(path) >= 2:
                            return probe_dw, path[1:], "tabpfn_source_prediction"

        # 5. Active Disambiguation via TabPFN UCB excluding stalled / recently visited points
        recent_pts = np.array([r.pos for r in samples[-16:]], dtype=np.float64)
        d_to_cands = np.linalg.norm(src_coords - drone_pos[None, :], axis=1)
        rel_est = src_coords - est_np[None, :]
        along_est = rel_est[:, 0] * ux + rel_est[:, 1] * uy
        upwind_probe_bonus = np.where((along_est >= -14.0) & (along_est <= -3.0) & (np.linalg.norm(rel_est, axis=1) <= 15.0), 0.28, 0.0)

        ucb = src_score + 1.50 * src_unc + upwind_probe_bonus - 0.006 * d_to_cands
        if self._stalled_goals:
            st_arr = np.asarray([sp for _, sp in self._stalled_goals], dtype=np.float64)
            d_st = np.min(np.linalg.norm(src_coords[:, None, :] - st_arr[None, :, :], axis=2), axis=1)
            ucb = np.where(d_st < 4.0, ucb - 5.0, ucb)
        if recent_pts.shape[0] > 0:
            d_rec = np.min(np.linalg.norm(src_coords[:, None, :] - recent_pts[None, :, :], axis=2), axis=1)
            ucb = np.where(d_rec < 3.5, ucb - 3.0, ucb)

        for idx in np.argsort(ucb)[::-1][:30]:
            probe = safe_goal_3d(planner, src_coords[int(idx)])
            if any(math.dist(probe, tuple(sp)) < 3.2 for _, sp in self._stalled_goals):
                continue
            path = planner.route(obs.pos, probe)
            if len(path) >= 2:
                return probe, path[1:], "tabpfn_source_prediction"

        path = planner.route(obs.pos, goal_est)
        if path:
            return goal_est, path[1:], "tabpfn_source_prediction"
        return obs.pos, [], "holding_for_clear_route"

    def _run_inference(self, obs: DroneObservation) -> tuple[Vec3, Optional[BeliefUpdatePayload]]:
        with self._data_lock:
            if self._fac is None or self._planner is None:
                return obs.pos, None
            fac = self._fac
            planner = self._planner
            samples = list(self._obs_hist.samples)
            if not samples:
                return obs.pos, None
            wind_now = self._obs_hist.current_wind
            prev_est = self._est_source
            has_contact = self._has_recent_contact()
            last_src_t = self._last_source_infer_t
            last_haz_t = self._last_hazard_infer_t
            interval = self.SOURCE_INTERVAL_TRACK_S if has_contact else self.SOURCE_INTERVAL_SEARCH_S
            need_src = (
                self.source_prediction_calls == 0
                or (has_contact and not self._had_contact_at_last_infer)
                or (obs.sim_time - last_src_t >= interval - 1e-9)
                or (obs.sim_time - last_src_t >= 0.5 - 1e-9 and self.goal is not None and math.dist(obs.pos, tuple(self.goal)) < 3.6)
            )

        if not need_src and not self.sync_mode:
            with self._data_lock:
                return (self.route[0] if self.route else obs.pos), None

        t0 = time.perf_counter()
        try:
            remote = self._get_remote()
            n_grid = fac.grid_np.shape[0]
            mode_phase = "search"
            search_map = source_map = None

            if has_contact:
                # TabPFN Model A: Pure-physics Post-Contact 3D Source Inversion (mean expectation + quantiles)
                self._ensure_post_data()
                cands, grid_idx = build_post_source_candidates(
                    fac, samples, wind_now, prev_estimate=prev_est, max_candidates=950
                )
                src_rows = build_post_source_feature_rows(fac, cands, samples, wind_now)
                s_mean, (sq10, sq50, sq90) = remote.predict_mean_and_quantiles(
                    self._post_cols,
                    self._post_x,
                    self._post_y,
                    src_rows,
                    quantiles=(0.1, 0.5, 0.9),
                    cache_key=self.POST_CACHE_KEY,
                    data_hash=self._post_hash,
                )
                src_score = np.clip(s_mean, 0.0, 1.0)
                src_unc = np.maximum(0.0, sq90 - sq10)
                best_idx = int(np.argmax(src_score))
                best_s = float(src_score[best_idx])
                best_pt = cands[best_idx]

                # Sharp local-cluster weighted centroid within 3.5m of argmax
                d_best = np.linalg.norm(cands - best_pt[None, :], axis=1)
                cluster = (d_best <= 3.5) & (src_score >= best_s - 0.06)
                est_pt = best_pt
                if int(np.sum(cluster)) >= 2:
                    w = (np.maximum(0.0, src_score[cluster] - (best_s - 0.12)) + 1e-4) ** 4
                    cent = np.sum(cands[cluster] * w[:, None], axis=0) / np.sum(w)
                    if not fac.inside_solid_mask(cent[None, :], margin=-0.15)[0]:
                        est_pt = cent
                if prev_est is not None and self._had_contact_at_last_infer:
                    prev_np = np.array(prev_est, dtype=np.float64)
                    if (
                        float(np.linalg.norm(est_pt - prev_np)) <= 4.5
                        and obs.ppm_measured >= 20.0
                        and float(np.linalg.norm(np.array(obs.pos, dtype=np.float64) - est_pt)) <= 5.0
                    ):
                        smoothed = 0.35 * est_pt + 0.65 * prev_np
                        if not fac.inside_solid_mask(smoothed[None, :], margin=-0.15)[0]:
                            est_pt = smoothed
                est_src: Vec3 = (float(est_pt[0]), float(est_pt[1]), float(est_pt[2]))

                grid_src = np.zeros(n_grid, dtype=np.float64)
                grid_unc = np.full(n_grid, 0.20, dtype=np.float64)
                if grid_idx.size > 0:
                    grid_src[grid_idx] = src_score[:grid_idx.size]
                    grid_unc[grid_idx] = src_unc[:grid_idx.size]
                scored_cells = [None] * n_grid
                for i in grid_idx:
                    scored_cells[int(i)] = round(float(grid_src[i]), 4)
                source_map = TabPFNMapPayload(sim_time=obs.sim_time, scores=scored_cells, estimate=est_src)

                # Check if gas signal expired & confidence collapsed -> drop back to search
                last_hit_t = max(
                    (r.sim_time for r in samples if (not r.is_clean and r.ppm_inst >= self.GAS_CONTACT_INST_PPM) or r.ppm_raw >= 4.0),
                    default=obs.sim_time,
                )
                last_strong_t = max(
                    (r.sim_time for r in samples if (not r.is_clean and r.ppm_inst >= self.LOCK_HOLD_PPM) or r.ppm_raw >= 15.0),
                    default=-1e9,
                )
                has_recent_strong = (obs.sim_time - last_strong_t) <= 60.0
                if (has_recent_strong and (obs.sim_time - last_hit_t) > 35.0 and best_s < 0.15) or (
                    not has_recent_strong and (obs.sim_time - last_hit_t) > 25.0 and best_s < 0.20
                ):
                    mode_phase = "search"
                else:
                    mode_phase = "track"

            if mode_phase == "search":
                # TabPFN Model B: Pre-Contact Spatial Prior & Clean-Air Falsification
                self._ensure_pre_data()
                cands = fac.base_src_coords
                pre_rows = build_pre_source_feature_rows(fac, cands, samples, wind_now, use_base_static=True)
                sq10, sq50, sq90 = remote.predict_quantiles(
                    self._pre_cols,
                    self._pre_x,
                    self._pre_y,
                    pre_rows,
                    quantiles=(0.1, 0.5, 0.9),
                    cache_key=self.PRE_CACHE_KEY,
                    data_hash=self._pre_hash,
                )
                src_score = np.clip(sq50, 0.0, 1.0)
                search_map = TabPFNMapPayload(sim_time=obs.sim_time, scores=np.round(src_score[:n_grid], 4).tolist())
                src_unc = np.maximum(0.0, sq90 - sq10)
                best_idx = int(np.argmax(src_score))
                best_s = float(src_score[best_idx])
                if not has_contact:
                    est_src = (
                        float(cands[best_idx, 0]),
                        float(cands[best_idx, 1]),
                        float(cands[best_idx, 2]),
                    )
                    grid_src = src_score[:n_grid] if n_grid > 0 else np.zeros(0)
                    grid_unc = src_unc[:n_grid] if n_grid > 0 else np.zeros(0)
                col_dwc = PRE_SOURCE_FEATURE_COLUMNS.index("downwind_clear_score")
                col_osc = PRE_SOURCE_FEATURE_COLUMNS.index("onsite_clear_score")
                dw_clear = np.array([float(r[col_dwc]) for r in pre_rows], dtype=np.float64)
                os_clear = np.array([float(r[col_osc]) for r in pre_rows], dtype=np.float64)

            # TabPFN Model C: Online + coordinate-masked prior Plume & Hazard Mapper (4 estimators)
            need_haz = (
                self._last_plume_q50 is None
                or (has_contact and not self._had_contact_at_last_infer)
                or (obs.sim_time - last_haz_t >= self.HAZARD_INTERVAL_S - 1e-9)
            )
            if need_haz and n_grid > 0:
                tx, ty = self._build_plume_train_rows(fac, samples, wind_now)
                qx = self._build_plume_query_rows(fac, wind_now)
                pq10, pq50, pq90 = self._get_online_remote().predict_quantiles(
                    self._plume_cols, tx, ty, qx, quantiles=(0.1, 0.5, 0.9), cache_key=None,
                )
                plume_q50 = np.maximum(0.0, pq50)
                plume_unc = np.maximum(0.0, pq90 - pq10)
                hazard_probs = self._hazard_prob_from_quantiles(pq10, pq50, pq90)
                hazard_obs_t = obs.sim_time
                haz_called = True
            else:
                plume_q50 = self._last_plume_q50 if self._last_plume_q50 is not None else np.zeros(n_grid)
                plume_unc = self._last_plume_unc if self._last_plume_unc is not None else np.ones(n_grid) * 0.5
                hazard_probs = self.hazard if self.hazard is not None else np.zeros(n_grid)
                hazard_obs_t = self._last_hazard_obs_t
                haz_called = False

            with self._data_lock:
                if fac is not self._fac:
                    return obs.pos, None
                if search_map is not None:
                    self._tabpfn_search_map = search_map
                if source_map is not None:
                    self._tabpfn_source_map = source_map
                if mode_phase == "track":
                    self.search_phase = "track"
                    next_goal, next_route, basis = self._select_phase2_target(
                        obs, fac, planner, samples, cands, src_score, src_unc, est_src, plume_q50, wind_now
                    )
                    decision_inc = 1
                else:
                    self.search_phase = "search"
                    next_goal, next_route, replanned = self._select_phase1_target(
                        obs, fac, planner, cands, src_score, src_unc, dw_clear, os_clear, plume_q50, plume_unc, wind_now
                    )
                    basis = "tabpfn_map_search"
                    decision_inc = 1 if replanned else 0

                elapsed = time.perf_counter() - t0
                if n_grid > 0 and float(grid_src.sum()) < 1e-6:
                    d_eg = np.linalg.norm(fac.grid_np - np.array(est_src)[None, :], axis=1)
                    grid_src = np.exp(-0.5 * (d_eg / 8.0) ** 2)
                norm_src = grid_src / max(float(grid_src.sum()), 1e-9) if n_grid > 0 else grid_src

                regional_mass = 0.0
                if n_grid > 0:
                    d_est_grid = np.linalg.norm(fac.grid_np - np.array(est_src)[None, :], axis=1)
                    regional_mass = float(np.sum(norm_src[d_est_grid <= 8.0]))

                self._last_source_infer_t = obs.sim_time
                if haz_called:
                    self._last_hazard_infer_t = obs.sim_time
                    self._last_hazard_obs_t = hazard_obs_t
                    self._last_plume_q50 = plume_q50
                    self._last_plume_unc = plume_unc
                    self.hazard = hazard_probs
                    self.hazard_prediction_calls += 1

                self._est_source = est_src
                self._last_src_cands = cands
                self._last_src_score = src_score
                self._last_src_unc = src_unc
                self._had_contact_at_last_infer = has_contact

                self.source_scores = norm_src
                self.route = list(next_route)
                self.goal = np.array(next_goal, dtype=np.float64)
                if decision_inc:
                    d_xy = math.hypot(next_goal[0] - obs.pos[0], next_goal[1] - obs.pos[1])
                    d_z = abs(next_goal[2] - obs.pos[2])
                    est_t = max(d_xy / 8.0, d_z / 3.0)
                    self.goal_at = obs.sim_time
                    self.goal_timeout = max(12.0, 2.0 * est_t + 4.0)
                    self.goal_wind = np.array(wind_now, dtype=np.float64)
                    self.completed_probes += 1
                    self.plan_revision += 1
                self.waypoint = next_goal
                self.planned_route = [obs.pos, *self.route]
                self.decision_basis = basis
                self.prediction_decisions += decision_inc
                self.source_prediction_calls += 1
                self.plan_state = "tabpfn_map"
                self.latency = elapsed
                self.error = None

                belief = BeliefUpdatePayload(
                    sim_time=obs.sim_time,
                    drone_id=self.drone_id,
                    q50=norm_src.tolist(),
                    uncertainty=grid_unc.tolist(),
                    estimated_source=est_src,
                    next_waypoint=next_goal,
                    source_scores=norm_src.tolist(),
                    hazard_probabilities=hazard_probs.tolist(),
                    hazard_observed_at=hazard_obs_t,
                    hazard_forecast_time=hazard_obs_t + 4.0,
                    source_prediction_backend="tabpfn-4090-regressor",
                    hazard_prediction_backend="tabpfn-4090-regressor",
                    navigation_basis=basis,
                    prediction_decisions=self.prediction_decisions,
                    route=self.planned_route.copy(),
                    inference_seconds=elapsed,
                    measurement_count=len(samples),
                    exposure_ppm_seconds=self.exposure,
                    score_kind="source_score",
                    source_model_score=best_s,
                    source_evidence=has_contact,
                    regional_score_mass=regional_mass,
                    tabpfn_search_map=self._tabpfn_search_map,
                    tabpfn_source_map=self._tabpfn_source_map,
                )
                return (self.route[0] if self.route else next_goal), belief

        except Exception as exc:
            with self._data_lock:
                if fac is not self._fac:
                    return obs.pos, None
                self.error = str(exc)
                if self.plan_state == "preparing":
                    self.plan_state = "error"
            logger.exception("ReliableTabPFNAgent inference failed: %s", exc)
            raise
