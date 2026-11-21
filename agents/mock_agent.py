"""Fast (<1ms) mock TabPFN agent for parallel backend/frontend/physics development."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from agents.base import BaseAgent
from shared.schemas import (
    BeliefUpdatePayload,
    DroneObservation,
    Vec3,
    WorldInitPayload,
)


class MockTabPFNAgent(BaseAgent):
    """Synthetic spatial belief & UCB waypoint generator with zero ML dependencies.

    Used when running the server with `--mock-agents` so frontend and physics work
    never block on TabPFN weights or GPU setup.
    """

    MAX_HISTORY = 120
    TRAVEL_PENALTY = 0.015
    REVISIT_RADIUS_M = 3.0

    def __init__(self, drone_id: str = "tabpfn") -> None:
        super().__init__(drone_id)
        self._grid_np = np.zeros((0, 3), dtype=np.float64)
        self._grid_norm2 = np.zeros((0, 1), dtype=np.float64)
        self._center_prior = np.zeros(0, dtype=np.float64)
        self._obs_pos: list[Vec3] = []
        self._obs_log_ppm: list[float] = []
        self._obs_upwind_est: list[Vec3] = []
        self._step_count: int = 0
        self._current_target: Vec3 = (0.0, 0.0, 5.0)
        self._latest_belief: Optional[BeliefUpdatePayload] = None
        self._last_wind: tuple[float, float] = (3.0, 0.8)

    def reset(self, world_init: WorldInitPayload) -> None:
        self._grid_np = np.array(world_init.grid_coords, dtype=np.float64)
        if self._grid_np.shape[0] > 0:
            self._grid_norm2 = np.sum(self._grid_np * self._grid_np, axis=1, keepdims=True)
            center_d2 = np.sum((self._grid_np - np.array([-4.0, -4.0, 5.0])) ** 2, axis=1)
            self._center_prior = 0.65 * np.exp(-0.5 * center_d2 / (24.0 ** 2))
        else:
            self._grid_norm2 = np.zeros((0, 1), dtype=np.float64)
            self._center_prior = np.zeros(0, dtype=np.float64)
        self._obs_pos.clear()
        self._obs_log_ppm.clear()
        self._obs_upwind_est.clear()
        self._step_count = 0
        self._latest_belief = None
        self._last_wind = (3.0, 0.8)
        if len(world_init.grid_coords) > 0:
            # Start steering toward the central industrial pipe corridor
            dists_center = np.linalg.norm(self._grid_np - np.array([-8.0, -6.0, 5.0]), axis=1)
            self._current_target = world_init.grid_coords[int(np.argmin(dists_center))]
        else:
            self._current_target = (-8.0, -6.0, 5.0)

    def update(self, obs: DroneObservation) -> Vec3:
        self._step_count += 1
        if obs.wind_measured[0] is not None and obs.wind_measured[1] is not None:
            self._last_wind = (obs.wind_measured[0], obs.wind_measured[1])

        log_ppm = math.log1p(max(0.0, obs.ppm_measured))
        self._obs_pos.append(obs.pos)
        self._obs_log_ppm.append(log_ppm)

        # Upwind-shifted source hypothesis when elevated gas is detected
        wx, wy = self._last_wind
        spd = max(0.3, math.hypot(wx, wy))
        shift = min(14.0, max(0.0, (obs.ppm_measured - 2.0) * 0.6)) if obs.ppm_measured > 2.5 else 0.0
        upwind_pt = (
            obs.pos[0] - (wx / spd) * shift,
            obs.pos[1] - (wy / spd) * shift,
            max(3.0, min(10.0, obs.pos[2])),
        )
        self._obs_upwind_est.append(upwind_pt)

        if len(self._obs_pos) > self.MAX_HISTORY:
            self._obs_pos.pop(0)
            self._obs_log_ppm.pop(0)
            self._obs_upwind_est.pop(0)

        if self._grid_np.shape[0] == 0:
            return self._current_target

        obs_arr = np.array(self._obs_pos, dtype=np.float64)
        upwind_arr = np.array(self._obs_upwind_est, dtype=np.float64)
        y_arr = np.array(self._obs_log_ppm, dtype=np.float64)

        # Pairwise squared distances via BLAS GEMM: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a @ b.T
        obs_norm2 = np.sum(obs_arr * obs_arr, axis=1)
        dist2_vis = self._grid_norm2 + obs_norm2[None, :] - 2.0 * (self._grid_np @ obs_arr.T)
        min_dist = np.sqrt(np.maximum(0.0, np.min(dist2_vis, axis=1)))

        # Smooth upwind-aware spatial belief estimate (q50)
        upw_norm2 = np.sum(upwind_arr * upwind_arr, axis=1)
        dist2_src = self._grid_norm2 + upw_norm2[None, :] - 2.0 * (self._grid_np @ upwind_arr.T)
        np.maximum(0.0, dist2_src, out=dist2_src)
        inv_2ls2 = -0.5 / (14.0 * 14.0)
        weights = np.exp(dist2_src * inv_2ls2, out=dist2_src)
        w_sum = np.sum(weights, axis=1) + 1e-6
        q50 = (weights @ y_arr) / w_sum

        # Add gentle central facility prior before any strong plume hit
        max_obs_ppm = float(math.expm1(float(np.max(y_arr))))
        if max_obs_ppm < 4.0:
            q50 = q50 + self._center_prior

        uncertainty = np.clip(1.6 * (1.0 - np.exp(-min_dist / 15.0)) + 0.12, 0.05, 2.2)

        beta = max(0.35, 2.0 * (0.91 ** self._step_count))
        drone_pos = np.array(obs.pos, dtype=np.float64)
        travel_dist = np.linalg.norm(self._grid_np - drone_pos, axis=1)
        revisit_mask = np.where(min_dist < self.REVISIT_RADIUS_M, -1e6, 0.0)

        scores = q50 + beta * uncertainty - self.TRAVEL_PENALTY * travel_dist + revisit_mask
        best_idx = int(np.argmax(scores))
        next_wp = (
            float(self._grid_np[best_idx, 0]),
            float(self._grid_np[best_idx, 1]),
            float(self._grid_np[best_idx, 2]),
        )

        est_idx = int(np.argmax(q50))
        est_src = (
            float(self._grid_np[est_idx, 0]),
            float(self._grid_np[est_idx, 1]),
            float(self._grid_np[est_idx, 2]),
        )

        self._current_target = next_wp

        # Emit BeliefUpdatePayload every 2nd observation (~1 Hz)
        if self._step_count % 2 == 0:
            self._latest_belief = BeliefUpdatePayload.model_construct(
                type="belief_update",
                sim_time=obs.sim_time,
                drone_id=self.drone_id,
                q50=q50.tolist(),
                uncertainty=uncertainty.tolist(),
                estimated_source=est_src,
                next_waypoint=next_wp,
            )

        return self._current_target

    def pop_belief_update(self) -> Optional[BeliefUpdatePayload]:
        belief = self._latest_belief
        self._latest_belief = None
        return belief
