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
        self._obs_pos: list[Vec3] = []
        self._obs_log_ppm: list[float] = []
        self._step_count: int = 0
        self._current_target: Vec3 = (0.0, 0.0, 5.0)
        self._latest_belief: Optional[BeliefUpdatePayload] = None

    def reset(self, world_init: WorldInitPayload) -> None:
        self._grid_np = np.array(world_init.grid_coords, dtype=np.float64)
        self._obs_pos.clear()
        self._obs_log_ppm.clear()
        self._step_count = 0
        self._latest_belief = None
        if len(world_init.grid_coords) > 0:
            self._current_target = world_init.grid_coords[len(world_init.grid_coords) // 2]
        else:
            self._current_target = (0.0, 0.0, 5.0)

    def update(self, obs: DroneObservation) -> Vec3:
        self._step_count += 1
        self._obs_pos.append(obs.pos)
        self._obs_log_ppm.append(math.log1p(max(0.0, obs.ppm_measured)))

        if len(self._obs_pos) > self.MAX_HISTORY:
            self._obs_pos.pop(0)
            self._obs_log_ppm.pop(0)

        if self._grid_np.shape[0] == 0:
            return self._current_target

        # Compute fast RBF spatial belief & distance-based uncertainty over grid_coords
        obs_arr = np.array(self._obs_pos, dtype=np.float64)  # (K, 3)
        y_arr = np.array(self._obs_log_ppm, dtype=np.float64)  # (K,)

        # Pairwise squared distances: (N, K)
        diff = self._grid_np[:, None, :] - obs_arr[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        min_dist = np.sqrt(np.min(dist2, axis=1))

        # Smooth kernel estimate for q50 (log-ppm scale)
        length_scale = 18.0
        weights = np.exp(-0.5 * dist2 / (length_scale ** 2))
        w_sum = np.sum(weights, axis=1) + 1e-6
        q50 = (weights @ y_arr) / w_sum

        # Uncertainty grows with distance from visited points
        uncertainty = np.clip(1.8 * (1.0 - np.exp(-min_dist / 16.0)) + 0.15, 0.05, 2.5)

        # Spatial UCB + travel penalty + 3m revisit mask
        beta = max(0.4, 2.5 * (0.93 ** self._step_count))
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
            self._latest_belief = BeliefUpdatePayload(
                sim_time=obs.sim_time,
                drone_id=self.drone_id,
                q50=[float(v) for v in q50],
                uncertainty=[float(v) for v in uncertainty],
                estimated_source=est_src,
                next_waypoint=next_wp,
            )

        return self._current_target

    def pop_belief_update(self) -> Optional[BeliefUpdatePayload]:
        belief = self._latest_belief
        self._latest_belief = None
        return belief
