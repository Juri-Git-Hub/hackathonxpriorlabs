"""TabPFN-powered active spatial gas-leak search agent (AsyncInferenceAgent).

OWNERSHIP: AI / TabPFN team.
This module is isolated from physics, server, and frontend. All non-blocking
threading and atomic target/belief swapping is handled by `AsyncInferenceAgent`.

To test locally once `tabpfn` and `pandas` are installed:
    python -m server.app          # (without --mock-agents)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from agents.base import AsyncInferenceAgent
from shared.schemas import (
    BeliefUpdatePayload,
    DroneObservation,
    Vec3,
    WorldInitPayload,
)

# Exact feature schema shared between training rows and candidate grid rows
FEATURE_COLUMNS = [
    "x",
    "y",
    "z",
    "wind_x",
    "wind_y",
    "upwind_pipe_score",
    "nearest_asset_type",
    "asset_age_years",
    "in_wind_shadow",
    "temp_delta",
]


class TabPFNAgent(AsyncInferenceAgent):
    """Active spatial search using TabPFNRegressor quantile UCB + travel penalty.

    How it works:
    1. `_record_observation(obs)` appends the observation to a rolling 120-row buffer
       on the fast sim thread (< 1ms).
    2. `_run_inference(obs)` runs in a single background worker thread:
       - Fits `TabPFNRegressor` on the rolling in-context table (target: `log1p(ppm)`).
       - Predicts `[0.1, 0.5, 0.9]` quantiles over all N candidate points in `grid_coords`.
       - Selects the next waypoint via Spatial UCB + travel penalty + 3m revisit mask:
           `score = q50 + beta * (q90 - q10) - travel_penalty * dist + revisit_mask`
       - Returns `(next_waypoint, BeliefUpdatePayload)` which is atomically swapped in.
    """

    MAX_HISTORY = 120
    MIN_SAMPLES_FOR_FIT = 6
    TRAVEL_PENALTY = 0.015
    REVISIT_RADIUS_M = 3.0

    def __init__(
        self,
        drone_id: str = "tabpfn",
        device: str = "auto",
    ) -> None:
        super().__init__(drone_id=drone_id)
        self.device = device
        self._world_init: Optional[WorldInitPayload] = None
        self._grid_np = np.zeros((0, 3), dtype=np.float64)
        self._history: list[dict[str, Any]] = []
        self._targets: list[float] = []
        self._step_count: int = 0
        self._regressor: Any = None
        self._warmup_waypoints: list[Vec3] = [
            (-30.0, -25.0, 5.0),
            (-10.0, -20.0, 5.0),
            (10.0, -10.0, 6.0),
            (15.0, 15.0, 6.0),
            (-20.0, 15.0, 6.0),
            (0.0, 0.0, 8.0),
        ]

    def _ensure_model(self) -> Any:
        if self._regressor is None:
            try:
                from tabpfn import TabPFNRegressor
            except ImportError as exc:
                raise ImportError(
                    "tabpfn is not installed. Install `tabpfn` and `pandas`, "
                    "or run the server with `--mock-agents`."
                ) from exc
            self._regressor = TabPFNRegressor(device=self.device)
        return self._regressor

    def _on_reset(self, world_init: WorldInitPayload) -> None:
        self._world_init = world_init
        self._grid_np = np.array(world_init.grid_coords, dtype=np.float64)
        self._history.clear()
        self._targets.clear()
        self._step_count = 0

    def _record_observation(self, obs: DroneObservation) -> Optional[Vec3]:
        self._step_count += 1
        row = {
            "x": obs.pos[0],
            "y": obs.pos[1],
            "z": obs.pos[2],
            "wind_x": obs.wind_measured[0] if obs.wind_measured[0] is not None else np.nan,
            "wind_y": obs.wind_measured[1] if obs.wind_measured[1] is not None else np.nan,
            "upwind_pipe_score": obs.upwind_pipe_score,
            "nearest_asset_type": obs.nearest_asset_type,
            "asset_age_years": obs.asset_age_years,
            "in_wind_shadow": float(obs.in_wind_shadow),
            "temp_delta": obs.temp_delta if obs.temp_delta is not None else np.nan,
        }
        self._history.append(row)
        self._targets.append(math.log1p(max(0.0, obs.ppm_measured)))

        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)
            self._targets.pop(0)

        # Warm-start exploration before enough rows exist for TabPFN in-context fit
        if len(self._history) < self.MIN_SAMPLES_FOR_FIT:
            return self._warmup_waypoints[self._step_count % len(self._warmup_waypoints)]
        return None

    def _build_candidate_dataframe(self, obs: DroneObservation) -> Any:
        """Build candidate feature table (N rows matching grid_coords) for TabPFN predict.

        TODO (TabPFN team): Enrich candidate rows with per-grid-point `nearest_asset_type`,
        `asset_age_years`, `in_wind_shadow`, and `upwind_pipe_score` computed from
        `self._world_init.assets` and `self._world_init.buildings` (or `World.get_grid_meta`)
        so TabPFN can leverage facility priors across the entire 3D grid!
        """
        import pandas as pd

        wx = obs.wind_measured[0] if obs.wind_measured[0] is not None else np.nan
        wy = obs.wind_measured[1] if obs.wind_measured[1] is not None else np.nan
        n = self._grid_np.shape[0]

        return pd.DataFrame(
            {
                "x": self._grid_np[:, 0],
                "y": self._grid_np[:, 1],
                "z": self._grid_np[:, 2],
                "wind_x": np.full(n, wx, dtype=np.float64),
                "wind_y": np.full(n, wy, dtype=np.float64),
                "upwind_pipe_score": np.zeros(n, dtype=np.float64),
                "nearest_asset_type": ["open_air"] * n,
                "asset_age_years": np.zeros(n, dtype=np.float64),
                "in_wind_shadow": np.zeros(n, dtype=np.float64),
                "temp_delta": np.full(n, np.nan, dtype=np.float64),
            },
            columns=FEATURE_COLUMNS,
        )

    def _run_inference(
        self, obs: DroneObservation
    ) -> tuple[Vec3, Optional[BeliefUpdatePayload]]:
        import pandas as pd

        if self._grid_np.shape[0] == 0:
            return obs.pos, None

        # Snapshot history safely
        rows = list(self._history)
        y_train = np.array(self._targets, dtype=np.float64)
        x_train = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        x_cand = self._build_candidate_dataframe(obs)

        model = self._ensure_model()
        model.fit(x_train, y_train)
        quantiles = model.predict(
            x_cand,
            output_type="quantiles",
            quantiles=[0.1, 0.5, 0.9],
        )
        q10 = np.asarray(quantiles[0], dtype=np.float64)
        q50 = np.asarray(quantiles[1], dtype=np.float64)
        q90 = np.asarray(quantiles[2], dtype=np.float64)
        uncertainty = np.maximum(0.0, q90 - q10)

        # Spatial UCB + travel penalty + 3m revisit mask
        beta = max(0.4, 2.5 * (0.93 ** self._step_count))
        drone_pos = np.array(obs.pos, dtype=np.float64)
        travel_dist = np.linalg.norm(self._grid_np - drone_pos, axis=1)

        visited = np.array([[r["x"], r["y"], r["z"]] for r in rows], dtype=np.float64)
        diff = self._grid_np[:, None, :] - visited[None, :, :]
        min_visit_dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        revisit_mask = np.where(min_visit_dist < self.REVISIT_RADIUS_M, -1e6, 0.0)

        scores = q50 + beta * uncertainty - self.TRAVEL_PENALTY * travel_dist + revisit_mask
        best_idx = int(np.argmax(scores))
        next_wp: Vec3 = (
            float(self._grid_np[best_idx, 0]),
            float(self._grid_np[best_idx, 1]),
            float(self._grid_np[best_idx, 2]),
        )

        est_idx = int(np.argmax(q50))
        est_src: Vec3 = (
            float(self._grid_np[est_idx, 0]),
            float(self._grid_np[est_idx, 1]),
            float(self._grid_np[est_idx, 2]),
        )

        belief = BeliefUpdatePayload(
            sim_time=obs.sim_time,
            drone_id=self.drone_id,
            q50=[float(v) for v in q50],
            uncertainty=[float(v) for v in uncertainty],
            estimated_source=est_src,
            next_waypoint=next_wp,
        )
        return next_wp, belief
