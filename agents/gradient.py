"""Baseline 2: Bio-inspired surge-cast / gradient ascent agent."""

from __future__ import annotations

import math

from agents.base import BaseAgent
from shared.schemas import DroneObservation, Vec3, WorldInitPayload


class GradientAgent(BaseAgent):
    """Bio-inspired surge-cast plume tracker.

    - Below threshold: broad cross-facility search.
    - Rising/high concentration: surge upwind into the plume.
    - Dropping concentration: crosswind cast sweeps to re-acquire the plume.
    Naturally vulnerable to downwind wake pockets and sudden wind shifts.
    """

    PPM_THRESHOLD = 4.0
    SURGE_STEP = 10.0

    def __init__(self, drone_id: str = "gradient") -> None:
        super().__init__(drone_id)
        self._bounds_min: Vec3 = (-54.0, -54.0, 2.5)
        self._bounds_max: Vec3 = (54.0, 54.0, 20.0)
        self._prev_ppm: float = 0.0
        self._prev_z=5.;self._vertical_dir=1.
        self._last_wind: tuple[float, float] = (3.0, 0.0)
        self._cast_dir: float = 1.0
        self._cast_steps: int = 0
        self._search_waypoints: list[Vec3] = []
        self._search_idx: int = 0
        self._mode: str = "search"

    def reset(self, world_init: WorldInitPayload) -> None:
        self._bounds_min = (
            world_init.bounds_min[0] + 6.0,
            world_init.bounds_min[1] + 6.0,
            2.5,
        )
        self._bounds_max = (
            world_init.bounds_max[0] - 6.0,
            world_init.bounds_max[1] - 6.0,
            world_init.bounds_max[2] - 2.0,
        )
        self._prev_ppm = 0.0
        self._prev_z=5.;self._vertical_dir=1.
        self._last_wind = (3.0, 0.0)
        self._cast_dir = 1.0
        self._cast_steps = 0
        self._mode = "search"
        self._search_waypoints = [
            (-30.0, -35.0, 13.0),
            (30.0, -30.0, 23.0),
            (35.0, 15.0, 8.0),
            (-25.0, 20.0, 18.0),
            (0.0, -10.0, 3.0),
            (15.0, 35.0, 28.0),
            (-40.0, 0.0, 5.0),
        ]
        import numpy as np
        from agents.lawnmower import _point_in_buildings
        grid=np.array(world_init.grid_coords)
        self._search_waypoints=[tuple(map(float,grid[np.argmin(np.linalg.norm(grid-np.array(wp),axis=1))])) if _point_in_buildings(wp,world_init.buildings,margin=2) else wp for wp in self._search_waypoints]
        self._search_idx = 0

    def _clamp(self, x: float, y: float, z: float) -> Vec3:
        return (
            max(self._bounds_min[0], min(self._bounds_max[0], x)),
            max(self._bounds_min[1], min(self._bounds_max[1], y)),
            max(self._bounds_min[2], min(self._bounds_max[2], z)),
        )

    def update(self, obs: DroneObservation) -> Vec3:
        wx, wy = obs.wind_measured
        if wx is not None and wy is not None:
            self._last_wind = (wx, wy)

        w_speed = math.hypot(self._last_wind[0], self._last_wind[1])
        if w_speed > 1e-3:
            # Upwind unit vector (opposite of wind direction)
            ux = -self._last_wind[0] / w_speed
            uy = -self._last_wind[1] / w_speed
        else:
            ux, uy = -1.0, 0.0

        # Crosswind unit vector
        cx, cy = -uy, ux

        ppm = obs.ppm_measured
        px, py, pz = obs.pos

        dz=pz-self._prev_z
        if abs(dz)>.3 and ppm>=self.PPM_THRESHOLD:
            self._vertical_dir=(1 if dz>0 else -1)*(1 if ppm>=self._prev_ppm else -1)
        if pz>=self._bounds_max[2]-1:self._vertical_dir=-1
        if pz<=self._bounds_min[2]+1:self._vertical_dir=1
        probe_z=pz+3*self._vertical_dir
        if ppm >= self.PPM_THRESHOLD:
            if ppm >= self._prev_ppm * 0.92:
                # Surge upwind
                self._mode = "surge"
                self._cast_steps = 0
                target = self._clamp(
                    px + ux * self.SURGE_STEP,
                    py + uy * self.SURGE_STEP,
                    probe_z,
                )
            else:
                # Concentration dropped: cast crosswind
                self._mode = "cast"
                self._cast_steps += 1
                self._cast_dir *= -1.0
                cast_amp = min(22.0, 7.0 + 3.0 * self._cast_steps)
                target = self._clamp(
                    px + cx * self._cast_dir * cast_amp + ux * 2.0,
                    py + cy * self._cast_dir * cast_amp + uy * 2.0,
                    probe_z,
                )
        else:
            if self._mode in ("surge", "cast") and self._cast_steps < 4:
                # Brief cast to re-enter lost plume before falling back to search
                self._mode = "cast"
                self._cast_steps += 1
                self._cast_dir *= -1.0
                cast_amp = 10.0 + 4.0 * self._cast_steps
                target = self._clamp(
                    px + cx * self._cast_dir * cast_amp,
                    py + cy * self._cast_dir * cast_amp,
                    probe_z,
                )
            else:
                self._mode = "search"
                self._cast_steps = 0
                wp = self._search_waypoints[self._search_idx]
                if math.dist(obs.pos, wp) <= 5.0:
                    self._search_idx = (self._search_idx + 1) % len(self._search_waypoints)
                    wp = self._search_waypoints[self._search_idx]
                target = wp

        self._prev_z=pz
        self._prev_ppm = ppm
        return target
