"""Baseline 1: Deterministic 3D serpentine raster sweep (Lawnmower)."""

from __future__ import annotations

import math

from agents.base import BaseAgent
from shared.schemas import BuildingBox, DroneObservation, Vec3, WorldInitPayload


def _point_in_buildings(
    pos: Vec3, buildings: list[BuildingBox], margin: float = 3.0
) -> bool:
    x, y, z = pos
    for b in buildings:
        if (
            b.min_xyz[0] - margin <= x <= b.max_xyz[0] + margin
            and b.min_xyz[1] - margin <= y <= b.max_xyz[1] + margin
            and b.min_xyz[2] <= z <= b.max_xyz[2] + margin
        ):
            return True
    return False


class LawnmowerAgent(BaseAgent):
    """Sweeps a deterministic 3D serpentine path at z = 4.0m and z = 10.0m."""

    WAYPOINT_TOLERANCE = 4.5

    def __init__(self, drone_id: str = "lawnmower") -> None:
        super().__init__(drone_id)
        self._waypoints: list[Vec3] = [(0.0, 0.0, 4.0)]
        self._idx: int = 0

    def reset(self, world_init: WorldInitPayload) -> None:
        waypoints: list[Vec3] = []
        x_coords = [-46.0, -30.0, -14.0, 2.0, 18.0, 34.0, 48.0]
        y_coords = [-46.0, -28.0, -10.0, 8.0, 26.0, 44.0]
        altitudes = [4.0, 10.0]

        for z in altitudes:
            for row_i, y in enumerate(y_coords):
                xs = x_coords if row_i % 2 == 0 else list(reversed(x_coords))
                for x in xs:
                    pt = (x, y, z)
                    if not _point_in_buildings(pt, world_init.buildings, margin=3.0):
                        waypoints.append(pt)

        self._waypoints = waypoints if waypoints else [(0.0, 0.0, 6.0)]
        self._idx = 0

    def update(self, obs: DroneObservation) -> Vec3:
        target = self._waypoints[self._idx]
        dx = obs.pos[0] - target[0]
        dy = obs.pos[1] - target[1]
        dz = obs.pos[2] - target[2]
        if math.sqrt(dx * dx + dy * dy + dz * dz) <= self.WAYPOINT_TOLERANCE:
            self._idx = (self._idx + 1) % len(self._waypoints)
            target = self._waypoints[self._idx]
        return target
