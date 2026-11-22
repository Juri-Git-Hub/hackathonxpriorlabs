"""Conservative flight collision geometry shared by planning and motion."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from shared.schemas import BuildingBox, PipeRun, Vec3

FLIGHT_CLEARANCE = 2.0
PIPE_FLANGE_RADIUS_SCALE = 1.42
Bounds = tuple[Vec3, Vec3]


def flight_bounds(
    buildings: Iterable[BuildingBox], pipes: Iterable[PipeRun], clearance: float = FLIGHT_CLEARANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clearance-expanded boxes for structures and the rendered pipe envelope."""
    pipes = tuple(pipes)
    boxes = [
        (building.min_xyz, building.max_xyz)
        for building in buildings
        if building.kind != "pipe_segment" or not pipes
    ]
    for pipe in pipes:
        for start, end in zip(pipe.points, pipe.points[1:]):
            length = math.dist(start, end)
            if length < 0.05:
                continue

            # Short boxes along diagonal legs preserve space under sloping runs.
            # The envelope covers both the pipe and its larger flange collars.
            a, b = np.asarray(start), np.asarray(end)
            diagonal = np.count_nonzero(a != b) > 1
            count = math.ceil(length / 2.0) if diagonal else 1
            points = np.linspace(a, b, count + 1)
            radius = pipe.radius * PIPE_FLANGE_RADIUS_SCALE
            for first, last in zip(points, points[1:]):
                boxes.append((np.minimum(first, last) - radius, np.maximum(first, last) + radius))

            # Match the renderer's midpoint stanchion on elevated horizontal legs.
            x, y, z = (a + b) * 0.5
            support_height = z - pipe.radius
            if length > 3.5 and abs(a[2] - b[2]) < 0.35 and z > 0.8 and support_height > 0.2:
                boxes.append(((x - 0.13, y - 0.13, 0.0), (x + 0.13, y + 0.13, support_height)))

    values = np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
    return values[:, 0] - clearance, values[:, 1] + clearance


def motion_candidates(bounds: tuple[Bounds, ...], start: Vec3, reach: Vec3) -> list[Bounds]:
    """Broad-phase query for all boxes a drone could reach during one tick."""
    x, y, z = start
    dx, dy, dz = reach
    return [
        (lo, hi) for lo, hi in bounds
        if lo[0] <= x + dx and hi[0] >= x - dx
        and lo[1] <= y + dy and hi[1] >= y - dy
        and lo[2] <= z + dz and hi[2] >= z - dz
    ]


def sweep_motion(bounds: Iterable[Bounds], start: Vec3, end: Vec3) -> tuple[float, int]:
    """Return the first entry fraction and its face axis, or (1, -1) for clear motion."""
    delta = tuple(b - a for a, b in zip(start, end))
    min_x, min_y, min_z = (min(a, b) for a, b in zip(start, end))
    max_x, max_y, max_z = (max(a, b) for a, b in zip(start, end))
    earliest, axis = 1.0, -1
    for lo, hi in bounds:
        if (
            lo[0] > max_x or hi[0] < min_x or lo[1] > max_y or hi[1] < min_y
            or lo[2] > max_z or hi[2] < min_z
        ):
            continue
        entry, leave, face = 0.0, earliest, -1
        for i, movement in enumerate(delta):
            if abs(movement) < 1e-12:
                if not lo[i] <= start[i] <= hi[i]:
                    break
                continue

            t0 = (lo[i] - start[i]) / movement
            t1 = (hi[i] - start[i]) / movement
            near, far = min(t0, t1), max(t0, t1)
            if near >= entry:
                entry, face = near, i
            leave = min(leave, far)
            if entry > leave:
                break
        else:
            # A drone already embedded by an external placement may move out.
            if face >= 0 and entry <= earliest:
                earliest, axis = entry, face
    return earliest, axis
