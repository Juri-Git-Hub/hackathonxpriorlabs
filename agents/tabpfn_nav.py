"""Fast obstacle-aware 3D A* navigation and safe goal projection for TabPFN search."""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from shared.flight_geometry import FLIGHT_CLEARANCE, flight_bounds
from shared.schemas import Vec3, WorldInitPayload

# Dense 0.75m local lattice offsets for exact narrow-corridor safe goal escape
_LOCAL_GRID = np.stack(
    np.meshgrid(
        np.arange(-4.5, 4.6, 0.75),
        np.arange(-4.5, 4.6, 0.75),
        np.arange(-4.5, 4.6, 0.75),
        indexing="ij",
    ),
    axis=-1,
).reshape(-1, 3)


def _axis_points(low: float, high: float, spacing: float) -> np.ndarray:
    if high < low:
        return np.array([0.5 * (low + high)], dtype=np.float64)
    pts = np.arange(low, high, spacing, dtype=np.float64)
    return np.unique(np.r_[pts, high])


class FastPlanner3D:
    """Obstacle-aware 3D A* planner derived from WorldInitPayload bounds and buildings."""

    CLEARANCE: float = FLIGHT_CLEARANCE + 0.5

    def __init__(self, world_init: WorldInitPayload, spacing_xy: float = 10.0) -> None:
        self.world = world_init
        self.bounds_lo = np.asarray(world_init.bounds_min, dtype=np.float64) + np.array([2.0, 2.0, 1.5])
        self.bounds_hi = np.asarray(world_init.bounds_max, dtype=np.float64) - np.array([2.0, 2.0, 1.5])

        self.bounds_hi[2] = min(self.bounds_hi[2], 28.0)
        self.lo, self.hi = flight_bounds(world_init.buildings, world_init.pipe_runs or world_init.pipes, self.CLEARANCE)

        # Build Z layers including rooftop clearances for elevated leaks
        z_base = _axis_points(float(self.bounds_lo[2]), float(self.bounds_hi[2]), 5.0)
        tops = [
            b.max_xyz[2] + self.CLEARANCE + 0.35
            for b in world_init.buildings
            if self.bounds_lo[2] <= b.max_xyz[2] + self.CLEARANCE + 0.35 <= self.bounds_hi[2]
        ]
        z_raw = np.sort(np.unique(np.r_[z_base, tops]))
        z_kept: list[float] = []
        for zv in z_raw:
            if not z_kept or abs(float(zv) - z_kept[-1]) >= 1.5:
                z_kept.append(float(zv))
        z_axis = np.array(z_kept, dtype=np.float64)

        self.axes = [
            _axis_points(float(self.bounds_lo[0]), float(self.bounds_hi[0]), spacing_xy),
            _axis_points(float(self.bounds_lo[1]), float(self.bounds_hi[1]), spacing_xy),
            z_axis,
        ]

        raw_pts: list[np.ndarray] = []
        for idx in itertools.product(*(range(len(a)) for a in self.axes)):
            pt = np.array([self.axes[0][idx[0]], self.axes[1][idx[1]], self.axes[2][idx[2]]], dtype=np.float64)
            if self.free(pt):
                raw_pts.append(pt)

        # Add compact face-midpoint portals around macro buildings so narrow corridors stay connected
        portal_z = [z for z in (4.5, 10.5, 18.0) if self.bounds_lo[2] <= z <= self.bounds_hi[2]]
        for b in world_init.buildings:
            if b.kind == "pipe_segment":
                continue
            x0, y0, _ = b.min_xyz
            x1, y1, z1 = b.max_xyz
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            off = self.CLEARANCE + 0.45
            xy_cands = [(x0 - off, ym), (x1 + off, ym), (xm, y0 - off), (xm, y1 + off)]
            for pz in portal_z:
                if pz <= z1 + off + 1.0:
                    for px, py in xy_cands:
                        pt = np.array([px, py, pz], dtype=np.float64)
                        if self.free(pt):
                            raw_pts.append(pt)

        if not raw_pts:
            mid = 0.5 * (self.bounds_lo + self.bounds_hi)
            mid[2] = self.bounds_hi[2]
            raw_pts.append(mid)

        pts_arr = np.asarray(raw_pts, dtype=np.float64)
        keys = np.round(pts_arr / 1.5).astype(np.int64)
        _, uniq_idx = np.unique(keys, axis=0, return_index=True)
        self.points = pts_arr[np.sort(uniq_idx)]
        self.n_points = self.points.shape[0]
        self.adjacency: dict[int, list[int]] = {}
        self._max_edge_dist = max(16.0, spacing_xy * 1.8)

    def free(self, p: np.ndarray) -> bool:
        if np.any(p < self.bounds_lo) or np.any(p > self.bounds_hi):
            return False
        if self.lo.shape[0] == 0:
            return True
        return not bool(np.any(np.all((p >= self.lo) & (p <= self.hi), axis=1)))

    def free_batch(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized collision check for (M, 3) points."""
        ok = np.all((pts >= self.bounds_lo) & (pts <= self.bounds_hi), axis=1)
        if self.lo.shape[0] > 0 and np.any(ok):
            sub = pts[ok]
            inside = np.any(
                np.all((sub[:, None, :] >= self.lo[None, :, :]) & (sub[:, None, :] <= self.hi[None, :, :]), axis=2),
                axis=1,
            )
            ok[ok] = ~inside
        return ok

    def visible(self, a: np.ndarray | Vec3, b: np.ndarray | Vec3) -> bool:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if self.lo.shape[0] == 0:
            return True
        delta = b - a
        lower = np.zeros(self.lo.shape[0], dtype=np.float64)
        upper = np.ones(self.lo.shape[0], dtype=np.float64)
        for axis in range(3):
            if abs(delta[axis]) < 1e-10:
                outside = (a[axis] < self.lo[:, axis]) | (a[axis] > self.hi[:, axis])
                lower = np.where(outside, np.inf, lower)
            else:
                inv = 1.0 / delta[axis]
                t1 = (self.lo[:, axis] - a[axis]) * inv
                t2 = (self.hi[:, axis] - a[axis]) * inv
                lower = np.maximum(lower, np.minimum(t1, t2))
                upper = np.minimum(upper, np.maximum(t1, t2))
        return not bool(np.any(lower <= upper))

    @staticmethod
    def cost(a: np.ndarray | Vec3, b: np.ndarray | Vec3) -> float:
        d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
        return max(float(np.hypot(d[0], d[1])) / 8.0, abs(float(d[2])) / 3.0)

    def _neighbors(self, idx: int) -> list[int]:
        if idx not in self.adjacency:
            p = self.points[idx]
            dists = np.linalg.norm(self.points - p, axis=1)
            cand = np.nonzero((dists > 1e-6) & (dists <= self._max_edge_dist))[0]
            cand = cand[np.argsort(dists[cand])]
            out: list[int] = []
            for j in cand[:26]:
                jj = int(j)
                if self.visible(p, self.points[jj]):
                    out.append(jj)
                    if len(out) >= 16:
                        break
            self.adjacency[idx] = out
        return self.adjacency[idx]

    def _connectors(self, p: np.ndarray) -> list[int]:
        dists = np.linalg.norm(self.points - p, axis=1)
        order = np.argsort(dists)
        out: list[int] = []
        for idx in order[:min(120, self.n_points)]:
            ii = int(idx)
            if self.visible(p, self.points[ii]):
                out.append(ii)
                if len(out) >= 10:
                    break
        return out

    def route(self, start: Vec3 | np.ndarray, goal: Vec3 | np.ndarray) -> list[Vec3]:
        start_arr = np.asarray(start, dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)
        if not self.free(goal_arr):
            return []

        prefix: list[Vec3] = []
        if not self.free(start_arr):
            entry = np.array(safe_goal_3d(self, start_arr), dtype=np.float64)
            if not self.free(entry):
                return []
            escape = entry - start_arr
            escape_distance = float(np.linalg.norm(escape))
            if 0.0 < escape_distance < 1.5:
                # A sub-metre projection can sit inside the flight controller's
                # position tolerance while the drone is still in the margin.
                farther_entry = start_arr + escape * (1.5 / escape_distance)
                if self.free(farther_entry) and self.visible(entry, farther_entry):
                    entry = farther_entry
            prefix = [(float(start_arr[0]), float(start_arr[1]), float(start_arr[2]))]
            start_arr = entry

        if self.visible(start_arr, goal_arr):
            path = [
                *prefix,
                (float(start_arr[0]), float(start_arr[1]), float(start_arr[2])),
                (float(goal_arr[0]), float(goal_arr[1]), float(goal_arr[2])),
            ]
            if len(path) >= 2 and np.allclose(path[0], path[1], atol=1e-6):
                path.pop(0)
            return path

        starts = self._connectors(start_arr)
        ends = set(self._connectors(goal_arr))
        if not starts or not ends:
            return []

        costs = {k: self.cost(start_arr, self.points[k]) for k in starts}
        parents: dict[int, int] = {}
        queue = [(v + self.cost(self.points[k], goal_arr), v, k) for k, v in costs.items()]
        heapq.heapify(queue)

        while queue:
            _, cur_cost, k = heapq.heappop(queue)
            if cur_cost > costs[k] + 1e-12:
                continue
            if k in ends:
                raw: list[Vec3] = [
                    (float(goal_arr[0]), float(goal_arr[1]), float(goal_arr[2])),
                    tuple(map(float, self.points[k])),  # type: ignore[arg-type]
                ]
                while k in parents:
                    k = parents[k]
                    raw.append(tuple(map(float, self.points[k])))  # type: ignore[arg-type]
                raw.append((float(start_arr[0]), float(start_arr[1]), float(start_arr[2])))
                raw.reverse()
                smooth = [raw[0]]
                i = 0
                while i < len(raw) - 1:
                    j = len(raw) - 1
                    while j > i + 1 and not self.visible(raw[i], raw[j]):
                        j -= 1
                    smooth.append(raw[j])
                    i = j
                if prefix and not np.allclose(prefix[0], smooth[0], atol=1e-6):
                    smooth.insert(0, prefix[0])
                return smooth

            for n in self._neighbors(k):
                nxt_cost = cur_cost + self.cost(self.points[k], self.points[n])
                if nxt_cost < costs.get(n, float("inf")):
                    costs[n] = nxt_cost
                    parents[n] = k
                    heapq.heappush(queue, (nxt_cost + self.cost(self.points[n], goal_arr), nxt_cost, n))

        return []


def safe_goal_3d(planner: FastPlanner3D, point: Vec3 | np.ndarray) -> Vec3:
    """Project any 3D hypothesis onto the closest collision-free drone waypoint."""
    p = np.clip(np.asarray(point, dtype=np.float64), planner.bounds_lo, planner.bounds_hi)
    if planner.free(p):
        return (float(p[0]), float(p[1]), float(p[2]))

    options: list[np.ndarray] = []
    for lo, hi in zip(planner.lo, planner.hi):
        if np.all(p >= lo) and np.all(p <= hi):
            for axis in range(3):
                for value in (lo[axis] - 0.15, hi[axis] + 0.15):
                    q = p.copy()
                    q[axis] = value
                    options.append(q)

    # Dense 0.75m local lattice handles narrow corridors & sandwiched valves cleanly
    local = p[None, :] + _LOCAL_GRID
    cand_arr = np.vstack([np.asarray(options, dtype=np.float64), local]) if options else local
    cand_arr = np.clip(cand_arr, planner.bounds_lo, planner.bounds_hi)
    valid = cand_arr[planner.free_batch(cand_arr)]
    if valid.shape[0] > 0:
        costs = np.linalg.norm((valid - p) * np.array([1.0, 1.0, 1.30]), axis=1)
        best = valid[int(np.argmin(costs))]
        return (float(best[0]), float(best[1]), float(best[2]))

    nearest = planner.points[int(np.argmin(np.linalg.norm(planner.points - p, axis=1)))]
    return (float(nearest[0]), float(nearest[1]), float(nearest[2]))
