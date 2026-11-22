"""Obstacle-aware 3D A* with swept-segment AABB collision checks (Z-up)."""

from __future__ import annotations

import heapq
import itertools
import threading

import numpy as np

from agents.base import AsyncInferenceAgent


class Planner3D:
    CLEARANCE = 2.0

    def __init__(self, world):
        self.world = world
        self.bounds_lo = np.array([world.bounds_min[0] + 2, world.bounds_min[1] + 2, 1.5])
        self.bounds_hi = np.array(
            [world.bounds_max[0] - 2, world.bounds_max[1] - 2, min(28.0, world.bounds_max[2] - 1.5)]
        )
        self.lo = (
            np.array([b.min_xyz for b in world.buildings], dtype=float).reshape(-1, 3)
            - self.CLEARANCE
            + 1e-6
        )
        self.hi = (
            np.array([b.max_xyz for b in world.buildings], dtype=float).reshape(-1, 3)
            + self.CLEARANCE
            - 1e-6
        )
        self.axes = [
            np.linspace(-52, 52, 12),
            np.linspace(-52, 52, 12),
            np.array([3.0, 8.0, 13.0, 18.0, 23.0, 28.0]),
        ]
        self.nodes = {}
        for idx in itertools.product(*[range(len(a)) for a in self.axes]):
            p = np.array([self.axes[i][idx[i]] for i in range(3)])
            if self.free(p):
                self.nodes[idx] = p
        self.indices = list(self.nodes)
        self.points = np.array(list(self.nodes.values()))
        self.adjacency = {}
        self.offsets = list(itertools.product([-1, 0, 1], repeat=3))
        self.offsets.remove((0, 0, 0))

    def free(self, p):
        if np.any(p < self.bounds_lo) or np.any(p > self.bounds_hi):
            return False
        return not np.any(np.all((p >= self.lo) & (p <= self.hi), axis=1))

    def visible(self, a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        delta = b - a
        lower = np.zeros(len(self.lo))
        upper = np.ones(len(self.lo))
        for axis in range(3):
            if abs(delta[axis]) < 1e-10:
                outside = (a[axis] < self.lo[:, axis]) | (a[axis] > self.hi[:, axis])
                lower = np.where(outside, np.inf, lower)
            else:
                t1 = (self.lo[:, axis] - a[axis]) / delta[axis]
                t2 = (self.hi[:, axis] - a[axis]) / delta[axis]
                lower = np.maximum(lower, np.minimum(t1, t2))
                upper = np.minimum(upper, np.maximum(t1, t2))
        return not np.any(lower <= upper)

    @staticmethod
    def cost(a, b):
        d = np.asarray(b) - a
        return max(float(np.linalg.norm(d[:2])) / 8, abs(float(d[2])) / 3)

    def neighbors(self, key):
        if key not in self.adjacency:
            p = self.nodes[key]
            result = []
            for offset in self.offsets:
                n = tuple(key[i] + offset[i] for i in range(3))
                if n in self.nodes and self.visible(p, self.nodes[n]):
                    result.append(n)
            self.adjacency[key] = result
        return self.adjacency[key]

    def connectors(self, p):
        ids = np.argsort(np.linalg.norm(self.points - p, axis=1))
        return [self.indices[i] for i in ids[:32] if self.visible(p, self.points[i])][:8]

    def route(self, start, goal):
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        if not self.free(start) or not self.free(goal):
            return []
        if self.visible(start, goal):
            return [tuple(start), tuple(goal)]
        starts = self.connectors(start)
        ends = set(self.connectors(goal))
        costs = {k: self.cost(start, self.nodes[k]) for k in starts}
        parents = {}
        queue = [(v + self.cost(self.nodes[k], goal), v, k) for k, v in costs.items()]
        heapq.heapify(queue)
        while queue:
            _, cost, k = heapq.heappop(queue)
            if cost > costs[k]:
                continue
            if k in ends:
                path = [tuple(goal), tuple(self.nodes[k])]
                while k in parents:
                    k = parents[k]
                    path.append(tuple(self.nodes[k]))
                path.append(tuple(start))
                path.reverse()
                # Smooth only across segments which are collision-free in 3D.
                smooth = [path[0]]
                i = 0
                while i < len(path) - 1:
                    j = len(path) - 1
                    while j > i + 1 and not self.visible(path[i], path[j]):
                        j -= 1
                    smooth.append(path[j])
                    i = j
                return smooth
            for n in self.neighbors(k):
                new = cost + self.cost(self.nodes[k], self.nodes[n])
                if new < costs.get(n, float("inf")):
                    costs[n] = new
                    parents[n] = k
                    heapq.heappush(queue, (new + self.cost(self.nodes[n], goal), new, n))
        return []


class RoutedAgent(AsyncInferenceAgent):
    """Baseline policy plus nonblocking 3D routing and fast waypoint following."""

    def __init__(self, policy):
        super().__init__(policy.drone_id)
        self.policy = policy
        self.nav_lock = threading.Lock()

    def _on_reset(self, world):
        planner = Planner3D(world)
        self.policy.reset(world)
        with self.nav_lock:
            self.planner = planner
            self.path = []
            self.goal = None
            self.desired = None

    def _record_observation(self, obs):
        target = self.policy.update(obs)
        with self.nav_lock:
            self.desired = target
        return None

    def _run_inference(self, obs):
        with self.nav_lock:
            goal = self.desired
            desired = goal
            planner = self.planner
            if self.goal == goal and self.path:
                return self.path[0], None
        if not planner.free(np.array(goal)):
            goal = tuple(
                planner.points[np.argmin(np.linalg.norm(planner.points - np.array(goal), axis=1))]
            )
        path = planner.route(obs.pos, goal)[1:]
        with self.nav_lock:
            if planner is self.planner:
                self.path = path
                self.goal = desired
        return (path[0] if path else obs.pos), None

    def update(self, obs):
        super().update(obs)
        with self.nav_lock:
            while len(self.path) > 1 and np.linalg.norm(np.array(self.path[0]) - obs.pos) < 2:
                self.path.pop(0)
            return self.path[0] if self.path else obs.pos
