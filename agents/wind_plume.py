"""Wind-relative flight policy using only onboard observations and the facility map."""
from collections import deque

import numpy as np

from agents.base import AsyncInferenceAgent
from agents.search_control import SearchController
from shared.schemas import BeliefUpdatePayload


class WindPlumeController(SearchController):
    DWELL_SECONDS = 2.5
    TRANSECT_ARRIVAL_RADIUS = 1.5

    def __init__(self, world):
        super().__init__(world)
        self.transect = deque()
        self.surveyed = set()
        self.transit_hit = None
        self.revisited_hits = []

    def _survey(self, obs):
        u, cross = self._wind(obs)
        if np.linalg.norm(u) < .5:
            u, cross = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        if not self.transect:
            options = []
            for index, asset in enumerate(self.world.assets):
                for band, offset in enumerate((1., 5., -4.)):
                    for setback in (9., 20.):
                        key = (index, band, setback)
                        if key in self.surveyed:
                            continue
                        center = np.asarray(asset.pos) + setback * u
                        center[2] = np.clip(center[2] + offset, 2.5, self.planner.bounds_hi[2])
                        ends = [center - 6 * cross, center + 6 * cross]
                        ends.sort(key=lambda p: np.linalg.norm(p - obs.pos))
                        cost = np.linalg.norm(ends[0] - obs.pos) + 12 * band + setback
                        options.append((cost, key, ends))
            if options:
                _, key, ends = min(options, key=lambda option: option[0])
                self.surveyed.add(key)
                self.transect.extend(ends)
        while self.transect:
            if self.set_goal(self.transect.popleft(), obs, 'downwind_transect'):
                return
        super()._survey(obs)

    def _advance_anchor(self, ppm):
        if self.probe_kind == "upwind":
            return ppm >= max(8., .2 * self.center_ppm)
        return super()._advance_anchor(ppm)

    def _next_probe(self, obs):
        if self.transit_hit is not None:
            hit = self.transit_hit
            self.transit_hit = None
            if obs.sim_time - hit.sim_time < 18 and self.set_goal(hit.pos, obs, "revisit"):
                self.revisited_hits.append(hit.pos)
                return
        if not self.cycle_active and self.center_ppm >= 8:
            u, cross = self._wind(obs)
            if np.linalg.norm(u) >= .5:
                self.cycle_active = True
                self.probes = [
                    self.anchor - cross * self.step,
                    self.anchor + cross * self.step,
                    self.anchor + [0, 0, 3],
                    self.anchor - [0, 0, 3],
                ]
                if self.set_goal(self.anchor - u * self.step, obs, "upwind"):
                    return
        super()._next_probe(obs)

    def ingest(self, obs):
        previous = self.phase
        if (
            self.phase == "track" and self.history and self.goal is not None
            and self.probe_kind != "revisit" and self.transit_hit is None
            and np.linalg.norm(np.asarray(obs.pos) - self.goal) > 4
            and np.linalg.norm(np.asarray(obs.pos) - self.anchor) > 5
            and obs.ppm_measured >= max(40., self.center_ppm * 1.5, self.history[-1].ppm_measured * 4)
            and all(np.linalg.norm(np.asarray(obs.pos) - hit) > 4 for hit in self.revisited_hits)
        ):
            self.transit_hit = obs
        super().ingest(obs)
        if (
            self.phase == "search" and self.probe_kind == "downwind_transect"
            and self.goal is not None and np.linalg.norm(np.asarray(obs.pos) - self.goal) < self.TRANSECT_ARRIVAL_RADIUS
        ):
            self._survey(obs)
        if previous != self.phase:
            self.transit_hit = None
            self.transect.clear()


class WindPlumeAgent(AsyncInferenceAgent):
    """Run collision-aware planning off the physics thread."""

    def __init__(self, drone_id="tabpfn", *, synchronous=False):
        super().__init__(drone_id)
        self.synchronous = synchronous

    def _on_reset(self, world_init):
        self.controller = WindPlumeController(world_init)
        self.search_phase = "search"
        self.decision_basis = "downwind_transect"
        self.plan_state = "wind_plume"
        self.completed_probes = 0
        self.planned_route = []
        self.history = []
        self.source_scores = None
        self.goal = None
        self.wind = (0., 0.)

    def _record_observation(self, obs):
        return None

    def update(self, obs):
        if not self.synchronous:
            return super().update(obs)
        target, belief = self._run_inference(obs)
        if belief is not None:
            self._latest_belief = belief
        return target

    def _run_inference(self, obs):
        controller = self.controller
        controller.ingest(obs)
        waypoint = controller.waypoint(obs)
        estimate = controller.anchor
        belief = None
        if controller.phase == "track" and estimate is not None and obs.sim_time - controller.last_contact <= 5:
            distance = np.linalg.norm(np.asarray(controller.world.grid_coords) - estimate, axis=1)
            scores = np.exp(-.5 * (distance / 5) ** 2)
            scores /= max(float(scores.sum()), 1e-12)
            belief = BeliefUpdatePayload(
                sim_time=obs.sim_time, drone_id=self.drone_id, q50=scores.tolist(),
                uncertainty=[1.] * len(scores), estimated_source=tuple(estimate),
                next_waypoint=waypoint, score_kind="sensor_peak",
                source_prediction_backend="wind_sensor", source_evidence=True,
                navigation_basis=controller.probe_kind, route=[obs.pos, *controller.route],
            )
        if controller is self.controller:
            self.goal = controller.goal.tolist() if controller.goal is not None else None
            self.wind = tuple(controller._wind(obs)[0][:2])
            self.search_phase = controller.phase
            self.decision_basis = controller.probe_kind if controller.phase == "track" else "downwind_transect"
            self.completed_probes = controller.completed_probes
            self.planned_route = [obs.pos, *controller.route]
            self.history = controller.history
            if belief is not None:
                self.source_scores = belief.q50
        return waypoint, belief
