"""Live PlumeSense agent: delegates to Dual-TabPFN `ReliableTabPFNAgent` by default,
while keeping injected-model support for unit tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import numpy as np

from agents.base import AsyncInferenceAgent
from agents.reliable_tabpfn import ReliableTabPFNAgent
from integration.acquisition_control import AcquisitionController
from integration.features import HAZARD, RICH_VERSION, SOURCE, rich_features
from integration.map_scoring import load_search_model
from integration.reduced_features import COMPACT, fit_reduced, lean_features
from integration.search_control import source_queries
from integration.wind_search import DEFAULT_ADAPTIVE_WIND
from plumesense import ROOT
from shared.schemas import BeliefUpdatePayload


def load_models(backend="tabpfn-api", dataset=None, context_rows=4096):
    from integration.dataset_io import sample_context

    folder = Path(dataset or os.environ.get("PLUMESENSE_DATASET") or ROOT / "data/datasets/main3d-v3")
    if backend != "tabpfn-api":
        raise ValueError("Validated search profile requires the local 4090 API")
    manifest = json.loads((folder / "manifest.json").read_text())
    verified = json.loads((folder / "verification.json").read_text())
    if (
        manifest["feature_version"] != RICH_VERSION
        or not verified["integrity_passed"]
        or not verified.get("quality_gate_passed", False)
        or verified.get("dataset_signature") != manifest["signature"]
    ):
        raise ValueError("Dataset must pass current version, integrity and coverage gates")
    source = fit_reduced(sample_context(folder, SOURCE, context_rows), SOURCE, context="source")
    hazard = fit_reduced(sample_context(folder, HAZARD, context_rows), HAZARD, COMPACT, context="hazard")
    return [source, hazard]


class PlumeSenseAgent(AsyncInferenceAgent):
    SOURCE_INTERVAL = 3.0
    HAZARD_INTERVAL = 10.0

    def __new__(cls, drone_id="tabpfn", models=None, adaptive_wind=DEFAULT_ADAPTIVE_WIND, search_model=None):
        if models is None and search_model is None:
            return ReliableTabPFNAgent(drone_id=drone_id, adaptive_wind=adaptive_wind)
        return super().__new__(cls)

    def __init__(self, drone_id="tabpfn", models=None, adaptive_wind=DEFAULT_ADAPTIVE_WIND, search_model=None):
        super().__init__(drone_id)
        self._history_lock = threading.Lock()
        self.models = models
        self.adaptive_wind = adaptive_wind
        self.search_model = search_model
        self.search_model_checked = search_model is not None
        self.error = None
        self.hazard = None
        self.source_scores = None
        self.latency = 0.0
        self.exposure = 0.0

    def _on_reset(self, world_init):
        controller = AcquisitionController(world_init, adaptive_wind=self.adaptive_wind)
        with self._history_lock:
            self.world = world_init
            self.controller = controller
            self.planner = controller.planner
            self.history = []
            self.grid = controller.search_grid.copy()
            self.prediction_grid = self.grid
            self.route = []
            self.waypoint = None
            self.error = None
            self.hazard = None
            self.source_scores = None
            self.exposure = 0.0
            self.latency = 0.0
            self.last_source = -100.0
            self.last_hazard = -100.0
            self.search_phase = "search"
            self.completed_probes = 0
            self.source_prediction_calls = 0
            self.hazard_prediction_calls = 0
            self.prediction_decisions = 0
            self.decision_basis = "coverage_until_gas_contact"
            self.wind_replans = 0
            self.plan_state = "preparing"
            self.planned_route = []
            self.plan_revision = 0
            self.search_candidate_count = len(controller.search_grid)
            if self.search_model is None:
                self.search_model_checked = False

    def update(self, obs):
        super().update(obs)
        # No model, path search or feature construction on the physics thread.
        with self._history_lock:
            while len(self.route) > 1 and np.linalg.norm(np.array(self.route[0]) - obs.pos) < 1.2:
                self.route.pop(0)
            return self.route[0] if self.route else obs.pos

    def _record_observation(self, obs):
        with self._history_lock:
            if not self.history or obs.sim_time > self.history[-1].sim_time:
                self.history.append(obs)
                self.exposure += obs.ppm_measured * 0.5
        return None

    def _publish_route(self, controller):
        with self._history_lock:
            if controller is not self.controller:
                return []
            position = self.history[-1].pos
        route = controller.planner.route(position, controller.goal)[1:] if controller.goal is not None else []
        with self._history_lock:
            if controller is self.controller:
                self.route = route.copy()
                self.waypoint = tuple(controller.goal) if controller.goal is not None else position
                self.search_phase = controller.phase
                self.completed_probes = controller.completed_probes
                self.decision_basis = controller.decision_basis
                self.prediction_decisions = controller.prediction_decisions
                self.wind_replans = getattr(controller, "wind_replans", 0)
                self.plan_state = controller.plan_state
                self.plan_revision = controller.plan_revision
                self.planned_route = controller.preview_route(position, route)
        return route

    def _run_inference(self, obs):
        with self._history_lock:
            history = list(self.history)
            grid = self.grid.copy()
            controller = self.controller
            world = self.world
        try:
            for reading in history[len(controller.history) :]:
                controller.ingest(reading)
            if controller.needs_plan:
                if not self.search_model_checked:
                    self.search_model = load_search_model(world)
                    self.search_model_checked = True
                controller.plan(self.search_model, history[-1], preflight=not controller.preflight_complete)
            with self._history_lock:
                if controller is not self.controller:
                    return obs.pos, None
            route = self._publish_route(controller)
            current = history[-1]
            if current.sim_time - self.last_source < self.SOURCE_INTERVAL:
                return (route[0] if route else current.pos), None
            start = time.perf_counter()
            if self.models is None:
                self.models = load_models()
            queries = source_queries(world, history, grid)
            build = lean_features if getattr(self.models[0], "feature_builder", None) == "lean" else rich_features
            f = build(history, queries)
            scores = self.models[0].predict(f)
            index = int(np.argmax(scores))
            estimate = tuple(map(float, queries[index]))
            # Fine source hypotheses can be near equipment where a drone cannot fly.
            hazard = self.hazard
            hazard_updated = hazard is None or current.sim_time - self.last_hazard >= self.HAZARD_INTERVAL
            if hazard_updated:
                hb = lean_features if getattr(self.models[1], "feature_builder", None) == "lean" else rich_features
                hf = f.iloc[: len(grid)] if hb is build else hb(history, grid)
                hazard = self.models[1].predict(hf)
                hazard_time = current.sim_time
            else:
                hazard_time = self.last_hazard
            # Inference can outlive multiple sensor ticks. Apply predictions to
            # current observations, never rewind the controller to an old pose.
            with self._history_lock:
                if controller is not self.controller:
                    return current.pos, None
                latest = list(self.history)
            for reading in latest[len(controller.history) :]:
                controller.ingest(reading)
            controller.set_source_prediction(queries, scores, current.sim_time)
            coarse = scores[: len(grid)]
            normalized = coarse / max(float(coarse.sum()), 1e-9)
            route = self._publish_route(controller)
            belief = BeliefUpdatePayload(
                sim_time=current.sim_time,
                drone_id=self.drone_id,
                q50=normalized.tolist(),
                uncertainty=(1 - np.abs(2 * coarse - 1)).tolist(),
                estimated_source=estimate,
                next_waypoint=tuple(controller.goal) if controller.goal is not None else current.pos,
                source_scores=normalized.tolist(),
                hazard_probabilities=hazard.tolist(),
                route=[current.pos, *route],
                inference_seconds=time.perf_counter() - start,
                measurement_count=len(history),
                exposure_ppm_seconds=self.exposure,
                score_kind="source_score",
                source_model_score=float(scores[index]),
            )
            belief.hazard_observed_at = hazard_time
            belief.hazard_forecast_time = hazard_time + 4
            belief.source_prediction_backend = getattr(self.models[0], "backend", "injected_test_model")
            belief.hazard_prediction_backend = getattr(self.models[1], "backend", "injected_test_model")
            belief.navigation_basis = controller.decision_basis
            belief.prediction_decisions = controller.prediction_decisions
            belief.source_evidence = any(o.ppm_measured > 3 for o in latest)
            belief.route = self.planned_route.copy()
            with self._history_lock:
                if controller is self.controller:
                    self.last_source = current.sim_time
                    self.last_hazard = hazard_time
                    self.error = None
                    self.hazard = hazard
                    self.source_scores = normalized
                    self.latency = time.perf_counter() - start
                    self.source_prediction_calls += 1
                    if hazard_updated:
                        self.hazard_prediction_calls += 1
            return (route[0] if route else current.pos), belief
        except Exception as exc:
            with self._history_lock:
                if controller is self.controller:
                    self.error = str(exc)
                    if not controller.preflight_complete:
                        self.plan_state = "error"
            raise
