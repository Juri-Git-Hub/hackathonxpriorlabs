"""Validated compact TabPFN inference plus observation-driven local 3D search."""
import json
import os
import threading
import time
from pathlib import Path
import numpy as np
from agents.base import AsyncInferenceAgent
from shared.schemas import BeliefUpdatePayload
from integration.features import rich_features,RICH_VERSION,SOURCE,HAZARD
from integration.reduced_features import fit_reduced,lean_features,COMPACT
from integration.search_control import SearchController,source_queries
from plumesense import ROOT


def load_models(backend='tabpfn-api',dataset=None,context_rows=4096):
    from integration.dataset_io import sample_context
    folder=Path(dataset or os.environ.get('PLUMESENSE_DATASET') or ROOT/'data/datasets/main3d-v3')
    if backend!='tabpfn-api':raise ValueError('Validated search profile requires the local 4090 API')
    manifest=json.loads((folder/'manifest.json').read_text());verified=json.loads((folder/'verification.json').read_text())
    if manifest['feature_version']!=RICH_VERSION or not verified['integrity_passed'] or not verified.get('quality_gate_passed',False) or verified.get('dataset_signature')!=manifest['signature']:
        raise ValueError('Dataset must pass current version, integrity and coverage gates')
    source=fit_reduced(sample_context(folder,SOURCE,context_rows),SOURCE,context='source')
    hazard=fit_reduced(sample_context(folder,HAZARD,context_rows),HAZARD,COMPACT,context='hazard')
    return [source,hazard]


class PlumeSenseAgent(AsyncInferenceAgent):
    SOURCE_INTERVAL=3.0
    HAZARD_INTERVAL=10.0
    def __init__(self,drone_id='tabpfn',models=None):
        super().__init__(drone_id)
        self._history_lock=threading.Lock();self.models=models
        self.error=None;self.hazard=None;self.source_scores=None;self.latency=0.;self.exposure=0.
    def _on_reset(self,world_init):
        controller=SearchController(world_init)
        with self._history_lock:
            self.world=world_init;self.controller=controller;self.planner=controller.planner
            self.history=[];self.grid=np.asarray(world_init.grid_coords)
            self.route=[];self.waypoint=None;self.error=None;self.hazard=None;self.source_scores=None;self.exposure=0.;self.latency=0.
            self.last_source=-100.;self.last_hazard=-100.;self.search_phase='search';self.completed_probes=0
    def update(self,obs):
        super().update(obs)
        # No model, path search or feature construction on the physics thread.
        with self._history_lock:
            while len(self.route)>1 and np.linalg.norm(np.array(self.route[0])-obs.pos)<1.2:self.route.pop(0)
            return self.route[0] if self.route else obs.pos
    def _record_observation(self,obs):
        with self._history_lock:
            if not self.history or obs.sim_time>self.history[-1].sim_time:
                self.history.append(obs);self.exposure+=obs.ppm_measured*.5
        return None
    def _publish_route(self,controller):
        with self._history_lock:
            if controller is not self.controller:return []
            position=self.history[-1].pos
        route=controller.planner.route(position,controller.goal)[1:] if controller.goal is not None else []
        with self._history_lock:
            if controller is self.controller:
                self.route=route.copy();self.waypoint=tuple(controller.goal) if controller.goal is not None else position
                self.search_phase=controller.phase;self.completed_probes=controller.completed_probes
        return route
    def _run_inference(self,obs):
        with self._history_lock:
            history=list(self.history);grid=self.grid.copy();controller=self.controller;world=self.world
        try:
            for reading in history[len(controller.history):]:controller.ingest(reading)
            route=self._publish_route(controller);current=history[-1]
            if current.sim_time-self.last_source<self.SOURCE_INTERVAL:
                return (route[0] if route else current.pos),None
            start=time.perf_counter()
            if self.models is None:self.models=load_models()
            queries=source_queries(world,history,grid)
            build=lean_features if getattr(self.models[0],'feature_builder',None)=='lean' else rich_features
            f=build(history,queries)
            scores=self.models[0].predict(f);index=int(np.argmax(scores));estimate=tuple(map(float,queries[index]))
            # Fine source hypotheses can be near equipment where a drone cannot fly.
            controller.set_source_hint(estimate,current.sim_time)
            hazard=self.hazard
            if hazard is None or current.sim_time-self.last_hazard>=self.HAZARD_INTERVAL:
                hb=lean_features if getattr(self.models[1],'feature_builder',None)=='lean' else rich_features
                hf=f.iloc[:len(grid)] if hb is build else hb(history,grid)
                hazard=self.models[1].predict(hf)
                hazard_time=current.sim_time
            else:hazard_time=self.last_hazard
            coarse=scores[:len(grid)];normalized=coarse/max(float(coarse.sum()),1e-9)
            route=self._publish_route(controller)
            belief=BeliefUpdatePayload(sim_time=current.sim_time,drone_id=self.drone_id,q50=normalized.tolist(),uncertainty=(1-np.abs(2*coarse-1)).tolist(),estimated_source=estimate,next_waypoint=tuple(controller.goal) if controller.goal is not None else current.pos,source_scores=normalized.tolist(),hazard_probabilities=hazard.tolist(),route=[current.pos,*route],inference_seconds=time.perf_counter()-start,measurement_count=len(history),exposure_ppm_seconds=self.exposure,score_kind='source_score',source_model_score=float(scores[index]))
            belief.hazard_observed_at=hazard_time
            belief.hazard_forecast_time=hazard_time+4
            with self._history_lock:
                if controller is self.controller:
                    self.last_source=current.sim_time;self.last_hazard=hazard_time;self.error=None
                    self.hazard=hazard;self.source_scores=normalized;self.latency=time.perf_counter()-start
            return (route[0] if route else current.pos),belief
        except Exception as exc:
            with self._history_lock:
                if controller is self.controller:self.error=str(exc)
            raise
