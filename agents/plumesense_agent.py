"""Our remote source/hazard classifiers adapted to main's async 3D contract."""
import threading
import time
import numpy as np
import pandas as pd
from agents.base import AsyncInferenceAgent
from shared.schemas import BeliefUpdatePayload
from integration.features import features,SOURCE,HAZARD
from integration.generate import DATA
from plumesense import Predictor,TARGET
from integration.navigation import Planner3D


def load_models(backend='tabpfn-api'):
    d=pd.read_csv(DATA);cut=int(d.episode_id.nunique()*.75)
    train=d[d.episode_id<cut]
    columns=[c for c in d if c not in [SOURCE,HAZARD,'episode_id','time_step']]
    models=[]
    for target,context in [(SOURCE,'source'),(HAZARD,'hazard')]:
        if train[target].nunique()!=2:raise ValueError(f'3D training data requires both classes for {target}; generate more episodes')
        f=train[columns].copy();f[TARGET]=train[target]
        m=Predictor(backend).fit(f,columns);m.context=context;models.append(m)
    return models

class PlumeSenseAgent(AsyncInferenceAgent):
    def __init__(self,drone_id='tabpfn',models=None):
        super().__init__(drone_id,default_target=(-30.,-25.,5.))
        self._history_lock=threading.Lock();self.models=models
        self.error=None;self.hazard=None;self.source_scores=None;self.latency=0.;self.exposure=0.
    def _on_reset(self,world_init):
        with self._history_lock:
            self.planner=Planner3D(world_init);self.route=[]
            self.history=[];self.grid=np.asarray(world_init.grid_coords)
            self.waypoint=None;self.target_time=-100.;self.error=None;self.hazard=None;self.source_scores=None;self.exposure=0.
    def update(self,obs):
        super().update(obs)
        with self._history_lock:
            while len(self.route)>1 and np.linalg.norm(np.array(self.route[0])-obs.pos)<2:
                self.route.pop(0)
            return self.route[0] if self.route else obs.pos

    def _record_observation(self,obs):
        with self._history_lock:
            if not self.history or obs.sim_time>self.history[-1].sim_time:
                self.history.append(obs);self.exposure+=obs.ppm_measured*.5
        return None
    def _run_inference(self,obs):
        with self._history_lock:
            history=list(self.history);grid=self.grid.copy();waypoint=self.waypoint;target_time=self.target_time
        try:
            start=time.perf_counter()
            if self.models is None:self.models=load_models()
            f=features(history,grid)
            source=self.models[0].predict(f);hazard=self.models[1].predict(f)
            positions=np.array([o.pos for o in history])
            nearest=np.linalg.norm(grid[:,None,:]-positions[None,:,:],axis=2).min(axis=1)
            novelty=1-np.exp(-nearest/10)
            regions=np.linalg.norm(grid[:,None,:]-grid[None,:,:],axis=2)<=15
            coverage=regions@novelty;coverage/=max(float(coverage.max()),1e-9)
            mass=regions@source;mass/=max(float(mass.max()),1e-9)
            signal=max(o.ppm_measured for o in history)>5
            alpha=.7 if signal else .15
            distance=np.linalg.norm(grid-np.array(obs.pos),axis=1)
            utility=(alpha*mass+(1-alpha)*coverage)/(1+.015*distance)
            utility[nearest<4]=-1
            if waypoint is None or np.linalg.norm(np.array(waypoint)-obs.pos)<5 or obs.sim_time-target_time>=6:
                waypoint=None
                for index in np.argsort(utility)[::-1][:30]:
                    candidate=tuple(map(float,grid[index]))
                    if self.planner.route(obs.pos,candidate):waypoint=candidate;break
                target_time=obs.sim_time
            if waypoint is None:raise RuntimeError('No reachable 3D measurement target')
            with self._history_lock:current_pos=self.history[-1].pos
            route=self.planner.route(current_pos,waypoint)[1:]
            estimate=tuple(map(float,grid[int(np.argmax(source))]))
            normalized=source/max(float(source.sum()),1e-9)
            belief=BeliefUpdatePayload(sim_time=obs.sim_time,drone_id=self.drone_id,q50=normalized.tolist(),uncertainty=(1-np.abs(2*source-1)).tolist(),estimated_source=estimate,next_waypoint=waypoint,source_scores=normalized.tolist(),hazard_probabilities=hazard.tolist(),route=[current_pos,*route],inference_seconds=time.perf_counter()-start,measurement_count=len(history),exposure_ppm_seconds=self.exposure,score_kind='source_score')
            # Publish diagnostics only for the same race generation.
            with self._history_lock:
                if self.history and history[0] is self.history[0]:
                    self.route=route;self.waypoint=waypoint;self.target_time=target_time;self.error=None
                    self.hazard=hazard;self.source_scores=normalized;self.latency=time.perf_counter()-start
            return (route[0] if route else obs.pos),belief
        except Exception as exc:
            with self._history_lock:
                if self.history and history[0] is self.history[0]:self.error=str(exc)
            raise
