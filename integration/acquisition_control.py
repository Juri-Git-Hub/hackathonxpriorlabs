"""Map-derived preflight planning and observation-driven search updates.

The map-search TabPFN dataset is a separate, versioned input. Until supplied,
use an explicitly labelled geometric coverage fallback, never fabricated scores.
"""
import numpy as np
from integration.prediction_control import PredictionController
from integration.map_features import MapPlanner3D,candidates,feature_table


class AcquisitionController(PredictionController):
    PLAN_TARGETS=8
    PLAN_SECONDS=90.

    def __init__(self,world,adaptive_wind=False):
        super().__init__(world)
        self.planner=MapPlanner3D(world)
        self.search_grid=candidates(world,self.planner)
        self.world=world;self.adaptive_wind=adaptive_wind;self.wind_replans=0
        self.preflight_complete=False;self.needs_plan=True
        self.plan_state='preparing';self.plan_revision=0;self.plan_scores=None
        self.plan_targets=[];self.tail_route=[];self.plan_seconds=0.
        self.search_targets_visited=0;self.first_contact_time=None
        self.plan_time=-100.;self.plan_wind=None
        self.decision_basis='preflight_planning'

    def plan(self,model,obs,preflight=False):
        history=[] if preflight else list(self.history)
        table=feature_table(self.world,self.search_grid,obs.pos,history,wind=obs.wind_measured,as_of=obs.sim_time)
        scores=None if model is None else np.asarray(model.predict(table),float)
        if scores is not None and (scores.shape!=(len(table),) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):
            raise ValueError('Map-search model must return one finite probability in [0,1] per candidate')
        current=np.array(obs.pos,float);visited=[np.array(o.pos) for o in history] or [current]
        targets=[];legs=[];seconds=0.;available=np.ones(len(table),bool)
        budget=min(self.PLAN_SECONDS,max(5.,obs.battery_pct*1.8-20))
        for _ in range(self.PLAN_TARGETS):
            distance=np.linalg.norm(self.search_grid[:,None,:]-np.asarray(visited)[None,:,:],axis=2).min(axis=1)
            novelty=1-np.exp(-distance/8)
            travel=np.array([self.planner.cost(current,p) for p in self.search_grid])
            interest=np.ones(len(table)) if scores is None else .05+.95*scores
            utility=interest*(.1+.9*novelty)/(1+.15*travel)
            utility[~available]=-1
            utility[distance<4]=-1
            selected=None
            for i in np.argsort(utility)[::-1]:
                if utility[i]<0:break
                route=self.planner.route(current,self.search_grid[i])
                if not route:continue
                duration=sum(self.planner.cost(a,b) for a,b in zip(route,route[1:]))
                if seconds+duration>budget:continue
                selected=int(i);break
            if selected is None:break
            goal=self.search_grid[selected].copy();available[selected]=False
            targets.append(tuple(map(float,goal)));legs.append(route[1:])
            visited.append(goal);current=goal;seconds+=duration
        if not targets:raise ValueError('No reachable unvisited search target in this map')
        # Publish a completed plan atomically from the inference worker.
        self.goal=np.asarray(targets[0]);self.route=list(legs[0]);self.goal_at=obs.sim_time;self.arrived=None
        first_leg=[obs.pos,*legs[0]]
        self.goal_timeout=max(15.,2*sum(self.planner.cost(a,b) for a,b in zip(first_leg,first_leg[1:]))+5)
        self.plan_targets=targets;self.tail_route=[p for leg in legs[1:] for p in leg]
        self.plan_seconds=float(seconds);self.plan_scores=scores
        self.plan_state='coverage_fallback' if scores is None else 'tabpfn_map' if getattr(model,'backend',None)=='tabpfn-api' else 'injected_test_model'
        self.decision_basis='map_coverage_fallback' if scores is None else 'tabpfn_map_search' if self.plan_state=='tabpfn_map' else 'test_map_search'
        self.needs_plan=False;self.preflight_complete=True;self.plan_revision+=1
        self.plan_time=obs.sim_time
        w=obs.wind_measured
        self.plan_wind=np.asarray(w,float) if all(v is not None and np.isfinite(v) for v in w) and np.linalg.norm(w)>=.5 else None

    def _survey(self,obs):
        # Parent source tracker calls this after sustained loss of gas.
        # Hold while the worker rebuilds the plan with current observations.
        self.needs_plan=True;self.goal=np.asarray(obs.pos);self.route=[];self.tail_route=[]
        self.decision_basis='map_replanning'

    def ingest(self,obs):
        if self.history and obs.sim_time<=self.history[-1].sim_time:return
        signal=float(np.median([o.ppm_measured for o in [*self.history[-2:],obs]]))
        if self.phase=='track' or (self.preflight_complete and signal>3):
            previous=self.phase;super().ingest(obs)
            if previous=='search' and self.phase=='track':
                self.first_contact_time=obs.sim_time;self.tail_route=[];self.plan_targets=[];self.needs_plan=False
            return
        self.history.append(obs)
        if not self.preflight_complete or self.needs_plan:return
        if self.adaptive_wind and self.plan_wind is not None and obs.sim_time-self.plan_time>=8:
            valid=[o.wind_measured for o in self.history[-3:] if all(v is not None and np.isfinite(v) for v in o.wind_measured)]
            if len(valid)==3:
                w=np.mean(valid,axis=0);norm=np.linalg.norm(w)
                if norm>=.5 and np.dot(w,self.plan_wind)/(norm*np.linalg.norm(self.plan_wind))<.5:
                    self.wind_replans+=1;self._survey(obs);return
        # Deadline is fixed at departure, not shortened as the drone advances.
        if np.linalg.norm(np.array(obs.pos)-self.goal)<3 or obs.sim_time-self.goal_at>self.goal_timeout:
            self.search_targets_visited+=1;self._survey(obs)

    def preview_route(self,position,current_route):
        return [tuple(position),*current_route,*(self.tail_route if self.phase=='search' else [])]
