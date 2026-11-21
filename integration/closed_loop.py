"""Headless closed-loop harness. Only the evaluator accesses the hidden leak."""
import argparse,json,time
from pathlib import Path
import numpy as np
from agents.base import BaseAgent
from shared.schemas import BeliefUpdatePayload,LeakSource
from sim.engine import SimEngine
from integration.features import rich_features
from integration.prediction_protocol import PROFILE, TEST_CASES
from integration.search_control import SearchController,source_queries
from integration.prediction_control import PredictionController
from integration.wind_search import WindReactiveController,DEFAULT_ADAPTIVE_WIND
from integration.dataset import write_json
from plumesense import ROOT

class SynchronousSearchAgent(BaseAgent):
    """Deterministic simulation harness; optional measured inference-latency delay."""
    def __init__(self,model=None,model_every=3,latency=False,fine=True,policy='prediction'):
        super().__init__('tabpfn');self.model=model;self.model_every=model_every;self.latency=latency;self.fine=fine;self.policy=policy
    def reset(self,world):
        self.world=world;self.controller=(SearchController(world) if self.policy=='legacy' else WindReactiveController(world) if self.policy=='wind_search' else PredictionController(world,wind_aware=self.policy=='wind'));self.belief=None;self.last_model=-100;self.pending=None;self.calls=0;self.latencies=[]
    def _apply_prediction(self,belief,points,scores):
        if isinstance(self.controller,PredictionController):self.controller.set_source_prediction(points,scores,belief.sim_time)
        elif self.model is not None:self.controller.set_source_hint(belief.estimated_source,belief.sim_time)
    def update(self,obs):
        self.controller.ingest(obs)
        if self.pending and obs.sim_time>=self.pending[0]:
            self.belief=self.pending[1];self._apply_prediction(self.belief,self.pending[2],self.pending[3]);self.pending=None
        if obs.sim_time-self.last_model>=self.model_every and self.pending is None:
            history=self.controller.history;grid=np.array(self.world.grid_coords)
            q=source_queries(self.world,history,grid) if self.fine else grid
            if self.model is None:
                peak=max(history,key=lambda o:o.ppm_measured);estimate=peak.pos;p=np.exp(-np.linalg.norm(q-np.array(estimate),axis=1)/5);scores=p[:len(grid)];score=None;latency=0.
            else:
                t=time.perf_counter();p=self.model.predict(rich_features(history,q));latency=time.perf_counter()-t
                estimate=tuple(map(float,q[int(np.argmax(p))]));score=float(p.max());scores=p[:len(grid)]
            normal=scores/max(float(scores.sum()),1e-9);goal=self.controller.goal
            b=BeliefUpdatePayload(sim_time=obs.sim_time,drone_id='tabpfn',q50=normal.tolist(),uncertainty=[0.]*len(grid),estimated_source=estimate,next_waypoint=tuple(goal) if goal is not None else obs.pos,source_scores=normal.tolist(),source_model_score=score,score_kind='source_score')
            self.calls+=1;self.latencies.append(latency);self.last_model=obs.sim_time
            if self.latency and latency>0:self.pending=(obs.sim_time+latency,b,q,p)
            else:self.belief=b;self._apply_prediction(b,q,p)
        return self.controller.waypoint(obs)
    def pop_belief_update(self):b=self.belief;self.belief=None;return b

def run(seed=42,seconds=180,model=None,source=None,wind=None,latency=False,fine=True,policy='prediction',wind_schedule=None):
    a=SynchronousSearchAgent(model,latency=latency,fine=fine,policy=policy);e=SimEngine({'tabpfn':a},seed=seed)
    if source is not None:e.reset_race([LeakSource(id='test',pos=tuple(source))])
    if wind is not None:e.airflow.set_wind(*wind)
    start=time.perf_counter();trace=[];exposure=0.;closest=1e9;max_ppm=0.
    changes=iter(sorted(wind_schedule or [],key=lambda change:change['time']));change=next(changes,None)
    for tick in range(int(seconds/e.DT)):
        if change is not None and e.sim_time+1e-9>=change['time']:
            e.airflow.set_wind(change['angle'],change['speed']);change=next(changes,None)
        t=e.step();d=t.drones[0];exposure+=d.ppm*e.DT;closest=min(closest,d.dist_to_nearest_leak);max_ppm=max(max_ppm,d.ppm)
        if tick%10==9:trace.append(dict(time=t.sim_time,pos=d.pos,ppm=d.ppm,phase=a.controller.phase,goal=a.controller.goal.tolist() if a.controller.goal is not None else None))
        if e.mission_report is not None or d.status=='depleted':break
    result=dict(policy=policy,seed=seed,wind=wind,wind_schedule=wind_schedule or [],source=e.leaks[0].pos,success=e.mission_report is not None,sim_seconds=e.sim_time,wall_seconds=time.perf_counter()-start,closest_distance_m=closest,max_ppm=max_ppm,exposure=exposure,report=e.mission_report.model_dump() if e.mission_report else None,model_calls=a.calls,mean_latency=float(np.mean(a.latencies)),latency_applied=latency,fine_source_queries=fine,probes=a.controller.completed_probes,prediction_decisions=getattr(a.controller,'prediction_decisions',None),wind_replans=getattr(a.controller,'wind_replans',0),transitions=a.controller.transitions,trace=trace)
    return result


def run_realtime(seed=42,seconds=180,models=None,source=None,adaptive_wind=DEFAULT_ADAPTIVE_WIND):
    """Exercise the actual async agent and 20Hz wall-clock physics, no time warp."""
    from integration.runtime import Runtime
    runtime=Runtime(seed=seed,models=models,adaptive_wind=adaptive_wind)
    trace=[];start=time.monotonic();last_sample=-1
    try:
        with runtime.engine._lock:
            if source is not None:runtime.engine.reset_race([LeakSource(id='test',pos=tuple(source))])
            runtime.engine.paused=False
        while time.monotonic()-start<seconds+60:
            if runtime.error or runtime.agent.error:
                raise RuntimeError(runtime.error or runtime.agent.error)
            tick,beliefs=runtime.snapshot();drone=tick.drones[0]
            if tick.sim_time-last_sample>=1:
                belief=beliefs.get('tabpfn')
                trace.append(dict(time=tick.sim_time,pos=drone.pos,ppm=drone.ppm,
                                  phase=runtime.agent.search_phase,
                                  estimated_source=belief.estimated_source if belief else None))
                last_sample=tick.sim_time
            if tick.mission_report or tick.sim_time>=seconds or drone.status=='depleted':break
            time.sleep(.1)
        else:raise TimeoutError('Wall-clock watchdog expired before the requested simulation duration')
        from integration.wind_protocol import PROFILE as WIND_PROFILE
        return dict(profile=WIND_PROFILE if adaptive_wind else PROFILE,adaptive_wind=adaptive_wind,seed=seed,real_time_async=True,success=tick.mission_report is not None,
                    wall_seconds=time.monotonic()-start,sim_time=tick.sim_time,
                    drone=drone.model_dump(),mission_report=tick.mission_report.model_dump() if tick.mission_report else None,
                    latency=runtime.agent.latency,source_inputs=len(runtime.agent.models[0].columns),
                    hazard_inputs=len(runtime.agent.models[1].columns),trace=trace)
    finally:runtime.close()


def run_suite(model,seconds,output,fine=True,policy='prediction'):
    """Keep every predeclared test, including failures; no model tuning here."""
    from integration.wind_protocol import PROFILE as WIND_PROFILE,TEST_CASES as WIND_CASES
    result=dict(profile=WIND_PROFILE if policy=='wind_search' else PROFILE,note='measured RPC delay applied in simulated time; not wall-clock async',tests=[])
    for case in (WIND_CASES if policy=='wind_search' else TEST_CASES):
        trial=run(case['seed'],seconds,model,source=case.get('source'),wind=case.get('wind'),wind_schedule=case.get('wind_schedule'),latency=True,fine=fine,policy=policy)
        trial['name']=case['name'];result['tests'].append(trial)
        result['successes']=sum(t['success'] for t in result['tests']);result['total']=len(result['tests'])
        print(json.dumps({k:v for k,v in trial.items() if k!='trace'}),flush=True)
        write_json(output,result)
    return result

if __name__=='__main__':
    from integration.reduced_features import fit_reduced
    from integration.dataset_io import sample_context
    from integration.dataset import DEFAULT
    from integration.features import SOURCE
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--seconds',type=int,default=180);p.add_argument('--source',type=float,nargs=3);p.add_argument('--dataset',type=Path,default=DEFAULT);p.add_argument('--sensor-only',action='store_true');p.add_argument('--coarse-only',action='store_true');p.add_argument('--suite',action='store_true');p.add_argument('--realtime',action='store_true');p.add_argument('--legacy-policy',action='store_true');p.add_argument('--adaptive-wind',action=argparse.BooleanOptionalAction,default=None);p.add_argument('--output',type=Path,default=ROOT/'reports/search_flight.json');args=p.parse_args()
    adaptive_wind=DEFAULT_ADAPTIVE_WIND if args.adaptive_wind is None else args.adaptive_wind
    if args.realtime:
        if args.sensor_only or args.coarse_only or args.suite or args.legacy_policy:p.error('--realtime uses the unchanged actual live agent; combine neither baseline nor suite flags')
        from agents.plumesense_agent import load_models
        r=run_realtime(args.seed,args.seconds,load_models(dataset=args.dataset),args.source,adaptive_wind=adaptive_wind)
    else:
        model=None if args.sensor_only else fit_reduced(sample_context(args.dataset,SOURCE,4096),SOURCE)
        policy='legacy' if args.legacy_policy else 'wind_search' if adaptive_wind else 'prediction'
        r=run_suite(model,args.seconds,args.output,fine=not args.coarse_only,policy=policy) if args.suite else run(args.seed,args.seconds,model,source=args.source,latency=True,fine=not args.coarse_only,policy=policy)
    write_json(args.output,r)
    if not args.suite:print(json.dumps({k:v for k,v in r.items() if k!='trace'},indent=2))
