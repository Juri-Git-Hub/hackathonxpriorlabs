"""Held-out source search versus fixed-goal baseline; truth only for scoring."""
import argparse,json,time
import numpy as np
from plumesense import *
from source_search import SourceSearch,fit_source_model

p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=9);p.add_argument('--steps',type=int,default=120);p.add_argument('--backend',default='tabpfn-api',choices=['tabpfn','tabpfn-api']);a=p.parse_args()
d=pd.read_csv(ROOT/'data/synthetic_drone_only.csv');cut=int(d.episode_id.nunique()*.75)
model=Predictor(a.backend).fit(d[d.episode_id<cut])
source_model=fit_source_model(a.backend)
results=[]
for seed in range(100,100+a.episodes):
    sim=Simulation(seed);drone=np.array([0,0]);search=SourceSearch(sim.walls,sim.wind,source_model)
    visits=np.zeros(sim.walls.shape);exposure=0.;sim.step();prev=sim.observe(drone)
    start=time.perf_counter()
    for step in range(a.steps):
        now=sim.observe(drone);search.update(sim.t,now,drone)
        if search.found:break
        risk,_=risk_map(sim,drone,now,prev,model)
        route=search.plan(drone,risk,visits)
        if len(route)>1:drone=np.array(route[1])
        visits[drone[1],drone[0]]+=1;exposure+=float(sim.gas[drone[1],drone[0]])
        prev=now;sim.step()
    error=float(np.linalg.norm(search.estimate-sim.source))
    result=dict(seed=seed,declared_found=search.found,success=search.found and error<=2,source_error=error,steps=step+1,confidence=search.confidence,exposure=exposure,seconds=time.perf_counter()-start)
    results.append(result);print(result,flush=True)
summary=dict(backend=a.backend,episodes=a.episodes,max_steps=a.steps,success_rate=float(np.mean([r['success'] for r in results])),results=results)
Path(f'reports/tabpfn_source_search_{a.backend}.json').write_text(json.dumps(summary,indent=2))
