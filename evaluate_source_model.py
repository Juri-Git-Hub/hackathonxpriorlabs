"""Full free-cell source ranking on held-out simulated trajectories."""
import json
import argparse
import numpy as np
from plumesense import Simulation,astar,ROOT
from source_search import fit_source_model
from source_learning import History

parser=argparse.ArgumentParser();parser.add_argument('--start-episode',type=int,default=80);args=parser.parse_args()
model=fit_source_model();results=[]
for ep in range(args.start_episode,args.start_episode+5):
    sim=Simulation(1000+ep);rng=np.random.default_rng(5000+ep);drone=np.array([0,0]);history=History();route=[]
    for t in range(90):
        sim.step();history.add(sim.t,sim.observe(drone),drone)
        if t in [29,59,89]:
            scores=model.predict(history.features(sim.free,sim.wind,sim.walls));guess=sim.free[scores.argmax()]
            error=float(np.linalg.norm(guess-sim.source))
            results.append(dict(episode=ep,time=sim.t,error=error,within_two_cells=error<=2,signal_seen=max(r[3] for r in history.rows)>.06))
            print(results[-1],flush=True)
        if len(route)<=1:
            goal=sim.free[rng.integers(len(sim.free))];route=astar(sim.walls,np.zeros(sim.walls.shape),drone,goal,weight=0)
        if len(route)>1:drone=np.array(route[1]);route=route[1:]
result=dict(backend='tabpfn-api',training_episodes=list(range(75)),test_episodes=list(range(args.start_episode,args.start_episode+5)),mean_error=float(np.mean([r['error'] for r in results])),within_two_cells_rate=float(np.mean([r['within_two_cells'] for r in results])),results=results)
(ROOT/'reports/tabpfn_source_full_grid.json').write_text(json.dumps(result,indent=2))
(ROOT/f'reports/tabpfn_source_grid_v2_{args.start_episode}_{args.start_episode+4}.json').write_text(json.dumps(result,indent=2))
print(result['mean_error'],result['within_two_cells_rate'])
