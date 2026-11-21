"""Generate independent 3D episodes using main's unmodified physics and sensors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from agents.lawnmower import LawnmowerAgent
from agents.gradient import GradientAgent
from sim.engine import SimEngine
from shared.schemas import LeakSource
from integration.features import features,SOURCE,HAZARD

DATA=Path(__file__).resolve().parents[1]/'data/main3d.csv'

class Recorder(LawnmowerAgent):
    def __init__(self,policy):super().__init__('tabpfn');self.history=[];self.policy=policy
    def reset(self,world):self.history=[];self.policy.reset(world)
    def update(self,obs):self.history.append(obs);return self.policy.update(obs)

def generate(episodes=24,seconds=40):
    rows=[]
    for ep in range(episodes):
        policy=LawnmowerAgent('tabpfn') if ep%2==0 else GradientAgent('tabpfn')
        rec=Recorder(policy);engine=SimEngine({'tabpfn':rec},seed=2000+ep)
        rng=np.random.default_rng(3000+ep);grid=np.array(engine.world.grid_coords)
        source=grid[rng.integers(len(grid))]
        engine.reset_race([LeakSource(id='training',pos=tuple(source),emission_rate=float(rng.uniform(.5,2.)))])
        engine.airflow.set_wind(float(rng.uniform(0,360)),float(rng.uniform(.5,4)))
        pending=[]
        for tick in range(int((seconds+4)/engine.DT)):
            engine.step()
            t=engine.sim_time
            if tick%100==99 and t<=seconds+.001 and rec.history:
                distance=np.linalg.norm(grid-source,axis=1)
                near=np.flatnonzero(distance<=6);far=np.flatnonzero(distance>6)
                idx=np.r_[near,rng.choice(far,min(16,len(far)),replace=False)]
                q=grid[idx];f=features(list(rec.history),q)
                f[SOURCE]=(np.linalg.norm(q-source,axis=1)<=6).astype(int)
                f['episode_id']=ep;f['time_step']=t
                pending.append((t+4,q,f))
            while pending and t+1e-6>=pending[0][0]:
                _,q,f=pending.pop(0)
                f[HAZARD]=[int(engine.plume.sample_concentration(tuple(p))>=10) for p in q]
                rows.append(f)
        print(f'episode {ep+1}/{episodes}',flush=True)
    d=pd.concat(rows,ignore_index=True);d.to_csv(DATA,index=False);return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=24);p.add_argument('--seconds',type=int,default=40);a=p.parse_args()
    if a.episodes<8:raise ValueError('At least eight independent episodes required')
    print(generate(a.episodes,a.seconds).shape)
