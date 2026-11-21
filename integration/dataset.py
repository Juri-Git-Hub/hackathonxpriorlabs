"""Resumable, process-parallel, provenance-tracked 3D dataset generator."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,dataclass
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from agents.base import BaseAgent
from agents.gradient import GradientAgent
from agents.lawnmower import LawnmowerAgent
from integration.navigation import Planner3D
from integration.features import rich_features,RICH_VERSION,SOURCE,HAZARD
from shared.schemas import LeakSource
from sim.engine import SimEngine

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/datasets/main3d-v3'

@dataclass(frozen=True)
class Config:
    episodes:int=512
    seconds:int=120
    snapshot_seconds:int=5
    candidates:int=64
    seed:int=73000
    eval_episodes:int=16
    dt:float=.05

def split_for(ep,episodes):
    return 'train' if ep<int(episodes*.6) else 'validation' if ep<int(episodes*.8) else 'test'

def fingerprint():
    paths=[*sorted((ROOT/'sim').glob('*.py')),ROOT/'integration/features.py',Path(__file__),ROOT/'integration/navigation.py',ROOT/'agents/gradient.py',ROOT/'agents/lawnmower.py',ROOT/'shared/schemas.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def parquet(path,frame):
    temp=path.with_suffix('.tmp');frame.to_parquet(temp,index=False);temp.replace(path)

class Collector(BaseAgent):
    """Mobile, source-blind policy. Curriculum affects start placement only."""
    def __init__(self,policy,seed):super().__init__('tabpfn');self.kind=policy;self.rng=np.random.default_rng(seed)
    def reset(self,world):
        self.history=[];self.grid=np.array(world.grid_coords);self.planner=Planner3D(world)
        self.gradient=GradientAgent('tabpfn');self.gradient.reset(world)
        self.raster=LawnmowerAgent('tabpfn');self.raster.reset(world)
        self.route=[];self.goal=None;self.goal_time=-100
    def update(self,obs):
        self.history.append(obs)
        while len(self.route)>1 and np.linalg.norm(np.array(self.route[0])-obs.pos)<2:self.route.pop(0)
        if not self.route or np.linalg.norm(np.array(self.goal)-obs.pos)<3 or obs.sim_time-self.goal_time>=8:
            if self.kind=='gradient' or (self.kind=='mixed' and obs.ppm_measured>5):
                goals=[self.gradient.update(obs)]
            elif self.kind=='raster':goals=[self.raster.update(obs)]
            else:
                visited=np.array([h.pos for h in self.history]);near=np.linalg.norm(self.grid[:,None,:]-visited[None,:,:],axis=2).min(axis=1)
                travel=np.linalg.norm(self.grid-np.array(obs.pos),axis=1)
                scores=near/(1+.04*travel)+self.rng.uniform(0,2,len(near))
                goals=self.grid[np.argsort(scores)[::-1][:20]]
            self.route=[]
            for goal in goals:
                goal=np.asarray(goal,float)
                if not self.planner.free(goal):goal=self.planner.points[np.linalg.norm(self.planner.points-goal,axis=1).argmin()]
                route=self.planner.route(obs.pos,goal)
                if route:self.route=route[1:];self.goal=tuple(goal);break
            self.goal_time=obs.sim_time
        return self.route[0] if self.route else obs.pos

def episode(ep,config,output,signature):
    c=Config(**config);out=Path(output);folder=out/'episodes'/f'{ep:05d}';folder.mkdir(parents=True,exist_ok=True)
    done=folder/'meta.json'
    if done.exists():
        meta=json.loads(done.read_text())
        if meta['signature']==signature and all((folder/name).exists() for name in meta['files']):return meta
        raise ValueError(f'Mismatched existing episode {ep}; use a new output directory')
    rng=np.random.default_rng(c.seed+ep);split=split_for(ep,c.episodes)
    curriculum=split=='train' and ep%5<3
    kind=['mixed','coverage','gradient','raster'][ep%4]
    collector=Collector(kind,c.seed+ep+900000);engine=SimEngine({'tabpfn':collector},seed=c.seed+ep)
    grid=np.array(engine.world.grid_coords)
    candidates=grid.copy()
    # Mix off-grid and asset-adjacent leaks while keeping source positions navigable.
    source=grid[rng.integers(len(grid))]+rng.uniform(-2,2,3)
    source[2]=np.clip(source[2],2,27)
    if ep%3==0:
        asset=engine.world.assets[rng.integers(len(engine.world.assets))]
        source=np.array(asset.pos)+rng.uniform(-2,2,3);source[2]=np.clip(source[2],2,27)
    if not collector.planner.free(source):source=grid[np.linalg.norm(grid-source,axis=1).argmin()]
    strength=float(np.exp(rng.uniform(np.log(.5),np.log(4))))
    angle=float(rng.uniform(0,360));speed=float(rng.uniform(.5,4.5))
    engine.reset_race([LeakSource(id='source',pos=tuple(source),emission_rate=strength,buoyancy_vz=float(rng.uniform(.05,.4)))])
    engine.airflow.set_wind(angle,speed);engine.plume.reset(seed=c.seed+ep+100)
    for _ in range(160):engine.airflow.step(c.dt);engine.plume.step(c.dt,engine.leaks)
    if curriculum:
        distance=np.linalg.norm(grid-source,axis=1)
        ids=np.flatnonzero((distance>=8)&(distance<=30))
        concentration=np.array([engine.plume.sample_concentration(tuple(grid[i])) for i in ids])
        eligible=ids[concentration>=5]
        start=grid[rng.choice(eligible)] if len(eligible) else grid[rng.choice(ids)] if len(ids) else grid[rng.integers(len(grid))]
    else:
        start=np.array([-48.,-45.,5.]) if ep%2==0 else grid[rng.integers(len(grid))]
    engine.drones['tabpfn'].reset(tuple(start))
    sensor=engine.sensors['tabpfn'];sensor.noise_std_frac=float(rng.uniform(.04,.16));sensor.tau_rise=float(rng.uniform(.3,1.2));sensor.tau_decay=float(rng.uniform(1,3))
    engine.sensor_dropout_active=ep%4==0
    frames=[];full=[];pending=[];feature_columns=None
    snapshots={round(float(t),5) for t in np.arange(c.snapshot_seconds,c.seconds+.001,c.snapshot_seconds)}
    ordered=sorted(snapshots);eval_times={ordered[0],ordered[len(ordered)//2],ordered[-1]}
    # Natural test/validation starts never depend on source or concentration.
    eval_full=split!='train' and ep<int(c.episodes*.6)+2*c.eval_episodes and ep%2==0
    if split=='test':eval_full=ep<int(c.episodes*.8)+2*c.eval_episodes and ep%2==0
    for tick in range(int(round((c.seconds+4)/c.dt))):
        engine.step();t=round(engine.sim_time,5)
        if ep%3==0 and t in {30.,60.,90.}:
            engine.airflow.set_wind((angle+45*int(t/30))%360,speed)
        if t in snapshots and collector.history:
            distance=np.linalg.norm(candidates-source,axis=1);positive=np.flatnonzero(distance<=6)
            hard=np.flatnonzero((distance>6)&(distance<=20));far=np.flatnonzero(distance>20)
            a=rng.choice(hard,min(len(hard),c.candidates//3),replace=False)
            b=rng.choice(far,min(len(far),c.candidates-len(a)),replace=False)
            sampled=np.unique(np.r_[positive,a,b]);full_at=eval_full and t in eval_times
            ids=np.arange(len(grid)) if full_at else sampled
            f=rich_features(list(collector.history),grid[ids]);feature_columns=list(f)
            f[SOURCE]=(np.linalg.norm(grid[ids]-source,axis=1)<=6).astype('int8')
            f['episode_id']=ep;f['snapshot_time']=t;f['label_time']=t+4;f['split']=split
            f['signal_seen']=max(o.ppm_measured for o in collector.history)>5
            f['phase']='cold' if not f['signal_seen'].iloc[0] else 'contact' if collector.history[-1].ppm_measured>5 else 'lost'
            f['snapshot_id']=f'{ep}:{t:g}'
            pending.append((t+4,ids,f,sampled,full_at))
        while pending and t+1e-5>=pending[0][0]:
            _,ids,f,sampled,full_at=pending.pop(0)
            truth=np.array([engine.plume.sample_concentration(tuple(p)) for p in grid[ids]])
            f['target_concentration_t4']=truth;f[HAZARD]=(truth>=10).astype('int8')
            if full_at:full.append(f.copy())
            frames.append(f.loc[np.isin(ids,sampled)].copy())
    if pending:raise RuntimeError('Future labels missing at episode end')
    if not frames:raise RuntimeError('No snapshots')
    data=pd.concat(frames,ignore_index=True);parquet(folder/'samples.parquet',data)
    files=['samples.parquet','observations.jsonl.gz']
    if full:parquet(folder/'full_grid.parquet',pd.concat(full,ignore_index=True));files.append('full_grid.parquet')
    with gzip.open(folder/'observations.jsonl.gz','wt') as file:
        for o in collector.history:file.write(o.model_dump_json()+'\n')
    history=[o for o in collector.history if o.sim_time<=c.seconds+1e-5]
    meta=dict(episode_id=ep,split=split,seed=c.seed+ep,signature=signature,feature_version=RICH_VERSION,feature_columns=feature_columns,files=files,rows=len(data),snapshots=int(data.snapshot_time.nunique()),curriculum=curriculum,policy=kind,source=source.tolist(),start=np.asarray(start).tolist(),emission_strength=strength,wind_angle=angle,wind_speed=speed,dropout=engine.sensor_dropout_active,noise=sensor.noise_std_frac,tau_rise=sensor.tau_rise,tau_decay=sensor.tau_decay,gas_contact=any(o.ppm_measured>5 for o in history),max_ppm=max(o.ppm_measured for o in history),final_battery=engine.drones['tabpfn'].battery_pct,grid=grid.tolist(),sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files})
    write_json(done,meta);return meta

def generate(config,output=DEFAULT,workers=6):
    if config.episodes<10 or config.seconds<10 or config.seconds%config.snapshot_seconds:raise ValueError('Need >=10 episodes, >=10 seconds and a snapshot-aligned duration')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()+fingerprint().encode()).hexdigest()
    run=out/'config.json';value=dict(config=asdict(config),signature=signature,feature_version=RICH_VERSION)
    if run.exists() and json.loads(run.read_text())!=value:raise ValueError('Output already belongs to a different configuration or code fingerprint')
    write_json(run,value);start=time.monotonic();metas=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(episode,ep,asdict(config),str(out),signature):ep for ep in range(config.episodes)}
        for future in as_completed(futures):
            meta=future.result();metas.append(meta)
            print(json.dumps(dict(completed=len(metas),total=config.episodes,episode=meta['episode_id'],contact=meta['gas_contact'],rows=meta['rows'],elapsed=round(time.monotonic()-start,1))),flush=True)
    metas.sort(key=lambda m:m['episode_id'])
    manifest=dict(**value,episodes=[{k:m[k] for k in ['episode_id','split','gas_contact','rows','snapshots','curriculum','policy']} for m in metas],feature_columns=metas[0]['feature_columns'],rows=sum(m['rows'] for m in metas),generation_seconds=time.monotonic()-start)
    write_json(out/'manifest.json',manifest);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=512);p.add_argument('--seconds',type=int,default=120);p.add_argument('--candidates',type=int,default=64);p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=73000);p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args()
    generate(Config(episodes=a.episodes,seconds=a.seconds,candidates=a.candidates,seed=a.seed),a.output,a.workers)
