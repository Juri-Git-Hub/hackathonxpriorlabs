"""Simulation, observation-only features, forecasting and navigation."""
from pathlib import Path
import heapq
import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TARGET = 'target_hazard_t4'
ORIGINAL_FEATURES = ['query_x','query_y','drone_x','drone_y','wind_x','wind_y','wall_distance','sensor_now','sensor_prev','sensor_idw_now','sensor_idw_prev']
FEATURES = ORIGINAL_FEATURES + ['nearest_sensor_distance','local_measurement_count']
THRESHOLD = 0.15
HORIZON = 4

def load_map():
    m = pd.read_csv(ROOT/'data/original/plumesense_map.csv')
    walls = np.zeros((24,24), bool)
    walls[m.y,m.x] = m.is_wall.astype(bool)
    sensors = m.loc[m.has_fixed_sensor == 1, ['x','y']].to_numpy(int)
    return walls, sensors

class Simulation:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.walls, _ = load_map()
        self.free = np.argwhere(~self.walls)[:, ::-1]
        self.source = self.free[self.rng.integers(len(self.free))]
        self.wind = self.rng.uniform(-0.7,0.7,2)
        self.emission = self.rng.uniform(0.4,1.0)
        self.gas = np.zeros(self.walls.shape)
        self.t = 0
    def step(self):
        # Conservative face fluxes: no wraparound and no transport across walls.
        for _ in range(4):
            g = self.gas
            delta = np.zeros_like(g)
            for axis, velocity in [(1,self.wind[0]),(0,self.wind[1])]:
                a,b = [slice(None)]*2,[slice(None)]*2
                a[axis], b[axis] = slice(None,-1),slice(1,None)
                a,b = tuple(a),tuple(b)
                flux = (0.12*(g[a]-g[b]) + max(velocity,0)*g[a] + min(velocity,0)*g[b])*0.25
                flux *= ~(self.walls[a] | self.walls[b])
                delta[a] -= flux; delta[b] += flux
            self.gas = np.maximum(0,(g+delta)*0.997)
            x,y = self.source
            self.gas[y,x] += self.emission*0.25
        self.t += 1
        return self.gas.copy()
    def observe(self, drone):
        positions = np.asarray(drone,dtype=int).reshape(1,2)
        values = self.gas[positions[:,1],positions[:,0]]
        return positions, np.maximum(0,values+self.rng.normal(0,0.015,len(values)))

def features(walls, queries, drone, wind, now, prev):
    """Only map, wind and measured observations enter the predictor.

    sensor_now/prev are drone readings (not unobserved query concentrations).
    IDW fields use the single current/previous mobile observation.
    With one sensor each interpolated field is spatially constant; this is
    explicitly not a measured raster. SourceSearch retains the time history.
    """
    q = np.asarray(queries)
    def interpolate(obs):
        pos,values = obs
        d = np.linalg.norm(q[:,None,:]-pos[None,:,:],axis=2)
        w = 1/np.maximum(d,0.25)**2
        return (w@values)/w.sum(axis=1),d
    idw,d = interpolate(now); old,_ = interpolate(prev)
    wall_positions = np.argwhere(walls)[:,::-1]
    wall_d = np.linalg.norm(q[:,None,:]-wall_positions[None,:,:],axis=2).min(axis=1)
    return pd.DataFrame(dict(query_x=q[:,0],query_y=q[:,1],drone_x=drone[0],drone_y=drone[1],wind_x=wind[0],wind_y=wind[1],wall_distance=wall_d,sensor_now=now[1][-1],sensor_prev=prev[1][-1],sensor_idw_now=idw,sensor_idw_prev=old,nearest_sensor_distance=d.min(axis=1),local_measurement_count=(d<=3).sum(axis=1)))[FEATURES]

def generate(episodes=36, steps=28, queries=16):
    rows=[]
    for ep in range(episodes):
        sim=Simulation(ep); drone=sim.free[sim.rng.integers(len(sim.free))]
        history=[]; observations=[]; drones=[]
        for t in range(steps+HORIZON+1):
            history.append(sim.step()); drones.append(drone.copy())
            observations.append(sim.observe(drone))
            neighbors=[p for p in sim.free if np.abs(p-drone).sum()==1]
            if neighbors: drone=np.array(neighbors[sim.rng.integers(len(neighbors))])
        for t in range(1,steps):
            q=sim.free[sim.rng.choice(len(sim.free),queries,replace=False)]
            frame=features(sim.walls,q,drones[t],sim.wind,observations[t],observations[t-1])
            frame[TARGET]=(history[t+HORIZON][q[:,1],q[:,0]]>=THRESHOLD).astype(int)
            frame['episode_id']=ep; frame['time_step']=t
            frame['query_seen_sensor']=np.linalg.norm(q[:,None,:]-observations[t][0][None,:,:],axis=2).min(axis=1)==0
            rows.append(frame)
    return pd.concat(rows,ignore_index=True)

class Predictor:
    def __init__(self, backend='baseline', device='auto'):
        if backend not in ('baseline','tabpfn','tabpfn-api'):
            raise ValueError(f'Unknown predictor backend: {backend}')
        self.backend=backend; self.device=device
        self._fitted=False
    def fit(self, frame, columns=FEATURES):
        self.columns=list(columns)
        if self.backend=='tabpfn-api':
            self.train_x=frame[self.columns].to_numpy().tolist()
            self.train_y=frame[TARGET].astype(int).tolist()
            self.classes=sorted(frame[TARGET].unique().tolist())
            if self.classes != [0,1]:
                raise ValueError('API training requires both binary classes 0 and 1')
        if self.backend=='tabpfn':
            from tabpfn import TabPFNClassifier
            self.model=TabPFNClassifier(device=self.device,n_estimators=2,random_state=0)
            self.model.fit(frame[self.columns],frame[TARGET])
        self._fitted=True
        return self
    def predict(self, frame):
        if not getattr(self,'_fitted',False):
            raise RuntimeError('Predictor is not fitted; call fit() before predict().')
        if self.backend=='baseline':
            # Persistence of interpolated observations, fixed noise-scale sigmoid.
            return 1/(1+np.exp(-np.clip((frame.sensor_idw_now.to_numpy()-THRESHOLD)/0.035,-30,30)))
        if self.backend=='tabpfn-api':
            payload=dict(X_train=self.train_x,y_train=self.train_y,X_test=frame[self.columns].to_numpy().tolist())
            if hasattr(self,'context'):payload['context']=self.context
            from tunnel import ensure_tunnel
            endpoint=os.environ.get('PLUMESENSE_API_URL','http://127.0.0.1:18000/predict')
            ensure_tunnel(endpoint)
            request=Request(endpoint,data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'},method='POST')
            try:
                with urlopen(request,timeout=300) as response:
                    result=json.loads(response.read())
            except HTTPError as e:
                raise RuntimeError(f'TabPFN API HTTP {e.code}: {e.read().decode()[:1000]}') from e
            except URLError as e:
                raise RuntimeError(f'TabPFN API unreachable: {e.reason}. Start service / tunnel on port 18000.') from e
            return api_probabilities(result,len(frame))
        return self.model.predict_proba(frame[self.columns])[:,list(self.model.classes_).index(1)]

def metrics(y,p):
    from sklearn.metrics import f1_score,roc_auc_score,brier_score_loss
    return dict(n=len(y),positives=int(np.sum(y)),f1=float(f1_score(y,p>=.5,zero_division=0)),auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,brier=float(brier_score_loss(y,p)))

def astar(walls,risk,start,goal,weight=8):
    start,goal=tuple(start),tuple(goal)
    queue=[(0,start)]; cost={start:0}; parent={}
    while queue:
        _,p=heapq.heappop(queue)
        if p==goal:
            path=[p]
            while p in parent: p=parent[p];path.append(p)
            return path[::-1]
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            n=(p[0]+dx,p[1]+dy)
            if not (0<=n[0]<walls.shape[1] and 0<=n[1]<walls.shape[0]) or walls[n[1],n[0]]: continue
            c=cost[p]+1+weight*risk[n[1],n[0]]
            if c<cost.get(n,float('inf')):
                cost[n]=c;parent[n]=p
                heapq.heappush(queue,(c+abs(n[0]-goal[0])+abs(n[1]-goal[1]),n))
    return []

def risk_map(sim,drone,now,prev,predictor):
    f=features(sim.walls,sim.free,drone,sim.wind,now,prev)
    t=time.perf_counter();p=predictor.predict(f);latency=time.perf_counter()-t
    risk=np.zeros(sim.walls.shape);risk[sim.free[:,1],sim.free[:,0]]=p
    return risk,latency

def next_path(sim,drone,goal,risk,visits,active=False):
    if active:
        # Probability ambiguity is a heuristic, not epistemic uncertainty.
        candidates=sim.free
        score=4*risk*(1-risk)/(1+visits)-0.15*risk
        ranked=sorted(candidates,key=lambda p:score[p[1],p[0]]-0.005*np.abs(p-drone).sum(),reverse=True)
        for target in ranked[:30]:
            if tuple(target)==tuple(drone): continue
            path=astar(sim.walls,risk,drone,target)
            if path:return path
        return []
    return astar(sim.walls,risk,drone,goal)

class NavigationController:
    """Commit to a measurement target and route until arrival.

    Replan early only when the next cell newly exceeds the emergency threshold.
    This threshold is a demo heuristic, not a safety guarantee.
    """
    def __init__(self):
        self.path=[]
        self.target=None
        self.active=None
        self.previous_risk=None

    def plan(self,sim,drone,goal,risk,visits,active=False):
        point=tuple(drone)
        if self.active!=active:
            self.path=[];self.target=None;self.active=active
        if point in self.path:
            self.path=self.path[self.path.index(point):]
        else:
            self.path=[]
        emergency=False
        if len(self.path)>1 and self.previous_risk is not None:
            x,y=self.path[1]
            emergency=risk[y,x]>=.85 and self.previous_risk[y,x]<.85
        if emergency and self.target is not None:
            self.path=astar(sim.walls,risk,drone,self.target)
        if len(self.path)<=1:
            self.path=next_path(sim,drone,goal,risk,visits,active)
            self.target=self.path[-1] if self.path else None
        self.previous_risk=risk.copy()
        return self.path.copy()


def rollout(predictor,seed=100,steps=50,active=False):
    sim=Simulation(seed);drone=np.array([0,0]);goal=np.array([23,23]);visits=np.zeros(sim.walls.shape);controller=NavigationController()
    sim.step();prev=sim.observe(drone);exposure=0.;latencies=[];reached=False
    for t in range(steps):
        now=sim.observe(drone);risk,latency=risk_map(sim,drone,now,prev,predictor);latencies.append(latency)
        path=controller.plan(sim,drone,goal,risk,visits,active)
        if len(path)>1:drone=np.array(path[1])
        exposure+=float(sim.gas[drone[1],drone[0]])
        visits[drone[1],drone[0]]+=1;prev=now;sim.step()
        if np.array_equal(drone,goal):reached=True;break
    return dict(seed=seed,exposure=exposure,reached_goal=reached,steps=t+1,prediction_seconds_mean=float(np.mean(latencies)))


def api_probabilities(result, count):
    """Require probabilities explicitly; hard predictions cannot produce risk maps."""
    if not isinstance(result,dict):
        raise ValueError('API must return an object containing probabilities')
    values=next((result[k] for k in ['probabilities','predict_proba','proba','y_proba'] if k in result),None)
    if values is None:
        raise ValueError(f'API response has no probability field; keys: {list(result)}. Need predict_proba, not hard predictions.')
    p=np.asarray(values,dtype=float)
    if p.shape==(count,2):
        classes=result.get('classes',result.get('classes_',[0,1]))
        if sorted(classes)!=[0,1]:raise ValueError('Expected binary class labels 0 and 1')
        if not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Class probabilities must sum to one')
        p=p[:,classes.index(1)]
    elif p.shape!=(count,):
        raise ValueError(f'Invalid probability shape {p.shape}, expected ({count}, 2) or ({count},)')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid probabilities: expected finite values in [0,1]')
    return p
