"""TabPFN source ranking. No transport simulator or hidden source access."""
import numpy as np
import pandas as pd
from plumesense import ROOT,Predictor,TARGET,astar
from source_learning import History,SOURCE_TARGET

class SourceSearch:
    def __init__(self,walls,wind,model=None):
        self.walls=walls.copy();self.wind=np.array(wind);self.cells=np.argwhere(~walls)[:,::-1]
        self.history=History();self.model=model;self.posterior=np.ones(len(self.cells))/len(self.cells)
        self.regions=np.linalg.norm(self.cells[:,None,:]-self.cells[None,:,:],axis=2)<=3
        self.plan_t=-100;self.mode='Erkundung';self.observed_t=-1;self.path=[];self.target=None;self.found=False;self.confirmations=0;self.estimate_history=[];self.latency=0.
    @property
    def estimate(self):return self.cells[self.posterior.argmax()]
    @property
    def confidence(self):return float(self.posterior[np.abs(self.cells-self.estimate).sum(axis=1)<=2].sum())
    def update(self,t,observations,drone):
        if t<=self.observed_t:return
        self.history.add(t,observations,drone)
        if self.model is None:raise RuntimeError('Source search requires a fitted TabPFN source model')
        import time
        start=time.perf_counter();scores=self.model.predict(self.history.features(self.cells,self.wind,self.walls));self.latency=time.perf_counter()-start
        # Normalized ranking scores, not a calibrated Bayesian posterior.
        scores=np.maximum(scores,1e-8);self.posterior=scores/scores.sum()
        self.observed_t=t;self.estimate_history.append(self.estimate.copy())
        stable=len(self.estimate_history)>=5 and all(np.linalg.norm(e-self.estimate)<=2 for e in self.estimate_history[-5:])
        close=np.linalg.norm(np.asarray(drone)-self.estimate)<=1
        positive=observations[1][0]>=.15
        self.confirmations=self.confirmations+1 if stable and close and positive else 0
        self.found=self.confirmations>=3
    def map(self):
        out=np.zeros(self.walls.shape);out[self.cells[:,1],self.cells[:,0]]=self.posterior;return out
    def plan(self,drone,risk,visits):
        point=tuple(drone)
        if self.found:return [point]
        if point in self.path:self.path=self.path[self.path.index(point):]
        else:self.path=[]
        # Hold a route for six simulation steps, not for an entire cross-map trip.
        if len(self.path)>1 and self.observed_t-self.plan_t<6:return self.path.copy()
        if np.linalg.norm(np.asarray(drone)-self.estimate)<=1 and self.confirmations>0:return [point]
        # Regional source mass prevents chasing an isolated noisy maximum.
        mass=self.regions@self.posterior
        mass/=max(float(mass.max()),1e-9)
        observed=np.asarray(self.history.rows)[:,1:3]
        distance_to_data=np.linalg.norm(self.cells[:,None,:]-observed[None,:,:],axis=2).min(axis=1)
        novelty=1-np.exp(-distance_to_data/2)
        # Interior measurements cover more unexplored neighborhood than border points.
        coverage=self.regions@novelty
        coverage/=max(float(coverage.max()),1e-9)
        signal=max(row[3] for row in self.history.rows)>.06
        self.mode='TabPFN-Quellen eingrenzen' if signal else 'Fläche erkunden · noch kein klares Gassignal'
        source_weight=.65 if signal else .15
        utility=source_weight*mass+(1-source_weight)*coverage
        # Compute actual reachable travel costs; walls must not distort distance ranking.
        import heapq
        distances={point:0.};parents={};queue=[(0.,point)]
        while queue:
            cost,p=heapq.heappop(queue)
            if cost>distances[p]:continue
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                q=(p[0]+dx,p[1]+dy)
                if not (0<=q[0]<self.walls.shape[1] and 0<=q[1]<self.walls.shape[0]) or self.walls[q[1],q[0]]:continue
                new=cost+1+.1*risk[q[1],q[0]]+.1*visits[q[1],q[0]]
                if new<distances.get(q,float('inf')):
                    distances[q]=new;parents[q]=p;heapq.heappush(queue,(new,q))
        scores=[]
        for i,cell in enumerate(self.cells):
            q=tuple(cell);d=distances.get(q,float('inf'))
            scores.append(utility[i]/(1+.08*d)/(1+.15*visits[q[1],q[0]]) if d>=2 and np.isfinite(d) else -1.)
        order=np.argsort(scores)[::-1]
        for index in order:
            if scores[index]<0:break
            cell=tuple(self.cells[index]);path=[cell];p=cell
            while p!=point:p=parents[p];path.append(p)
            self.path=path[::-1];self.target=tuple(map(int,cell));self.plan_t=self.observed_t
            return self.path.copy()
        return [point]

def fit_source_model(backend='tabpfn-api'):
    d=pd.read_csv(ROOT/'data/source_histories.csv');cut=int(d.episode_id.nunique()*.75)
    train=d[d.episode_id<cut];train=train.sample(n=min(8000,len(train)),random_state=42);columns=[c for c in d if c not in [SOURCE_TARGET,'episode_id','time_step']]
    if backend=='baseline':raise ValueError('Source localization requires TabPFN, choose tabpfn-api or tabpfn')
    m=Predictor(backend).fit(train.rename(columns={SOURCE_TARGET:TARGET}),columns);m.context='source';return m
