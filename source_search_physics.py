"""Bayesian source search using known transport physics and noisy observations.

No reference to Simulation, its gas field, source or emission strength.
Each free cell and three possible emission strengths form source hypotheses.
"""
import numpy as np
from plumesense import astar

class SourceSearch:
    def __init__(self,walls,wind):
        self.walls=walls.copy();self.wind=np.array(wind)
        self.cells=np.argwhere(~walls)[:,::-1]
        self.fields=np.zeros((len(self.cells),*walls.shape))
        self.strengths=np.array([.4,.7,1.])
        self.log_weights=np.zeros((len(self.cells),3))
        self.t=0;self.observed_t=-1;self.path=[];self.target=None;self.confirmations=0;self.found=False

    def advance(self):
        for _ in range(4):
            g=self.fields;delta=np.zeros_like(g)
            for axis,v in [(2,self.wind[0]),(1,self.wind[1])]:
                a,b=[slice(None)]*3,[slice(None)]*3
                a[axis],b[axis]=slice(None,-1),slice(1,None)
                a,b=tuple(a),tuple(b)
                flux=(.12*(g[a]-g[b])+max(v,0)*g[a]+min(v,0)*g[b])*.25
                flux*=~(self.walls[a[1:]]|self.walls[b[1:]])
                delta[a]-=flux;delta[b]+=flux
            self.fields=np.maximum(0,(g+delta)*.997)
            self.fields[np.arange(len(self.cells)),self.cells[:,1],self.cells[:,0]]+=.25
        self.t+=1

    @property
    def weights(self):
        w=np.exp(self.log_weights-self.log_weights.max());return w/w.sum()

    @property
    def posterior(self):return self.weights.sum(axis=1)

    @property
    def estimate(self):return self.cells[self.posterior.argmax()]

    @property
    def confidence(self):
        return float(self.posterior[np.abs(self.cells-self.estimate).sum(axis=1)<=1].sum())

    def update(self,t,observations,drone):
        if t<=self.observed_t:return
        while self.t<t:self.advance()
        positions,values=observations
        if len(positions)!=1 or tuple(positions[0])!=tuple(drone):
            raise ValueError('Source search accepts only the sensor at the current drone position')
        # One mobile measurement per timestamp.
        _,indices=np.unique(positions,axis=0,return_index=True)
        positions,values=positions[indices],values[indices]
        expected=self.fields[:,positions[:,1],positions[:,0]][:,None,:]*self.strengths[None,:,None]
        sigma=.03+.05*expected
        residual=(values[None,None,:]-expected)/sigma
        self.log_weights+=(-.5*residual**2-np.log(sigma)).sum(axis=2)
        self.log_weights-=self.log_weights.max();self.observed_t=t
        close=np.abs(np.asarray(drone)-self.estimate).sum()<=1
        self.confirmations=self.confirmations+1 if self.confidence>=.9 and close else 0
        self.found=self.confirmations>=3

    def map(self):
        result=np.zeros(self.walls.shape);result[self.cells[:,1],self.cells[:,0]]=self.posterior
        return result

    def plan(self,drone,risk,visits):
        point=tuple(drone)
        if self.found:return [point]
        if point in self.path:self.path=self.path[self.path.index(point):]
        else:self.path=[]
        # Commit until arrival, except when strong source evidence changes the target.
        if self.confidence>=.8 and self.target is not None and np.abs(np.asarray(self.target)-self.estimate).sum()>2:
            self.path=[]
        if len(self.path)>1:return self.path.copy()
        if self.confidence>=.8:
            targets=[self.estimate]
        else:
            # Expected measurement disagreement across hypotheses (information heuristic).
            w=self.weights
            mean=np.einsum('ke,kij,e->ij',w,self.fields,self.strengths)
            second=np.einsum('ke,kij,e->ij',w,self.fields**2,self.strengths**2)
            variance=np.maximum(0,second-mean**2)
            info=np.sqrt(variance)
            info/=max(float(info.max()),1e-9)
            mass=self.map();mass/=max(float(mass.max()),1e-9)
            distance=np.abs(self.cells-np.asarray(drone)).sum(axis=1)
            score=(info+.7*mass+.1*risk)/(1+.15*visits)
            scores=score[self.cells[:,1],self.cells[:,0]]/(1+.04*distance)
            targets=self.cells[np.argsort(scores)[::-1]]
        for target in targets:
            if tuple(target)==point:continue
            # Gas is a search signal; risk has only a small travel-cost influence.
            path=astar(self.walls,risk,drone,target,weight=.25)
            if path:
                self.path=path;self.target=tuple(map(int,target));return path.copy()
        # At estimated source: stay and gather confirmation measurements.
        self.path=[point];self.target=point;return [point]
