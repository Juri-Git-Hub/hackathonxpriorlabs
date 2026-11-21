"""Tabular summaries of mobile measurement histories for TabPFN source ranking."""
import numpy as np
import pandas as pd
from plumesense import Simulation,ROOT
SOURCE_TARGET='target_source_near'

class History:
    def __init__(self):self.rows=[]
    def add(self,t,obs,drone):
        positions,values=obs
        if len(values)!=1 or tuple(positions[0])!=tuple(drone):raise ValueError('Only mobile measurements accepted')
        if self.rows and t<=self.rows[-1][0]:return
        self.rows.append([t,*map(float,drone),float(values[0])])
    def features(self,cells,wind,walls):
        h=np.asarray(self.rows);q=np.asarray(cells);n=len(q)
        f={'query_x':q[:,0],'query_y':q[:,1],'wind_x':np.full(n,wind[0]),'wind_y':np.full(n,wind[1]),'time':np.full(n,h[-1,0]),'count':np.full(n,len(h))}
        # Every observation contributes to spatial bins and global statistics.
        weight=np.maximum(h[:,3]-.015,0);total=max(weight.sum(),1e-9)
        for key,val in [('mean',h[:,3].mean()),('max',h[:,3].max()),('positive_count',(h[:,3]>.05).sum()),('center_x',weight@h[:,1]/total),('center_y',weight@h[:,2]/total)]:f[key]=np.full(n,val)
        for by in range(4):
            for bx in range(4):
                values=h[(h[:,1]//6==bx)&(h[:,2]//6==by),3]
                for name,val in [('n',len(values)),('mean',values.mean() if len(values) else 0),('max',values.max() if len(values) else 0)]:f[f'bin_{bx}_{by}_{name}']=np.full(n,val)
        top=h[np.argsort(h[:,3])[::-1][:4]]
        for i in range(4):
            row=top[min(i,len(top)-1)]
            for key,val in zip(['time','x','y','value'],row):f[f'peak_{i}_{key}']=np.full(n,val)
            f[f'peak_{i}_dx']=q[:,0]-row[1];f[f'peak_{i}_dy']=q[:,1]-row[2]
        for i in range(4):
            row=h[max(0,len(h)-1-i)]
            for key,val in zip(['time','x','y','value'],row):f[f'recent_{i}_{key}']=np.full(n,val)
        distance=np.linalg.norm(q[:,None,:]-h[None,:,1:3],axis=2)
        order=np.argsort(distance,axis=1)
        for i in range(4):
            idx=order[:,min(i,len(h)-1)]
            f[f'nearest_{i}_distance']=distance[np.arange(n),idx]
            f[f'nearest_{i}_value']=h[idx,3]
            f[f'nearest_{i}_age']=h[-1,0]-h[idx,0]
        temporal=np.exp(-.02*(h[-1,0]-h[:,0]))
        weights=temporal[None,:]/np.maximum(distance,.5)**2
        f['history_idw']=(weights@h[:,3])/weights.sum(axis=1)
        local=distance<=3
        f['local_count']=local.sum(axis=1)
        f['local_mean']=(local@h[:,3])/np.maximum(local.sum(axis=1),1)
        f['local_peak']=np.max(np.where(local,h[None,:,3],0),axis=1)
        for i in range(4):
            dx=f[f'peak_{i}_dx'];dy=f[f'peak_{i}_dy']
            f[f'peak_{i}_along_wind']=dx*wind[0]+dy*wind[1]
            f[f'peak_{i}_cross_wind']=dx*wind[1]-dy*wind[0]
        wall=np.argwhere(walls)[:,::-1]
        f['wall_distance']=np.linalg.norm(q[:,None,:]-wall[None,:,:],axis=2).min(axis=1)
        return pd.DataFrame(f)

def generate_sources(episodes=100,steps=90):
    frames=[]
    for ep in range(episodes):
        sim=Simulation(1000+ep);rng=np.random.default_rng(5000+ep);drone=np.array([0,0]);history=History();drone=drone if ep%2==0 else sim.free[rng.integers(len(sim.free))].copy();visits=np.zeros(sim.walls.shape);goal=sim.free[rng.integers(len(sim.free))]
        from plumesense import astar
        route=[]
        for t in range(steps):
            sim.step();history.add(sim.t,sim.observe(drone),drone)
            if t%8==0 or t==steps-1:
                near=sim.free[np.abs(sim.free-sim.source).sum(axis=1)<=1]
                far=sim.free[np.abs(sim.free-sim.source).sum(axis=1)>1]
                q=np.vstack([near,far[rng.choice(len(far),16,replace=False)]])
                f=history.features(q,sim.wind,sim.walls)
                f[SOURCE_TARGET]=(np.abs(q-sim.source).sum(axis=1)<=1).astype(int)
                f['episode_id']=ep;f['time_step']=sim.t;frames.append(f)
            if len(route)<=1:
                goal=sim.free[rng.integers(len(sim.free))];route=astar(sim.walls,visits*.1,drone,goal,weight=1)
            if len(route)>1:drone=np.array(route[1]);route=route[1:]
            visits[drone[1],drone[0]]+=1
    return pd.concat(frames,ignore_index=True)

if __name__=='__main__':
    d=generate_sources();d.to_csv(ROOT/'data/source_histories.csv',index=False);print(d.shape,flush=True)
