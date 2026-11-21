"""Observation-only 3D history features; no simulator state or leak coordinates."""
import numpy as np
import pandas as pd

SOURCE='target_source_near'
HAZARD='target_hazard_t4'
DATA_VERSION='main3d-v2-volumetric'

def features(history,grid):
    q=np.asarray(grid,float);n=len(q);last=history[-1]
    h=np.array([[o.sim_time,*o.pos,np.log1p(o.ppm_measured)] for o in history])
    f={f'query_{axis}':q[:,i] for i,axis in enumerate('xyz')}
    for i,axis in enumerate('xy'):
        value=last.wind_measured[i]
        f[f'wind_{axis}']=np.full(n,value if value is not None else 0.)
        f[f'wind_{axis}_missing']=np.full(n,float(value is None))
    f['time']=np.full(n,last.sim_time);f['measurement_count']=np.full(n,len(h))
    for name,value in [('mean',h[:,4].mean()),('max',h[:,4].max()),('std',h[:,4].std())]:f[name]=np.full(n,value)
    for kind,rows in [('peak',h[np.argsort(h[:,4])[::-1]]),('recent',h[::-1])]:
        for i in range(4):
            row=rows[min(i,len(rows)-1)]
            for j,name in enumerate(['time','x','y','z','log_ppm']):f[f'{kind}_{i}_{name}']=np.full(n,row[j])
            for j,axis in enumerate('xyz'):f[f'{kind}_{i}_d{axis}']=q[:,j]-row[j+1]
    d=np.linalg.norm(q[:,None,:]-h[None,:,1:4],axis=2)
    ids=np.argsort(d,axis=1)
    for i in range(3):
        idx=ids[:,min(i,len(h)-1)]
        f[f'nearest_{i}_distance']=d[np.arange(n),idx]
        f[f'nearest_{i}_log_ppm']=h[idx,4]
        f[f'nearest_{i}_age']=h[-1,0]-h[idx,0]
    w=np.exp(-.02*(h[-1,0]-h[:,0]))[None,:]/np.maximum(d,1.)**2
    f['history_idw']=w@h[:,4]/w.sum(axis=1)
    return pd.DataFrame(f)

RICH_VERSION='main3d-history-v3'

def rich_features(history,grid):
    """V3 adds measured wind history, gradients and spatial evidence; no truth."""
    f=features(history,grid);q=np.asarray(grid,float);n=len(q)
    h=np.array([[o.sim_time,*o.pos,np.log1p(o.ppm_measured)] for o in history])
    for kind,indices in [('peak',np.argsort(h[:,4])[::-1]),('recent',np.arange(len(h)-1,-1,-1))]:
        for i in range(4):
            idx=int(indices[min(i,len(indices)-1)]);o=history[idx]
            for axis,value in zip('xy',o.wind_measured):
                f[f'{kind}_{i}_wind_{axis}']=np.full(n,0 if value is None else value)
                f[f'{kind}_{i}_wind_{axis}_missing']=np.full(n,int(value is None))
            f[f'{kind}_{i}_temp']=np.full(n,0 if o.temp_delta is None else o.temp_delta)
            f[f'{kind}_{i}_temp_missing']=np.full(n,int(o.temp_delta is None))
    times=h[:,0];age=times[-1]-times
    for seconds in [5,15,30]:
        mask=age<=seconds;values=h[mask,4]
        f[f'window_{seconds}_mean']=np.full(n,values.mean())
        f[f'window_{seconds}_std']=np.full(n,values.std())
        f[f'window_{seconds}_n']=np.full(n,len(values))
        f[f'window_{seconds}_delta']=np.full(n,values[-1]-values[0])
    f['visited_extent_z']=np.full(n,np.ptp(h[:,3]))
    for axis in range(3):
        f[f'visited_extent_{axis}']=np.full(n,np.ptp(h[:,axis+1]))
    d=np.linalg.norm(q[:,None,:]-h[None,:,1:4],axis=2)
    for radius in [6,15]:
        near=d<=radius
        f[f'local_{radius}_n']=near.sum(axis=1)
        f[f'local_{radius}_mean']=near@h[:,4]/np.maximum(near.sum(axis=1),1)
        f[f'local_{radius}_max']=np.max(np.where(near,h[None,:,4],0),axis=1)
    latest=history[-1]
    for i,axis in enumerate('xyz'):f[f'velocity_{axis}']=np.full(n,latest.vel[i])
    return f
