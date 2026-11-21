"""Observation-only 3D history features; no simulator state or leak coordinates."""
import numpy as np
import pandas as pd

SOURCE='target_source_near'
HAZARD='target_hazard_t4'
DATA_VERSION='main3d-v1'

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
