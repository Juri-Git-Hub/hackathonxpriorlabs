"""Measured wind-frame features: no absolute location, simulator truth or clock."""
import numpy as np
import pandas as pd
from plumesense import Predictor,TARGET

COMPACT=['peak_0_dx','peak_0_dy','peak_0_dz','peak_0_log_ppm','peak_0_wind_x','peak_0_wind_y','recent_0_dx','recent_0_dy','recent_0_dz','recent_0_log_ppm','nearest_0_distance','nearest_0_log_ppm','nearest_0_age','history_idw','local_6_n','local_6_mean','local_6_max','window_5_delta','wind_x','wind_y']
MEDIUM=COMPACT+[f'peak_{i}_{name}' for i in [1,2,3] for name in ['dx','dy','dz','log_ppm','wind_x','wind_y']]+[f'window_{t}_{name}' for t in [15,30] for name in ['mean','std','delta']]+['velocity_x','velocity_y','velocity_z']

RELATIVE_COLUMNS=['along_wind','cross_wind','vertical_offset','peak_log_ppm','wind_speed','nearest_distance','nearest_log_ppm','history_idw','current_log_ppm','gas_delta_5s','local_mean','local_peak']

def relative_features(frame):
    wx=frame.peak_0_wind_x.to_numpy();wy=frame.peak_0_wind_y.to_numpy();missing=np.hypot(wx,wy)<.2
    wx=np.where(missing,frame.wind_x.to_numpy(),wx);wy=np.where(missing,frame.wind_y.to_numpy(),wy);speed=np.hypot(wx,wy)
    norm=np.maximum(speed,.2);dx=frame.peak_0_dx.to_numpy();dy=frame.peak_0_dy.to_numpy()
    return pd.DataFrame(dict(along_wind=(dx*wx+dy*wy)/norm,cross_wind=(dx*wy-dy*wx)/norm,vertical_offset=frame.peak_0_dz.to_numpy(),peak_log_ppm=frame.peak_0_log_ppm.to_numpy(),wind_speed=speed,nearest_distance=frame.nearest_0_distance.to_numpy(),nearest_log_ppm=frame.nearest_0_log_ppm.to_numpy(),history_idw=frame.history_idw.to_numpy(),current_log_ppm=frame.recent_0_log_ppm.to_numpy(),gas_delta_5s=frame.window_5_delta.to_numpy(),local_mean=frame.local_6_mean.to_numpy(),local_peak=frame.local_6_max.to_numpy()))[RELATIVE_COLUMNS]

class ReducedPredictor:
    def __init__(self,model,columns=None):
        self.model=model;self.selected_columns=columns;self.feature_version='main3d-history-v3'
        self.context=model.context;self.columns=model.columns;self.feature_builder='lean'
    def predict(self,frame):
        return self.model.predict(relative_features(frame) if self.selected_columns is None else frame[self.selected_columns])

def fit_reduced(train,target,columns=None,context='source'):
    frame=relative_features(train) if columns is None else train[columns].copy()
    selected=list(frame);frame[TARGET]=train[target].to_numpy()
    model=Predictor('tabpfn-api');model.context=context;model.fit(frame,selected)
    return ReducedPredictor(model,columns)


def lean_features(history,grid):
    q=np.asarray(grid,float);h=np.array([[o.sim_time,*o.pos,np.log1p(o.ppm_measured)] for o in history]);n=len(q)
    peak_index=np.argsort(h[:,4])[::-1][0];peak=h[peak_index];recent=h[-1];po=history[peak_index]
    d=np.linalg.norm(q[:,None,:]-h[None,:,1:4],axis=2);idx=np.argsort(d,axis=1)[:,0]
    w=np.exp(-.02*(recent[0]-h[:,0]))[None,:]/np.maximum(d,1.)**2;near=d<=6
    f={}
    for i,axis in enumerate('xyz'):
        f[f'peak_0_d{axis}']=q[:,i]-peak[i+1];f[f'recent_0_d{axis}']=q[:,i]-recent[i+1]
    for i,axis in enumerate('xy'):
        f[f'peak_0_wind_{axis}']=np.full(n,po.wind_measured[i] or 0.)
        f[f'wind_{axis}']=np.full(n,history[-1].wind_measured[i] or 0.)
    f.update(peak_0_log_ppm=np.full(n,peak[4]),recent_0_log_ppm=np.full(n,recent[4]),nearest_0_distance=d[np.arange(n),idx],nearest_0_log_ppm=h[idx,4],nearest_0_age=recent[0]-h[idx,0],history_idw=w@h[:,4]/w.sum(axis=1),local_6_n=near.sum(axis=1),local_6_mean=near@h[:,4]/np.maximum(near.sum(axis=1),1),local_6_max=np.max(np.where(near,h[None,:,4],0),axis=1),window_5_delta=np.full(n,recent[4]-h[recent[0]-h[:,0]<=5,4][0]))
    return pd.DataFrame(f)[COMPACT]
