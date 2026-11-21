"""Full-grid source ranking + hazard benchmark on source-independent holdouts.

No validation/test rows enter model context. Both tree and TabPFN receive the
same fixed context. This is prediction evaluation, not navigation success.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score,roc_auc_score,brier_score_loss
from plumesense import Predictor,TARGET,ROOT
from integration.dataset import DEFAULT,write_json
from integration.features import SOURCE,HAZARD
from integration.dataset_io import sample_context



def evaluation(dataset,split,limit_episodes=16):
    manifest=json.loads((Path(dataset)/'manifest.json').read_text());frames=[];sources={}
    for m in manifest['episodes']:
        folder=Path(dataset)/'episodes'/f"{m['episode_id']:05d}"
        if m['split']!=split or not (folder/'full_grid.parquet').exists():continue
        frames.append(pd.read_parquet(folder/'full_grid.parquet'));sources[m['episode_id']]=json.loads((folder/'meta.json').read_text())['source']
        if len(frames)>=limit_episodes:break
    if not frames:raise ValueError('No full-grid holdout snapshots')
    return pd.concat(frames,ignore_index=True),sources

def metrics(y,p,threshold=.5):
    return dict(rows=len(y),positives=int(np.sum(y)),f1=float(f1_score(y,p>=threshold,zero_division=0)),auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,brier=float(brier_score_loss(y,p)))

def ranking(frame,p,sources):
    errors=[];signal_errors=[];cold_errors=[];top5=[];by_episode={}
    for (ep,t),g in frame.groupby(['episode_id','snapshot_time']):
        indices=g.index.to_numpy();scores=p[indices];q=g[['query_x','query_y','query_z']].to_numpy();source=np.array(sources[int(ep)])
        order=np.argsort(scores)[::-1];error=float(np.linalg.norm(q[order[0]]-source))
        by_episode.setdefault(int(ep),[]).append(error)
        errors.append(error);(signal_errors if g.signal_seen.iloc[0] else cold_errors).append(error)
        top5.append(bool(np.any(np.linalg.norm(q[order[:5]]-source,axis=1)<=6)))
    def summary(e):return dict(snapshots=len(e),mean_error_m=float(np.mean(e)) if e else None,median_error_m=float(np.median(e)) if e else None,within_6m=float(np.mean(np.array(e)<=6)) if e else None,within_5m=float(np.mean(np.array(e)<=5)) if e else None)
    episode_errors=[np.array(v) for v in by_episode.values()];rng=np.random.default_rng(42)
    bootstrap=np.array([[np.mean(np.concatenate([episode_errors[i] for i in draw])),np.mean(np.concatenate([episode_errors[i] for i in draw])<=6)] for draw in rng.integers(0,len(episode_errors),(1000,len(episode_errors)))])
    ci=dict(mean_error_m_95=np.quantile(bootstrap[:,0],[.025,.975]).tolist(),within_6m_95=np.quantile(bootstrap[:,1],[.025,.975]).tolist(),resampling_unit='episode',episodes=len(episode_errors))
    return dict(episode_bootstrap=ci,all=summary(errors),signal_seen=summary(signal_errors),no_signal=summary(cold_errors),top5_within_6m=float(np.mean(top5)))

def batch_predict(model,frame,batch):
    parts=[];latencies=[]
    for start in range(0,len(frame),batch):
        t=time.perf_counter();parts.append(model.predict(frame.iloc[start:start+batch]));latencies.append(time.perf_counter()-t)
    return np.concatenate(parts),dict(total_seconds=sum(latencies),mean_request_seconds=float(np.mean(latencies)),requests=len(latencies),rows_per_request=batch)

def benchmark(dataset=DEFAULT,context_rows=8192,episodes=16,tabpfn=False,batch=4096):
    dataset=Path(dataset);manifest=json.loads((dataset/'manifest.json').read_text());columns=manifest['feature_columns']
    val,val_sources=evaluation(dataset,'validation',episodes);test,test_sources=evaluation(dataset,'test',episodes)
    result=dict(dataset_signature=manifest['signature'],feature_count=len(columns),context_limit=context_rows,training_episode_count=sum(m['split']=='train' for m in manifest['episodes']),test_episode_ids=sorted(test.episode_id.unique().tolist()),validation_episode_ids=sorted(val.episode_id.unique().tolist()),full_grid=True,models={},note='Complete 3D candidate grids and source-independent holdout starts; no navigation success claimed. Candidate probabilities may be miscalibrated due to context sampling. F1 threshold selected using validation only.')
    for name,target in [('source',SOURCE),('hazard',HAZARD)]:
        train=sample_context(dataset,target,context_rows)
        assert not set(train.episode_id)&(set(test.episode_id)|set(val.episode_id))
        models={'prior':None,'observation_baseline':None,'extra_trees':ExtraTreesClassifier(n_estimators=120,min_samples_leaf=4,n_jobs=4,random_state=42)}
        if tabpfn:
            m=Predictor('tabpfn-api');m.context=name
            f=train[columns].copy();f[TARGET]=train[target]
            m.fit(f,columns);models['tabpfn']=m
        for kind,model in models.items():
            if kind=='prior':
                # Constant prior baseline cannot locate a source; ranking ties are arbitrary.
                pv=np.full(len(val),val[target].mean());pt=np.full(len(test),val[target].mean());latency={'prior_from_validation':float(val[target].mean())}
            elif kind=='observation_baseline':
                def observed(frame):
                    if name=='source':
                        distance2=sum(frame[f'peak_0_d{axis}'].to_numpy()**2 for axis in 'xyz')
                        p=np.exp(-.5*distance2/36)
                        p[frame['max'].to_numpy()<=np.log1p(5)]=0.001
                        return p
                    return 1/(1+np.exp(-np.clip((frame.history_idw.to_numpy()-np.log1p(10))/.5,-30,30)))
                pv=observed(val);pt=observed(test);latency={}
            elif kind=='extra_trees':
                t=time.perf_counter();model.fit(train[columns],train[target]);fit=time.perf_counter()-t
                t=time.perf_counter();pv=model.predict_proba(val[columns])[:,list(model.classes_).index(1)];pt=model.predict_proba(test[columns])[:,list(model.classes_).index(1)];latency=dict(fit_seconds=fit,predict_seconds=time.perf_counter()-t)
            else:
                pv,lv=batch_predict(model,val[columns],batch);pt,lt=batch_predict(model,test[columns],batch);latency=dict(validation=lv,test=lt)
            thresholds=np.linspace(.01,.99,99);threshold=float(thresholds[np.argmax([f1_score(val[target],pv>=v,zero_division=0) for v in thresholds])])
            record=dict(context_rows=0 if kind in ('prior','observation_baseline') else len(train),context_episode_count=0 if kind in ('prior','observation_baseline') else int(train.episode_id.nunique()),context_snapshot_count=0 if kind in ('prior','observation_baseline') else int(train.snapshot_id.nunique()),context_positive_rate=None if kind in ('prior','observation_baseline') else float(train[target].mean()),threshold_from_validation=threshold,test=metrics(test[target],pt,threshold),test_at_point5=metrics(test[target],pt),validation=metrics(val[target],pv,threshold),latency=latency)
            if name=='source':record['source_ranking']=ranking(test,pt,test_sources)
            result['models'][f'{name}_{kind}']=record
            print(json.dumps({name+'_'+kind:record}),flush=True)
            write_json(dataset/'benchmark.json',result)
    reports=ROOT/'reports';reports.mkdir(exist_ok=True);write_json(reports/'dataset_v3_benchmark.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',type=Path,default=DEFAULT);p.add_argument('--context-rows',type=int,default=8192);p.add_argument('--episodes',type=int,default=16);p.add_argument('--tabpfn',action='store_true');p.add_argument('--batch',type=int,default=4096);a=p.parse_args()
    benchmark(a.dataset,a.context_rows,a.episodes,a.tabpfn,a.batch)
