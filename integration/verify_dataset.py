"""Integrity, provenance, label and temporal-leakage checks for V3 shards."""
import argparse,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from shared.schemas import DroneObservation
from integration.dataset import DEFAULT,write_json,split_for
from integration.features import rich_features,SOURCE,HAZARD

EXTRA={SOURCE,HAZARD,'target_concentration_t4','episode_id','snapshot_time','label_time','split','signal_seen','snapshot_id','phase'}

def _verify_episode(item,out,manifest,columns,recompute):
    episodes=manifest['episodes']
    folder=out/'episodes'/f"{item['episode_id']:05d}";meta=json.loads((folder/'meta.json').read_text())
    assert meta['signature']==manifest['signature']
    assert meta['split']==split_for(meta['episode_id'],len(episodes))==item['split']
    assert meta['split']=='train' or not meta['curriculum'],'Source-conditioned evaluation is forbidden'
    for name,digest in meta['sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,f'Checksum mismatch: {folder/name}'
    with gzip.open(folder/'observations.jsonl.gz','rt') as f:history=[DroneObservation.model_validate_json(line) for line in f]
    times=np.array([o.sim_time for o in history]);assert len(times) and np.all(np.diff(times)>0)
    assert abs(meta['max_ppm']-max(o.ppm_measured for o in history if o.sim_time<=manifest['config']['seconds']+1e-5))<1e-8
    data=pd.read_parquet(folder/'samples.parquet')
    assert set(data)==set(columns)|EXTRA,'Unexpected metadata/schema (potential feature leakage)'
    assert np.isfinite(data[columns].to_numpy()).all()
    assert set(data[SOURCE])<={0,1} and set(data[HAZARD])<={0,1}
    assert data.episode_id.eq(meta['episode_id']).all() and data.split.eq(meta['split']).all()
    assert not data.duplicated(['snapshot_id','query_x','query_y','query_z']).any()
    assert np.allclose(data.label_time-data.snapshot_time,4)
    assert (data['time']<=data.snapshot_time+1e-5).all()
    q=data[['query_x','query_y','query_z']].to_numpy();source=np.array(meta['source'])
    assert np.array_equal(data[SOURCE],(np.linalg.norm(q-source,axis=1)<=6).astype(int)),'Source label mismatch'
    assert np.array_equal(data[HAZARD],(data.target_concentration_t4>=10).astype(int)),'Hazard label mismatch'
    snapshots=[]
    for t,g in data.groupby('snapshot_time'):
        observed=[o for o in history if o.sim_time<=t+1e-5]
        assert observed and g.measurement_count.eq(len(observed)).all()
        assert all(g.filter(regex=r'^(peak|recent)_\d_time$').max().to_numpy()<=t+1e-5)
        if recompute:
            # Rebuild from raw prefix, never from any future observation.
            sample=g.iloc[[0,-1]]
            rebuilt=rich_features(observed,sample[['query_x','query_y','query_z']].to_numpy())
            assert list(rebuilt)==columns
            assert np.allclose(rebuilt.to_numpy(),sample[columns].to_numpy(),rtol=1e-6,atol=1e-6),'Features differ from raw history prefix'
        snapshots.append(bool(g.signal_seen.iloc[0]))
    for filename in ['full_grid.parquet']:
        if (folder/filename).exists():
            f=pd.read_parquet(folder/filename)
            for t,g in f.groupby('snapshot_time'):
                assert len(g)==len(meta['grid'])
                assert not g.duplicated(['query_x','query_y','query_z']).any()
                assert np.allclose(g[['query_x','query_y','query_z']],meta['grid'])
                assert np.isfinite(g[columns].to_numpy()).all()
                observed=[o for o in history if o.sim_time<=t+1e-5]
                if recompute:
                    rebuilt=rich_features(observed,g[['query_x','query_y','query_z']].iloc[[0,-1]].to_numpy())
                    assert np.allclose(rebuilt.to_numpy(),g[columns].iloc[[0,-1]].to_numpy(),rtol=1e-6,atol=1e-6)
                assert np.array_equal(g[SOURCE],(np.linalg.norm(g[['query_x','query_y','query_z']].to_numpy()-source,axis=1)<=6).astype(int))
                assert np.array_equal(g[HAZARD],(g.target_concentration_t4>=10).astype(int))
    positions=np.array([o.pos for o in history])
    return dict(episode=meta['episode_id'],split=meta['split'],curriculum=meta['curriculum'],contact=meta['gas_contact'],signal_snapshots=sum(snapshots),snapshots=len(snapshots),rows=len(data),height_span=float(np.ptp(positions[:,2])),height_min=float(positions[:,2].min()),height_max=float(positions[:,2].max()),final_battery=meta['final_battery'],source_positive=int(data[SOURCE].sum()),hazard_positive=int(data[HAZARD].sum()),policy=meta['policy'],phase_counts=data.drop_duplicates('snapshot_time').phase.value_counts().to_dict())

def verify(output=DEFAULT,recompute=True,workers=1):
    out=Path(output);manifest=json.loads((out/'manifest.json').read_text());columns=manifest['feature_columns'];records=[];warnings=[]
    assert not any(c.startswith('target') or c in EXTRA or 'source' in c or 'seed' in c for c in columns),'Forbidden feature'
    assert len(columns)==len(set(columns))
    episodes=manifest['episodes'];ids=[m['episode_id'] for m in episodes]
    assert ids==list(range(manifest['config']['episodes'])),'Missing/duplicate episodes'
    total=0
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor
        from itertools import repeat
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=list(pool.map(_verify_episode,episodes,repeat(out),repeat(manifest),repeat(columns),repeat(recompute)))
    else:
        records=[_verify_episode(item,out,manifest,columns,recompute) for item in episodes]
    total=sum(record['rows'] for record in records)
    assert total==manifest['rows']
    summaries={}
    for split in ['train','validation','test']:
        r=[m for m in records if m['split']==split]
        summaries[split]=dict(episodes=len(r),rows=sum(m['rows'] for m in r),contact_episodes=sum(m['contact'] for m in r),contact_episode_rate=float(np.mean([m['contact'] for m in r])),signal_snapshot_rate=sum(m['signal_snapshots'] for m in r)/sum(m['snapshots'] for m in r),mean_height_span=float(np.mean([m['height_span'] for m in r])),source_positive=sum(m['source_positive'] for m in r),hazard_positive=sum(m['hazard_positive'] for m in r))
        if summaries[split]['contact_episode_rate']<.2:warnings.append(f'{split}: less than 20% episodes have gas contact; source inference may be unidentifiable.')
        if not summaries[split]['source_positive'] or not summaries[split]['hazard_positive']:warnings.append(f'{split}: missing positive class')
    phases=Counter()
    for record in records:
        if record['split']=='train':phases.update(record['phase_counts'])
    quality_checks=dict(at_least_128_episodes=len(records)>=128,training_contact_rate_at_least_50_percent=summaries['train']['contact_episode_rate']>=.5,training_has_cold_contact_and_lost=all(phases[p]>0 for p in ['cold','contact','lost']),mean_vertical_span_at_least_10m=summaries['train']['mean_height_span']>=10,all_splits_have_positive_labels=all(v['source_positive']>0 and v['hazard_positive']>0 for v in summaries.values()))
    report=dict(quality_checks=quality_checks,quality_gate_passed=all(quality_checks.values()),training_phase_snapshots=dict(phases),dataset_signature=manifest['signature'],integrity_passed=True,temporal_prefix_recomputed=recompute,feature_count=len(columns),rows=total,splits=summaries,warnings=warnings,episodes=records,note='Passed integrity/coverage gates do not establish accurate source localization or calibrated probabilities. Fixed map shared across splits; no claim of generalization to unseen facilities. Hazard labels checked against saved future simulator truth, not an independent CFD model.')
    write_json(out/'verification.json',report);print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',type=Path,default=DEFAULT);p.add_argument('--skip-recompute',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args();verify(a.dataset,not a.skip_recompute,a.workers)
