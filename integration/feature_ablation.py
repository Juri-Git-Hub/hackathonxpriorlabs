"""Validation-only feature/group ablation with identical training contexts."""
import argparse
import json
import time
from pathlib import Path

from integration.dataset import DEFAULT, write_json
from integration.dataset_io import sample_context
from integration.benchmark_dataset import evaluation, ranking, metrics, batch_predict
from integration.features import SOURCE, HAZARD
from integration.reduced_features import COMPACT, MEDIUM, RELATIVE_COLUMNS, relative_features
from plumesense import Predictor, TARGET, ROOT


def run(dataset=DEFAULT, episodes=8, context_rows=4096, groups=False, target=SOURCE, output=None):
    dataset=Path(dataset)
    manifest=json.loads((dataset/'manifest.json').read_text())
    train=sample_context(dataset,target,context_rows)
    val,sources=evaluation(dataset,'validation',episodes)
    variants=[('relative12',None),('compact20',COMPACT),('medium47',MEDIUM),('full159',manifest['feature_columns'])]
    if groups:
        drops={
            'all12':[],
            'without_wind_alignment':['along_wind','cross_wind','wind_speed'],
            'without_gas_levels':['peak_log_ppm','nearest_log_ppm','history_idw','current_log_ppm','local_mean','local_peak'],
            'without_local_evidence':['nearest_distance','nearest_log_ppm','history_idw','local_mean','local_peak'],
            'without_gas_change':['gas_delta_5s'],
        }
        variants=[(name,[c for c in RELATIVE_COLUMNS if c not in drop]) for name,drop in drops.items()]
    output=output or ROOT/'reports'/('validation_feature_groups.json' if groups else 'validation_feature_ablation.json')
    report=dict(split='validation_only',episodes=sorted(val.episode_id.unique().tolist()),
                context_rows=len(train),target=target,variants={},dataset_signature=manifest['signature'])
    for name,columns in variants:
        use_relative=groups or columns is None
        training=relative_features(train) if use_relative else train
        validation=relative_features(val) if use_relative else val
        columns=columns or RELATIVE_COLUMNS
        model=Predictor('tabpfn-api');model.context='source' if target==SOURCE else 'hazard'
        frame=training[columns].copy();frame[TARGET]=train[target].to_numpy()
        model.fit(frame,columns)
        start=time.perf_counter();p,latency=batch_predict(model,validation[columns],4096)
        record=dict(columns=columns,features=len(columns),metrics=metrics(val[target],p),
                    latency=latency,seconds=time.perf_counter()-start)
        if target==SOURCE:record['ranking']=ranking(val,p,sources)
        report['variants'][name]=record
        print(json.dumps({name:record}),flush=True);write_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT)
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--context-rows',type=int,default=4096)
    parser.add_argument('--groups',action='store_true')
    parser.add_argument('--target',choices=['source','hazard'],default='source')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.dataset,args.episodes,args.context_rows,args.groups,SOURCE if args.target=='source' else HAZARD,args.output)
