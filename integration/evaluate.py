"""Evaluate the actual compact live profile on episode-separated full grids.

Feature selection uses validation only; test is used once for final reporting.
Navigation success is measured separately by integration.closed_loop.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from agents.plumesense_agent import load_models
from integration.benchmark_dataset import batch_predict, evaluation, metrics, ranking
from integration.dataset import DEFAULT, write_json
from integration.features import SOURCE, HAZARD
from integration.search_protocol import PROFILE
from plumesense import ROOT


def observation_baseline(frame, target):
    if target == SOURCE:
        distance2=sum(frame[f'peak_0_d{axis}'].to_numpy()**2 for axis in 'xyz')
        result=np.exp(-.5*distance2/36)
        result[frame['max'].to_numpy() <= np.log1p(5)]=.001
        return result
    return 1/(1+np.exp(-np.clip((frame.history_idw.to_numpy()-np.log1p(10))/.5,-30,30)))


def evaluate(dataset=DEFAULT, episodes=16, output=ROOT/'reports/compact_profile_predictions.json'):
    dataset=Path(dataset)
    manifest=json.loads((dataset/'manifest.json').read_text())
    models=load_models(dataset=dataset, context_rows=PROFILE['context_rows'])
    val,_=evaluation(dataset,'validation',episodes)
    test,sources=evaluation(dataset,'test',episodes)
    training={m['episode_id'] for m in manifest['episodes'] if m['split']=='train'}
    assert not training & (set(val.episode_id)|set(test.episode_id))
    result=dict(profile=PROFILE,dataset_signature=manifest['signature'],
                validation_episodes=sorted(val.episode_id.unique().tolist()),
                test_episodes=sorted(test.episode_id.unique().tolist()),models={},
                note='Complete coarse XYZ grids; not a navigation success rate. '
                     'Fine adaptive source hypotheses are tested in closed-loop flights. '
                     'F1 threshold uses validation only. Scores are uncalibrated.')
    for name,model,target in zip(['source','hazard'],models,[SOURCE,HAZARD]):
        for kind in ['tabpfn','observation_baseline']:
            if kind=='tabpfn':
                pv,lv=batch_predict(model,val,4096)
                pt,lt=batch_predict(model,test,4096)
                latency=dict(validation=lv,test=lt)
            else:
                pv=observation_baseline(val,target)
                pt=observation_baseline(test,target)
                latency={}
            thresholds=np.linspace(.01,.99,99)
            threshold=float(thresholds[np.argmax([f1_score(val[target],pv>=t,zero_division=0) for t in thresholds])])
            record=dict(feature_count=len(model.columns) if kind=='tabpfn' else None,
                        threshold_from_validation=threshold,
                        validation=metrics(val[target],pv,threshold),
                        test=metrics(test[target],pt,threshold),
                        test_at_point5=metrics(test[target],pt),latency=latency)
            if target==SOURCE:record['source_ranking']=ranking(test,pt,sources)
            result['models'][f'{name}_{kind}']=record
            print(json.dumps({f'{name}_{kind}':record}),flush=True)
            write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT)
    parser.add_argument('--episodes',type=int,default=16)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/compact_profile_predictions.json')
    args=parser.parse_args()
    evaluate(args.dataset,args.episodes,args.output)
