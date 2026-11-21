import argparse,json,time
from pathlib import Path
import pandas as pd
from plumesense import *

p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','generate','evaluate']);p.add_argument('--backend',choices=['baseline','tabpfn','tabpfn-api'],default='baseline');p.add_argument('--dataset',choices=['original','synthetic'],default='synthetic');p.add_argument('--device',default='auto');p.add_argument('--episodes',type=int,default=36);p.add_argument('--navigation-episodes',type=int,default=3)
a=p.parse_args();Path('reports').mkdir(exist_ok=True)
if a.command=='audit':
    train=pd.read_csv('data/original/plumesense_train.csv');small=pd.read_csv('data/original/plumesense_tabpfn_train.csv');test=pd.read_csv('data/original/plumesense_tabpfn_test.csv');ep=pd.read_csv('data/original/plumesense_episodes.csv')
    assert train[small.columns].equals(small)
    assert set(train.episode_id)==set(ep.loc[ep.dataset_split=='train','episode_id'])
    result=dict(train_rows=len(train),test_rows=len(test),fixed_positions=len(train[['query_x','query_y']].drop_duplicates()),constants=list(train.columns[train.nunique()==1]),train_positive_rate=train[TARGET].mean(),full_test_metadata_available=False,note='Test episode membership cannot be independently verified without full test metadata.')
    Path('reports/audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
elif a.command=='generate':
    if a.episodes<4: raise ValueError('At least four episodes required')
    d=generate(a.episodes);d.to_csv('data/synthetic_drone_only.csv',index=False);print(d.shape,d[TARGET].mean())
else:
    if a.dataset=='original':
        train=pd.read_csv('data/original/plumesense_tabpfn_train.csv');test=pd.read_csv('data/original/plumesense_tabpfn_test.csv');cols=ORIGINAL_FEATURES
    else:
        d=pd.read_csv('data/synthetic_drone_only.csv');cut=int(d.episode_id.nunique()*.75);train=d[d.episode_id<cut];test=d[d.episode_id>=cut];cols=FEATURES
        assert not set(train.episode_id)&set(test.episode_id)
    model=Predictor(a.backend,a.device);t=time.perf_counter();model.fit(train,cols);fit=time.perf_counter()-t
    t=time.perf_counter();prob=model.predict(test);elapsed=time.perf_counter()-t
    result=dict(sensor_mode='drone_only' if a.dataset=='synthetic' else 'original_fixed_sensors',backend=a.backend,dataset=a.dataset,fit_seconds=fit,predict_seconds=elapsed,metrics=metrics(test[TARGET],prob))
    if a.dataset=='synthetic':
        result['spatial_slices']={str(v):metrics(test.loc[test.query_seen_sensor==v,TARGET],prob[test.query_seen_sensor.to_numpy()==v]) for v in [False,True] if (test.query_seen_sensor==v).any()}
        result['navigation']=[rollout(model,100+i) for i in range(a.navigation_episodes)]
    Path(f'reports/{a.dataset}_{a.backend}{"_drone_only" if a.dataset=="synthetic" else ""}.json').write_text(json.dumps(result,indent=2));test.assign(predicted_hazard_probability_t4=prob).to_csv(f'reports/{a.dataset}_{a.backend}_predictions.csv',index=False);print(json.dumps(result,indent=2))
