"""Strict TabPFN adapter for the map-search dataset supplied in the next step."""
import json
import os
from pathlib import Path
import pandas as pd
from integration.map_features import VERSION,TARGET,CONTACT_PPM,HORIZON_SECONDS,feature_table
from plumesense import Predictor,TARGET as PREDICTOR_TARGET,ROOT


def load_search_model(world,dataset=None,context_rows=4096):
    configured=dataset or os.environ.get('PLUMESENSE_SEARCH_DATASET')
    folder=Path(configured or ROOT/'data/datasets/map-search-v1')
    if not folder.exists():
        if configured:raise ValueError(f'Configured map-search dataset does not exist: {folder}')
        return None
    manifest=json.loads((folder/'manifest.json').read_text())
    columns=list(feature_table(world,[[*world.bounds_min]],world.bounds_min))
    if manifest.get('feature_version')!=VERSION or manifest.get('target')!=TARGET or manifest.get('feature_columns')!=columns or manifest.get('contact_threshold_ppm')!=CONTACT_PPM or manifest.get('horizon_seconds')!=HORIZON_SECONDS:
        raise ValueError('Map-search dataset must match map-search-v1 features and target_gas_contact_t4')
    train=pd.read_parquet(folder/'train.parquet')
    if 'split' in train and not train['split'].eq('train').all():raise ValueError('Search context contains non-training rows')
    if 'episode_id' in train:
        held=set(map(str,manifest.get('validation_episodes',[])))|set(map(str,manifest.get('test_episodes',[])))
        if set(map(str,train.episode_id))&held:raise ValueError('Search context overlaps held-out episodes')
    if train[TARGET].nunique()!=2 or not set(train[TARGET].unique())<={0,1}:raise ValueError('Search context needs both binary classes')
    positive=train[train[TARGET]==1];negative=train[train[TARGET]==0]
    npositive=min(len(positive),max(1,context_rows//4));nnegative=min(len(negative),context_rows-npositive)
    train=pd.concat([positive.sample(n=npositive,random_state=42),negative.sample(n=nnegative,random_state=43)])
    frame=train[columns].copy();frame[PREDICTOR_TARGET]=train[TARGET].to_numpy()
    # Current remote API supports source/hazard contexts. A distinct fit hash
    # prevents accidental reuse of the old 20-feature hazard model.
    model=Predictor('tabpfn-api');model.context='hazard';model.fit(frame,columns)
    model.feature_version=VERSION
    return model
