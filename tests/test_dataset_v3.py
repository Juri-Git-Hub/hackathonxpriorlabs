import json
from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest
from integration.dataset import Config,episode,split_for
from integration.features import rich_features,SOURCE,HAZARD
from shared.schemas import DroneObservation
from integration.dataset_io import sample_context

def observation(t=1,ppm=2,wind=(1,0)):
    return DroneObservation(sim_time=t,drone_id='tabpfn',pos=(-48,-45,5),vel=(1,2,3),wind_measured=wind,ppm_measured=ppm,temp_delta=None,nearest_asset_type='valve',asset_age_years=10,in_wind_shadow=False,upwind_pipe_score=0,battery_pct=100)

def test_wind_history_contributes_even_when_last_wind_is_identical():
    a=rich_features([observation(1,wind=(1,0)),observation(2,wind=(0,1))],[[0,0,3]])
    b=rich_features([observation(1,wind=(-1,0)),observation(2,wind=(0,1))],[[0,0,3]])
    assert a.wind_x.iloc[0]==b.wind_x.iloc[0]
    assert not a.equals(b)
    assert np.isfinite(a.to_numpy()).all()
    assert not any(c.startswith('target') for c in a)

def test_episode_is_reproducible_and_resume_preserves_completed_files(tmp_path):
    c=Config(episodes=10,seconds=10,candidates=16)
    a=episode(0,asdict(c),str(tmp_path/'a'),'test')
    b=episode(0,asdict(c),str(tmp_path/'b'),'test')
    assert a['sha256']['samples.parquet']==b['sha256']['samples.parquet']
    import gzip
    with gzip.open(tmp_path/'a/episodes/00000/observations.jsonl.gz','rt') as x, gzip.open(tmp_path/'b/episodes/00000/observations.jsonl.gz','rt') as y:
        assert x.read()==y.read()
    assert a['curriculum'] and a['split']=='train'
    assert episode(0,asdict(c),str(tmp_path/'a'),'test')['sha256']==a['sha256']
    with pytest.raises(ValueError):episode(0,asdict(c),str(tmp_path/'a'),'changed')

def test_held_out_start_is_not_source_conditioned(tmp_path):
    c=Config(episodes=10,seconds=10,candidates=16)
    for ep in [6,8]:
        m=episode(ep,asdict(c),str(tmp_path),'test')
        assert not m['curriculum'] and m['start']==[-48,-45,5]
        assert m['split']==split_for(ep,10)!='train'

def test_context_excludes_validation_and_test(tmp_path):
    rows=[]
    episodes=[]
    for ep,split in enumerate(['train','validation','test']):
        folder=tmp_path/'episodes'/f'{ep:05d}';folder.mkdir(parents=True)
        pd.DataFrame({'episode_id':[ep]*4,SOURCE:[0,1,0,1],'query_x':[ep]*4}).to_parquet(folder/'samples.parquet')
        episodes.append(dict(episode_id=ep,split=split))
    (tmp_path/'manifest.json').write_text(json.dumps({'episodes':episodes}))
    c=sample_context(tmp_path,SOURCE,4)
    assert set(c.episode_id)=={0}

@pytest.fixture(scope='module')
def verified_dataset(tmp_path_factory):
    from integration.dataset import generate
    from integration.verify_dataset import verify
    p=tmp_path_factory.mktemp('v3-complete')
    generate(Config(episodes=10,seconds=10,candidates=16),p,workers=2)
    report=verify(p)
    assert report['integrity_passed'] and report['temporal_prefix_recomputed']
    return p

def test_integrity_detects_modified_shard(verified_dataset):
    from integration.verify_dataset import verify
    file=verified_dataset/'episodes/00000/samples.parquet';original=file.read_bytes()
    try:
        file.write_bytes(original+b'tampered')
        with pytest.raises(AssertionError,match='Checksum mismatch'):verify(verified_dataset)
    finally:file.write_bytes(original)

def test_feature_recomputation_detects_fabricated_or_future_values(verified_dataset):
    import hashlib
    from integration.verify_dataset import verify
    file=verified_dataset/'episodes/00000/samples.parquet';meta_file=file.parent/'meta.json'
    original=file.read_bytes();original_meta=meta_file.read_text()
    try:
        d=pd.read_parquet(file);d.loc[0,'recent_0_log_ppm']=999;d.to_parquet(file,index=False)
        meta=json.loads(original_meta);meta['sha256']['samples.parquet']=hashlib.sha256(file.read_bytes()).hexdigest();meta_file.write_text(json.dumps(meta))
        with pytest.raises(AssertionError,match='Features differ'):verify(verified_dataset)
    finally:file.write_bytes(original);meta_file.write_text(original_meta)

def test_parallel_verification_agrees_with_sequential(verified_dataset):
    from integration.verify_dataset import verify
    a=verify(verified_dataset,workers=1)
    b=verify(verified_dataset,workers=2)
    assert a==b

def test_live_loader_rejects_dataset_that_fails_coverage_gate(verified_dataset):
    from agents.plumesense_agent import load_models
    with pytest.raises(ValueError,match='coverage gates'):load_models(dataset=verified_dataset)

def test_source_benchmark_counts_complete_snapshots_and_bootstraps_episodes():
    from integration.benchmark_dataset import ranking
    f=pd.DataFrame({'episode_id':[0,0,1,1],'snapshot_time':[5]*4,'query_x':[0,10,0,10],'query_y':[0]*4,'query_z':[3]*4,'signal_seen':[False,False,True,True]})
    r=ranking(f,np.array([.9,.1,.1,.9]),{0:[0,0,3],1:[0,0,3]})
    assert r['all']['snapshots']==2 and r['all']['mean_error_m']==5
    assert r['all']['within_6m']==.5
    assert r['signal_seen']['mean_error_m']==10
    assert r['episode_bootstrap']['episodes']==2
