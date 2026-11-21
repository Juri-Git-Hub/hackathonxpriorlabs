"""Episode-separated evaluation for the new domain (sampled candidate rows)."""
import json
import time
import numpy as np
import pandas as pd
from agents.plumesense_agent import load_models
from integration.generate import DATA
from integration.features import features,SOURCE,HAZARD
from plumesense import metrics,ROOT
from integration.runtime import Runtime

def main():
    d=pd.read_csv(DATA);cut=int(d.episode_id.nunique()*.75)
    train=d[d.episode_id<cut];test=d[d.episode_id>=cut]
    assert not set(train.episode_id)&set(test.episode_id)
    source,hazard=load_models();columns=source.columns
    result=dict(training_episodes=sorted(train.episode_id.unique().tolist()),test_episodes=sorted(test.episode_id.unique().tolist()),note='Metrics on sampled candidate rows; source scores not calibrated; not navigation success or full-grid source accuracy.')
    for name,model,target in [('source',source,SOURCE),('hazard',hazard,HAZARD)]:
        start=time.perf_counter();p=model.predict(test[columns])
        result[name]=dict(metrics=metrics(test[target],p),request_seconds=time.perf_counter()-start)
    runtime=Runtime(seed=42,models=[source,hazard])
    try:
        with runtime.engine._lock:runtime.engine.paused=False
        start=time.monotonic()
        while time.monotonic()-start<30:
            with runtime.engine._lock:
                if runtime.agent.source_scores is not None and 'tabpfn' in runtime.beliefs:break
            if runtime.agent.error:raise RuntimeError(runtime.agent.error)
            time.sleep(.1)
        if runtime.agent.source_scores is None or 'tabpfn' not in runtime.beliefs:raise RuntimeError('No real API belief within 30s')
        tick,belief=runtime.snapshot()
        result['live_smoke']=dict(sim_time=tick.sim_time,wall_seconds=time.monotonic()-start,grid_points=len(runtime.agent.source_scores),hazard_points=len(runtime.agent.hazard),inference_seconds=runtime.agent.latency,belief_received='tabpfn' in belief)
    finally:runtime.close()
    (ROOT/'reports/main3d_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
