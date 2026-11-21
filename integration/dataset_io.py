"""Shared training-only context selection for benchmarking and explicit deployment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def sample_context(dataset,target,limit=8192):
    manifest=json.loads((Path(dataset)/'manifest.json').read_text());train=[m for m in manifest['episodes'] if m['split']=='train']
    quota=int(np.ceil(limit/len(train)));chunks=[]
    for m in train:
        f=pd.read_parquet(Path(dataset)/'episodes'/f"{m['episode_id']:05d}"/'samples.parquet')
        rng=np.random.default_rng(19000+m['episode_id']);positive=f[f[target]==1];negative=f[f[target]==0]
        npos=min(len(positive),max(1,quota//4));nneg=min(len(negative),quota-npos)
        if npos:chunks.append(positive.iloc[rng.choice(len(positive),npos,replace=False)])
        if nneg:chunks.append(negative.iloc[rng.choice(len(negative),nneg,replace=False)])
    f=pd.concat(chunks,ignore_index=True)
    return f.sample(n=min(limit,len(f)),random_state=42).reset_index(drop=True)
