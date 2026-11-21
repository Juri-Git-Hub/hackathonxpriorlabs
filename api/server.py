"""Local CUDA TabPFN hazard classification service."""
import hashlib
import threading
import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

app = FastAPI(title='PlumeSense TabPFN API',version='1.0.0')
model = None
fit_key = None
lock = threading.Lock()
contexts = {}

class PredictRequest(BaseModel):
    context: str = 'hazard'
    X_train: list[list[float]]
    y_train: list[float]
    X_test: list[list[float]]

@app.get('/health')
def health():
    return dict(status='ok',cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,task='binary_classification',model_version='v3.5-fast')

@app.post('/predict')
def predict(data: PredictRequest):
    global model,fit_key
    try:
        train=np.asarray(data.X_train,dtype=np.float32)
        labels=np.asarray(data.y_train,dtype=np.float32)
        test=np.asarray(data.X_test,dtype=np.float32)
        if train.ndim!=2 or test.ndim!=2 or not train.size or not test.size:
            raise ValueError('Nonempty 2D feature arrays required')
        if len(train)!=len(labels) or train.shape[1]!=test.shape[1]:
            raise ValueError('Training rows/labels or feature counts do not match')
        if not all(np.isfinite(x).all() for x in (train,labels,test)):
            raise ValueError('All inputs must be finite')
        if set(np.unique(labels))!={0,1}:
            raise ValueError('Both binary hazard labels 0 and 1 required')
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e)) from e
    # Protect shared fit context from interleaved client requests.
    key=hashlib.sha256(str(train.shape).encode()+train.tobytes()+labels.tobytes()).hexdigest()
    with lock:
        if data.context not in ('hazard','source'):raise HTTPException(status_code=400,detail='Unknown context')
        model,fit_key=contexts.get(data.context,(None,None))
        start=time.perf_counter()
        if model is None:
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST,device='cuda',n_estimators=1)
        reused=fit_key==key
        if not reused:
            fit_key=None
            model.fit(train,labels.astype(int))
            fit_key=key
        contexts[data.context]=(model,fit_key)
        p=model.predict_proba(test)
        classes=model.classes_.tolist()
        return dict(probabilities=p.tolist(),classes=classes,predictions=model.classes_[p.argmax(axis=1)].tolist(),fit_reused=reused,elapsed_seconds=time.perf_counter()-start)
