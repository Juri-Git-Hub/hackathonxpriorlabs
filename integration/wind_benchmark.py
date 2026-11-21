"""Compare measured-wind policies with identical real TabPFN context and seeds."""
import argparse
import hashlib
import json
from pathlib import Path

from integration.closed_loop import run
from integration.dataset import DEFAULT,write_json
from integration.dataset_io import sample_context
from integration.features import SOURCE
from integration.reduced_features import fit_reduced
from plumesense import ROOT


CASES=[
    dict(name='default41',seed=41),dict(name='default42',seed=42),dict(name='default43',seed=43),
    dict(name='north_wind',seed=44,wind=[90,3.5]),
    dict(name='west_wind',seed=45,wind=[180,4.5]),
    dict(name='changing_wind',seed=46,wind_schedule=[dict(time=35,angle=135,speed=4)]),
    dict(name='early_wind_reversal',seed=47,wind_schedule=[dict(time=5,angle=195,speed=3.5)]),
]


def benchmark(dataset=DEFAULT,policies=('prediction','wind','wind_search'),output=ROOT/'reports/wind_policy_comparison.json'):
    files=['integration/wind_planning.py','integration/wind_search.py','integration/prediction_control.py']
    report=dict(split='validation_only',cases=CASES,policies=list(policies),context_rows=4096,
                code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},runs=[],
                note='prediction=working default; wind=head/crosswind cost experiment; '
                     'wind_search=reactive acquisition. Same real TabPFN context and judge; '
                     'measured RPC delay applied to simulated time, not wall-clock async.')
    model=fit_reduced(sample_context(Path(dataset),SOURCE,4096),SOURCE)
    for i,case in enumerate(CASES):
        for policy in (policies if i%2==0 else list(reversed(policies))):
            result=run(seconds=180,model=model,latency=True,policy=policy,**{k:v for k,v in case.items() if k!='name'})
            result['name']=case['name'];report['runs'].append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
            write_json(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT)
    p.add_argument('--policies',nargs='+',choices=['prediction','wind','wind_search'],default=['prediction','wind','wind_search'])
    p.add_argument('--output',type=Path,default=ROOT/'reports/wind_policy_comparison.json')
    args=p.parse_args();benchmark(args.dataset,args.policies,args.output)
