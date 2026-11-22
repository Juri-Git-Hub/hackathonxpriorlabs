"""Start the Three.js frontend and WebSocket backend."""
import argparse
import subprocess
import sys
from pathlib import Path
from agents.tabpfn_remote import check_tabpfn_backend

DEFAULT_ADAPTIVE_WIND = False

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mock-agents',action='store_true');p.add_argument('--port',type=int,default=8501);p.add_argument('--seed',type=int,default=42);p.add_argument('--adaptive-wind',action=argparse.BooleanOptionalAction,default=DEFAULT_ADAPTIVE_WIND)
    p.add_argument('--randomized-layout',action='store_true',help='Start the MVP on a procedural refinery layout (the default)')
    p.add_argument('--mvp',action='store_true',help='Run the repeatable local demonstration')
    p.add_argument('--tabpfn',action='store_true',help='Use the TabPFN flight policy and GPU backend')
    p.add_argument('--source-pre-data',type=Path,help='Alternate pre-contact TabPFN context')
    p.add_argument('--source-post-data',type=Path,help='Alternate post-contact TabPFN context')
    p.add_argument('--layout-seed',type=int,default=None,help='Seed for a connected procedural facility (default: random)')
    a=p.parse_args()
    if a.randomized_layout and not a.mvp:
        p.error('--randomized-layout requires --mvp')
    if a.mvp and (a.mock_agents or a.tabpfn):
        p.error('--mvp requires the wind-sensor agent')
    if (a.source_pre_data or a.source_post_data) and (a.mock_agents or not a.tabpfn):
        p.error('Alternate TabPFN options require --tabpfn without --mock-agents')
    root=Path(__file__).resolve().parent;frontend=root/'frontend'
    if not (frontend/'node_modules').exists():subprocess.run(['npm','ci'],cwd=frontend,check=True)
    subprocess.run(['npm','run','build'],cwd=frontend,check=True)
    if a.tabpfn and not a.mock_agents:
        print('TabPFN backend:', check_tabpfn_backend(), flush=True)
    command=[sys.executable,'-m','server.app','--host','127.0.0.1','--port',str(a.port),'--seed',str(a.seed)]
    if a.mvp:
        command.append('--mvp')
        if a.randomized_layout:command.append('--randomized-layout')
    elif a.mock_agents:command.append('--mock-agents')
    elif a.tabpfn:
        command.append('--tabpfn')
        for flag, path in (('--source-pre-data', a.source_pre_data), ('--source-post-data', a.source_post_data)):
            if path is not None:command.extend([flag,str(path.resolve())])
    if a.layout_seed is not None:command.extend(['--layout-seed',str(a.layout_seed)])
    command.append('--adaptive-wind' if a.adaptive_wind else '--no-adaptive-wind')
    try:subprocess.run(command,cwd=root,check=True)
    except KeyboardInterrupt:pass
