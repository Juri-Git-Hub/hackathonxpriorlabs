"""Start main's Three.js frontend and WebSocket backend with our GPU tunnel."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from tunnel import ensure_tunnel

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mock-agents',action='store_true');p.add_argument('--port',type=int,default=8501);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    root=Path(__file__).resolve().parent;frontend=root/'frontend'
    if not (frontend/'node_modules').exists():subprocess.run(['npm','ci'],cwd=frontend,check=True)
    subprocess.run(['npm','run','build'],cwd=frontend,check=True)
    if not a.mock_agents:
        endpoint=os.environ.get('PLUMESENSE_API_URL','http://127.0.0.1:18000/predict');ensure_tunnel(endpoint)
        with urlopen(endpoint.rsplit('/',1)[0]+'/health',timeout=15) as response:print('GPU API:',response.read().decode(),flush=True)
    command=[sys.executable,'-m','server.app','--host','127.0.0.1','--port',str(a.port),'--seed',str(a.seed)]
    if a.mock_agents:command.append('--mock-agents')
    try:subprocess.run(command,cwd=root,check=True)
    except KeyboardInterrupt:pass
