"""Start the API tunnel and Streamlit together; keep the tunnel until exit."""
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from tunnel import ensure_tunnel

if __name__ == '__main__':
    endpoint=os.environ.get('PLUMESENSE_API_URL','http://127.0.0.1:18000/predict')
    ensure_tunnel(endpoint)
    with urlopen(endpoint.rsplit('/',1)[0]+'/health',timeout=15) as response:
        print('API:',response.read().decode(),flush=True)
    try:
        subprocess.run([sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('app.py')),'--server.headless=true','--server.port=8501','--browser.gatherUsageStats=false'],check=True)
    except KeyboardInterrupt:
        pass
