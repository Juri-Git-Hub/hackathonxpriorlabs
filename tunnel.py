"""Own only the SSH tunnel started by this Python process."""
import atexit
import os
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

_process = None
_lock = threading.Lock()

def _listening(host, port):
    try:
        with socket.create_connection((host,port),timeout=.3):return True
    except OSError:return False

def _cleanup():
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:_process.wait(timeout=3)
        except subprocess.TimeoutExpired:_process.kill();_process.wait()
    _process=None

atexit.register(_cleanup)

def ensure_tunnel(url='http://127.0.0.1:18000/predict'):
    """Forward local 18000 to my4090:8000; external URLs bypass SSH."""
    global _process
    parsed=urlparse(url)
    if parsed.hostname not in ('localhost','127.0.0.1') or parsed.port!=18000:return
    if os.environ.get('PLUMESENSE_AUTO_TUNNEL','1')=='0':return
    with _lock:
        if _listening('127.0.0.1',18000):return
        if _process is not None and _process.poll() is None:_cleanup()
        host=os.environ.get('PLUMESENSE_SSH_HOST','my4090')
        remote_port=int(os.environ.get('PLUMESENSE_REMOTE_PORT','8000'))
        command=['ssh','-N','-T','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-L',f'127.0.0.1:18000:127.0.0.1:{remote_port}',host]
        _process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if _process.poll() is not None:
                error=_process.stderr.read().strip()
                raise RuntimeError(f'SSH tunnel failed: {error}')
            if _listening('127.0.0.1',18000):return
            time.sleep(.1)
        _cleanup()
        raise RuntimeError('SSH tunnel did not open port 18000 within 15 seconds')
