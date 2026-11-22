"""Own only the SSH tunnel started by this Python process."""
import atexit
import os
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

_process = None
_processes: dict[int, subprocess.Popen[str]] = {}
_lock = threading.Lock()


def _listening(host, port):
    try:
        with socket.create_connection((host, port), timeout=.3):
            return True
    except OSError:
        return False


def _terminate_proc(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _cleanup():
    global _process
    _terminate_proc(_process)
    _process = None
    for proc in list(_processes.values()):
        _terminate_proc(proc)
    _processes.clear()


atexit.register(_cleanup)


def resolve_ssh_target(ssh_host: str) -> tuple[list[str], str]:
    """Avoid inherited LocalForward collisions (e.g. Host viewer) by bypassing matching alias."""
    try:
        res = subprocess.run(
            ['ssh', '-G', ssh_host],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode != 0:
            return [], ssh_host
    except Exception:
        return [], ssh_host

    opts: dict[str, list[str]] = {}
    for line in res.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            opts.setdefault(parts[0].lower(), []).append(parts[1].strip())

    if 'localforward' not in opts:
        return [], ssh_host

    hostname = opts.get('hostname', [ssh_host])[0]
    user = opts.get('user', [''])[0]
    port = opts.get('port', ['22'])[0]
    proxyjump = opts.get('proxyjump', [''])[0]
    extra: list[str] = []
    if proxyjump and proxyjump.lower() != 'none':
        extra.extend(['-J', proxyjump])
    if port and port != '22':
        extra.extend(['-p', port])
    target = f'{user}@{hostname}' if user else hostname
    if target == ssh_host:
        extra.extend(['-o', 'ExitOnForwardFailure=no'])
    return extra, target


def ensure_tunnel(url='http://127.0.0.1:18000/predict'):
    """Forward local 18000/18001 to viewer:8000/8001; external URLs bypass SSH."""
    global _process
    parsed = urlparse(url)
    if parsed.hostname not in ('localhost', '127.0.0.1') or parsed.port not in (18000, 18001):
        return
    if os.environ.get('PLUMESENSE_AUTO_TUNNEL', '1') == '0':
        return
    local_port = int(parsed.port)
    with _lock:
        if _listening('127.0.0.1', local_port):
            return
        existing = _processes.pop(local_port, None)
        _terminate_proc(existing)
        if _process is not None and _process.poll() is not None:
            _process = None
        host = os.environ.get('PLUMESENSE_SSH_HOST', 'viewer')
        default_remote = 8001 if local_port == 18001 else 8000
        remote_port = int(os.environ.get('PLUMESENSE_REMOTE_PORT', str(default_remote)))
        extra_args, target = resolve_ssh_target(host)
        command = [
            'ssh', '-N', '-T',
            '-o', 'BatchMode=yes',
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3',
            *extra_args,
            '-L', f'127.0.0.1:{local_port}:127.0.0.1:{remote_port}',
            target,
        ]
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        _processes[local_port] = proc
        _process = proc
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                error = proc.stderr.read().strip() if proc.stderr else ''
                _processes.pop(local_port, None)
                if _process is proc:
                    _process = None
                raise RuntimeError(f'SSH tunnel failed: {error}')
            if _listening('127.0.0.1', local_port):
                return
            time.sleep(.1)
        _processes.pop(local_port, None)
        _terminate_proc(proc)
        if _process is proc:
            _process = None
        raise RuntimeError(f'SSH tunnel did not open port {local_port} within 15 seconds')
