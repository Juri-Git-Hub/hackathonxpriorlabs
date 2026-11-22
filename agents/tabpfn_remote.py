"""Remote CUDA TabPFNRegressor HTTP client & automatic SSH tunnel manager.

Connects to `tabpfn_gpu_server.py` running on the 4090 host (`viewer:8001`)
via a local SSH port forward (`127.0.0.1:18001 -> 127.0.0.1:8001`).

Environment overrides:
- TABPFN_API_URL:       default "http://127.0.0.1:18001/regress"
- TABPFN_SSH_HOST:      default "viewer"
- TABPFN_REMOTE_PORT:   default "8001"
- TABPFN_AUTO_TUNNEL:   default "1" (set "0" to disable automatic SSH forwarding)
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from tunnel import resolve_ssh_target

DEFAULT_API_URL = "http://127.0.0.1:18001/regress"
DEFAULT_SSH_HOST = "viewer"
DEFAULT_REMOTE_PORT = 8001

_tunnel_proc: subprocess.Popen[str] | None = None
_tunnel_lock = threading.Lock()


def _port_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _cleanup_tunnel() -> None:
    global _tunnel_proc
    if _tunnel_proc is not None and _tunnel_proc.poll() is None:
        _tunnel_proc.terminate()
        try:
            _tunnel_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _tunnel_proc.kill()
            _tunnel_proc.wait()
    _tunnel_proc = None


atexit.register(_cleanup_tunnel)


def ensure_tunnel(url: str = DEFAULT_API_URL) -> None:
    """Ensure local port forward to the 4090 TabPFNRegressor service is open."""
    global _tunnel_proc
    parsed = urlparse(url)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return
    if os.environ.get("TABPFN_AUTO_TUNNEL", "1") == "0":
        return

    local_port = parsed.port or 18001
    with _tunnel_lock:
        if _port_listening("127.0.0.1", local_port):
            return
        if _tunnel_proc is not None and _tunnel_proc.poll() is None:
            _cleanup_tunnel()

        ssh_host = os.environ.get("TABPFN_SSH_HOST", DEFAULT_SSH_HOST)
        remote_port = int(os.environ.get("TABPFN_REMOTE_PORT", str(DEFAULT_REMOTE_PORT)))
        extra_args, target = resolve_ssh_target(ssh_host)
        cmd = [
            "ssh",
            "-N",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            *extra_args,
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            target,
        ]
        _tunnel_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if _tunnel_proc.poll() is not None:
                err = (_tunnel_proc.stderr.read() if _tunnel_proc.stderr else "").strip()
                _cleanup_tunnel()
                raise RuntimeError(f"SSH tunnel to {ssh_host}:{remote_port} failed: {err}")
            if _port_listening("127.0.0.1", local_port):
                return
            time.sleep(0.1)

        _cleanup_tunnel()
        raise RuntimeError(
            f"SSH tunnel to {ssh_host}:{remote_port} did not open port {local_port} within 12s"
        )


def _json_safe_cell(v: Any) -> Any:
    """Convert numpy scalars and NaN/inf to JSON-safe Python values (`null` for NaN)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    return v


class RemoteTabPFNRegressor:
    """Zero-local-dependency HTTP client for the remote CUDA TabPFNRegressor service."""

    def __init__(
        self,
        api_url: str | None = None,
        n_estimators: int = 4,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_url = api_url or os.environ.get("TABPFN_API_URL", DEFAULT_API_URL)
        self.n_estimators = max(1, min(8, int(n_estimators)))
        self.timeout_s = float(timeout_s)
        self.last_elapsed_s: float = 0.0
        self.last_fit_reused: bool = False
        self._registered_contexts: dict[str, str] = {}

    def health(self) -> dict[str, Any]:
        ensure_tunnel(self.api_url)
        health_url = self.api_url.rsplit("/", 1)[0] + "/health"
        req = Request(health_url, method="GET")
        with urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read())

    def _predict_raw(
        self,
        columns: Sequence[str],
        x_train_rows: Sequence[Sequence[Any]],
        y_train: Sequence[float],
        x_test_rows: Sequence[Sequence[Any]],
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        cache_key: str | None = None,
        data_hash: str | None = None,
        return_mean: bool = False,
    ) -> dict[str, Any]:
        ensure_tunnel(self.api_url)
        safe_x_test = [[_json_safe_cell(c) for c in row] for row in x_test_rows]
        payload: dict[str, Any] = {
            "columns": list(columns),
            "X_test": safe_x_test,
            "quantiles": [float(q) for q in quantiles],
            "n_estimators": self.n_estimators,
        }
        if return_mean:
            payload["return_mean"] = True

        computed_hash: str | None = data_hash
        safe_x_train: list[list[Any]] | None = None
        safe_y_train: list[float] | None = None

        def _materialize_train() -> tuple[list[list[Any]], list[float], str]:
            nonlocal safe_x_train, safe_y_train, computed_hash
            if safe_x_train is None:
                safe_x_train = [[_json_safe_cell(c) for c in row] for row in x_train_rows]
                safe_y_train = [float(y) for y in y_train]
                if computed_hash is None:
                    sig = json.dumps(
                        [payload["columns"], self.n_estimators, safe_x_train, safe_y_train],
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    computed_hash = hashlib.sha256(sig).hexdigest()
            assert computed_hash is not None
            return safe_x_train, safe_y_train, computed_hash

        if cache_key is not None:
            payload["cache_key"] = cache_key
            if computed_hash is not None and self._registered_contexts.get(cache_key) == computed_hash:
                payload["data_hash"] = computed_hash
            else:
                tx, ty, dh = _materialize_train()
                if self._registered_contexts.get(cache_key) == dh:
                    payload["data_hash"] = dh
                else:
                    payload["X_train"] = tx
                    payload["y_train"] = ty
                    payload["data_hash"] = dh
        else:
            payload["X_train"] = [[_json_safe_cell(c) for c in row] for row in x_train_rows]
            payload["y_train"] = [float(y) for y in y_train]

        def send(data: dict[str, Any]) -> dict[str, Any]:
            import http.client
            import time as _time
            body = json.dumps(data, allow_nan=False).encode("utf-8")
            last_exc: Exception | None = None
            for attempt in range(3):
                req = Request(
                    self.api_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(req, timeout=self.timeout_s) as resp:
                        return json.loads(resp.read())
                except (http.client.HTTPException, ConnectionResetError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < 2:
                        _time.sleep(0.25 * (attempt + 1))
                        continue
                    raise
            raise last_exc  # pragma: no cover

        try:
            try:
                result = send(payload)
            except HTTPError as exc:
                if exc.code == 409 and cache_key is not None and "X_train" not in payload:
                    tx, ty, dh = _materialize_train()
                    full_payload = dict(payload)
                    full_payload["X_train"] = tx
                    full_payload["y_train"] = ty
                    full_payload["data_hash"] = dh
                    result = send(full_payload)
                else:
                    raise
            if cache_key is not None and result.get("cache_reference_protocol") == 1 and computed_hash is not None:
                self._registered_contexts[cache_key] = computed_hash
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"TabPFN API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"TabPFN API unreachable at {self.api_url}: {exc.reason}"
            ) from exc

        self.last_elapsed_s = float(result.get("elapsed_seconds", 0.0))
        self.last_fit_reused = bool(result.get("fit_reused", False))
        raw_q = result["quantiles"]
        if len(raw_q) != len(quantiles):
            raise RuntimeError(
                f"Expected {len(quantiles)} quantile arrays, got {len(raw_q)}"
            )
        return result

    def predict_quantiles(
        self,
        columns: Sequence[str],
        x_train_rows: Sequence[Sequence[Any]],
        y_train: Sequence[float],
        x_test_rows: Sequence[Sequence[Any]],
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        cache_key: str | None = None,
        data_hash: str | None = None,
    ) -> list[np.ndarray]:
        result = self._predict_raw(
            columns,
            x_train_rows,
            y_train,
            x_test_rows,
            quantiles=quantiles,
            cache_key=cache_key,
            data_hash=data_hash,
            return_mean=False,
        )
        return [np.asarray(q, dtype=np.float64) for q in result["quantiles"]]

    def predict_mean_and_quantiles(
        self,
        columns: Sequence[str],
        x_train_rows: Sequence[Sequence[Any]],
        y_train: Sequence[float],
        x_test_rows: Sequence[Sequence[Any]],
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        cache_key: str | None = None,
        data_hash: str | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        result = self._predict_raw(
            columns,
            x_train_rows,
            y_train,
            x_test_rows,
            quantiles=quantiles,
            cache_key=cache_key,
            data_hash=data_hash,
            return_mean=True,
        )
        mean_arr = np.asarray(result["mean"], dtype=np.float64)
        q_arrs = [np.asarray(q, dtype=np.float64) for q in result["quantiles"]]
        return mean_arr, q_arrs
