"""Unified TabPFNRegressor backend adapter.

Supports three backends, selectable via ``TABPFN_BACKEND`` (default: ``auto``):

1. ``client`` — Prior Labs free hosted TabPFN API (``pip install tabpfn-client``).
   Authenticate once with ``python -c "from tabpfn_client import interactive_login; interactive_login()"``
   or export ``TABPFN_TOKEN`` from https://platform.priorlabs.ai/account/api-keys. No local GPU required.
2. ``local``  — Local ``TabPFNRegressor`` on your own computer (``pip install tabpfn``).
   Uses CUDA/MPS/CPU automatically (override with ``TABPFN_DEVICE=cuda|mps|cpu``).
3. ``http``   — Optional custom self-hosted HTTP ``/regress`` endpoint via
   ``TABPFN_API_URL`` (e.g. ``http://127.0.0.1:8001/regress``).

In ``auto`` mode, the backend is chosen in order:
``TABPFN_API_URL`` (if set) -> ``tabpfn_client`` -> ``tabpfn`` (local).
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
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
    """Ensure local SSH port forward to a remote TabPFNRegressor service is open."""
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

_INSTALL_HELP = (
    "No TabPFN backend is available. Choose one of:\n"
    "  1) Free Prior Labs hosted API (no GPU required):\n"
    "       pip install tabpfn-client\n"
    '       python -c "from tabpfn_client import interactive_login; interactive_login()"\n'
    "     (or set TABPFN_TOKEN from https://platform.priorlabs.ai/account/api-keys)\n"
    "  2) Local on your own computer (GPU or CPU):\n"
    "       pip install tabpfn\n"
    "  3) Custom HTTP endpoint:\n"
    "       export TABPFN_API_URL=http://<host>:<port>/regress"
)


def resolve_backend(api_url: str | None = None, backend: str | None = None) -> str:
    """Determine which TabPFN backend to use: 'client', 'local', or 'http'."""
    chosen = (backend or os.environ.get("TABPFN_BACKEND", "auto")).strip().lower()
    explicit_url = api_url or os.environ.get("TABPFN_API_URL")
    explicit_ssh = bool(os.environ.get("TABPFN_SSH_HOST"))
    if chosen == "ssh":
        return "http"
    if chosen in ("client", "local", "http"):
        return chosen
    if chosen != "auto":
        raise RuntimeError(
            f"Unknown TABPFN_BACKEND={chosen!r}. Expected 'auto', 'client', 'local', 'http', or 'ssh'."
        )
    if explicit_url or explicit_ssh:
        return "http"
    if importlib.util.find_spec("tabpfn_client") is not None:
        return "client"
    if importlib.util.find_spec("tabpfn") is not None:
        return "local"
    raise RuntimeError(_INSTALL_HELP)


def check_tabpfn_backend(api_url: str | None = None) -> str:
    """Validate that a TabPFN backend is configured and return a status summary."""
    backend = resolve_backend(api_url=api_url)
    if backend == "http":
        url = api_url or os.environ.get("TABPFN_API_URL") or DEFAULT_API_URL
        ensure_tunnel(url)
        health_url = url.rsplit("/", 1)[0] + "/health"
        with urlopen(health_url, timeout=15) as resp:
            return f"http ({url}): {resp.read().decode().strip()}"
    if backend == "client":
        token = os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_API_KEY")
        if token:
            import tabpfn_client  # type: ignore

            if hasattr(tabpfn_client, "set_access_token"):
                tabpfn_client.set_access_token(token)
        return "client (Prior Labs hosted TabPFN API via tabpfn-client)"
    device = os.environ.get("TABPFN_DEVICE", "auto")
    return f"local (tabpfn on device={device})"


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


def _to_tabular(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> Any:
    """Convert row sequences to a pandas DataFrame (if installed) or float32 ndarray."""
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(rows, columns=list(columns))
    except ImportError:
        arr = np.empty((len(rows), len(columns)), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                arr[i, j] = np.nan if cell is None else float(cell)
        return arr


def _normalize_quantiles(raw: Any, n_quantiles: int, n_samples: int) -> list[np.ndarray]:
    """Normalize TabPFN quantile output to a list of ``n_quantiles`` 1D float64 arrays."""
    if isinstance(raw, (list, tuple)):
        if len(raw) == n_quantiles:
            return [np.asarray(q, dtype=np.float64).reshape(-1) for q in raw]
        arr = np.asarray(raw, dtype=np.float64)
    else:
        arr = np.asarray(raw, dtype=np.float64)

    if arr.ndim == 2:
        if arr.shape == (n_quantiles, n_samples):
            return [arr[i] for i in range(n_quantiles)]
        if arr.shape == (n_samples, n_quantiles):
            return [arr[:, i] for i in range(n_quantiles)]
    if arr.ndim == 1 and n_quantiles == 1:
        return [arr]
    raise RuntimeError(
        f"Unexpected quantile output shape {arr.shape} for {n_quantiles} quantiles and {n_samples} samples"
    )


class RemoteTabPFNRegressor:
    """TabPFNRegressor adapter supporting Prior Labs API, local execution, or custom HTTP."""

    def __init__(
        self,
        api_url: str | None = None,
        n_estimators: int = 4,
        timeout_s: float = 30.0,
        backend: str | None = None,
    ) -> None:
        self.api_url = api_url or os.environ.get("TABPFN_API_URL")
        self.backend = resolve_backend(api_url=self.api_url, backend=backend)
        if self.backend == "http" and not self.api_url:
            self.api_url = DEFAULT_API_URL
        self.n_estimators = max(1, min(8, int(n_estimators)))
        self.timeout_s = float(timeout_s)
        self.last_elapsed_s: float = 0.0
        self.last_fit_reused: bool = False
        self._registered_contexts: dict[str, str] = {}
        self._fitted_models: dict[str, tuple[str, Any]] = {}
        self._lock = threading.Lock()
        self._token_applied = False

    def health(self) -> dict[str, Any]:
        if self.backend == "http":
            assert self.api_url is not None
            ensure_tunnel(self.api_url)
            health_url = self.api_url.rsplit("/", 1)[0] + "/health"
            req = Request(health_url, method="GET")
            with urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read())
        return {"status": "ok", "backend": self.backend}

    def _create_sdk_model(self) -> Any:
        if self.backend == "client":
            import tabpfn_client  # type: ignore

            if not self._token_applied:
                token = os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_API_KEY")
                if token and hasattr(tabpfn_client, "set_access_token"):
                    tabpfn_client.set_access_token(token)
                self._token_applied = True
            return tabpfn_client.TabPFNRegressor(n_estimators=self.n_estimators)

        from tabpfn import TabPFNRegressor  # type: ignore

        device = os.environ.get("TABPFN_DEVICE", "auto")
        return TabPFNRegressor(device=device, n_estimators=self.n_estimators)

    def _predict_sdk(
        self,
        columns: Sequence[str],
        x_train_rows: Sequence[Sequence[Any]],
        y_train: Sequence[float],
        x_test_rows: Sequence[Sequence[Any]],
        quantiles: Sequence[float],
        cache_key: str | None,
        data_hash: str | None,
        return_mean: bool,
    ) -> tuple[np.ndarray | None, list[np.ndarray]]:
        t0 = time.perf_counter()
        sig_hash = data_hash
        if sig_hash is None and cache_key is not None:
            safe_tx = [[_json_safe_cell(c) for c in row] for row in x_train_rows]
            safe_ty = [float(y) for y in y_train]
            sig = json.dumps(
                [list(columns), self.n_estimators, safe_tx, safe_ty],
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            sig_hash = hashlib.sha256(sig).hexdigest()

        with self._lock:
            cached = self._fitted_models.get(cache_key) if cache_key is not None else None
            if cached is not None and sig_hash is not None and cached[0] == sig_hash:
                model = cached[1]
                reused = True
            else:
                model = self._create_sdk_model()
                x_tr = _to_tabular(columns, x_train_rows)
                y_tr = np.asarray(y_train, dtype=np.float64)
                model.fit(x_tr, y_tr)
                reused = False
                if cache_key is not None and sig_hash is not None:
                    self._fitted_models[cache_key] = (sig_hash, model)

            x_te = _to_tabular(columns, x_test_rows)
            q_list = [float(q) for q in quantiles]
            raw_q = model.predict(x_te, output_type="quantiles", quantiles=q_list)
            q_arrs = _normalize_quantiles(raw_q, len(q_list), len(x_test_rows))
            mean_arr: np.ndarray | None = None
            if return_mean:
                raw_mean = model.predict(x_te, output_type="mean")
                mean_arr = np.asarray(raw_mean, dtype=np.float64).reshape(-1)

        self.last_elapsed_s = time.perf_counter() - t0
        self.last_fit_reused = reused
        return mean_arr, q_arrs

    def _predict_http(
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
        assert self.api_url is not None
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
                        time.sleep(0.25 * (attempt + 1))
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
            if (
                cache_key is not None
                and result.get("cache_reference_protocol") == 1
                and computed_hash is not None
            ):
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
        if self.backend != "http":
            _, q_arrs = self._predict_sdk(
                columns,
                x_train_rows,
                y_train,
                x_test_rows,
                quantiles=quantiles,
                cache_key=cache_key,
                data_hash=data_hash,
                return_mean=False,
            )
            return q_arrs
        result = self._predict_http(
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
        if self.backend != "http":
            mean_arr, q_arrs = self._predict_sdk(
                columns,
                x_train_rows,
                y_train,
                x_test_rows,
                quantiles=quantiles,
                cache_key=cache_key,
                data_hash=data_hash,
                return_mean=True,
            )
            assert mean_arr is not None
            return mean_arr, q_arrs
        result = self._predict_http(
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
