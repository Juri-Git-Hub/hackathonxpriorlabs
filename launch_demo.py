"""Build the Three.js frontend and launch SeeCH4 (Triple-TabPFN autonomous methane-leak drones)."""

import argparse
import socket
import subprocess
import sys
from pathlib import Path

from agents.tabpfn_remote import check_tabpfn_backend

DEFAULT_ADAPTIVE_WIND = False


def _pick_free_port(host: str, preferred: int, max_tries: int = 20) -> int:
    for port in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return preferred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Three.js frontend and launch SeeCH4 Triple-TabPFN methane-leak detection drones."
    )
    parser.add_argument("--port", type=int, default=8501, help="HTTP/WebSocket port (default: 8501)")
    parser.add_argument("--seed", type=int, default=42, help="Simulation RNG seed")
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=None,
        help="Seed for a connected procedural facility layout (default: random)",
    )
    parser.add_argument(
        "--adaptive-wind",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_WIND,
        help="Refresh acquisition targets after sustained changes in measured wind",
    )
    parser.add_argument("--source-pre-data", type=Path, help="Alternate pre-contact TabPFN context")
    parser.add_argument("--source-post-data", type=Path, help="Alternate post-contact TabPFN context")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    frontend = root / "frontend"
    if not (frontend / "node_modules").exists():
        subprocess.run(["npm", "ci"], cwd=frontend, check=True)
    subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)

    print("TabPFN backend:", check_tabpfn_backend(), flush=True)

    host = "127.0.0.1"
    port = _pick_free_port(host, args.port)
    if port != args.port:
        print(f"Port {args.port} is already in use; using http://{host}:{port} instead.", flush=True)
    else:
        print(f"Serving SeeCH4 at http://{host}:{port}", flush=True)

    command = [
        sys.executable,
        "-m",
        "server.app",
        "--host",
        host,
        "--port",
        str(port),
        "--seed",
        str(args.seed),
    ]
    for flag, path in (
        ("--source-pre-data", args.source_pre_data),
        ("--source-post-data", args.source_post_data),
    ):
        if path is not None:
            command.extend([flag, str(path.resolve())])
    if args.layout_seed is not None:
        command.extend(["--layout-seed", str(args.layout_seed)])
    command.append("--adaptive-wind" if args.adaptive_wind else "--no-adaptive-wind")
    try:
        subprocess.run(command, cwd=root, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
