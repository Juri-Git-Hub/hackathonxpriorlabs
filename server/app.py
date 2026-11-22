"""FastAPI WebSocket broadcaster & judge control handler for Drone Gas Leak Detection.

Run the TabPFN simulation server:
    python -m server.app --port 8501
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import random
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from agents.base import BaseAgent
from agents.mock_agent import MockTabPFNAgent
from shared.schemas import BackendStatusPayload, JudgeCommand
from sim.engine import SimEngine

logger = logging.getLogger("gas_leak_server")

DEFAULT_ADAPTIVE_WIND = False


def build_agents(
    mock_agents: bool = False,
    adaptive_wind: bool = DEFAULT_ADAPTIVE_WIND,
    *,
    source_pre_data: Path | None = None,
    source_post_data: Path | None = None,
) -> dict[str, BaseAgent]:
    """Instantiate the primary TabPFN mission drone (or mock agent in unit tests)."""
    if mock_agents:
        primary: BaseAgent = MockTabPFNAgent(drone_id="tabpfn")
    else:
        from agents.reliable_tabpfn import ReliableTabPFNAgent

        primary = ReliableTabPFNAgent(
            drone_id="tabpfn",
            adaptive_wind=adaptive_wind,
            source_pre_data_path=source_pre_data,
            source_post_data_path=source_post_data,
        )
    return {"tabpfn": primary}


def create_app(
    mock_agents: bool = False,
    seed: int = 42,
    adaptive_wind: bool = DEFAULT_ADAPTIVE_WIND,
    layout_seed: int | None = None,
    *,
    source_pre_data: Path | None = None,
    source_post_data: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    agents = build_agents(
        mock_agents=mock_agents,
        adaptive_wind=adaptive_wind,
        source_pre_data=source_pre_data,
        source_post_data=source_post_data,
    )
    if layout_seed is None:
        layout_seed = random.randrange(2**31)
    engine = SimEngine(
        agents=agents,
        seed=seed,
        layout_seed=layout_seed,
        responsive_sensor=False,
        randomize_initial_maps=True,
    )
    clients: set[WebSocket] = set()
    latest_beliefs = {}
    stream_lock = asyncio.Lock()

    cached_wind_key: tuple[int, int] = (-1, -1)
    cached_wind_json = ""

    def get_cached_wind_field_json() -> str:
        nonlocal cached_wind_key, cached_wind_json
        key = (id(engine.airflow), engine.airflow._field_version)
        if key != cached_wind_key:
            cached_wind_json = engine.get_wind_field().model_dump_json()
            cached_wind_key = key
        return cached_wind_json

    async def broadcast_json(payload_text: str) -> None:
        if not clients:
            return
        client_list = list(clients)
        if len(client_list) == 1:
            ws = client_list[0]
            try:
                await ws.send_text(payload_text)
            except Exception:
                clients.discard(ws)
            return
        results = await asyncio.gather(
            *(ws.send_text(payload_text) for ws in client_list),
            return_exceptions=True,
        )
        for ws, res in zip(client_list, results):
            if isinstance(res, Exception):
                clients.discard(ws)

    async def sim_loop() -> None:
        dt = engine.DT
        status_at = 0.0
        while True:
            t0 = time.perf_counter()
            async with stream_lock:
                has_clients = bool(clients)
                tick = engine.step(dt, include_puffs=has_clients)
                if has_clients:
                    await broadcast_json(tick.model_dump_json())
                    for belief in engine.pop_belief_updates():
                        latest_beliefs[belief.drone_id] = belief
                        await broadcast_json(belief.model_dump_json())
                else:
                    for belief in engine.pop_belief_updates():
                        latest_beliefs[belief.drone_id] = belief

                if has_clients and time.monotonic() >= status_at:
                    await broadcast_json(backend_status().model_dump_json())
                    status_at = time.monotonic() + 1
            elapsed = time.perf_counter() - t0
            sleep_for = max(0.0, dt - elapsed)
            await asyncio.sleep(sleep_for)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(sim_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            for agent in agents.values():
                close_fn = getattr(agent, "close", None)
                if callable(close_fn):
                    close_fn()

    app = FastAPI(title="Drone Gas Leak Detection Server", lifespan=lifespan)
    app.state.engine = engine
    app.state.mock_agents = mock_agents

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def backend_status() -> BackendStatusPayload:
        agent = agents["tabpfn"]
        error = getattr(agent, "error", None)
        models = getattr(agent, "models", None)
        return BackendStatusPayload(
            sim_time=engine.sim_time,
            mode="mock" if mock_agents else "tabpfn-api",
            state=(
                "mock"
                if mock_agents
                else "error"
                if error
                else "ready"
                if getattr(agent, "source_scores", None) is not None
                else "pending"
            ),
            error=error,
            inference_seconds=getattr(agent, "latency", 0),
            measurement_count=len(getattr(agent, "history", [])),
            exposure_ppm_seconds=getattr(agent, "exposure", 0),
            paused=engine.paused,
            active_leaks=len(engine.leaks),
            battery_pct=engine.drones["tabpfn"].battery_pct,
            mission_report=engine.mission_report,
            mission_state=(
                "found"
                if engine.mission_report is not None
                else "no_leak"
                if not engine.leaks
                else "depleted"
                if engine.drones["tabpfn"].status == "depleted"
                else "paused"
                if engine.paused
                else "searching"
            ),
            search_phase=getattr(agent, "search_phase", "search"),
            completed_probes=getattr(agent, "completed_probes", 0),
            source_features=len(models[0].columns) if models else 0,
            hazard_features=len(models[1].columns) if models else 0,
            navigation_basis=getattr(agent, "decision_basis", "unspecified"),
            prediction_decisions=getattr(agent, "prediction_decisions", 0),
            source_prediction_calls=getattr(agent, "source_prediction_calls", 0),
            hazard_prediction_calls=getattr(agent, "hazard_prediction_calls", 0),
            adaptive_wind=getattr(agent, "adaptive_wind", False),
            wind_replans=getattr(agent, "wind_replans", 0),
            plan_state=getattr(agent, "plan_state", "mock" if mock_agents else "preparing"),
            plan_revision=getattr(agent, "plan_revision", 0),
            search_candidate_count=getattr(agent, "search_candidate_count", 0),
            planned_route=getattr(agent, "planned_route", []),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mock_agents": mock_agents,
            "connected_clients": len(clients),
            "sim_time": round(engine.sim_time, 3),
            "inference_error": getattr(agents["tabpfn"], "error", None),
            "layout_seed": engine.world.layout_seed,
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            async with stream_lock:
                await websocket.send_text(engine.get_world_init().model_dump_json())
                await websocket.send_text(backend_status().model_dump_json())
                for belief in latest_beliefs.values():
                    await websocket.send_text(belief.model_dump_json())
                await websocket.send_text(get_cached_wind_field_json())
                clients.add(websocket)

            while True:
                raw_text = await websocket.receive_text()
                try:
                    cmd = JudgeCommand.model_validate_json(raw_text)
                except ValidationError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue
                async with stream_lock:
                    previous_world = engine.world
                    if cmd.action == "randomize_layout" and cmd.seed is None:
                        cmd = cmd.model_copy(update={"seed": random.randrange(2**31)})
                    try:
                        engine.apply_command(cmd)
                    except ValueError as exc:
                        await websocket.send_json({"type": "error", "detail": str(exc)})
                        continue
                    if (
                        engine.world is not previous_world
                        or cmd.action == "reset_race"
                        or (cmd.action == "place_leak" and cmd.replace_existing)
                    ):
                        latest_beliefs.clear()
                        await broadcast_json(engine.get_world_init().model_dump_json())
                    await broadcast_json(backend_status().model_dump_json())
                    if engine.world is not previous_world or cmd.action in ("set_wind", "reset_race"):
                        await broadcast_json(get_cached_wind_field_json())
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone Gas Leak Detection WebSocket Server")
    parser.add_argument("--source-pre-data", type=Path, help="Alternate pre-contact TabPFN context.")
    parser.add_argument("--source-post-data", type=Path, help="Alternate post-contact TabPFN context.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--seed", type=int, default=42, help="Simulation RNG seed")
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=None,
        help="Seed for connected procedural facility layout (default: random)",
    )
    parser.add_argument(
        "--adaptive-wind",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_WIND,
        help="Refresh acquisition targets after sustained changes in measured wind",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()
    app = create_app(
        seed=args.seed,
        adaptive_wind=args.adaptive_wind,
        layout_seed=args.layout_seed,
        source_pre_data=args.source_pre_data,
        source_post_data=args.source_post_data,
    )
    uvicorn.run(app, host=args.host, port=args.port)
