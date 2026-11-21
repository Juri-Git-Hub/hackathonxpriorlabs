"""FastAPI WebSocket broadcaster & judge control handler for Drone Gas Leak Detection.

Run locally in mock mode (no TabPFN weights required):
    python -m server.app --mock-agents

Run with real TabPFN agent:
    python -m server.app
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from agents.base import BaseAgent
from agents.mock_agent import MockTabPFNAgent
from integration.wind_search import DEFAULT_ADAPTIVE_WIND
from agents.plumesense_agent import PlumeSenseAgent
from shared.schemas import JudgeCommand, BackendStatusPayload
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sim.engine import SimEngine

logger = logging.getLogger("gas_leak_server")


def build_agents(mock_agents: bool = True,adaptive_wind: bool = DEFAULT_ADAPTIVE_WIND) -> dict[str, BaseAgent]:
    """Instantiate only the TabPFN drone for the live demo."""
    if mock_agents:
        primary: BaseAgent = MockTabPFNAgent(drone_id="tabpfn")
    else:
        primary = PlumeSenseAgent(drone_id="tabpfn",adaptive_wind=adaptive_wind)

    return {
        "tabpfn": primary,
    }


def create_app(mock_agents: bool = True, seed: int = 42,adaptive_wind: bool = DEFAULT_ADAPTIVE_WIND) -> FastAPI:
    """Create and configure the FastAPI application."""
    agents = build_agents(mock_agents=mock_agents,adaptive_wind=adaptive_wind)
    engine = SimEngine(agents=agents, seed=seed)
    clients: set[WebSocket] = set()
    latest_beliefs = {}

    async def broadcast_json(payload_text: str) -> None:
        if not clients:
            return
        dead: list[WebSocket] = []
        for ws in list(clients):
            try:
                await ws.send_text(payload_text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    async def sim_loop() -> None:
        dt = engine.DT
        status_at=0.0
        while True:
            t0 = time.perf_counter()
            tick = engine.step(dt)
            if clients:
                await broadcast_json(tick.model_dump_json())
                for belief in engine.pop_belief_updates():
                    latest_beliefs[belief.drone_id]=belief
                    await broadcast_json(belief.model_dump_json())
            else:
                # Drain pending beliefs when nobody is connected
                for belief in engine.pop_belief_updates():latest_beliefs[belief.drone_id]=belief

            if clients and time.monotonic()>=status_at:
                await broadcast_json(backend_status().model_dump_json())
                status_at=time.monotonic()+1
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

    def backend_status():
        agent=agents['tabpfn'];error=getattr(agent,'error',None)
        models=getattr(agent,'models',None)
        return BackendStatusPayload(sim_time=engine.sim_time,mode='mock' if mock_agents else 'tabpfn-api',state='mock' if mock_agents else 'error' if error else 'ready' if getattr(agent,'source_scores',None) is not None else 'pending',error=error,inference_seconds=getattr(agent,'latency',0),measurement_count=len(getattr(agent,'history',[])),exposure_ppm_seconds=getattr(agent,'exposure',0),paused=engine.paused,active_leaks=len(engine.leaks),battery_pct=engine.drones['tabpfn'].battery_pct,mission_report=engine.mission_report,mission_state='found' if engine.mission_report is not None else 'no_leak' if not engine.leaks else 'depleted' if engine.drones['tabpfn'].status=='depleted' else 'paused' if engine.paused else 'searching',search_phase=getattr(agent,'search_phase','search'),completed_probes=getattr(agent,'completed_probes',0),source_features=len(models[0].columns) if models else 0,hazard_features=len(models[1].columns) if models else 0,navigation_basis=getattr(agent,'decision_basis','unspecified'),prediction_decisions=getattr(agent,'prediction_decisions',0),source_prediction_calls=getattr(agent,'source_prediction_calls',0),hazard_prediction_calls=getattr(agent,'hazard_prediction_calls',0),adaptive_wind=getattr(agent,'adaptive_wind',False),wind_replans=getattr(agent,'wind_replans',0))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mock_agents": mock_agents,
            "connected_clients": len(clients),
            "sim_time": round(engine.sim_time, 3),
            "inference_error": getattr(agents["tabpfn"], "error", None),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        clients.add(websocket)
        try:
            # Immediately send static world geometry on connect
            world_init = engine.get_world_init()
            await websocket.send_text(world_init.model_dump_json())
            await websocket.send_text(backend_status().model_dump_json())
            for belief in list(latest_beliefs.values()):await websocket.send_text(belief.model_dump_json())

            while True:
                raw_text = await websocket.receive_text()
                try:
                    cmd = JudgeCommand.model_validate_json(raw_text)
                except ValidationError as exc:
                    await websocket.send_json(
                        {"type": "error", "detail": str(exc)}
                    )
                    continue
                engine.apply_command(cmd)
                if cmd.action=='reset_race':
                    latest_beliefs.clear()
                    await broadcast_json(engine.get_world_init().model_dump_json())
                await broadcast_json(backend_status().model_dump_json())
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    dist=Path(__file__).resolve().parents[1]/'frontend'/'dist'
    if dist.exists():app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone Gas Leak Detection WebSocket Server")
    parser.add_argument(
        "--mock-agents",
        action="store_true",
        default=os.environ.get("MOCK_AGENTS", "0") == "1",
        help="Use fast MockTabPFNAgent instead of loading TabPFN weights.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--seed", type=int, default=42, help="Simulation RNG seed")
    parser.add_argument("--adaptive-wind",action=argparse.BooleanOptionalAction,default=DEFAULT_ADAPTIVE_WIND,help="Refresh acquisition targets after sustained changes in measured wind")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()
    app = create_app(mock_agents=args.mock_agents, seed=args.seed,adaptive_wind=args.adaptive_wind)
    uvicorn.run(app, host=args.host, port=args.port)
