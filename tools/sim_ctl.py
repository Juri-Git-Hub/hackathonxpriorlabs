"""Unified CLI controller for BOTH headless and live (non-headless) simulations.

Designed for both humans and AI agents to inspect, debug, benchmark, and control
the simulation from the command line with machine-readable JSON output.

Examples:
    # Physics realism audit
    python -m tools.sim_ctl audit --json

    # List standardized scenarios
    python -m tools.sim_ctl scenarios

    # Headless run with mid-run scheduled wind shift & trace export
    python -m tools.sim_ctl headless run --scenario wake_trap --duration 45 \\
        --wind-at 15:90:5.0 --trace-out /tmp/trace.jsonl --json

    # Instant 3D spatial physics probe (wind, wake, turbulence, spatial context, ppm)
    python -m tools.sim_ctl headless probe --pos 18,24,5 --wind-angle 0 --wind-speed 4.0

    # Control a live running server (non-headless / 3D frontend connected)
    python -m tools.sim_ctl live health
    python -m tools.sim_ctl live status
    python -m tools.sim_ctl live cmd set-wind --angle 90 --speed 5.5
    python -m tools.sim_ctl live cmd place-leak --pos 12,-5,4
    python -m tools.sim_ctl live cmd toggle-dropout --state on
    python -m tools.sim_ctl live scenario wake_trap
    python -m tools.sim_ctl live watch --ticks 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np

from shared.schemas import JudgeCommand, LeakSource
from sim.airflow import Airflow
from sim.plume import Plume
from sim.world import World
from tools.eval_runner import run_single_eval
from tools.physics_audit import run_physics_audit
from tools.scenarios import SCENARIOS, get_scenario, list_scenario_names


def _parse_vec3(raw: str) -> tuple[float, float, float]:
    parts = [float(p.strip()) for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'x,y,z' (3 floats), got: {raw}")
    return (parts[0], parts[1], parts[2])


def _parse_wind_event(raw: str) -> tuple[float, JudgeCommand]:
    """Parse 'time:angle_deg:speed' into (at_time, JudgeCommand)."""
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected 'time:angle:speed', got: {raw}")
    t_s, ang, spd = float(parts[0]), float(parts[1]), float(parts[2])
    return (t_s, JudgeCommand(action="set_wind", wind_angle_deg=ang, wind_speed=spd))


def _parse_dropout_event(raw: str) -> tuple[float, JudgeCommand]:
    """Parse 'time:on|off' into (at_time, JudgeCommand)."""
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'time:on|off', got: {raw}")
    t_s = float(parts[0])
    state = parts[1].strip().lower() in ("on", "true", "1")
    return (t_s, JudgeCommand(action="toggle_dropout", dropout_enabled=state))


# ---------------------------------------------------------------------------
# Headless commands
# ---------------------------------------------------------------------------


def _cmd_headless_run(args: argparse.Namespace) -> int:
    spec = get_scenario(args.scenario)
    extra_events: list[tuple[float, JudgeCommand]] = []
    for raw in args.wind_at or []:
        extra_events.append(_parse_wind_event(raw))
    for raw in args.dropout_at or []:
        extra_events.append(_parse_dropout_event(raw))
    for raw in args.place_leak or []:
        extra_events.append((0.0, JudgeCommand(action="place_leak", leak_pos=_parse_vec3(raw))))

    agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]
    result = run_single_eval(
        scenario=spec,
        seed=args.seed,
        duration_s=args.duration,
        agent_names=agent_names,
        use_real_tabpfn=args.real_tabpfn,
        extra_events=extra_events,
        trace_out=args.trace_out,
        trace_stride=args.trace_stride,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_headless_probe(args: argparse.Namespace) -> int:
    pos = _parse_vec3(args.pos)
    world = World()
    airflow = Airflow(
        world,
        base_wind_angle_deg=args.wind_angle,
        base_wind_speed=args.wind_speed,
        sigma_w=0.0 if args.deterministic else 0.6,
        seed=args.seed,
    )
    wind_local, turb_mult = airflow.sample_wind_and_turbulence(np.array(pos), compute_turb=True)
    ctx = world.query_spatial_context(pos, airflow.current_wind_vec())

    ppm = None
    if args.leak:
        leak_pos = _parse_vec3(args.leak)
        plume = Plume(world, airflow, seed=args.seed + 100)
        leaks = [LeakSource(id="probe_leak", pos=leak_pos, emission_rate=1.0, buoyancy_vz=0.2)]
        for _ in range(int(args.warmup / 0.05)):
            airflow.step(0.05)
            plume.step(0.05, leaks)
        ppm = round(plume.sample_concentration(pos), 4)

    out = {
        "pos": list(pos),
        "inside_building": world.is_inside_building(pos, margin=0.0),
        "inside_safety_margin_2m": world.is_inside_building(pos, margin=2.0),
        "height_above_surface_m": round(world.height_above_surface(*pos), 3),
        "ambient_wind_vec": [round(v, 3) for v in airflow.current_wind_vec()],
        "local_wind_3d_mps": [round(float(v), 3) for v in wind_local],
        "local_wind_speed_mps": round(float(np.linalg.norm(wind_local)), 3),
        "turbulence_multiplier": round(float(turb_mult), 3),
        "spatial_context": ctx,
        "sampled_ppm": ppm,
    }
    print(json.dumps(out, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Live (Non-Headless WebSocket + HTTP) commands
# ---------------------------------------------------------------------------


def _cmd_live_health(args: argparse.Namespace) -> int:
    url = args.host.rstrip("/") + "/health"
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print(json.dumps(body, indent=2))
    return 0


async def _ws_send_and_read(
    ws_url: str,
    commands: list[JudgeCommand],
    wait_ticks: int = 1,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    import websockets

    collected: list[dict[str, Any]] = []
    async with websockets.connect(ws_url, open_timeout=5.0) as ws:
        # Consume initial WorldInitPayload
        init_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        init_msg = json.loads(init_raw)

        for cmd in commands:
            await ws.send(cmd.model_dump_json())

        ticks_seen = 0
        out_fp = out_path.open("w", encoding="utf-8") if out_path is not None else None
        try:
            while ticks_seen < wait_ticks:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg.get("type") == "sim_tick":
                    ticks_seen += 1
                    summary = {
                        "sim_time": round(float(msg["sim_time"]), 3),
                        "wind_vec": [round(float(v), 3) for v in msg["wind_vec"]],
                        "sensor_dropout_active": bool(msg["sensor_dropout_active"]),
                        "n_puffs": len(msg.get("puffs", [])),
                        "leaks": msg.get("leaks", []),
                        "drones": msg.get("drones", []),
                    }
                    collected.append(summary)
                    if out_fp is not None:
                        out_fp.write(json.dumps(summary) + "\n")
        finally:
            if out_fp is not None:
                out_fp.close()

    if not collected:
        return [{"world_init_buildings": len(init_msg.get("buildings", []))}]
    return collected


def _cmd_live_status(args: argparse.Namespace) -> int:
    ticks = asyncio.run(_ws_send_and_read(args.ws_url, commands=[], wait_ticks=1))
    print(json.dumps(ticks[-1], indent=2))
    return 0


def _cmd_live_watch(args: argparse.Namespace) -> int:
    ticks = asyncio.run(
        _ws_send_and_read(
            args.ws_url,
            commands=[],
            wait_ticks=max(1, args.ticks),
            out_path=args.out,
        )
    )
    if args.out is not None:
        print(json.dumps({"written_ticks": len(ticks), "path": str(args.out)}, indent=2))
    else:
        for t in ticks:
            print(json.dumps(t))
    return 0


def _cmd_live_cmd(args: argparse.Namespace) -> int:
    action = args.judge_action
    if action == "set-wind":
        cmd = JudgeCommand(action="set_wind", wind_angle_deg=args.angle, wind_speed=args.speed)
    elif action == "place-leak":
        cmd = JudgeCommand(action="place_leak", leak_pos=_parse_vec3(args.pos))
    elif action == "clear-leaks":
        cmd = JudgeCommand(action="clear_leaks")
    elif action == "toggle-dropout":
        state = None if args.state is None else (args.state == "on")
        cmd = JudgeCommand(action="toggle_dropout", dropout_enabled=state)
    elif action == "reset-race":
        cmd = JudgeCommand(action="reset_race")
    elif action == "pause":
        cmd = JudgeCommand(action="pause")
    else:
        raise ValueError(f"Unsupported judge action: {action}")

    ticks = asyncio.run(_ws_send_and_read(args.ws_url, commands=[cmd], wait_ticks=2))
    print(json.dumps({"sent": cmd.model_dump(), "latest_tick": ticks[-1]}, indent=2))
    return 0


def _cmd_live_scenario(args: argparse.Namespace) -> int:
    spec = get_scenario(args.name)
    cmds: list[JudgeCommand] = [
        JudgeCommand(action="reset_race"),
        JudgeCommand(action="clear_leaks"),
        JudgeCommand(action="set_wind", wind_angle_deg=spec.wind_angle_deg, wind_speed=spec.wind_speed),
        JudgeCommand(action="toggle_dropout", dropout_enabled=spec.dropout_enabled),
    ]
    for lk in spec.leaks:
        cmds.append(JudgeCommand(action="place_leak", leak_pos=lk.pos))

    ticks = asyncio.run(_ws_send_and_read(args.ws_url, commands=cmds, wait_ticks=2))
    print(
        json.dumps(
            {
                "applied_scenario": spec.name,
                "description": spec.description,
                "commands_sent": len(cmds),
                "latest_tick": ticks[-1],
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.sim_ctl",
        description="Unified CLI controller for headless and live Drone Gas Leak Detection sim.",
    )
    sub = parser.add_subparsers(dest="top_cmd", required=True)

    # audit
    p_audit = sub.add_parser("audit", help="Run quantitative physics realism audit")
    p_audit.add_argument("--json", action="store_true", help="Output JSON report")

    # scenarios
    sub.add_parser("scenarios", help="List available evaluation & demo scenarios")

    # headless
    p_hl = sub.add_parser("headless", help="Run and probe the simulation headlessly (no server)")
    hl_sub = p_hl.add_subparsers(dest="hl_cmd", required=True)

    p_run = hl_sub.add_parser("run", help="Run a headless scenario race")
    p_run.add_argument("--scenario", default="open_corridor", choices=list_scenario_names())
    p_run.add_argument("--duration", type=float, default=60.0)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--agents", default="tabpfn,gradient,lawnmower")
    p_run.add_argument("--real-tabpfn", action="store_true")
    p_run.add_argument("--wind-at", action="append", help="Scheduled wind shift 'time:angle:speed'")
    p_run.add_argument("--dropout-at", action="append", help="Scheduled dropout 'time:on|off'")
    p_run.add_argument("--place-leak", action="append", help="Extra leak at 'x,y,z'")
    p_run.add_argument("--trace-out", type=Path, default=None, help="Export JSONL debug trace")
    p_run.add_argument("--trace-stride", type=int, default=10)

    p_probe = hl_sub.add_parser("probe", help="Inspect 3D airflow, wake, context & ppm at a point")
    p_probe.add_argument("--pos", required=True, help="3D point 'x,y,z'")
    p_probe.add_argument("--wind-angle", type=float, default=15.0)
    p_probe.add_argument("--wind-speed", type=float, default=3.5)
    p_probe.add_argument("--leak", default=None, help="Optional leak 'x,y,z' to sample ppm")
    p_probe.add_argument("--warmup", type=float, default=8.0, help="Plume warmup seconds")
    p_probe.add_argument("--deterministic", action="store_true", help="Zero turbulent noise")
    p_probe.add_argument("--seed", type=int, default=42)

    # live
    p_live = sub.add_parser("live", help="Inspect and control a live running FastAPI/WS server")
    live_sub = p_live.add_subparsers(dest="live_cmd", required=True)

    p_health = live_sub.add_parser("health", help="GET /health from live server")
    p_health.add_argument("--host", default="http://localhost:8000")

    p_status = live_sub.add_parser("status", help="Fetch latest live SimTick snapshot over WS")
    p_status.add_argument("--ws-url", default="ws://localhost:8000/ws")

    p_watch = live_sub.add_parser("watch", help="Stream N live SimTicks as JSONL")
    p_watch.add_argument("--ws-url", default="ws://localhost:8000/ws")
    p_watch.add_argument("--ticks", type=int, default=10)
    p_watch.add_argument("--out", type=Path, default=None)

    p_scen = live_sub.add_parser("scenario", help="Apply a named scenario preset to live server")
    p_scen.add_argument("name", choices=list_scenario_names())
    p_scen.add_argument("--ws-url", default="ws://localhost:8000/ws")

    p_cmd = live_sub.add_parser("cmd", help="Send a JudgeCommand to the live server")
    p_cmd.add_argument("--ws-url", default="ws://localhost:8000/ws")
    cmd_sub = p_cmd.add_subparsers(dest="judge_action", required=True)

    p_sw = cmd_sub.add_parser("set-wind")
    p_sw.add_argument("--angle", type=float, default=None)
    p_sw.add_argument("--speed", type=float, default=None)

    p_pl = cmd_sub.add_parser("place-leak")
    p_pl.add_argument("--pos", required=True, help="'x,y,z'")

    cmd_sub.add_parser("clear-leaks")

    p_td = cmd_sub.add_parser("toggle-dropout")
    p_td.add_argument("--state", choices=["on", "off"], default=None)

    cmd_sub.add_parser("reset-race")
    cmd_sub.add_parser("pause")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.top_cmd == "audit":
        report = run_physics_audit()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            from tools.physics_audit import _print_report

            _print_report(report)
        return 0 if report["all_passed"] else 1

    if args.top_cmd == "scenarios":
        out = [
            {
                "name": s.name,
                "description": s.description,
                "wind_angle_deg": s.wind_angle_deg,
                "wind_speed": s.wind_speed,
                "n_leaks": len(s.leaks),
                "dropout_enabled": s.dropout_enabled,
                "n_scheduled_events": len(s.events),
            }
            for s in SCENARIOS.values()
        ]
        print(json.dumps(out, indent=2))
        return 0

    if args.top_cmd == "headless":
        if args.hl_cmd == "run":
            return _cmd_headless_run(args)
        if args.hl_cmd == "probe":
            return _cmd_headless_probe(args)

    if args.top_cmd == "live":
        if args.live_cmd == "health":
            return _cmd_live_health(args)
        if args.live_cmd == "status":
            return _cmd_live_status(args)
        if args.live_cmd == "watch":
            return _cmd_live_watch(args)
        if args.live_cmd == "scenario":
            return _cmd_live_scenario(args)
        if args.live_cmd == "cmd":
            return _cmd_live_cmd(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
