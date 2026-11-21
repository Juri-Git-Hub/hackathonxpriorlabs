"""Headless multi-scenario evaluation & debug trace runner for AI agents.

Runs standardized races directly on `SimEngine` at >500 steps/sec (no server
or browser required) and outputs structured metrics + optional JSONL debug traces.

Usage:
    python -m tools.eval_runner --scenarios open_corridor,wake_trap --seeds 42,43
    python -m tools.eval_runner --scenarios all --duration 90 --json
    python -m tools.eval_runner --scenarios wake_trap --trace-out /tmp/wake_trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Optional

from agents.base import BaseAgent
from agents.gradient import GradientAgent
from agents.lawnmower import LawnmowerAgent
from agents.mock_agent import MockTabPFNAgent
from agents.tabpfn_agent import TabPFNAgent
from shared.schemas import JudgeCommand
from sim.engine import SimEngine
from tools.scenarios import ScenarioSpec, get_scenario, list_scenario_names


def build_eval_agents(
    agent_names: list[str],
    use_real_tabpfn: bool = False,
) -> dict[str, BaseAgent]:
    """Instantiate requested agent roster for headless evaluation."""
    roster: dict[str, BaseAgent] = {}
    for name in agent_names:
        if name == "tabpfn":
            roster[name] = TabPFNAgent("tabpfn") if use_real_tabpfn else MockTabPFNAgent("tabpfn")
        elif name == "gradient":
            roster[name] = GradientAgent("gradient")
        elif name == "lawnmower":
            roster[name] = LawnmowerAgent("lawnmower")
        else:
            raise ValueError(f"Unknown agent '{name}'. Expected: tabpfn, gradient, lawnmower")
    return roster


def run_single_eval(
    scenario: ScenarioSpec,
    seed: int = 42,
    duration_s: float = 90.0,
    agent_names: list[str] | None = None,
    use_real_tabpfn: bool = False,
    extra_events: list[tuple[float, JudgeCommand]] | None = None,
    trace_out: Optional[Path] = None,
    trace_stride: int = 10,
) -> dict[str, Any]:
    """Run one headless scenario race and collect per-drone metrics & optional JSONL trace."""
    names = agent_names or ["tabpfn", "gradient", "lawnmower"]
    agents = build_eval_agents(names, use_real_tabpfn=use_real_tabpfn)
    engine = SimEngine(agents=agents, seed=seed)

    # Configure scenario wind, leaks, and dropout state
    engine.airflow.set_wind(scenario.wind_angle_deg, scenario.wind_speed)
    engine.sensor_dropout_active = scenario.dropout_enabled
    engine.reset_race(leaks=list(scenario.leaks))

    # Merge scenario scheduled events with any CLI-injected events
    pending_events: list[tuple[float, JudgeCommand]] = [
        (ev.at_time, ev.command) for ev in scenario.events
    ]
    if extra_events:
        pending_events.extend(extra_events)
    pending_events.sort(key=lambda item: item[0])
    next_ev_idx = 0

    n_steps = max(1, int(round(duration_s / engine.DT)))

    # Per-drone accumulators
    time_to_lock: dict[str, Optional[float]] = {k: None for k in names}
    samples_to_lock: dict[str, Optional[int]] = {k: None for k in names}
    min_dist: dict[str, float] = {k: 999.0 for k in names}
    peak_ppm: dict[str, float] = {k: 0.0 for k in names}
    max_tilt_deg: dict[str, float] = {k: 0.0 for k in names}
    speed_sum: dict[str, float] = {k: 0.0 for k in names}

    trace_fp = trace_out.open("w", encoding="utf-8") if trace_out is not None else None
    t_wall_0 = time.perf_counter()

    try:
        for step_i in range(n_steps):
            # Fire any scheduled commands whose time has arrived
            while next_ev_idx < len(pending_events) and engine.sim_time >= pending_events[next_ev_idx][0] - 1e-9:
                engine.apply_command(pending_events[next_ev_idx][1])
                next_ev_idx += 1

            tick = engine.step()
            beliefs = engine.pop_belief_updates()

            for d_state in tick.drones:
                did = d_state.id
                if d_state.dist_to_nearest_leak < min_dist[did]:
                    min_dist[did] = d_state.dist_to_nearest_leak
                if d_state.ppm > peak_ppm[did]:
                    peak_ppm[did] = d_state.ppm
                if d_state.status == "locked" and time_to_lock[did] is None:
                    time_to_lock[did] = round(tick.sim_time, 3)
                    samples_to_lock[did] = d_state.samples_taken

                drone_obj = engine.drones[did]
                tilt = math.degrees(drone_obj.tilt_rad)
                if tilt > max_tilt_deg[did]:
                    max_tilt_deg[did] = tilt
                speed_sum[did] += float(
                    math.sqrt(
                        drone_obj.vel[0] ** 2
                        + drone_obj.vel[1] ** 2
                        + drone_obj.vel[2] ** 2
                    )
                )

            if trace_fp is not None and (step_i % max(1, trace_stride) == 0):
                record = {
                    "sim_time": round(tick.sim_time, 3),
                    "wind_vec": [round(tick.wind_vec[0], 3), round(tick.wind_vec[1], 3)],
                    "sensor_dropout_active": tick.sensor_dropout_active,
                    "n_puffs": len(tick.puffs),
                    "drones": {
                        d.id: {
                            "pos": [round(v, 2) for v in d.pos],
                            "rot_rpy_deg": [round(math.degrees(v), 1) for v in d.rot_rpy],
                            "target": [round(v, 2) for v in d.target_pos],
                            "ppm": round(d.ppm, 2),
                            "dist_to_leak": round(d.dist_to_nearest_leak, 2),
                            "battery_pct": round(d.battery_pct, 2),
                            "samples": d.samples_taken,
                            "status": d.status,
                        }
                        for d in tick.drones
                    },
                    "belief_sources": {
                        b.drone_id: [round(v, 2) for v in b.estimated_source] for b in beliefs
                    },
                }
                trace_fp.write(json.dumps(record) + "\n")
    finally:
        if trace_fp is not None:
            trace_fp.close()
        for agent in agents.values():
            close_fn = getattr(agent, "close", None)
            if callable(close_fn):
                close_fn()

    wall_s = time.perf_counter() - t_wall_0
    final_tick = engine.get_tick_payload()

    drone_results: dict[str, Any] = {}
    for d_state in final_tick.drones:
        did = d_state.id
        drone_results[did] = {
            "locked": time_to_lock[did] is not None,
            "time_to_lock_s": time_to_lock[did],
            "samples_to_lock": samples_to_lock[did],
            "min_dist_to_leak_m": round(min_dist[did], 2),
            "final_dist_to_leak_m": round(d_state.dist_to_nearest_leak, 2),
            "peak_ppm": round(peak_ppm[did], 2),
            "final_battery_pct": round(d_state.battery_pct, 2),
            "mean_speed_mps": round(speed_sum[did] / n_steps, 2),
            "max_tilt_deg": round(max_tilt_deg[did], 2),
            "final_status": d_state.status,
        }

    return {
        "scenario": scenario.name,
        "seed": seed,
        "duration_s": round(final_tick.sim_time, 2),
        "wall_time_s": round(wall_s, 3),
        "steps_per_sec": round(n_steps / max(wall_s, 1e-6), 1),
        "drones": drone_results,
    }


def _print_table(runs: list[dict[str, Any]]) -> None:
    print("\n=== Headless Evaluation Results ===")
    for run in runs:
        print(
            f"\nScenario: {run['scenario']} | seed={run['seed']} | "
            f"sim={run['duration_s']}s | wall={run['wall_time_s']}s ({run['steps_per_sec']} steps/s)"
        )
        print(
            f"  {'Agent':10s} {'Locked':6s} {'T_lock':7s} {'MinDist':8s} "
            f"{'PeakPPM':8s} {'Batt%':6s} {'AvgSpd':7s} {'MaxTilt':8s}"
        )
        for did, m in run["drones"].items():
            t_lock = f"{m['time_to_lock_s']:.1f}s" if m["time_to_lock_s"] is not None else "-"
            print(
                f"  {did:10s} {str(m['locked']):6s} {t_lock:7s} "
                f"{m['min_dist_to_leak_m']:6.2f}m  {m['peak_ppm']:7.1f} "
                f"{m['final_battery_pct']:5.1f}% {m['mean_speed_mps']:5.2f}ms "
                f"{m['max_tilt_deg']:6.1f}deg"
            )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless multi-scenario evaluation runner")
    parser.add_argument(
        "--scenarios",
        default="open_corridor,wake_trap",
        help=f"Comma-separated scenario names or 'all'. Available: {', '.join(list_scenario_names())}",
    )
    parser.add_argument("--seeds", default="42", help="Comma-separated RNG seeds (default: 42)")
    parser.add_argument("--duration", type=float, default=60.0, help="Sim duration in seconds (default: 60)")
    parser.add_argument(
        "--agents",
        default="tabpfn,gradient,lawnmower",
        help="Comma-separated agents (default: tabpfn,gradient,lawnmower)",
    )
    parser.add_argument(
        "--real-tabpfn",
        action="store_true",
        help="Use real TabPFNAgent instead of MockTabPFNAgent",
    )
    parser.add_argument("--trace-out", type=Path, default=None, help="Write JSONL debug trace (single run)")
    parser.add_argument("--trace-stride", type=int, default=10, help="Record every N physics ticks (default: 10 = 2Hz)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted table")
    args = parser.parse_args(argv)

    scenario_names = list_scenario_names() if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]

    runs: list[dict[str, Any]] = []
    for s_name in scenario_names:
        spec = get_scenario(s_name)
        for seed in seeds:
            trace_path = args.trace_out if (len(scenario_names) == 1 and len(seeds) == 1) else None
            runs.append(
                run_single_eval(
                    scenario=spec,
                    seed=seed,
                    duration_s=args.duration,
                    agent_names=agent_names,
                    use_real_tabpfn=args.real_tabpfn,
                    trace_out=trace_path,
                    trace_stride=args.trace_stride,
                )
            )

    if args.json:
        print(json.dumps({"runs": runs}, indent=2))
    else:
        _print_table(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
