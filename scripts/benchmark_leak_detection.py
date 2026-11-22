"""Comprehensive validation & held-out benchmark for TabPFN leak detection.

Runs the real Dual-TabPFN agent against the unchanged `SimEngine._check_completion()`
judge (180s max per mission, optional measured RPC latency applied to simulated clock).

Tiers:
- `val`  (15 cases): Fast iteration & failure diagnosis across all maps & regimes
- `test` (44 cases): Test suite across:
  * All 7 standardized scenarios (`tools/scenarios.py`)
  * All 5 named worlds (`refinery`, `pipe_loop`, `tank_terminal`, `compressor_yard`, `cooling_works`)
  * 10 unseen procedural `generated` layouts (layout seeds 9001..9010)
  * All 7 cases from `integration/prediction_protocol.py` (seeds 201..207)
  * All 7 cases from `integration/wind_protocol.py` (seeds 301..307)
  * Edge regimes (low wind 0.8 m/s, weak emission 0.5, elevated off-asset z=18m)
- `async` (5 cases): True real-time 20 Hz multi-threaded async runtime test

Usage:
    python -m scripts.benchmark_leak_detection --tier val
    python -m scripts.benchmark_leak_detection --tier test
    python -m scripts.benchmark_leak_detection --tier async
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Optional, TextIO

import numpy as np

from agents.reliable_tabpfn import ReliableTabPFNAgent
from agents.wind_plume import WindPlumeAgent
from shared.schemas import JudgeCommand, LeakSource, Vec3
from sim.engine import SimEngine
from sim.world import World
from tools.scenarios import SCENARIOS

PRED_CASES = [
    dict(name="default_seed201", seed=201),
    dict(name="default_seed202", seed=202),
    dict(name="default_seed203", seed=203),
    dict(name="other_valve", seed=204, source=[-17.5, -2.0, 3.5]),
    dict(name="east_tank_port", seed=205, source=[20.5, -7.0, 6.0]),
    dict(name="off_asset", seed=206, source=[8.0, -10.0, 4.0]),
    dict(name="elevated", seed=207, source=[15.0, 8.0, 13.0]),
]

WIND_CASES = [
    dict(name="default", seed=301),
    dict(name="north_wind", seed=302, wind=[90, 3.5]),
    dict(name="west_wind", seed=303, wind=[180, 4.5]),
    dict(name="early_wind_reversal", seed=304, wind_schedule=[dict(time=5, angle=195, speed=3.5)]),
    dict(name="later_wind_change", seed=305, wind_schedule=[dict(time=35, angle=135, speed=4)]),
    dict(name="off_asset", seed=306, source=[8.0, -10.0, 4.0], wind=[225, 3.0]),
    dict(name="elevated", seed=307, source=[15.0, 8.0, 13.0]),
]


@dataclass(frozen=True)
class EvalCase:
    name: str
    seed: int
    map_id: str = "refinery"
    layout_seed: Optional[int] = None
    wind: Optional[tuple[float, float]] = None
    leaks: Optional[list[dict[str, Any]]] = None
    dropout: bool = False
    wind_schedule: tuple[dict[str, Any], ...] = ()
    duration_s: float = 180.0


def _build_val_cases() -> list[EvalCase]:
    return [
        EvalCase("val_refinery_default_41", seed=41),
        EvalCase("val_refinery_layout42", seed=42, layout_seed=42),
        EvalCase("val_wake_trap", seed=142, wind=(0.0, 4.2), leaks=[{"pos": (11.5, 24.0, 6.0), "rate": 1.1, "buoyancy": 0.08}]),
        EvalCase("val_wind_shift_90", seed=143, wind=(0.0, 4.0), leaks=[{"pos": (-6.0, -7.2, 4.0), "rate": 1.0, "buoyancy": 0.20}], wind_schedule=({"time": 20.0, "angle": 90.0, "speed": 4.5},)),
        EvalCase("val_heavy_gas", seed=144, wind=(-20.0, 3.0), leaks=[{"pos": (4.0, -7.2, 5.0), "rate": 1.1, "buoyancy": -0.22}]),
        EvalCase("val_high_wind", seed=145, wind=(40.0, 7.5), leaks=[{"pos": (-10.0, 5.5, 4.0), "rate": 1.3, "buoyancy": 0.15}]),
        EvalCase("val_dropouts", seed=146, wind=(15.0, 3.5), leaks=[{"pos": (-6.0, -7.2, 4.0)}], dropout=True),
        EvalCase("val_pipe_loop_z21", seed=147, map_id="pipe_loop", wind=(30.0, 3.5)),
        EvalCase("val_tank_terminal", seed=148, map_id="tank_terminal", wind=(140.0, 3.5)),
        EvalCase("val_compressor_yard", seed=149, map_id="compressor_yard", wind=(-60.0, 4.0)),
        EvalCase("val_cooling_works_z13", seed=150, map_id="cooling_works", wind=(210.0, 3.5)),
        EvalCase("val_generated_8001", seed=151, map_id="generated", layout_seed=8001, wind=(75.0, 3.8)),
        EvalCase("val_generated_8002_offasset", seed=152, map_id="generated", layout_seed=8002, wind=(250.0, 3.2), leaks=[{"pos": (12.0, -14.0, 9.0), "rate": 1.0, "buoyancy": 0.18}]),
        EvalCase("val_elevated_offasset", seed=153, wind=(115.0, 3.0), leaks=[{"pos": (15.0, 8.0, 14.0), "rate": 1.0, "buoyancy": 0.25}]),
        EvalCase("val_multi_leak", seed=154, wind=(25.0, 3.8), leaks=[{"pos": (-17.5, -2.0, 3.5), "rate": 0.9, "buoyancy": 0.25}, {"pos": (20.5, -7.0, 6.0), "rate": 1.0, "buoyancy": 0.20}]),
    ]


def _build_test_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []

    # 1. All 7 standardized scenarios from tools/scenarios.py (7)
    for name, sc in SCENARIOS.items():
        cases.append(
            EvalCase(
                name=f"sc_{name}",
                seed=700 + len(cases),
                wind=(sc.wind_angle_deg, sc.wind_speed),
                leaks=[{"pos": lk.pos, "rate": lk.emission_rate, "buoyancy": lk.buoyancy_vz} for lk in sc.leaks],
                dropout=sc.dropout_enabled,
                wind_schedule=tuple(
                    {"time": ev.at_time, "angle": ev.command.wind_angle_deg, "speed": ev.command.wind_speed}
                    for ev in sc.events
                    if ev.command.action == "set_wind"
                ),
            )
        )

    # 2. All 5 named worlds (default + alt asset/wind) (10)
    named_specs = [
        ("refinery", None, (35.0, 3.6), None),
        ("refinery", 42, (200.0, 4.2), [{"pos": (20.5, -7.0, 6.0)}]),  # note layout 42 shifts assets -> use world asset 8 if None
        ("pipe_loop", None, (120.0, 3.5), None),
        ("pipe_loop", None, (-45.0, 4.5), [{"pos": (-18.0, -7.4, 3.0), "buoyancy": -0.18}]),
        ("tank_terminal", None, (10.0, 3.8), None),
        ("tank_terminal", None, (260.0, 3.5), [{"pos": (-17.0, 12.8, 4.0)}]),
        ("compressor_yard", None, (85.0, 4.0), None),
        ("compressor_yard", None, (315.0, 3.2), [{"pos": (1.0, 30.0, 8.0)}]),
        ("cooling_works", None, (-110.0, 3.8), None),
        ("cooling_works", None, (45.0, 4.0), [{"pos": (-11.0, -7.8, 6.0), "buoyancy": -0.15}]),
    ]
    for i, (m, lseed, w, lks) in enumerate(named_specs):
        # For refinery layout_seed=42 alt asset, read asset dynamically
        if m == "refinery" and lseed == 42:
            w_tmp = World(layout_seed=42, map_id="refinery")
            lks = [{"pos": w_tmp.assets[8].pos}]
        cases.append(EvalCase(name=f"world_{m}_{i}", seed=800 + i, map_id=m, layout_seed=lseed, wind=w, leaks=lks))

    # 3. 10 completely unseen procedural generated layouts (9001..9010) (10)
    rng = np.random.default_rng(9999)
    for k in range(10):
        lseed = 9001 + k
        w_tmp = World(layout_seed=lseed, map_id="generated")
        ang = float(rng.uniform(0.0, 360.0))
        spd = float(rng.uniform(1.5, 6.5))
        buoy = float(rng.choice([-0.22, 0.0, 0.25]))
        if k % 3 == 0:
            # Off-asset leak
            g = np.array(w_tmp.grid_coords, dtype=np.float64)
            pt = g[int(rng.integers(len(g)))]
            lk = [{"pos": (float(pt[0]), float(pt[1]), float(np.clip(pt[2], 3.0, 16.0))), "buoyancy": buoy}]
        else:
            ast = w_tmp.assets[int(rng.integers(len(w_tmp.assets)))]
            lk = [{"pos": ast.pos, "buoyancy": buoy}]
        w_sched = ({"time": 22.0, "angle": (ang + 110.0) % 360.0, "speed": 4.0},) if k % 4 == 1 else ()
        cases.append(
            EvalCase(
                name=f"gen_{lseed}",
                seed=900 + k,
                map_id="generated",
                layout_seed=lseed,
                wind=(ang, spd),
                leaks=lk,
                dropout=(k % 3 == 2),
                wind_schedule=w_sched,
            )
        )

    # 4. All 7 prediction_protocol cases (201..207) (7)
    for pc in PRED_CASES:
        lks = [{"pos": tuple(pc["source"])}] if "source" in pc else None
        cases.append(EvalCase(name=f"pred_{pc['name']}", seed=pc["seed"], leaks=lks))

    # 5. All 7 wind_protocol cases (301..307) (7)
    for wc in WIND_CASES:
        lks = [{"pos": tuple(wc["source"])}] if "source" in wc else None
        w = tuple(wc["wind"]) if "wind" in wc else None
        ws = tuple(wc.get("wind_schedule", ()))
        cases.append(EvalCase(name=f"wind_{wc['name']}", seed=wc["seed"], wind=w, leaks=lks, wind_schedule=ws))

    # 6. Extra edge regimes (3)
    cases.extend(
        [
            EvalCase("edge_low_wind_0p8", seed=950, wind=(135.0, 0.8), leaks=[{"pos": (8.0, -10.0, 4.0), "rate": 0.9, "buoyancy": 0.15}]),
            EvalCase("edge_weak_emission_0p5", seed=951, wind=(-40.0, 3.2), leaks=[{"pos": (-17.5, -2.0, 3.5), "rate": 0.5, "buoyancy": 0.20}]),
            EvalCase("edge_high_offasset_z18", seed=952, wind=(55.0, 3.5), leaks=[{"pos": (-12.0, 10.0, 18.0), "rate": 1.1, "buoyancy": 0.15}]),
        ]
    )
    return cases


def run_case(agent: ReliableTabPFNAgent | WindPlumeAgent, case: EvalCase, trace: TextIO | None = None, *, realtime: bool = False) -> dict[str, Any]:
    engine = SimEngine(
        agents={"tabpfn": agent},
        seed=case.seed,
        layout_seed=case.layout_seed,
        map_id=case.map_id,
        responsive_sensor=isinstance(agent, WindPlumeAgent),
    )
    leaks = (
        [
            LeakSource(
                id=f"lk_{i}",
                pos=tuple(lk["pos"]),
                emission_rate=float(lk.get("rate", 1.0)),
                buoyancy_vz=float(lk.get("buoyancy", 0.25)),
            )
            for i, lk in enumerate(case.leaks)
        ]
        if case.leaks is not None
        else None
    )
    if case.wind is not None:
        engine.airflow.set_wind(case.wind[0], case.wind[1])
    engine.sensor_dropout_active = case.dropout
    engine.reset_race(leaks=leaks)

    changes = iter(sorted(case.wind_schedule, key=lambda c: c["time"]))
    next_change = next(changes, None)

    t0 = time.perf_counter()
    closest = 999.0
    max_ppm = 0.0
    first_contact_t: Optional[float] = None
    last_est: Optional[Vec3] = None
    last_est_err: Optional[float] = None

    traced_calls = 0
    samples_seen = 0
    near_gas_samples = 0
    max_near_gas_samples = 0
    max_lock_seconds = 0.0
    n_ticks = int(round(case.duration_s / engine.DT))
    for _ in range(n_ticks):
        if next_change is not None and engine.sim_time + 1e-9 >= next_change["time"]:
            engine.airflow.set_wind(float(next_change["angle"]), float(next_change["speed"]))
            next_change = next(changes, None)

        if realtime:
            delay = t0 + _ * engine.DT - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        tick = engine.step(include_puffs=False)
        for b in engine.pop_belief_updates():
            last_est = b.estimated_source
            last_est_err = engine._dist_to_nearest_leak(last_est)

        d = tick.drones[0]
        if d.samples_taken != samples_seen:
            samples_seen = d.samples_taken
            max_lock_seconds = max(max_lock_seconds, engine._lock_timers[d.id])
            if d.ppm > 15 and d.dist_to_nearest_leak <= engine.LOCK_DISTANCE_M:
                near_gas_samples += 1
                max_near_gas_samples = max(max_near_gas_samples, near_gas_samples)
            else:
                near_gas_samples = 0
        calls = getattr(agent, "source_prediction_calls", 0)
        if trace is not None and (_ % 10 == 0 or traced_calls != calls):
            position = np.asarray(d.pos)
            if isinstance(agent, ReliableTabPFNAgent):
                with agent._data_lock:
                    wind = np.asarray(agent._obs_hist.current_wind).copy()
                    candidates = agent._last_src_cands
                    scores = agent._last_src_score
                    calls = agent.source_prediction_calls
                    goal = agent.goal.tolist() if agent.goal is not None else None
                    phase = agent.search_phase
                    basis = agent.decision_basis
            else:
                wind = np.asarray(agent.wind)
                candidates = scores = None
                goal = agent.goal
                phase = agent.search_phase
                basis = agent.decision_basis
            speed = np.linalg.norm(wind)
            refreshed = traced_calls != calls
            top = []
            if refreshed and scores is not None and candidates is not None:
                top = [
                    {"pos": candidates[i].tolist(), "score": float(scores[i])}
                    for i in np.argsort(scores)[-5:][::-1]
                ]
            errors = []
            for leak in engine.leaks:
                delta = position - leak.pos
                distances = None if candidates is None else np.linalg.norm(candidates - leak.pos, axis=1)
                errors.append({
                    "id": leak.id, "source": leak.pos, "drone_dz": float(delta[2]),
                    "downwind_m": float(delta[:2] @ wind / speed) if speed >= .2 else None,
                    "estimate_delta": (np.asarray(last_est) - leak.pos).tolist() if last_est else None,
                    "candidate_min_error_m": float(distances.min()) if distances is not None and distances.size else None,
                })
            belief = engine._latest_agent_beliefs.get(d.id)
            trace.write(json.dumps({
                "case": case.name, "sim_time": tick.sim_time, "pos": d.pos,
                "target": d.target_pos, "goal": goal,
                "phase": phase, "basis": basis, "ppm": d.ppm,
                "wind": wind.tolist(), "top_candidates": top, "errors": errors,
                "belief_age_s": tick.sim_time - belief.sim_time if belief is not None else None,
                "lock_seconds": engine._lock_timers[d.id],
            }) + "\n")
            traced_calls = calls
        closest = min(closest, d.dist_to_nearest_leak)
        max_ppm = max(max_ppm, d.ppm)
        if first_contact_t is None and d.ppm >= 3.5:
            first_contact_t = round(tick.sim_time, 1)
        if engine.mission_report is not None or d.status == "depleted":
            break

    wall_s = time.perf_counter() - t0
    success = engine.mission_report is not None
    failure_reason = None
    if not success:
        if first_contact_t is None:
            failure_reason = "no_gas_contact"
        elif closest > 5.0:
            failure_reason = "drone_never_within_5m"
        elif last_est_err is None or last_est_err > 5.0:
            failure_reason = "estimate_error_gt_5m"
        else:
            failure_reason = "lock_sustain_incomplete"

    return {
        "name": case.name,
        "map_id": case.map_id,
        "layout_seed": case.layout_seed,
        "seed": case.seed,
        "success": success,
        "sim_seconds": round(engine.sim_time, 1),
        "wall_seconds": round(wall_s, 2),
        "first_contact_s": first_contact_t,
        "closest_dist_m": round(closest, 2),
        "final_est_err_m": round(last_est_err, 2) if last_est_err is not None else None,
        "max_ppm": round(max_ppm, 1),
        "near_source_gas_seconds": round(max_near_gas_samples * engine.OBS_INTERVAL, 2),
        "best_lock_seconds": round(max_lock_seconds, 2),
        "src_calls": getattr(agent, "source_prediction_calls", 0),
        "haz_calls": getattr(agent, "hazard_prediction_calls", 0),
        "est_error_m": round(engine.mission_report.estimated_error_m, 2) if engine.mission_report else None,
        "failure_reason": failure_reason,
    }


def run_async_smoke() -> list[dict[str, Any]]:
    """True 20 Hz multi-threaded async runtime verification against 4090."""
    cases = [
        EvalCase("async_refinery_42", seed=42, layout_seed=42),
        EvalCase("async_pipe_loop_z21", seed=147, map_id="pipe_loop", wind=(30.0, 3.5)),
        EvalCase("async_wake_trap", seed=142, wind=(0.0, 4.2), leaks=[{"pos": (11.5, 24.0, 6.0), "rate": 1.1, "buoyancy": 0.08}]),
        EvalCase("async_wind_shift", seed=143, wind=(0.0, 4.0), leaks=[{"pos": (-6.0, -7.2, 4.0), "rate": 1.0, "buoyancy": 0.20}], wind_schedule=({"time": 20.0, "angle": 90.0, "speed": 4.5},)),
        EvalCase("async_generated_9001", seed=900, map_id="generated", layout_seed=9001, wind=(75.0, 3.8)),
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        agent = ReliableTabPFNAgent(drone_id="tabpfn", sync_mode=False)
        agent.warmup_remote()
        engine = SimEngine(agents={"tabpfn": agent}, seed=case.seed, layout_seed=case.layout_seed, map_id=case.map_id)
        leaks = (
            [LeakSource(id="lk_0", pos=tuple(lk["pos"]), emission_rate=lk.get("rate", 1.0), buoyancy_vz=lk.get("buoyancy", 0.25)) for lk in case.leaks]
            if case.leaks is not None
            else None
        )
        if case.wind is not None:
            engine.airflow.set_wind(case.wind[0], case.wind[1])
        engine.reset_race(leaks=leaks)
        changes = iter(sorted(case.wind_schedule, key=lambda c: c["time"]))
        nxt = next(changes, None)

        start = time.monotonic()
        try:
            while engine.sim_time < case.duration_s:
                t_loop = time.monotonic()
                if nxt is not None and engine.sim_time >= nxt["time"]:
                    engine.airflow.set_wind(float(nxt["angle"]), float(nxt["speed"]))
                    nxt = next(changes, None)
                tick = engine.step(include_puffs=False)
                if agent.error:
                    raise RuntimeError(agent.error)
                if tick.mission_report is not None or tick.drones[0].status == "depleted":
                    break
                time.sleep(max(0.0, engine.DT - (time.monotonic() - t_loop)))
            row = {
                "name": case.name,
                "success": engine.mission_report is not None,
                "sim_seconds": round(engine.sim_time, 1),
                "wall_seconds": round(time.monotonic() - start, 1),
                "est_error_m": round(engine.mission_report.estimated_error_m, 2) if engine.mission_report else None,
                "src_calls": getattr(agent, "source_prediction_calls", 0),
                "haz_calls": getattr(agent, "hazard_prediction_calls", 0),
            }
            print(json.dumps(row), flush=True)
            results.append(row)
        finally:
            agent.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Dual-TabPFN leak detection")
    parser.add_argument("--tier", choices=["val", "test", "async"], default="val")
    parser.add_argument("--out", type=Path, default=Path("reports/reliable_tabpfn_benchmark.json"))
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--policy", choices=["tabpfn", "wind"], default="tabpfn")
    parser.add_argument("--source-pre-data", type=Path, help="Alternate pre-contact TabPFN context.")
    parser.add_argument("--source-post-data", type=Path, help="Alternate post-contact TabPFN context.")
    parser.add_argument("--target-rate", type=float, help="Stop once the target fraction is mathematically unreachable.")
    parser.add_argument("--random-seeds", type=int, help="Number of randomized refinery layouts to evaluate.")
    parser.add_argument("--random-seed-base", type=int, default=20261122, help="Seed for generating randomized cases.")
    parser.add_argument("--case", action="append", help="Run only the named case. Repeat for multiple cases.")
    parser.add_argument("--realtime", action="store_true", help="Replay selected cases using the live async scheduler at 20 Hz.")
    parser.add_argument("--trace", type=Path, help="Write flight and source-ranking diagnostics to JSONL.")
    args = parser.parse_args()

    if (args.source_post_data is not None or args.source_pre_data is not None) and (
        args.policy != "tabpfn" or args.tier == "async"
    ):
        parser.error("Alternate TabPFN options require --policy tabpfn with --tier val or test")
    if args.target_rate is not None and (not 0 < args.target_rate <= 1 or args.tier == "async"):
        parser.error("--target-rate must be in (0, 1] for --tier val or test")
    if args.tier == "async":
        if args.case or args.trace or args.realtime or args.policy != "tabpfn":
            parser.error("Use --tier val or test for case selection and flight traces.")
        res = run_async_smoke()
        ok = sum(1 for r in res if r["success"])
        print(f"\nAsync real-time summary: {ok}/{len(res)} ({100.0*ok/len(res):.1f}%)")
        return 0 if ok / len(res) >= 0.8 else 1

    cases = _build_val_cases() if args.tier == "val" else _build_test_cases()
    if args.random_seeds is not None:
        if args.random_seeds < 1 or args.tier == "async":
            parser.error("--random-seeds requires a positive count and cannot be combined with --tier async")
        rng = np.random.default_rng(args.random_seed_base)
        cases = [EvalCase(f"random_{i:03d}", seed=int(rng.integers(1, 2**31)), layout_seed=int(rng.integers(1, 2**31))) for i in range(args.random_seeds)]
    if args.case:
        known = {case.name for case in cases}
        if unknown := set(args.case) - known:
            parser.error(f"Unknown cases for {args.tier}: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.name in args.case]
    if args.policy == "wind":
        agent = WindPlumeAgent(synchronous=not args.realtime)
    else:
        agent = ReliableTabPFNAgent(
            sync_mode=not args.realtime,
            apply_rpc_latency=not args.no_latency,
            source_post_data_path=args.source_post_data,
            source_pre_data_path=args.source_pre_data,
        )
        print(f"Warming up 4090 TabPFN cache before running {len(cases)} {args.tier} cases...", flush=True)
        agent.warmup_remote()

    results: list[dict[str, Any]] = []
    trace = None
    try:
        if args.trace is not None:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = args.trace.open("w")
        for case in cases:
            row = run_case(agent, case, trace=trace, realtime=args.realtime)
            results.append(row)
            summary = write_checkpoint(args, cases, results)
            status = "PASS" if row["success"] else f"FAIL({row['failure_reason']})"
            print(
                f"[{len(results):2d}/{len(cases)}] {case.name:28s} {status:28s} "
                f"t={row['sim_seconds']:5.1f}s dmin={row['closest_dist_m']:5.2f}m "
                f"est_err={str(row['final_est_err_m']):>5s} ppm={row['max_ppm']:6.1f} wall={row['wall_seconds']:4.1f}s",
                flush=True,
            )
            if summary["target_reachable"] is False:
                print("Target is unreachable even if every remaining case passes.", flush=True)
                break
    finally:
        if trace is not None:
            trace.close()
        agent.close()

    print(
        f"\n=== {args.tier.upper()} SUMMARY: {summary['successes']}/{len(results)} "
        f"({summary['detection_rate_pct']:.1f}%) -> {args.out} ==="
    )
    return 1 if summary["target_reachable"] is False else 0


def write_checkpoint(args, cases, results):
    successes = [row for row in results if row["success"]]
    required = math.ceil(args.target_rate * len(cases)) if args.target_rate is not None else None
    max_successes = len(successes) + len(cases) - len(results)
    summary = {
        "tier": args.tier,
        "random_seed_base": args.random_seed_base if args.random_seeds is not None else None,
        "realtime": args.realtime,
        "simulated_rpc_latency": args.policy == "tabpfn" and not (args.no_latency or args.realtime),
        "policy": args.policy,
        "source_post_data": str(args.source_post_data) if args.source_post_data else None,
        "source_pre_data": str(args.source_pre_data) if args.source_pre_data else None,
        "total": len(results),
        "planned_total": len(cases),
        "complete": len(results) == len(cases),
        "target_rate": args.target_rate,
        "required_successes": required,
        "target_reachable": max_successes >= required if required is not None else None,
        "successes": len(successes),
        "detection_rate_pct": round(100 * len(successes) / len(results), 2),
        "mean_sim_seconds_success": round(float(np.mean([r["sim_seconds"] for r in successes])), 2) if successes else None,
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
