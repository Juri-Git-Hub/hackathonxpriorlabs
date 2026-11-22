"""Compare time to first contact and confirmed leak with one and three drones."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from agents.reliable_tabpfn import ReliableTabPFNAgent
from server.app import build_agents
from shared.schemas import LeakSource
from sim.engine import SimEngine
from sim.world import World


def trial(
    count: int, *, seed: int, leak: LeakSource, duration: float, mock: bool
) -> dict:
    """Run one fleet on the fixed leak until confirmation or the time limit."""
    agents = build_agents(mock_agents=mock, drone_count=count)
    try:
        for agent in agents.values():
            if isinstance(agent, ReliableTabPFNAgent):
                agent.warmup_remote()
        engine = SimEngine(agents, seed=seed, layout_seed=seed, mission_drone_ids=frozenset(agents))
        engine.reset_race([leak])
        first_contact = None
        contact_drone = None
        start = time.perf_counter()
        for step in range(round(duration / engine.DT)):
            engine.advance()
            if first_contact is None:
                for drone in engine.drones.values():
                    if drone.last_ppm >= 3.5:
                        first_contact = round(engine.sim_time, 2)
                        contact_drone = drone.id
                        break
            if engine.mission_report is not None:
                break
            if not mock:
                # Give the background inference workers the same wall-time budget as the demo.
                time.sleep(max(0, start + (step + 1) * engine.DT - time.perf_counter()))

        report = engine.mission_report
        return {
            "first_contact_s": first_contact,
            "contact_drone": contact_drone,
            "confirmed_s": round(report.elapsed_sim_seconds, 2) if report else None,
            "confirming_drone": report.drone_id if report else None,
            "sim_seconds": round(engine.sim_time, 2),
            "wall_seconds": round(time.perf_counter() - start, 2),
            "team_samples": sum(drone.samples_taken for drone in engine.drones.values()),
        }
    finally:
        for agent in agents.values():
            close = getattr(agent, "close", None)
            if close is not None:
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use the demo mock policy without TabPFN")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 73])
    parser.add_argument("--duration", type=float, default=90.0, help="Sim seconds per trial")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < SimEngine.DT:
        parser.error("duration must cover at least one simulation step")

    results = []
    for seed in args.seeds:
        world = World(layout_seed=seed)
        rng = np.random.default_rng(seed)
        sites = {}
        for _ in range(200):
            pos = world.random_pipe_leak_pos(rng)
            sector = int(np.searchsorted([-20, 20], pos[0], side="right"))
            sites.setdefault(sector, pos)
            if len(sites) == 3:
                break
        if len(sites) != 3:
            raise RuntimeError(f"Could not sample a pipe leak in every sector for seed {seed}")
        for sector, pos in sorted(sites.items()):
            leak = LeakSource(id="benchmark", pos=pos)
            pair = {
                str(count): trial(count, seed=seed, leak=leak, duration=args.duration, mock=args.mock)
                for count in (1, 3)
            }
            row = {"seed": seed, "sector": sector, "leak_pos": pos, "trials": pair}
            results.append(row)
            print(json.dumps(row), flush=True)

    summary = {"backend": "mock" if args.mock else "tabpfn", "duration_s": args.duration}
    for count in ("1", "3"):
        trials = [row["trials"][count] for row in results]
        summary[count] = {}
        for metric in ("first_contact_s", "confirmed_s"):
            times = [run[metric] for run in trials if run[metric] is not None]
            capped_times = [args.duration if run[metric] is None else run[metric] for run in trials]
            summary[count][metric] = {
                "successes": len(times),
                "trials": len(trials),
                "median_success_s": statistics.median(times) if times else None,
                "mean_capped_s": round(statistics.mean(capped_times), 2),
            }
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
