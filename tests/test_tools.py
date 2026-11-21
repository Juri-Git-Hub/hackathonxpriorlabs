"""Tests for physics realism audit, headless evaluation runner, and unified sim_ctl CLI."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import uvicorn

from server.app import create_app
from tools import eval_runner, physics_audit, sim_ctl


def test_physics_audit_all_checks_pass() -> None:
    report = physics_audit.run_physics_audit()
    assert report["total_count"] >= 10
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert report["all_passed"], f"Failed physics audit checks: {failed}"


def test_eval_runner_headless_and_trace_export(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    rc = eval_runner.main(
        [
            "--scenarios",
            "wake_trap",
            "--seeds",
            "42",
            "--duration",
            "10",
            "--trace-out",
            str(trace_file),
            "--trace-stride",
            "10",
            "--json",
        ]
    )
    assert rc == 0
    assert trace_file.exists()
    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 15
    first = json.loads(lines[0])
    assert "drones" in first
    assert "tabpfn" in first["drones"]


def test_sim_ctl_headless_commands(capsys, tmp_path: Path) -> None:
    # 1. scenarios list
    assert sim_ctl.main(["scenarios"]) == 0
    scenarios_out = json.loads(capsys.readouterr().out)
    names = {s["name"] for s in scenarios_out}
    assert {"open_corridor", "wake_trap", "wind_shift_90", "sensor_dropouts"} <= names

    # 2. headless probe
    assert (
        sim_ctl.main(
            [
                "headless",
                "probe",
                "--pos",
                "18,24,5",
                "--wind-angle",
                "0",
                "--wind-speed",
                "4.0",
                "--leak",
                "12,24,4",
                "--warmup",
                "4.0",
                "--deterministic",
            ]
        )
        == 0
    )
    probe_out = json.loads(capsys.readouterr().out)
    assert probe_out["spatial_context"]["in_wind_shadow"] is True
    assert probe_out["sampled_ppm"] > 1.0

    # 3. headless run with scheduled wind shift
    trace_file = tmp_path / "ctl_trace.jsonl"
    assert (
        sim_ctl.main(
            [
                "headless",
                "run",
                "--scenario",
                "open_corridor",
                "--duration",
                "8",
                "--wind-at",
                "3:90:5.0",
                "--dropout-at",
                "4:on",
                "--trace-out",
                str(trace_file),
            ]
        )
        == 0
    )
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["scenario"] == "open_corridor"
    assert run_out["steps_per_sec"] > 200.0
    assert trace_file.exists()


def test_sim_ctl_live_commands_against_running_server(capsys) -> None:
    """Spin up an ephemeral local uvicorn server on a free port and test live CLI control."""
    app = create_app(mock_agents=True, seed=42)
    port = 18765
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        # Wait for server startup
        deadline = time.perf_counter() + 5.0
        while not server.started and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert server.started

        host_url = f"http://127.0.0.1:{port}"
        ws_url = f"ws://127.0.0.1:{port}/ws"

        assert sim_ctl.main(["live", "health", "--host", host_url]) == 0
        health = json.loads(capsys.readouterr().out)
        assert health["status"] == "ok"

        assert sim_ctl.main(["live", "status", "--ws-url", ws_url]) == 0
        status = json.loads(capsys.readouterr().out)
        assert [drone["id"] for drone in status["drones"]] == ["tabpfn"]

        assert (
            sim_ctl.main(
                ["live", "cmd", "--ws-url", ws_url, "toggle-dropout", "--state", "on"]
            )
            == 0
        )
        cmd_res = json.loads(capsys.readouterr().out)
        assert cmd_res["latest_tick"]["sensor_dropout_active"] is True

        assert sim_ctl.main(["live", "scenario", "wake_trap", "--ws-url", ws_url]) == 0
        scen_res = json.loads(capsys.readouterr().out)
        assert scen_res["applied_scenario"] == "wake_trap"
        assert len(scen_res["latest_tick"]["leaks"]) == 1
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)
