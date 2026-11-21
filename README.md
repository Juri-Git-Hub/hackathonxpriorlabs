# Drone Gas Leak Detection (Hackathon x Prior Labs)

Active 3D spatial gas-leak search simulation powered by **TabPFN**. Three autonomous drones (`tabpfn`, `gradient`, `lawnmower`) race to localize hidden gas leaks inside an industrial refinery with turbulent wind, building wake pockets, asymmetric chemical sensor lag, and toggleable `NaN` sensor dropouts.

## Links
- [Notion](https://quick-chestnut-0cd.notion.site/2nd-edition-Berlin-Robotics-AI-Hackathon-01c30736e47a83fe85388156949d2240)

---

## Core Architectural Rules (For Humans & AI Agents)

1. **Single Source of Truth (`shared/schemas.py`):** All cross-module payloads (`WorldInitPayload`, `DroneObservation`, `SimTickPayload`, `BeliefUpdatePayload`, `JudgeCommand`) are frozen Pydantic v2 models. Do not invent ad-hoc dicts or rename fields without updating all consumers.
2. **Coordinate System Convention:**
   - **Python Physics & Agents:** Right-handed **Z-up**, in meters. Facility bounds: `x ∈ [-60, 60]`, `y ∈ [-60, 60]`, `z ∈ [0, 30]` (ground plane at `z = 0`).
   - **Three.js Frontend (Y-up):** Convert strictly at the rendering boundary using `three_pos = (sim_x, sim_z, -sim_y)`. Never convert coordinates in Python.
3. **Strict Async Isolation for TabPFN:** Physics runs at a fixed **20 Hz** (`dt = 0.05s`) and must **never block** on TabPFN inference. `agents/tabpfn_agent.py` subclasses `AsyncInferenceAgent` (`agents/base.py`), which runs inference in a single background worker thread and atomically swaps `target_pos` and `BeliefUpdatePayload`.
4. **Headless-First Execution:** `sim/` and `agents/` have **zero dependency** on FastAPI, WebSockets, Three.js, or serial hardware, and run headless at `>500 steps/sec` for fast benchmarking.
5. **Mock-First Parallelism:**
   - Backend runs with `--mock-agents` (`MockTabPFNAgent`) so physics and frontend can be developed and demoed without loading TabPFN weights.
   - Frontend should support `?mock=true` for standalone UI development.
   - Hardware bridge (`hardware/drone_bridge.py`) subscribes externally via WebSocket and never blocks or crashes the sim.

---

## Repository Structure & Ownership Boundaries

```text
.
├── shared/
│   └── schemas.py          # FROZEN CONTRACT: Pydantic v2 models for all boundaries
├── sim/                    # OWNER: Simulation / Physics (headless, zero server deps)
│   ├── world.py            # 120x120x30m refinery AABBs, assets, 3D candidate grid
│   ├── airflow.py          # Base wind + OU gusts + building deflection & wake cavities
│   ├── plume.py            # Vectorized Lagrangian 3D Gaussian puff dispersion (<=400 puffs)
│   ├── drone.py            # 2nd-order dynamics, drag, thrust attitude, obstacle repulsion, battery
│   ├── sensor.py           # Asymmetric rise/decay lag, noise, toggleable NaN dropouts
│   └── engine.py           # Master 20 Hz simulation loop + multi-drone race coordinator
├── agents/                 # OWNER: AI / TabPFN
│   ├── base.py             # BaseAgent ABC + non-blocking AsyncInferenceAgent worker wrapper
│   ├── lawnmower.py        # Baseline 1: deterministic 3D serpentine raster sweep
│   ├── gradient.py         # Baseline 2: bio-inspired surge-cast / gradient ascent
│   ├── mock_agent.py       # Fast (<1ms) mock TabPFN agent for --mock-agents mode
│   └── tabpfn_agent.py     # TabPFNRegressor quantile UCB + travel penalty (READY FOR AI TEAM)
├── server/                 # OWNER: Backend / Glue
│   └── app.py              # FastAPI WebSocket broadcaster (/ws, /health) + judge commands
├── frontend/               # OWNER: Three.js UI (Vite + Three.js, see frontend/README.md)
├── hardware/               # OWNER: Physical Drone Bridge (optional, isolated WS client)
│   └── drone_bridge.py
├── tests/
│   ├── test_physics.py     # Plume bending, wake trapping, drone stability, >500 steps/s check
│   ├── test_agents.py      # Agent contracts, async non-blocking test, headless 3-drone race
│   └── test_ws_contract.py # Schema serialization + FastAPI WebSocket end-to-end tests
└── requirements.txt
```

---

## Quickstart

### 1. Setup Environment (Python 3.12 recommended for TabPFN/PyTorch)
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest
```

### 3. Run the WebSocket Server (Mock Mode — No GPU/TabPFN Needed)
```bash
python -m server.app --mock-agents --port 8000
```
- Health check: `GET http://localhost:8000/health`
- WebSocket stream: `ws://localhost:8000/ws`
  - Emits `WorldInitPayload` on connect
  - Emits `SimTickPayload` at 20 Hz
  - Emits `BeliefUpdatePayload` at ~1 Hz
  - Accepts `JudgeCommand` JSON messages (`set_wind`, `place_leak`, `clear_leaks`, `toggle_dropout`, `reset_race`, `pause`)

### 4. Run the 3D Three.js Frontend MVP
```bash
cd frontend
npm install
npm run dev
```
- Open `http://localhost:5173` (connects live to `ws://localhost:8000/ws`)
- Standalone UI mock mode (no Python required): `http://localhost:5173/?mock=true`

---

## Guide for Parallel Workstreams

### AI / TabPFN Integration (`agents/tabpfn_agent.py`)
Only edit `agents/tabpfn_agent.py` (do not touch `sim/` or `server/`).
1. Install TabPFN dependencies in your venv:
   ```bash
   uv pip install tabpfn pandas
   ```
2. `TabPFNAgent` already subclasses `AsyncInferenceAgent`, so threading, generation-safe resets, and non-blocking `update()` are already wired up.
3. Key things to iterate on inside `agents/tabpfn_agent.py`:
   - `_build_candidate_dataframe(obs)`: enrich the candidate grid rows (`world_init.grid_coords`) with static/wind-aware features (`nearest_asset_type`, `asset_age_years`, `in_wind_shadow`, `upwind_pipe_score`) using `world_init.assets` and `world_init.buildings` so TabPFN exploits refinery priors.
   - Tune `TabPFNRegressor` hyperparameters, device (`cuda` / `cpu`), warm-start waypoints, `beta` exploration schedule, and `TRAVEL_PENALTY`.
   - Test with NaN dropouts enabled (`obs.wind_measured` and `obs.temp_delta` can be `None` -> `np.nan`).
4. Run the server with real TabPFN (no `--mock-agents` flag):
   ```bash
   python -m server.app --port 8000
   ```
5. Benchmark headlessly against `GradientAgent` and `LawnmowerAgent` using `SimEngine` directly in Python (no server required).

### Three.js Frontend (`frontend/`)
- Build a Vite + Three.js app in `frontend/` connecting to `ws://localhost:8000/ws` (start backend with `--mock-agents`).
- Convert coordinates at the rendering boundary: `three_pos = (sim_x, sim_z, -sim_y)`.
- See `frontend/README.md` and `shared/schemas.py` for exact message fields.

### Hardware Bridge (`hardware/drone_bridge.py`)
- Standalone WebSocket client connecting to `ws://localhost:8000/ws` and forwarding `tabpfn` telemetry (`ppm`, `status`) to an ESP32 / Flight Controller over `pyserial`.
