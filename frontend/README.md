# Frontend (Three.js + Vite)

OWNER: Frontend / Visualization team.

## Contract & Integration
- Connect to backend WebSocket at `ws://localhost:8000/ws` (start backend with `python -m server.app --mock-agents`).
- Support `?mock=true` query param to replay synthetic `world_init`, `sim_tick`, and `belief_update` payloads without needing Python running.
- **Coordinate conversion (strictly at the rendering boundary):**
  - Python physics is **Right-handed Z-up** (`x ∈ [-60, 60]`, `y ∈ [-60, 60]`, `z ∈ [0, 30]`).
  - Three.js is **Y-up**: convert with `threePos = (sim_x, sim_z, -sim_y)`.
- See [`shared/schemas.py`](../shared/schemas.py) for all WebSocket message shapes (`WorldInitPayload`, `SimTickPayload`, `BeliefUpdatePayload`, `JudgeCommand`).

## PlumeSense backend integration

Production/demo: `python launch_demo.py`, open http://localhost:8501. Development: run `python -m server.app --port 8501`, then `npm run dev` here. Vite proxies `/ws` and `/health`; built files are served by FastAPI. The frontend remains the main Three.js app; there is no Streamlit UI.

Optional `belief_update` fields: `source_scores`, `hazard_probabilities`, `route`, `inference_seconds`, `measurement_count`, `exposure_ppm_seconds`, `score_kind`. `backend_status` carries live inference status/errors. Existing fields remain compatible. `judge_cmd` additionally accepts `step` while paused and `seed` with `reset_race`. See `shared/schemas.py`.
