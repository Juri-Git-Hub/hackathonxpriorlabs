# Frontend (Three.js + Vite)

OWNER: Frontend / Visualization team.

## Contract & Integration
- Connect to backend WebSocket at `ws://localhost:8000/ws` (start backend with `python -m server.app --mock-agents`).
- Support `?mock=true` query param to replay synthetic `world_init`, `sim_tick`, and `belief_update` payloads without needing Python running.
- **Coordinate conversion (strictly at the rendering boundary):**
  - Python physics is **Right-handed Z-up** (`x ∈ [-60, 60]`, `y ∈ [-60, 60]`, `z ∈ [0, 30]`).
  - Three.js is **Y-up**: convert with `threePos = (sim_x, sim_z, -sim_y)`.
- See [`shared/schemas.py`](../shared/schemas.py) for all WebSocket message shapes (`WorldInitPayload`, `SimTickPayload`, `BeliefUpdatePayload`, `JudgeCommand`).
