"""SeeCH4 Triple-TabPFN leak detection agent interface.

Uses `ReliableTabPFNAgent` (Triple-TabPFN regressors + 3D A* navigation) as the
active spatial leak search agent for `tools/eval_runner.py --real-tabpfn` and
`server/app.py`.
"""

from __future__ import annotations

from agents.reliable_tabpfn import ReliableTabPFNAgent
from agents.tabpfn_features import SOURCE_FEATURE_COLUMNS as FEATURE_COLUMNS


class TabPFNAgent(ReliableTabPFNAgent):
    """SeeCH4 Triple-TabPFN agent backed by Prior Labs' TabPFN API, local TabPFN, or HTTP."""

    def __init__(
        self,
        drone_id: str = "tabpfn",
        api_url: str | None = None,
        n_estimators: int = 8,
        sync_mode: bool = False,
        apply_rpc_latency: bool = False,
        adaptive_wind: bool = True,
    ) -> None:
        super().__init__(
            drone_id=drone_id,
            api_url=api_url,
            n_estimators=n_estimators,
            sync_mode=sync_mode,
            apply_rpc_latency=apply_rpc_latency,
            adaptive_wind=adaptive_wind,
        )


__all__ = ["TabPFNAgent", "FEATURE_COLUMNS"]
