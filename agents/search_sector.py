"""Disjoint horizontal search sectors for the simulator's drone team."""

import numpy as np

from shared.schemas import WorldInitPayload


def sector_mask(
    points: np.ndarray, world: WorldInitPayload, sector: int, count: int
) -> np.ndarray:
    """Assign each point to exactly one west-to-east sector, at every altitude."""
    cuts = np.linspace(world.bounds_min[0], world.bounds_max[0], count + 1)[1:-1]
    return np.searchsorted(cuts, points[:, 0], side="right") == sector
