"""Chemical gas sensor model with asymmetric rise/decay lag, noise, and NaN dropouts."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class GasSensor:
    """1st-order asymmetric chemical sensor with multiplicative noise and fault injection."""

    def __init__(
        self,
        tau_rise: float = 0.6,
        tau_decay: float = 1.8,
        noise_std_frac: float = 0.08,
        baseline_ppm: float = 0.5,
        dropout_prob: float = 0.30,
        seed: int = 202,
    ) -> None:
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.noise_std_frac = noise_std_frac
        self.baseline_ppm = baseline_ppm
        self.dropout_prob = dropout_prob
        self._rng = np.random.default_rng(seed)

        self.filtered_ppm: float = 0.0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.filtered_ppm = 0.0

    def step(self, true_ppm: float, dt: float) -> float:
        """Advance asymmetric 1st-order chemical lag filter by dt seconds."""
        tau = self.tau_rise if true_ppm >= self.filtered_ppm else self.tau_decay
        alpha = 1.0 - math.exp(-dt / tau)
        self.filtered_ppm += alpha * (true_ppm - self.filtered_ppm)
        return self.filtered_ppm

    def read(
        self,
        wind_local: np.ndarray,
        dropout_active: bool = False,
    ) -> tuple[float, tuple[Optional[float], Optional[float]], Optional[float]]:
        """Sample noisy ppm, wind_measured, and temp_delta (with optional NaN dropouts)."""
        noise_mult = 1.0 + float(self._rng.normal(0.0, self.noise_std_frac))
        ppm_measured = max(self.baseline_ppm, self.filtered_ppm * noise_mult + self.baseline_ppm)

        # Slight temperature anomaly correlated with gas concentration + sensor noise
        temp_delta: Optional[float] = float(
            0.012 * math.log1p(self.filtered_ppm) + self._rng.normal(0.0, 0.08)
        )

        wx = float(wind_local[0] + self._rng.normal(0.0, 0.15))
        wy = float(wind_local[1] + self._rng.normal(0.0, 0.15))
        wind_measured: tuple[Optional[float], Optional[float]] = (wx, wy)

        if dropout_active:
            if float(self._rng.random()) < self.dropout_prob:
                temp_delta = None
            if float(self._rng.random()) < self.dropout_prob:
                wind_measured = (None, None)

        return ppm_measured, wind_measured, temp_delta
