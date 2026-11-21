"""Realistic chemical gas, thermal, and ultrasonic anemometer sensor suite.

Features:
- Asymmetric 1st-order chemical adsorption/desorption lag (tau_rise=0.6s, tau_decay=1.8s)
- Multiplicative Gaussian noise (±8%), baseline floor (0.5 ppm), slow zero-drift,
  sensor saturation (5000 ppm), and 0.01 ppm ADC quantization
- Lagged thermal anomaly channel (tau_temp=1.2s)
- Drone-tilt-coupled ultrasonic anemometer noise
- Toggleable 30% NaN dropout fault injection on wind_measured and temp_delta
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class GasSensor:
    """Asymmetric 1st-order chemical sensor with realistic noise, drift, saturation, and faults."""

    MAX_PPM: float = 5000.0
    ADC_STEP_PPM: float = 0.01
    TAU_TEMP: float = 1.2

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
        self.filtered_temp_delta: float = 0.0
        self._zero_drift: float = 0.0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.filtered_ppm = 0.0
        self.filtered_temp_delta = 0.0
        self._zero_drift = 0.0

    def step(self, true_ppm: float, dt: float) -> float:
        """Advance asymmetric 1st-order chemical lag, thermal lag, and slow baseline drift."""
        clamped_ppm = max(0.0, min(self.MAX_PPM, float(true_ppm)))
        tau = self.tau_rise if clamped_ppm >= self.filtered_ppm else self.tau_decay
        alpha = 1.0 - math.exp(-dt / tau)
        self.filtered_ppm += alpha * (clamped_ppm - self.filtered_ppm)

        # 1st-order lagged thermal anomaly correlated with gas plume concentration
        target_temp = 0.012 * math.log1p(self.filtered_ppm)
        alpha_t = 1.0 - math.exp(-dt / self.TAU_TEMP)
        self.filtered_temp_delta += alpha_t * (target_temp - self.filtered_temp_delta)

        # Slow bounded OU zero-drift (±0.08 ppm)
        self._zero_drift += (-0.02 * self._zero_drift * dt) + 0.008 * math.sqrt(dt) * float(
            self._rng.normal(0.0, 1.0)
        )
        self._zero_drift = max(-0.08, min(0.08, self._zero_drift))

        return self.filtered_ppm

    def read(
        self,
        wind_local: np.ndarray,
        dropout_active: bool = False,
        drone_tilt_rad: float = 0.0,
    ) -> tuple[float, tuple[Optional[float], Optional[float]], Optional[float]]:
        """Sample noisy ppm, wind_measured, and temp_delta (with optional NaN dropouts)."""
        noise_mult = 1.0 + float(self._rng.normal(0.0, self.noise_std_frac))
        raw_ppm = self.filtered_ppm * noise_mult + self.baseline_ppm + self._zero_drift
        clamped_ppm = max(self.baseline_ppm, min(self.MAX_PPM, raw_ppm))
        # Realistic ADC quantization (0.01 ppm resolution)
        ppm_measured = round(clamped_ppm / self.ADC_STEP_PPM) * self.ADC_STEP_PPM

        # Thermal anomaly channel with measurement noise
        temp_delta: Optional[float] = float(
            self.filtered_temp_delta + self._rng.normal(0.0, 0.08)
        )

        # Ultrasonic anemometer: higher tilt slightly degrades horizontal wind measurement
        anem_std = 0.15 + 0.30 * abs(math.sin(drone_tilt_rad))
        wx = float(wind_local[0] + self._rng.normal(0.0, anem_std))
        wy = float(wind_local[1] + self._rng.normal(0.0, anem_std))
        wind_measured: tuple[Optional[float], Optional[float]] = (wx, wy)

        if dropout_active:
            if float(self._rng.random()) < self.dropout_prob:
                temp_delta = None
            if float(self._rng.random()) < self.dropout_prob:
                wind_measured = (None, None)

        return ppm_measured, wind_measured, temp_delta
