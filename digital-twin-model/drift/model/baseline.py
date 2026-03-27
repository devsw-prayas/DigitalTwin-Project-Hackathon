"""
baseline.py — Personal Baseline Recalibration (EMA BaselineTracker)

All signals are normalized relative to a person's own slow EMA baseline
BEFORE entering the model. This is Stage 0 of the full architecture pipeline.

Why personal baseline normalization?
    A resting HR of 72 bpm is normal for one person and elevated for another.
    Population norms destroy individual signal. The model should see deviations
    from *your* baseline, not from average humans.

From the spec (Section 6.3 + Appendix A.7):
    slow_t = 0.005 * x_t + 0.995 * slow_{t-1}   # ~140-day half-life
    fast_t = 0.05  * x_t + 0.95  * fast_{t-1}    # ~20-day half-life
    normalized = (x_t - slow_t) / (|fast_t - slow_t| + 1e-6)

Two EMAs:
    slow: long-term personal baseline (what's "normal" for you over months)
    fast: short-term personal baseline (what's "normal" for you this week)
    spread = |fast - slow| captures how much your baseline is currently drifting

Context-aware suspension:
    During illness or travel, signals are physiologically atypical.
    Updating the baseline on these readings would corrupt it.
    The tracker skips updates when these flags are set.

Cold start:
    New users (<14 days) fall back to population norms for initialization.
    After 14 days the personal baseline dominates.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


# Population norms used as cold-start initialization (mean values)
# Keyed by signal name — expand as more signals are added
POPULATION_NORMS: Dict[str, float] = {
    "hrv_rmssd":        55.0,   # ms
    "resting_hr":       65.0,   # bpm
    "spo2":             97.5,   # %
    "skin_temp":        36.6,   # Celsius
    "respiratory_rate": 15.0,   # breaths/min
    "sleep_efficiency": 0.85,   # fraction
    "rem_duration":     1.8,    # hours
    "deep_sleep":       1.2,    # hours
    "steps":            8000.0, # steps/day
    "active_minutes":   30.0,   # min/day
    "screen_off_time":  7.0,    # hours (proxy for non-screen time)
    "aqi_pm25":         12.0,   # µg/m³ (good air quality baseline)
    "ambient_temp":     22.0,   # Celsius
    "stress_score":     0.3,    # normalized [0,1]
}


@dataclass
class SignalBaseline:
    """Tracks slow and fast EMA for a single signal."""
    slow: float      # long-term baseline (~140-day half-life)
    fast: float      # short-term baseline (~20-day half-life)
    n_updates: int = 0  # how many real updates have been applied


class BaselineTracker:
    """
    Per-user EMA baseline tracker.

    Maintains two EMAs per signal:
        alpha_fast = 0.05   ->  half-life ~= ln(2)/0.05 ~= 13.9 days (~20-day)
        alpha_slow = 0.005  ->  half-life ~= ln(2)/0.005 ~= 138.6 days (~140-day)

    Usage:
        tracker = BaselineTracker()
        tracker.initialize(population_norms)         # cold start
        normalized = tracker.update_and_normalize(signal_dict, context_flags)
    """

    ALPHA_FAST: float = 0.05    # ~20-day half-life
    ALPHA_SLOW: float = 0.005   # ~140-day half-life
    COLD_START_DAYS: int = 14   # use population norms until this many updates

    def __init__(self):
        self.baselines: Dict[str, SignalBaseline] = {}
        self._initialize_from_population()

    def _initialize_from_population(self):
        """Seed both EMAs from population norms (cold-start fallback)."""
        for signal, norm in POPULATION_NORMS.items():
            self.baselines[signal] = SignalBaseline(slow=norm, fast=norm)

    def update(self, signal: str, value: float, illness: bool = False, travel: bool = False):
        """
        Update EMA baselines for one signal with one new observation.

        Skips update during illness or travel to prevent baseline corruption.

        Args:
            signal:  signal name (must be in POPULATION_NORMS)
            value:   observed value
            illness: if True, skip update (physiologically atypical)
            travel:  if True, skip update (environment atypical)
        """
        if illness or travel:
            return  # suspend — don't corrupt baseline with atypical readings

        if signal not in self.baselines:
            # New signal: initialize from value itself
            self.baselines[signal] = SignalBaseline(slow=value, fast=value)
            return

        b = self.baselines[signal]
        b.fast = self.ALPHA_FAST * value + (1 - self.ALPHA_FAST) * b.fast
        b.slow = self.ALPHA_SLOW * value + (1 - self.ALPHA_SLOW) * b.slow
        b.n_updates += 1

    def normalize(self, signal: str, value: float) -> float:
        """
        Normalize a value relative to personal baseline.

        normalized = (value - slow) / (|fast - slow| + 1e-6)

        A value of 0.0 means "exactly at your long-term baseline".
        A value of +1.0 means "one spread-unit above your long-term baseline".

        Args:
            signal: signal name
            value:  raw observed value

        Returns:
            normalized float
        """
        if signal not in self.baselines:
            return 0.0  # unknown signal: return neutral

        b = self.baselines[signal]
        spread = abs(b.fast - b.slow) + 1e-6
        return (value - b.slow) / spread

    def update_and_normalize(
        self,
        signals: Dict[str, float],
        illness: bool = False,
        travel: bool = False
    ) -> Dict[str, float]:
        """
        Convenience: update all signals then return normalized dict.

        Args:
            signals: {signal_name: raw_value}
            illness: context flag
            travel:  context flag

        Returns:
            {signal_name: normalized_value}
        """
        for signal, value in signals.items():
            self.update(signal, value, illness=illness, travel=travel)

        return {s: self.normalize(s, v) for s, v in signals.items()}

    def is_cold_start(self, signal: str) -> bool:
        """True if this signal hasn't accumulated enough personal data yet."""
        if signal not in self.baselines:
            return True
        return self.baselines[signal].n_updates < self.COLD_START_DAYS

    def get_state(self) -> Dict:
        """Serialize tracker state for storage (e.g. DB persistence)."""
        return {
            signal: {"slow": b.slow, "fast": b.fast, "n_updates": b.n_updates}
            for signal, b in self.baselines.items()
        }

    @classmethod
    def from_state(cls, state: Dict) -> "BaselineTracker":
        """Restore tracker from serialized state."""
        tracker = cls.__new__(cls)
        tracker.baselines = {
            signal: SignalBaseline(
                slow=v["slow"],
                fast=v["fast"],
                n_updates=v.get("n_updates", 0)
            )
            for signal, v in state.items()
        }
        return tracker


class BatchBaselineNormalizer(torch.nn.Module):
    """
    Vectorized batch version of baseline normalization for training.

    During training with synthetic data, we have pre-computed slow/fast baselines
    for each persona. This module applies the same normalization formula in
    batched tensor form.

    Args:
        n_signals: number of signals being normalized
    """

    def __init__(self, n_signals: int = 20):
        super().__init__()
        self.n_signals = n_signals

    def forward(
        self,
        x: torch.Tensor,        # (B, N, n_signals)   raw signal values
        slow: torch.Tensor,     # (B, N, n_signals)   slow EMA baseline per timestep
        fast: torch.Tensor,     # (B, N, n_signals)   fast EMA baseline per timestep
    ) -> torch.Tensor:
        """
        Apply personal normalization: (x - slow) / (|fast - slow| + eps)

        Returns:
            normalized: (B, N, n_signals)
        """
        spread = (fast - slow).abs() + 1e-6
        return (x - slow) / spread
