"""
heads.py — Prediction Heads

Full architecture uses quantile regression heads (P10/P50/P90), not Gaussian NLL.

From the spec (Section 5, Stage 7):
    Output per specialist, per horizon h in {7d, 30d, 90d, 180d}:
        P10_h, P50_h, P90_h = QuantileHead(H_i)   # risk quantiles
        velocity_h           = VelocityHead(H_i)   # signed slope
        time_to_threshold    = ThresholdHead(H_i)  # days to threshold, >= 0

Why quantile regression instead of Gaussian NLL (which the MVP uses)?
    The Gaussian assumption fails for asymmetric health trajectories.
    Recovery from illness is right-skewed (takes longer than expected).
    Burnout accumulation is left-skewed (faster than expected).
    Quantile regression makes no distributional assumption —
    it directly estimates P10/P50/P90 without assuming a bell curve.

    Gaussian NLL is fine for the MVP demo. The full architecture needs
    honest uncertainty for clinical credibility.

Velocity head:
    Signed slope of risk trajectory — positive = worsening, negative = improving.
    This is more actionable than raw risk scores.
    "Your cardiovascular risk is increasing at +2%/month" is useful.
    "Your risk is 0.34" is not.

Threshold head:
    Estimated days until risk crosses a concern threshold.
    Output is clamped >= 0 (can't be in the past).
    A value of 0 means "already at or past threshold".

Conformalized quantile regression (post-training calibration):
    After training, quantile coverage is measured on held-out data.
    The P10-P90 interval must achieve 80% empirical coverage.
    Temperature scaling per specialist per horizon corrects miscalibration.
    This is applied once after training, not during.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# Prediction horizons in days
HORIZONS = [7, 30, 90, 180]
N_HORIZONS = len(HORIZONS)

# Quantile levels
QUANTILES = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILES)


@dataclass
class AgentPrediction:
    """Structured output from a single agent's prediction heads."""
    # Quantile predictions: (B, N_HORIZONS, N_QUANTILES)
    quantiles: torch.Tensor  # [P10, P50, P90] per horizon

    # Signed velocity (rate of change) per horizon: (B, N_HORIZONS)
    velocity: torch.Tensor

    # Days to risk threshold per horizon: (B, N_HORIZONS), clamped >= 0
    time_to_threshold: torch.Tensor

    @property
    def p10(self) -> torch.Tensor:
        return self.quantiles[..., 0]  # (B, N_HORIZONS)

    @property
    def p50(self) -> torch.Tensor:
        return self.quantiles[..., 1]  # (B, N_HORIZONS)

    @property
    def p90(self) -> torch.Tensor:
        return self.quantiles[..., 2]  # (B, N_HORIZONS)


class QuantileHead(nn.Module):
    """
    Quantile regression prediction head.

    Directly predicts P10, P50, P90 for each horizon without assuming
    a distribution.

    Monotonicity constraint:
        Quantile crossing (P90 < P50 or P50 < P10) is a training instability.
        We enforce monotonicity via cumulative softplus:
            P10 = base
            P50 = P10 + softplus(delta_1)   # always > P10
            P90 = P50 + softplus(delta_2)   # always > P50

    Args:
        d_model:    input hidden dimension
        n_horizons: number of prediction horizons (default: 4)
        d_hidden:   intermediate hidden size
    """

    def __init__(self, d_model: int, n_horizons: int = N_HORIZONS, d_hidden: int = None):
        super().__init__()

        d_hidden = d_hidden or d_model // 2

        # Base quantile (P10) per horizon
        self.base_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_horizons),
        )

        # Positive increments to enforce monotonicity
        # delta_1: P50 - P10  (always positive via softplus)
        # delta_2: P90 - P50  (always positive via softplus)
        self.delta1_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_horizons),
        )
        self.delta2_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_horizons),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_model)  agent representation

        Returns:
            quantiles: (B, N_HORIZONS, 3)  [P10, P50, P90] per horizon
        """
        p10 = self.base_head(h)                          # (B, N_HORIZONS)
        d1 = F.softplus(self.delta1_head(h))             # strictly positive
        d2 = F.softplus(self.delta2_head(h))             # strictly positive

        p50 = p10 + d1                                   # always > P10
        p90 = p50 + d2                                   # always > P50

        # Stack: (B, N_HORIZONS, 3)
        return torch.stack([p10, p50, p90], dim=-1)


class VelocityHead(nn.Module):
    """
    Signed velocity prediction head.

    Predicts the rate of change of risk trajectory at each horizon.
    Positive = risk worsening, Negative = risk improving.

    Used in velocity consistency loss:
        L_vel = || velocity_h - (P50_{h+1} - P50_h) / delta_t ||^2

    Args:
        d_model:    input hidden dimension
        n_horizons: number of prediction horizons
    """

    def __init__(self, d_model: int, n_horizons: int = N_HORIZONS):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_horizons),
            nn.Tanh(),  # constrain to [-1, 1] — velocity is normalized
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_model)

        Returns:
            velocity: (B, N_HORIZONS)  signed slope in [-1, 1]
        """
        return self.head(h)


class ThresholdHead(nn.Module):
    """
    Time-to-threshold prediction head.

    Predicts how many days until risk crosses a concern threshold.
    Clamped >= 0: the threshold is either in the future or already crossed.
    A value of 0 means "at or past threshold now".

    Note: This is not a standard regression — the target has a mass at zero
    (many users never cross threshold). Consider a two-part model for v2.

    Args:
        d_model:    input hidden dimension
        n_horizons: number of prediction horizons
        max_days:   maximum days (clamp upper bound, e.g. 365)
    """

    def __init__(self, d_model: int, n_horizons: int = N_HORIZONS, max_days: float = 365.0):
        super().__init__()
        self.max_days = max_days
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_horizons),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_model)

        Returns:
            days: (B, N_HORIZONS)  time to threshold, clamped to [0, max_days]
        """
        raw = self.head(h)
        return torch.clamp(raw, min=0.0, max=self.max_days)


class AgentHead(nn.Module):
    """
    Combined prediction head for one specialist agent.
    Wraps QuantileHead + VelocityHead + ThresholdHead.

    Args:
        d_model:    input hidden dimension
        n_horizons: number of prediction horizons
    """

    def __init__(self, d_model: int, n_horizons: int = N_HORIZONS):
        super().__init__()
        self.quantile_head = QuantileHead(d_model, n_horizons)
        self.velocity_head = VelocityHead(d_model, n_horizons)
        self.threshold_head = ThresholdHead(d_model, n_horizons)

    def forward(self, h: torch.Tensor) -> AgentPrediction:
        """
        Args:
            h: (B, d_model)

        Returns:
            AgentPrediction with quantiles, velocity, time_to_threshold
        """
        return AgentPrediction(
            quantiles=self.quantile_head(h),
            velocity=self.velocity_head(h),
            time_to_threshold=self.threshold_head(h),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Post-training calibration (conformalized quantile regression)
# ─────────────────────────────────────────────────────────────────────────────

class QuantileCalibrator(nn.Module):
    """
    Post-training temperature scaling for quantile calibration.

    After training, if the P10-P90 interval doesn't achieve 80% empirical
    coverage, this module learns a per-specialist-per-horizon temperature
    that stretches/shrinks the interval.

    From the spec (Stage 8):
        Quantile coverage measured on held-out validation data.
        Corrected via conformalized quantile regression.
        P10-P90 interval must achieve 80% empirical coverage.

    Usage:
        calibrator = QuantileCalibrator(n_agents=2, n_horizons=4)
        # After training, fit calibrator on validation set
        # During inference, pass quantiles through calibrator before serving

    Args:
        n_agents:   number of specialist agents
        n_horizons: number of prediction horizons
    """

    def __init__(self, n_agents: int = 2, n_horizons: int = N_HORIZONS):
        super().__init__()

        # Temperature per agent per horizon: initialized to 1.0 (no-op)
        # Trained via ECE minimization on validation data
        self.temperature = nn.Parameter(
            torch.ones(n_agents, n_horizons)
        )

    def forward(self, quantiles: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """
        Apply temperature scaling to stretch/shrink the quantile interval.

        Args:
            quantiles:  (B, N_HORIZONS, 3)  [P10, P50, P90]
            agent_idx:  which agent (for temperature lookup)

        Returns:
            calibrated: (B, N_HORIZONS, 3)  calibrated [P10, P50, P90]
        """
        temp = self.temperature[agent_idx]  # (N_HORIZONS,)

        p10 = quantiles[..., 0]
        p50 = quantiles[..., 1]
        p90 = quantiles[..., 2]

        # Stretch interval around P50 by temperature
        # temp > 1: wider interval (more uncertain)
        # temp < 1: narrower interval (more confident)
        p10_cal = p50 - temp * (p50 - p10)
        p90_cal = p50 + temp * (p90 - p50)

        return torch.stack([p10_cal, p50, p90_cal], dim=-1)
