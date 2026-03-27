"""
simulate.py — Counterfactual Simulation Engine

Allows users to ask: "What happens to my risk trajectory if I sleep 1 more hour
every night for the next 6 months?"

From the spec (Section 6.6):

Simulation protocol:
    1. Take real token sequence X = [X_{t-N}, ..., X_t]
    2. Construct future tokens F by applying adherence-shaped delta to personal baseline
    3. Append X_cf = concat(X, F), shape (N+K, 104)
    4. Run forward pass; compare agent outputs vs baseline
    5. Report: delta trajectory, best leverage horizon, leverage score per specialist

Adherence profiles (from spec):
    'optimistic':    lambda t, d: d                      # full adherence forever
    'realistic':     lambda t, d: d * exp(-t/60)         # 60-day half-life
    'conservative':  lambda t, d: d * exp(-t/30)         # 30-day half-life
    'ramp':          lambda t, d: d * min(1, t/14)       # 2-week ramp-up

Why adherence modeling?
    Without decay, the simulator would show unrealistically large benefits.
    "If you sleep 8h every night forever, your risk drops 40%" is not useful.
    "If you increase sleep realistically (adherence decays over 60 days),
     your risk drops 11% at 90 days" is actionable.

Leverage score:
    Normalized measure of intervention impact.
    leverage = (baseline_risk - cf_risk) / baseline_risk
    A score of 0.1 means a 10% relative improvement.
    Used to gate nudge generation (only nudge if leverage > threshold).

Available interventions:
    sleep_hours:    +X hours per night (delta on sleep_efficiency, rem_duration, deep_sleep)
    steps:          +X steps per day (delta on steps, active_minutes)
    stress:         X% reduction (delta on stress_score, hrv_rmssd)
    screen_time:    -X hours (delta on screen_off_time)
    aqi:            move to lower AQI environment (delta on aqi_pm25)
"""

import torch
import math
from dataclasses import dataclass
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Adherence profiles
# ─────────────────────────────────────────────────────────────────────────────

ADHERENCE_PROFILES: dict[str, Callable[[int, float], float]] = {
    "optimistic":   lambda t, d: d,
    "realistic":    lambda t, d: d * math.exp(-t / 60),
    "conservative": lambda t, d: d * math.exp(-t / 30),
    "ramp":         lambda t, d: d * min(1.0, t / 14),
}


def adherence_decay(
    t: int,                     # day index (0 = first day of intervention)
    delta: float,               # intended daily delta
    profile: str = "realistic",
) -> float:
    """
    Apply adherence decay to an intended intervention delta.

    Args:
        t:       days since intervention start
        delta:   intended change (e.g. +1.0 hour of sleep)
        profile: adherence profile name

    Returns:
        effective delta on day t
    """
    fn = ADHERENCE_PROFILES.get(profile, ADHERENCE_PROFILES["realistic"])
    return fn(t, delta)


# ─────────────────────────────────────────────────────────────────────────────
# Intervention definitions
# ─────────────────────────────────────────────────────────────────────────────

# Map intervention name -> which token dimensions it affects and how
# Token layout (from spec §6.2):
#   [0:32]  physio embedding (8 signals projected to 32)
#   [32:56] behavior embedding (6 signals projected to 24)
#   [56:72] environment embedding (4 signals projected to 16)
#   [72:88] context embedding (8 ctx projected to 16)
#   [88:104] uncertainty encoding (20 signals projected to 16)
#
# Since tokens are already projected embeddings (not raw signals), we apply
# deltas proportionally to the relevant embedding subspace.
# In production with a tokenizer, you'd apply deltas pre-tokenization.

INTERVENTION_DIMS: dict[str, dict] = {
    "sleep_hours": {
        "description": "+X hours of sleep per night",
        "dim_start": 0,      # physio subspace (sleep stages are physio signals)
        "dim_end": 32,
        "scale": 0.1,        # rough embedding-space magnitude per +1 hour
        "direction": +1,     # positive = beneficial (more sleep = lower risk)
    },
    "steps": {
        "description": "+X steps per day",
        "dim_start": 32,     # behavior subspace
        "dim_end": 56,
        "scale": 0.02,       # per +1000 steps
        "direction": +1,
    },
    "stress": {
        "description": "X% stress reduction",
        "dim_start": 72,     # context subspace
        "dim_end": 88,
        "scale": 0.05,       # per 10% reduction
        "direction": -1,     # negative stress delta = positive health direction
    },
    "screen_time": {
        "description": "-X hours screen time per day",
        "dim_start": 32,     # behavior subspace
        "dim_end": 56,
        "scale": 0.03,
        "direction": +1,     # less screen time = better
    },
    "aqi": {
        "description": "Move to lower AQI environment",
        "dim_start": 56,     # environment subspace
        "dim_end": 72,
        "scale": 0.05,       # per 10 µg/m³ reduction
        "direction": -1,     # lower AQI = better
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Output from one counterfactual simulation run.

    Attributes:
        baseline:    dict { agent_name: {"p50": (N_HORIZONS,), "p10": ..., "p90": ...} }
        counterfactual: same structure, under the intervention
        delta:       absolute risk change: cf_p50 - baseline_p50
        leverage:    relative improvement: (baseline - cf) / baseline
        best_horizon: horizon index with highest leverage per agent
        adherence_profile: which profile was used
        intervention: what was simulated
    """
    baseline: dict
    counterfactual: dict
    delta: dict               # per agent per horizon
    leverage: dict            # per agent per horizon
    best_horizon: dict        # per agent
    adherence_profile: str
    intervention: dict


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class CounterfactualSimulator:
    """
    Runs counterfactual simulations over the DRIFT model.

    Usage:
        simulator = CounterfactualSimulator(model, device="cuda")
        result = simulator.simulate(
            X=token_sequence,           # (1, 90, 104)
            interventions={"sleep_hours": 1.0, "steps": 2000},
            n_future=180,               # simulate 180 days forward
            adherence_profile="realistic",
        )

    Args:
        model:   trained DRIFT model
        device:  computation device
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def simulate(
        self,
        X: torch.Tensor,                    # (1, N, 104)  real token sequence
        interventions: dict[str, float],    # {intervention_name: magnitude}
        n_future: int = 180,                # days to simulate forward
        adherence_profile: str = "realistic",
        static_input: torch.Tensor = None,
    ) -> SimulationResult:
        """
        Run counterfactual simulation.

        Args:
            X:                  (1, N, 104)  observed token sequence (last 90 days)
            interventions:      dict of intervention name -> magnitude
                                e.g. {"sleep_hours": 1.0, "steps": 2000}
            n_future:           how many future days to simulate
            adherence_profile:  "realistic", "optimistic", "conservative", "ramp"
            static_input:       (1, d_static_raw) user metadata

        Returns:
            SimulationResult with baseline vs counterfactual trajectories
        """
        self.model.eval()
        X = X.to(self.device)

        # --- Baseline forward pass ---
        baseline_output = self.model(X, static_input=static_input, return_weights=False)

        # --- Build counterfactual future tokens ---
        X_cf = self._build_counterfactual_sequence(
            X, interventions, n_future, adherence_profile
        )  # (1, N + n_future, 104)

        # --- Counterfactual forward pass ---
        cf_output = self.model(X_cf, static_input=static_input, return_weights=False)

        # --- Compute delta and leverage ---
        result = self._compute_result(
            baseline_output, cf_output, interventions, adherence_profile
        )
        return result

    def _build_counterfactual_sequence(
        self,
        X: torch.Tensor,
        interventions: dict[str, float],
        n_future: int,
        adherence_profile: str,
    ) -> torch.Tensor:
        """
        Extend the observed sequence with intervention-modified future tokens.

        For each future timestep t:
            future_token = last_observed_token + sum(adherence_delta(t, intervention))

        Args:
            X:               (1, N, 104)
            interventions:   intervention dict
            n_future:        number of future days
            adherence_profile: decay profile name

        Returns:
            X_cf: (1, N + n_future, 104)  — truncated to last 90 if longer
        """
        B, N, D = X.shape

        # Anchor: use mean of last 7 days as the "current baseline" for future tokens
        anchor = X[:, -7:, :].mean(dim=1, keepdim=True)  # (1, 1, 104)

        future_tokens = []
        for t in range(n_future):
            token = anchor.clone()  # (1, 1, 104)

            # Apply each intervention with adherence decay
            for name, magnitude in interventions.items():
                if name not in INTERVENTION_DIMS:
                    continue

                spec = INTERVENTION_DIMS[name]
                effective_delta = adherence_decay(t, magnitude, adherence_profile)
                effective_delta *= spec["scale"] * spec["direction"]

                # Apply delta to the relevant embedding subspace
                d_start = spec["dim_start"]
                d_end = spec["dim_end"]
                token[:, :, d_start:d_end] += effective_delta

            future_tokens.append(token)

        future_tokens = torch.cat(future_tokens, dim=1)  # (1, n_future, 104)

        # Concatenate: observed + future
        X_extended = torch.cat([X, future_tokens], dim=1)  # (1, N + n_future, 104)

        # The model expects a fixed-length window — use the last 90 timesteps
        if X_extended.shape[1] > 90:
            X_cf = X_extended[:, -90:, :]
        else:
            X_cf = X_extended

        return X_cf

    def _compute_result(
        self,
        baseline_output,
        cf_output,
        interventions: dict,
        adherence_profile: str,
    ) -> SimulationResult:
        """Compute delta, leverage, and best horizon from baseline vs cf outputs."""
        from model.model import CORE_SPECIALISTS

        baseline_dict = {}
        cf_dict = {}
        delta_dict = {}
        leverage_dict = {}
        best_horizon_dict = {}

        for name in CORE_SPECIALISTS:
            if name not in baseline_output.agents:
                continue

            b_pred = baseline_output.agents[name]
            c_pred = cf_output.agents[name]

            b_p50 = b_pred.p50.squeeze(0).cpu()  # (N_HORIZONS,)
            c_p50 = c_pred.p50.squeeze(0).cpu()

            baseline_dict[name] = {
                "p10": b_pred.p10.squeeze(0).cpu(),
                "p50": b_p50,
                "p90": b_pred.p90.squeeze(0).cpu(),
                "velocity": b_pred.velocity.squeeze(0).cpu(),
            }
            cf_dict[name] = {
                "p10": c_pred.p10.squeeze(0).cpu(),
                "p50": c_p50,
                "p90": c_pred.p90.squeeze(0).cpu(),
                "velocity": c_pred.velocity.squeeze(0).cpu(),
            }

            # Delta: positive = risk reduced (improvement)
            delta = b_p50 - c_p50
            delta_dict[name] = delta

            # Leverage: relative improvement
            leverage = delta / (b_p50.abs() + 1e-6)
            leverage_dict[name] = leverage

            # Best horizon: where leverage is highest
            best_horizon_dict[name] = leverage.argmax().item()

        return SimulationResult(
            baseline=baseline_dict,
            counterfactual=cf_dict,
            delta=delta_dict,
            leverage=leverage_dict,
            best_horizon=best_horizon_dict,
            adherence_profile=adherence_profile,
            intervention=interventions,
        )

    def format_summary(self, result: SimulationResult) -> str:
        """
        Format simulation result as a human-readable summary.

        Example output:
            Cardiovascular: -11% at 90d horizon
            Mental: -8% at 30d horizon
            Recovery fastest (leverage: +0.14 at 7d)
        """
        from model.heads import HORIZONS

        lines = []
        for name, leverage in result.leverage.items():
            best_h_idx = result.best_horizon[name]
            best_leverage = leverage[best_h_idx].item()
            best_days = HORIZONS[best_h_idx]
            delta_pct = result.delta[name][best_h_idx].item() * 100

            direction = "↓" if delta_pct > 0 else "↑"
            lines.append(
                f"{name.capitalize()}: {direction}{abs(delta_pct):.1f}% "
                f"at {best_days}d | leverage={best_leverage:.2f}"
            )

        return "\n".join(lines)
