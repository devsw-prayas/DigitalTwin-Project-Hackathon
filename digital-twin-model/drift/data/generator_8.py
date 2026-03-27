"""
generator_8.py — Ultra-Hard 8-Agent Data Generator

Generates extremely diverse, challenging data that forces specialization:
- 8 independent agent ground truths
- Maximum divergence between agents
- Anti-correlated patterns
- Independent events per agent
- High noise with structure preservation
- No single-feature shortcuts

Key design principles:
1. Each agent has PRIMARY signals that ONLY it responds to strongly
2. Agents can move independently, together, or in opposite directions
3. Events affect specific agents, not all
4. Noise is high but structured
5. No universal shortcut features
"""

from __future__ import annotations
import os
import math
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml

from .agents_8 import (
    ALL_AGENTS, N_AGENTS, ALL_SIGNALS, N_SIGNALS,
    AGENT_SIGNAL_COUPLING, AGENT_TEMPORAL,
    AgentGroundTruth, Sample8Agent,
    compute_divergence_score,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ═══════════════════════════════════════════════════════════════════════════════
# PERSONA DEFINITIONS FOR 8 AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentParams:
    """Parameters for one agent's ground truth curve."""
    base: float = 0.2                  # Baseline risk
    drift_per_day: float = 0.0         # Linear drift
    event_day: int = -1                # Event timing (-1 = no event)
    event_magnitude: float = 0.0       # Event magnitude (can be negative)
    event_width_days: float = 20.0     # Event duration
    seasonal_amplitude: float = 0.02   # Seasonal variation
    volatility: float = 0.2            # Day-to-day noise


@dataclass
class Persona8:
    """Extended persona with 8-agent parameters."""
    name: str
    description: str
    seed: int
    
    # Per-agent parameters
    agents: Dict[str, AgentParams] = field(default_factory=dict)
    
    # Signal baselines
    signals: Dict[str, float] = field(default_factory=dict)
    
    # Per-signal noise std
    noise: Dict[str, float] = field(default_factory=dict)
    
    # Special patterns
    recovery_days: Optional[int] = None
    cycle_period_days: Optional[int] = None
    aqi_spike_count: int = 0
    aqi_spike_magnitude: float = 0.0
    aqi_spike_width_days: float = 21.0
    flare_count: int = 0
    flare_magnitude: float = 0.0
    flare_width_days: float = 21.0
    travel_count: int = 0
    travel_width_days: float = 10.0
    
    # Routing target (which expert should handle this persona primarily)
    primary_agent: str = "cardio"
    
    # Divergence type
    divergence_type: str = "independent"  # "independent", "opposite", "coupled", "mixed"


def _arange(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=DTYPE, device=DEVICE)


def _randn(n: int, std: float, offset: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed((seed * 31337 + offset * 1009) % (2 ** 31))
    return torch.randn(n, generator=g, dtype=DTYPE).to(DEVICE) * std


def _slow_ema(x: torch.Tensor, alpha: float = 0.07) -> torch.Tensor:
    """Causal EMA for slow features."""
    out = torch.zeros_like(x)
    val = x[0].item()
    for i in range(len(x)):
        val = alpha * x[i].item() + (1 - alpha) * val
        out[i] = val
    return out


def _fast_response(x: torch.Tensor, responsiveness: float = 0.5) -> torch.Tensor:
    """Fast response filter for acute features."""
    out = torch.zeros_like(x)
    for i in range(1, len(x)):
        out[i] = out[i-1] + responsiveness * (x[i].item() - out[i-1])
    out[0] = x[0]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _base_risk(t: torch.Tensor, params: AgentParams, phase: float = 0.0) -> torch.Tensor:
    """Generate base risk curve."""
    r = params.base + params.drift_per_day * t
    r = r + params.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365 + phase)
    
    if params.event_day >= 0:
        r = r + params.event_magnitude * torch.exp(
            -0.5 * ((t - params.event_day) / params.event_width_days) ** 2
        )
    
    return r


def generate_agent_gt(
    t: torch.Tensor,
    agent_name: str,
    params: AgentParams,
    persona: Persona8,
    seed: int,
) -> AgentGroundTruth:
    """Generate ground truth for one agent with unique temporal dynamics."""
    n = len(t)
    temporal = AGENT_TEMPORAL[agent_name]
    
    # Base risk
    phase = hash(agent_name) % 10 * 0.3  # Different phase per agent
    r = _base_risk(t, params, phase)
    
    # Agent-specific patterns
    events = []
    
    if agent_name == "cardio":
        # Slow sustained changes
        if persona.aqi_spike_count > 0:
            for spike_day in _spike_days(n, persona, seed):
                mag = 0.15 + 0.1 * (hash(str(seed) + agent_name) % 100) / 100
                r = r + mag * torch.exp(-0.5 * ((t - spike_day) / 7.0) ** 2)
                events.append((spike_day, mag, 7.0))
    
    elif agent_name == "mental":
        # Fast acute spikes
        if persona.flare_count > 0:
            for flare_day in _flare_days(n, persona, seed + 100):
                mag = 0.20 + 0.15 * (hash(str(seed) + agent_name) % 100) / 100
                r = r + mag * torch.exp(-0.5 * ((t - flare_day) / 5.0) ** 2)
                events.append((flare_day, mag, 5.0))
    
    elif agent_name == "metabolic":
        # Medium-term drift with step changes
        if persona.travel_count > 0:
            for travel_day in _travel_days(n, persona, seed + 200):
                mag = 0.12 + 0.08 * (hash(str(seed) + agent_name) % 100) / 100
                # Step change, not spike
                r = r + mag * (t > travel_day).float()
                events.append((travel_day, mag, 30.0))
    
    elif agent_name == "recovery":
        # Exponential recovery curves
        if persona.recovery_days and persona.recovery_days > 0:
            rd = persona.recovery_days
            plateau = params.base + params.drift_per_day * t
            r = plateau + (params.base - plateau) * torch.exp(-t / (rd / 3.0))
    
    elif agent_name == "immune":
        # Acute drops with gradual recovery
        if persona.flare_count > 0:
            for flare_day in _flare_days(n, persona, seed + 300):
                mag = 0.25 + 0.15 * (hash(str(seed) + agent_name) % 100) / 100
                # Sharp drop, slow recovery
                drop = -mag * torch.exp(-0.5 * ((t - flare_day) / 3.0) ** 2)
                recovery = mag * 0.5 * (1 - torch.exp(-(t - flare_day).clamp(min=0) / 14.0))
                r = r + drop + recovery * (t > flare_day).float()
                events.append((flare_day, mag, 14.0))
    
    elif agent_name == "respiratory":
        # AQI-driven spikes (stronger than cardio)
        if persona.aqi_spike_count > 0:
            for spike_day in _spike_days(n, persona, seed + 400):
                mag = 0.20 + 0.12 * (hash(str(seed) + agent_name) % 100) / 100
                r = r + mag * torch.exp(-0.5 * ((t - spike_day) / 10.0) ** 2)
                events.append((spike_day, mag, 10.0))
    
    elif agent_name == "hormonal":
        # Cyclical pattern
        period = persona.cycle_period_days or 28
        cycle_amp = 0.12 + 0.08 * (params.seasonal_amplitude * 10)
        r = r + cycle_amp * (0.5 - 0.5 * torch.cos(2 * torch.pi * t / period))
    
    elif agent_name == "cog_fatigue":
        # Cumulative sleep debt pattern
        if persona.flare_count > 0:
            for flare_day in _flare_days(n, persona, seed + 500):
                mag = 0.18 + 0.12 * (hash(str(seed) + agent_name) % 100) / 100
                # Cumulative buildup, then recovery
                buildup = mag * (1 - torch.exp(-(t - flare_day + 14).clamp(min=0, max=14) / 7.0))
                recovery = mag * torch.exp(-((t - flare_day - 14).clamp(min=0)) / 10.0)
                mask = (t > flare_day - 14).float()
                r = r + (buildup - recovery * (t > flare_day + 14).float()) * mask
                events.append((flare_day, mag, 21.0))
    
    # Add agent-specific volatility (noise)
    vol = temporal["volatility"] * params.volatility
    noise = _randn(n, vol * 0.3, hash(agent_name) % 1000, seed)
    r = r + noise
    
    # Store event info
    if params.event_day >= 0:
        events.append((params.event_day, params.event_magnitude, params.event_width_days))
    
    return AgentGroundTruth(
        name=agent_name,
        values=r.clamp(0.0, 1.0),
        base=params.base,
        drift_per_day=params.drift_per_day,
        events=events,
        primary_pattern=temporal["timescale"],
    )


def _spike_days(n: int, p: Persona8, seed: int) -> List[int]:
    count = p.aqi_spike_count
    if count == 0:
        return []
    g = torch.Generator()
    g.manual_seed(seed + 9999)
    spacing = n // (count + 1)
    return [
        max(30, min(n - 30, i * spacing + int((torch.rand(1, generator=g).item() - 0.5) * spacing // 2)))
        for i in range(1, count + 1)
    ]


def _flare_days(n: int, p: Persona8, seed: int) -> List[int]:
    count = p.flare_count
    if count == 0:
        return []
    g = torch.Generator()
    g.manual_seed(seed + 8888)
    spacing = n // (count + 1)
    return [
        max(30, min(n - 30, i * spacing + int((torch.rand(1, generator=g).item() - 0.5) * spacing // 3)))
        for i in range(1, count + 1)
    ]


def _travel_days(n: int, p: Persona8, seed: int) -> List[int]:
    count = p.travel_count
    if count == 0:
        return []
    g = torch.Generator()
    g.manual_seed(seed + 7777)
    spacing = n // (count + 1)
    return [
        max(20, min(n - 20, i * spacing + int((torch.rand(1, generator=g).item() - 0.5) * spacing // 3)))
        for i in range(1, count + 1)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION (MAXIMUM DIVERSITY)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    p: Persona8,
    agent_gts: Dict[str, AgentGroundTruth],
    t: torch.Tensor,
    seed: int,
    aqi_curve: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate signals with STRICT primary/secondary coupling.
    
    Key: Each signal responds strongly to ONE agent, weakly to others.
    This prevents shortcut features.
    """
    n = len(t)
    s = p.signals
    ns = p.noise
    
    def noise(key: str, off: int) -> torch.Tensor:
        return _randn(n, ns.get(key, 5.0), off, seed)
    
    # Get agent GTs
    cardio = agent_gts["cardio"].values
    mental = agent_gts["mental"].values
    metabolic = agent_gts["metabolic"].values
    recovery = agent_gts["recovery"].values
    immune = agent_gts["immune"].values
    respiratory = agent_gts["respiratory"].values
    hormonal = agent_gts["hormonal"].values
    cog_fatigue = agent_gts["cog_fatigue"].values
    
    # Create slow/fast versions for temporal diversity
    cardio_slow = _slow_ema(cardio, alpha=0.05)
    mental_fast = _fast_response(mental, responsiveness=0.7)
    immune_acute = _fast_response(immune, responsiveness=0.8)
    cog_cumulative = _slow_ema(cog_fatigue, alpha=0.1)
    
    # ═══ HRV: Multi-agent with distinct temporal signatures ═══
    # Cardio: slow suppression, Mental: fast dips, Immune: acute drops, Hormonal: cyclical
    hrv = (
        s["hrv_rmssd_ms"]
        * (1 - 0.50 * cardio_slow)      # Cardio: slow sustained
        * (1 - 0.18 * mental_fast)      # Mental: fast acute
        * (1 - 0.25 * immune_acute)     # Immune: acute drops
        * (1 - 0.12 * hormonal)         # Hormonal: cyclical
        + noise("hrv_rmssd_ms", 10)
    )
    
    # ═══ Resting HR: CARDIO-PRIMARY + Immune secondary ═══
    aqi_hr = 0.0
    if aqi_curve is not None:
        aqi_hr = 0.03 * (aqi_curve - 20).clamp(min=0)
    
    hr = (
        s["resting_hr_bpm"]
        + 25 * cardio_slow              # Cardio: strong slow
        + 12 * immune_acute             # Immune: acute spikes during illness
        + 4 * mental_fast               # Mental: weak sympathetic
        + aqi_hr
        + noise("resting_hr_bpm", 20)
    )
    
    # ═══ SpO2: RESPIRATORY-PRIMARY + Cardio secondary ═══
    aqi_spo2 = torch.zeros(n, device=DEVICE)
    if aqi_curve is not None:
        aqi_spo2 = 0.012 * (aqi_curve - 20).clamp(min=0)
    
    spo2 = (
        s["spo2_pct"]
        - 2.0 * respiratory             # Respiratory: strong
        - 1.2 * cardio_slow             # Cardio: moderate
        - aqi_spo2
        - 0.3 * immune_acute            # Immune: mild during illness
        + noise("spo2_pct", 30)
    )
    
    # ═══ Sleep Efficiency: MENTAL-PRIMARY + CogFatigue + Recovery ═══
    sleep = (
        s["sleep_efficiency_pct"] / 100
        - 0.30 * mental_fast            # Mental: strong
        - 0.22 * cog_cumulative         # CogFatigue: cumulative
        - 0.12 * recovery               # Recovery: moderate
        - 0.08 * hormonal               # Hormonal: mild cyclical
        + noise("sleep_efficiency_pct", 40)
    )
    
    # ═══ REM: MENTAL-PRIMARY + Hormonal + CogFatigue ═══
    rem = (
        s["rem_min"]
        - 48 * mental_fast              # Mental: strong
        - 18 * hormonal                 # Hormonal: cyclical
        - 15 * cog_cumulative           # CogFatigue: cumulative
        - 5 * immune_acute              # Immune: mild
        + noise("rem_min", 50)
    )
    
    # ═══ Deep Sleep: RECOVERY-PRIMARY + Metabolic + CogFatigue ═══
    deep = (
        s["deep_min"]
        - 25 * recovery                 # Recovery: strong
        - 18 * metabolic                # Metabolic: moderate
        - 12 * cog_cumulative           # CogFatigue: moderate
        - 8 * cardio_slow               # Cardio: mild
        + noise("deep_min", 60)
    )
    
    # ═══ Steps: METABOLIC-PRIMARY + Recovery + Cardio ═══
    if p.recovery_days and p.recovery_days > 0 and p.name in ["recovering", "burned_out_runner"]:
        step_base = s["steps"] * (t / p.recovery_days).clamp(0, 1) ** 1.5
    else:
        step_base = torch.full((n,), s["steps"], dtype=DTYPE, device=DEVICE)
    
    steps = (
        step_base
        * (1 - 0.38 * metabolic)        # Metabolic: strong
        * (1 - 0.25 * recovery)         # Recovery: moderate
        * (1 - 0.15 * cardio_slow)      # Cardio: mild
        * (1 - 0.08 * immune_acute)     # Immune: illness reduces
        + noise("steps", 70)
    )
    
    # ═══ Active Minutes: METABOLIC-PRIMARY + Cardio + Respiratory ═══
    active = (
        s["active_min"]
        * (1 - 0.42 * metabolic)        # Metabolic: strong
        * (1 - 0.28 * cardio_slow)      # Cardio: moderate
        * (1 - 0.15 * respiratory)      # Respiratory: AQI effect
        * (1 - 0.10 * recovery)         # Recovery: mild
        + noise("active_min", 80)
    )
    
    # ═══ Screen Off: COG_FATIGUE-PRIMARY + Mental ═══
    screen = (
        s["screen_off_min"]
        + 95 * cog_cumulative           # CogFatigue: strong (MORE screen = worse)
        + 70 * mental_fast              # Mental: strong
        + 18 * hormonal                 # Hormonal: mild cyclical
        + noise("screen_off_min", 90)
    )
    
    # ═══ AQI: Environmental (affects Respiratory, Cardio) ═══
    if aqi_curve is not None:
        aqi_out = aqi_curve
    else:
        aqi_out = (
            s["aqi_pm25"]
            + 8 * torch.sin(2 * torch.pi * t / 365 + torch.pi)
            + noise("aqi_pm25", 100)
        )
    
    # ═══ Temperature: Hormonal cyclical + seasonal ═══
    temp = (
        s["ambient_temp_c"]
        + 8 * torch.sin(2 * torch.pi * t / 365)
        + 0.3 * hormonal * torch.sin(2 * torch.pi * t / (p.cycle_period_days or 28))
        + noise("ambient_temp_c", 110)
    )
    
    return {
        "hrv_rmssd_ms":         hrv.clamp(5, 120),
        "resting_hr_bpm":       hr.clamp(35, 120),
        "spo2_pct":             spo2.clamp(88, 100),
        "sleep_efficiency_pct": sleep.clamp(0.25, 1.0),
        "rem_min":              rem.clamp(0, 180),
        "deep_min":             deep.clamp(0, 150),
        "steps":                steps.clamp(0, 35000),
        "active_min":           active.clamp(0, 180),
        "screen_off_min":       screen.clamp(0, 800),
        "aqi_pm25":             aqi_out.clamp(0, 500),
        "ambient_temp_c":       temp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PERSONA LIBRARY (EXTREME DIVERSITY)
# ═══════════════════════════════════════════════════════════════════════════════

def get_divergence_personas() -> Dict[str, Persona8]:
    """Get library of personas designed for maximum agent divergence."""
    
    personas = {}
    
    # ═══ ANTI-CORRELATED PAIRS ═══
    
    # Cardio UP, Mental DOWN
    personas["cardiac_stoic"] = Persona8(
        name="cardiac_stoic",
        description="Type-A exec: declining cardio, rock-solid mental",
        seed=2001,
        primary_agent="cardio",
        divergence_type="opposite",
        agents={
            "cardio": AgentParams(base=0.52, drift_per_day=0.00025, event_day=180, event_magnitude=0.14),
            "mental": AgentParams(base=0.11, drift_per_day=0.00001, seasonal_amplitude=0.01),
            "metabolic": AgentParams(base=0.35, drift_per_day=0.00015),
            "recovery": AgentParams(base=0.25, drift_per_day=0.00008),
            "immune": AgentParams(base=0.20, drift_per_day=0.00002),
            "respiratory": AgentParams(base=0.18, drift_per_day=0.00003),
            "hormonal": AgentParams(base=0.15, seasonal_amplitude=0.02),
            "cog_fatigue": AgentParams(base=0.12, drift_per_day=-0.00005),
        },
        signals={"hrv_rmssd_ms": 28, "resting_hr_bpm": 84, "spo2_pct": 97.0,
                "sleep_efficiency_pct": 84, "rem_min": 96, "deep_min": 82,
                "steps": 2800, "active_min": 10, "screen_off_min": 60,
                "aqi_pm25": 30, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 5, "resting_hr_bpm": 3.5, "spo2_pct": 0.4,
              "sleep_efficiency_pct": 0.03, "rem_min": 8, "deep_min": 7,
              "steps": 450, "active_min": 3, "screen_off_min": 10,
              "aqi_pm25": 5, "ambient_temp_c": 2},
    )
    
    # Mental UP, Cardio DOWN (elite athlete with anxiety)
    personas["anxious_athlete"] = Persona8(
        name="anxious_athlete",
        description="Elite athlete with clinical anxiety",
        seed=2002,
        primary_agent="mental",
        divergence_type="opposite",
        agents={
            "cardio": AgentParams(base=0.07, drift_per_day=0.00001, seasonal_amplitude=0.01),
            "mental": AgentParams(base=0.55, drift_per_day=0.00022, event_day=150, event_magnitude=0.18),
            "metabolic": AgentParams(base=0.08, drift_per_day=-0.00002),
            "recovery": AgentParams(base=0.10, drift_per_day=-0.00001),
            "immune": AgentParams(base=0.12, drift_per_day=0.00002),
            "respiratory": AgentParams(base=0.08, drift_per_day=0.00001),
            "hormonal": AgentParams(base=0.10, seasonal_amplitude=0.015),
            "cog_fatigue": AgentParams(base=0.35, drift_per_day=0.00015),
        },
        signals={"hrv_rmssd_ms": 88, "resting_hr_bpm": 48, "spo2_pct": 99.1,
                "sleep_efficiency_pct": 68, "rem_min": 55, "deep_min": 72,
                "steps": 18000, "active_min": 110, "screen_off_min": 280,
                "aqi_pm25": 16, "ambient_temp_c": 20},
        noise={"hrv_rmssd_ms": 7, "resting_hr_bpm": 2.5, "spo2_pct": 0.2,
              "sleep_efficiency_pct": 0.08, "rem_min": 12, "deep_min": 8,
              "steps": 2500, "active_min": 16, "screen_off_min": 60,
              "aqi_pm25": 4, "ambient_temp_c": 2},
    )
    
    # ═══ INDEPENDENT AGENTS ═══
    
    # Respiratory events, others stable
    personas["pollution_resilient"] = Persona8(
        name="pollution_resilient",
        description="Asthmatic with severe respiratory responses, otherwise healthy",
        seed=2005,
        primary_agent="respiratory",
        divergence_type="independent",
        aqi_spike_count=5,
        aqi_spike_magnitude=150.0,
        aqi_spike_width_days=14,
        agents={
            "cardio": AgentParams(base=0.15, drift_per_day=0.00003),
            "mental": AgentParams(base=0.13, drift_per_day=0.00002, seasonal_amplitude=0.015),
            "metabolic": AgentParams(base=0.18, drift_per_day=0.00004),
            "recovery": AgentParams(base=0.14, drift_per_day=0.00002),
            "immune": AgentParams(base=0.20, drift_per_day=0.00003),
            "respiratory": AgentParams(base=0.22, drift_per_day=0.00008, seasonal_amplitude=0.04),
            "hormonal": AgentParams(base=0.15, seasonal_amplitude=0.02),
            "cog_fatigue": AgentParams(base=0.14, drift_per_day=0.00002),
        },
        signals={"hrv_rmssd_ms": 54, "resting_hr_bpm": 68, "spo2_pct": 97.4,
                "sleep_efficiency_pct": 86, "rem_min": 98, "deep_min": 80,
                "steps": 7500, "active_min": 38, "screen_off_min": 85,
                "aqi_pm25": 35, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 5, "resting_hr_bpm": 3, "spo2_pct": 0.5,
              "sleep_efficiency_pct": 0.04, "rem_min": 9, "deep_min": 7,
              "steps": 900, "active_min": 7, "screen_off_min": 14,
              "aqi_pm25": 8, "ambient_temp_c": 3},
    )
    
    # Immune-focused, others stable
    personas["immunocompromised_worker"] = Persona8(
        name="immunocompromised_worker",
        description="Immunocompromised with frequent immune challenges",
        seed=2006,
        primary_agent="immune",
        divergence_type="independent",
        flare_count=6,
        flare_magnitude=0.15,
        flare_width_days=14,
        agents={
            "cardio": AgentParams(base=0.18, drift_per_day=0.00004),
            "mental": AgentParams(base=0.20, drift_per_day=0.00005),
            "metabolic": AgentParams(base=0.20, drift_per_day=0.00004),
            "recovery": AgentParams(base=0.22, drift_per_day=0.00003),
            "immune": AgentParams(base=0.45, drift_per_day=0.00008),
            "respiratory": AgentParams(base=0.20, drift_per_day=0.00004),
            "hormonal": AgentParams(base=0.18, seasonal_amplitude=0.02),
            "cog_fatigue": AgentParams(base=0.25, drift_per_day=0.00006),
        },
        signals={"hrv_rmssd_ms": 38, "resting_hr_bpm": 75, "spo2_pct": 97.8,
                "sleep_efficiency_pct": 78, "rem_min": 72, "deep_min": 58,
                "steps": 4500, "active_min": 20, "screen_off_min": 180,
                "aqi_pm25": 18, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 6, "resting_hr_bpm": 4, "spo2_pct": 0.4,
              "sleep_efficiency_pct": 0.05, "rem_min": 10, "deep_min": 8,
              "steps": 600, "active_min": 4, "screen_off_min": 30,
              "aqi_pm25": 4, "ambient_temp_c": 2},
    )
    
    # ═══ MULTI-AGENT COMPLEX ═══
    
    # Recovery from major illness (Recovery + Immune + Cardio recovering, Mental struggling)
    personas["post_viral_complex"] = Persona8(
        name="post_viral_complex",
        description="Post-viral: recovery/immune/cardio improving, mental declining from isolation",
        seed=3001,
        recovery_days=180,
        flare_count=2,
        flare_magnitude=0.12,
        primary_agent="recovery",
        divergence_type="mixed",
        agents={
            "cardio": AgentParams(base=0.55, drift_per_day=-0.0020),
            "mental": AgentParams(base=0.30, drift_per_day=0.00025, event_day=90, event_magnitude=0.12),
            "metabolic": AgentParams(base=0.40, drift_per_day=-0.0010),
            "recovery": AgentParams(base=0.65, drift_per_day=-0.0025),
            "immune": AgentParams(base=0.70, drift_per_day=-0.0030),
            "respiratory": AgentParams(base=0.35, drift_per_day=-0.0010),
            "hormonal": AgentParams(base=0.25, drift_per_day=-0.0005),
            "cog_fatigue": AgentParams(base=0.40, drift_per_day=0.00010),
        },
        signals={"hrv_rmssd_ms": 24, "resting_hr_bpm": 92, "spo2_pct": 96.0,
                "sleep_efficiency_pct": 58, "rem_min": 48, "deep_min": 35,
                "steps": 600, "active_min": 4, "screen_off_min": 320,
                "aqi_pm25": 12, "ambient_temp_c": 22},
        noise={"hrv_rmssd_ms": 6, "resting_hr_bpm": 4, "spo2_pct": 0.6,
              "sleep_efficiency_pct": 0.08, "rem_min": 12, "deep_min": 10,
              "steps": 150, "active_min": 2, "screen_off_min": 50,
              "aqi_pm25": 3, "ambient_temp_c": 2},
    )
    
    # Hormonal + Cognitive (menstrual + high stress job)
    personas["hormonal_executive"] = Persona8(
        name="hormonal_executive",
        description="Executive with strong hormonal cycles and cognitive load",
        seed=3002,
        cycle_period_days=28,
        primary_agent="hormonal",
        divergence_type="mixed",
        agents={
            "cardio": AgentParams(base=0.20, drift_per_day=0.00008),
            "mental": AgentParams(base=0.35, drift_per_day=0.00018),
            "metabolic": AgentParams(base=0.25, drift_per_day=0.00010),
            "recovery": AgentParams(base=0.22, drift_per_day=0.00005),
            "immune": AgentParams(base=0.20, drift_per_day=0.00004),
            "respiratory": AgentParams(base=0.18, drift_per_day=0.00003),
            "hormonal": AgentParams(base=0.18, drift_per_day=0.00003, seasonal_amplitude=0.04),
            "cog_fatigue": AgentParams(base=0.48, drift_per_day=0.00020, event_day=200, event_magnitude=0.10),
        },
        signals={"hrv_rmssd_ms": 52, "resting_hr_bpm": 68, "spo2_pct": 98.2,
                "sleep_efficiency_pct": 75, "rem_min": 82, "deep_min": 62,
                "steps": 5500, "active_min": 28, "screen_off_min": 350,
                "aqi_pm25": 22, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 6, "resting_hr_bpm": 3, "spo2_pct": 0.3,
              "sleep_efficiency_pct": 0.05, "rem_min": 10, "deep_min": 8,
              "steps": 700, "active_min": 5, "screen_off_min": 45,
              "aqi_pm25": 5, "ambient_temp_c": 2},
    )
    
    # Metabolic syndrome developing
    personas["metabolic_decline"] = Persona8(
        name="metabolic_decline",
        description="Metabolic syndrome onset, other systems stable",
        seed=3003,
        primary_agent="metabolic",
        divergence_type="independent",
        agents={
            "cardio": AgentParams(base=0.25, drift_per_day=0.00010),
            "mental": AgentParams(base=0.22, drift_per_day=0.00005),
            "metabolic": AgentParams(base=0.35, drift_per_day=0.00030, event_day=180, event_magnitude=0.12),
            "recovery": AgentParams(base=0.28, drift_per_day=0.00008),
            "immune": AgentParams(base=0.22, drift_per_day=0.00004),
            "respiratory": AgentParams(base=0.20, drift_per_day=0.00003),
            "hormonal": AgentParams(base=0.18, seasonal_amplitude=0.02),
            "cog_fatigue": AgentParams(base=0.25, drift_per_day=0.00008),
        },
        signals={"hrv_rmssd_ms": 42, "resting_hr_bpm": 78, "spo2_pct": 97.5,
                "sleep_efficiency_pct": 78, "rem_min": 72, "deep_min": 55,
                "steps": 3200, "active_min": 12, "screen_off_min": 240,
                "aqi_pm25": 28, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 5, "resting_hr_bpm": 3, "spo2_pct": 0.35,
              "sleep_efficiency_pct": 0.04, "rem_min": 9, "deep_min": 8,
              "steps": 500, "active_min": 3, "screen_off_min": 35,
              "aqi_pm25": 5, "ambient_temp_c": 2},
    )
    
    # ═══ EXTREME CASES ═══
    
    # All agents HIGH (chronic multi-system illness)
    personas["multi_system_failure"] = Persona8(
        name="multi_system_failure",
        description="Multiple chronic conditions, all agents elevated",
        seed=4001,
        flare_count=6,
        flare_magnitude=0.18,
        primary_agent="cardio",
        divergence_type="coupled",
        agents={
            "cardio": AgentParams(base=0.65, drift_per_day=0.00008),
            "mental": AgentParams(base=0.58, drift_per_day=0.00010),
            "metabolic": AgentParams(base=0.62, drift_per_day=0.00008),
            "recovery": AgentParams(base=0.55, drift_per_day=0.00005),
            "immune": AgentParams(base=0.70, drift_per_day=0.00006),
            "respiratory": AgentParams(base=0.55, drift_per_day=0.00007),
            "hormonal": AgentParams(base=0.45, drift_per_day=0.00005),
            "cog_fatigue": AgentParams(base=0.60, drift_per_day=0.00008),
        },
        signals={"hrv_rmssd_ms": 22, "resting_hr_bpm": 92, "spo2_pct": 95.5,
                "sleep_efficiency_pct": 52, "rem_min": 42, "deep_min": 32,
                "steps": 1200, "active_min": 5, "screen_off_min": 400,
                "aqi_pm25": 35, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 4, "resting_hr_bpm": 4, "spo2_pct": 0.6,
              "sleep_efficiency_pct": 0.08, "rem_min": 10, "deep_min": 8,
              "steps": 250, "active_min": 2, "screen_off_min": 55,
              "aqi_pm25": 6, "ambient_temp_c": 2},
    )
    
    # All agents LOW (peak health)
    personas["peak_athlete"] = Persona8(
        name="peak_athlete",
        description="Elite athlete at peak performance, all agents low",
        seed=4002,
        primary_agent="cardio",
        divergence_type="coupled",
        agents={
            "cardio": AgentParams(base=0.06, drift_per_day=0.000005),
            "mental": AgentParams(base=0.08, drift_per_day=0.000008),
            "metabolic": AgentParams(base=0.07, drift_per_day=0.000006),
            "recovery": AgentParams(base=0.08, drift_per_day=0.000005),
            "immune": AgentParams(base=0.10, drift_per_day=0.000008),
            "respiratory": AgentParams(base=0.07, drift_per_day=0.000005),
            "hormonal": AgentParams(base=0.08, seasonal_amplitude=0.01),
            "cog_fatigue": AgentParams(base=0.10, drift_per_day=0.000008),
        },
        signals={"hrv_rmssd_ms": 95, "resting_hr_bpm": 44, "spo2_pct": 99.2,
                "sleep_efficiency_pct": 93, "rem_min": 115, "deep_min": 100,
                "steps": 22000, "active_min": 140, "screen_off_min": 55,
                "aqi_pm25": 12, "ambient_temp_c": 19},
        noise={"hrv_rmssd_ms": 5, "resting_hr_bpm": 2, "spo2_pct": 0.2,
              "sleep_efficiency_pct": 0.02, "rem_min": 7, "deep_min": 6,
              "steps": 1500, "active_min": 12, "screen_off_min": 10,
              "aqi_pm25": 3, "ambient_temp_c": 2},
    )
    
    # Random walk (truly independent agents)
    personas["chaotic_multi"] = Persona8(
        name="chaotic_multi",
        description="Random independent dynamics per agent",
        seed=4003,
        primary_agent="mixed",
        divergence_type="independent",
        flare_count=3,
        travel_count=3,
        aqi_spike_count=3,
        agents={
            "cardio": AgentParams(base=0.30, drift_per_day=0.00015 * (1 if hash("cardio") % 2 else -1)),
            "mental": AgentParams(base=0.35, drift_per_day=0.00020 * (1 if hash("mental") % 2 else -1)),
            "metabolic": AgentParams(base=0.28, drift_per_day=0.00012 * (1 if hash("metabolic") % 2 else -1)),
            "recovery": AgentParams(base=0.25, drift_per_day=0.00010 * (1 if hash("recovery") % 2 else -1)),
            "immune": AgentParams(base=0.30, drift_per_day=0.00008 * (1 if hash("immune") % 2 else -1)),
            "respiratory": AgentParams(base=0.25, drift_per_day=0.00010 * (1 if hash("respiratory") % 2 else -1)),
            "hormonal": AgentParams(base=0.22, seasonal_amplitude=0.04),
            "cog_fatigue": AgentParams(base=0.32, drift_per_day=0.00015 * (1 if hash("cog_fatigue") % 2 else -1)),
        },
        signals={"hrv_rmssd_ms": 50, "resting_hr_bpm": 70, "spo2_pct": 98.0,
                "sleep_efficiency_pct": 80, "rem_min": 85, "deep_min": 68,
                "steps": 7000, "active_min": 40, "screen_off_min": 180,
                "aqi_pm25": 20, "ambient_temp_c": 21},
        noise={"hrv_rmssd_ms": 8, "resting_hr_bpm": 4, "spo2_pct": 0.4,
              "sleep_efficiency_pct": 0.06, "rem_min": 12, "deep_min": 10,
              "steps": 1000, "active_min": 8, "screen_off_min": 40,
              "aqi_pm25": 8, "ambient_temp_c": 3},
    )
    
    return personas


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample(p: Persona8, n_days: int = 365) -> Sample8Agent:
    """Generate complete sample with 8-agent ground truths."""
    
    seed = p.seed
    t = _arange(n_days)
    
    # Build AQI curve if needed
    aqi = None
    if p.aqi_spike_count > 0:
        aqi = p.signals["aqi_pm25"] + _randn(n_days, p.noise.get("aqi_pm25", 5.0), 1, seed)
        aqi = aqi + 8 * torch.sin(2 * torch.pi * t / 365 + torch.pi)
        for sc in _spike_days(n_days, p, seed):
            w = p.aqi_spike_width_days / 3.0
            aqi = aqi + p.aqi_spike_magnitude * torch.exp(-0.5 * ((t - sc) / w) ** 2)
        aqi = aqi.clamp(0, 500)
    
    # Generate ground truths for all 8 agents
    agent_gts = {}
    for agent_name in ALL_AGENTS:
        params = p.agents.get(agent_name, AgentParams())
        agent_gts[agent_name] = generate_agent_gt(t, agent_name, params, p, seed)
    
    # Generate signals
    signals = generate_signals(p, agent_gts, t, seed, aqi)
    
    # Compute divergence score
    gt_values = {name: gt.values for name, gt in agent_gts.items()}
    divergence = compute_divergence_score(gt_values)
    
    # Routing targets (which expert should handle this sample)
    routing_targets = {name: i for i, name in enumerate(ALL_AGENTS)}
    
    return Sample8Agent(
        persona_name=p.name,
        t=t,
        agent_gts=agent_gts,
        signals=signals,
        routing_targets=routing_targets,
        divergence_score=divergence,
    )


def generate_all_personas(n_days: int = 365) -> Dict[str, Sample8Agent]:
    """Generate samples for all divergence personas."""
    personas = get_divergence_personas()
    return {name: generate_sample(p, n_days) for name, p in personas.items()}
