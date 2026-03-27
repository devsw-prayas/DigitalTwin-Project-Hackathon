"""
agents_8.py — 8-Agent Ground Truth Framework

Each agent represents a distinct physiological domain with:
- Primary signals (high coupling)
- Secondary signals (low coupling)  
- Unique temporal patterns
- Independent or anti-correlated behavior from other agents

═══════════════════════════════════════════════════════════════════════════════
AGENT DEFINITIONS
═══════════════════════════════════════════════════════════════════════════════

1. CARDIO (cardiovascular health)
   Primary signals: hrv_rmssd, resting_hr, spo2, active_min
   Pattern: Slow sustained changes (weeks-months)
   Events: Hypertensive episodes, cardiac stress

2. MENTAL (psychological wellbeing)
   Primary signals: rem_min, screen_off_min, sleep_efficiency
   Pattern: Fast acute changes (days), high volatility
   Events: Anxiety spikes, depression episodes

3. METABOLIC (metabolic health)
   Primary signals: steps, active_min, deep_sleep, resting_hr
   Pattern: Medium-term drift (weeks), activity-dependent
   Events: Metabolic syndrome onset, weight gain periods

4. RECOVERY (physical recovery)
   Primary signals: hrv_rmssd (trend), deep_min, sleep_efficiency
   Pattern: Exponential recovery curves, post-exertion dips
   Events: Post-illness recovery, overtraining recovery

5. IMMUNE (immune system status)
   Primary signals: hrv_rmssd (acute drops), resting_hr (spikes)
   Pattern: Acute suppression during illness, gradual return
   Events: Infections, immune challenges

6. RESPIRATORY (respiratory health)
   Primary signals: spo2, aqi_pm25, resting_hr
   Pattern: AQI-driven spikes, seasonal variation
   Events: Pollution exposure, asthma flares

7. HORMONAL (hormonal cycles)
   Primary signals: hrv_rmssd (cyclical), rem_min, temp
   Pattern: 28-day cycles (or other periods)
   Events: Cycle irregularities, hormonal shifts

8. COG_FATIGUE (cognitive fatigue)
   Primary signals: screen_off_min, rem_min, sleep_efficiency
   Pattern: Cumulative sleep debt, screen-overload
   Events: Burnout, exam periods, crunch time

═══════════════════════════════════════════════════════════════════════════════
SIGNAL DIVERSITY MATRIX
═══════════════════════════════════════════════════════════════════════════════

Signal            | Cardio | Mental | Metab | Recov | Immune | Resp | Horm | CogFat
------------------|--------|--------|-------|-------|--------|------|------|-------
hrv_rmssd_ms      | HIGH   | MED    | LOW   | HIGH  | HIGH   | LOW  | MED  | MED
resting_hr_bpm    | HIGH   | LOW    | MED   | MED   | HIGH   | MED  | LOW  | LOW
spo2_pct          | HIGH   | NONE   | NONE  | LOW   | LOW    | HIGH | NONE | NONE
sleep_efficiency  | LOW    | HIGH   | LOW   | HIGH  | MED    | NONE | MED  | HIGH
rem_min           | LOW    | HIGH   | NONE  | MED   | LOW    | NONE | MED  | HIGH
deep_min          | MED    | MED    | MED   | HIGH  | LOW    | NONE | LOW  | MED
steps             | MED    | LOW    | HIGH  | MED   | LOW    | NONE | NONE | LOW
active_min        | HIGH   | LOW    | HIGH  | MED   | LOW    | MED  | LOW  | LOW
screen_off_min    | NONE   | HIGH   | NONE  | NONE  | NONE   | NONE | LOW  | HIGH
aqi_pm25          | MED    | NONE   | NONE  | NONE  | NONE   | HIGH | NONE | NONE
ambient_temp_c    | LOW    | LOW    | LOW   | LOW   | LOW    | MED  | MED  | LOW

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import math

# Agent names in canonical order
ALL_AGENTS = ["cardio", "mental", "metabolic", "recovery", "immune", "respiratory", "hormonal", "cog_fatigue"]
N_AGENTS = len(ALL_AGENTS)

# Signal names in canonical order  
ALL_SIGNALS = [
    "hrv_rmssd_ms", "resting_hr_bpm", "spo2_pct", "sleep_efficiency_pct",
    "rem_min", "deep_min", "steps", "active_min", "screen_off_min",
    "aqi_pm25", "ambient_temp_c"
]
N_SIGNALS = len(ALL_SIGNALS)

# Agent → Primary signals mapping (for ground truth generation)
AGENT_PRIMARY_SIGNALS = {
    "cardio":     ["hrv_rmssd_ms", "resting_hr_bpm", "spo2_pct", "active_min"],
    "mental":     ["rem_min", "screen_off_min", "sleep_efficiency_pct"],
    "metabolic":  ["steps", "active_min", "deep_min", "resting_hr_bpm"],
    "recovery":   ["hrv_rmssd_ms", "deep_min", "sleep_efficiency_pct"],
    "immune":     ["hrv_rmssd_ms", "resting_hr_bpm"],
    "respiratory":["spo2_pct", "aqi_pm25", "resting_hr_bpm"],
    "hormonal":   ["hrv_rmssd_ms", "rem_min", "ambient_temp_c"],
    "cog_fatigue":["screen_off_min", "rem_min", "sleep_efficiency_pct"],
}

# Agent coupling coefficients for signal generation
# Format: agent -> (primary_coeff, secondary_coeff, none_coeff)
AGENT_SIGNAL_COUPLING = {
    "cardio": {
        "hrv_rmssd_ms": ("primary", 0.55), "resting_hr_bpm": ("primary", 0.85), 
        "spo2_pct": ("primary", 0.70), "active_min": ("primary", 0.50),
        "deep_min": ("secondary", 0.20), "sleep_efficiency_pct": ("secondary", 0.10),
        "steps": ("secondary", 0.25), "aqi_pm25": ("secondary", 0.15),
        "rem_min": ("none", 0.05), "screen_off_min": ("none", 0.02),
        "ambient_temp_c": ("none", 0.03),
    },
    "mental": {
        "rem_min": ("primary", 0.75), "screen_off_min": ("primary", 0.80), 
        "sleep_efficiency_pct": ("primary", 0.60),
        "deep_min": ("secondary", 0.25), "hrv_rmssd_ms": ("secondary", 0.15),
        "steps": ("none", 0.05), "resting_hr_bpm": ("none", 0.08),
        "spo2_pct": ("none", 0.02), "active_min": ("none", 0.03),
        "aqi_pm25": ("none", 0.01), "ambient_temp_c": ("none", 0.02),
    },
    "metabolic": {
        "steps": ("primary", 0.70), "active_min": ("primary", 0.65),
        "deep_min": ("primary", 0.45), "resting_hr_bpm": ("primary", 0.40),
        "hrv_rmssd_ms": ("secondary", 0.20), "spo2_pct": ("secondary", 0.10),
        "sleep_efficiency_pct": ("secondary", 0.15), "rem_min": ("none", 0.05),
        "screen_off_min": ("none", 0.03), "aqi_pm25": ("none", 0.02),
        "ambient_temp_c": ("none", 0.02),
    },
    "recovery": {
        "hrv_rmssd_ms": ("primary", 0.60), "deep_min": ("primary", 0.55),
        "sleep_efficiency_pct": ("primary", 0.50),
        "resting_hr_bpm": ("secondary", 0.25), "spo2_pct": ("secondary", 0.15),
        "steps": ("secondary", 0.20), "active_min": ("secondary", 0.20),
        "rem_min": ("secondary", 0.15), "screen_off_min": ("none", 0.05),
        "aqi_pm25": ("none", 0.02), "ambient_temp_c": ("none", 0.02),
    },
    "immune": {
        "hrv_rmssd_ms": ("primary", 0.70), "resting_hr_bpm": ("primary", 0.60),
        "sleep_efficiency_pct": ("secondary", 0.25), "deep_min": ("secondary", 0.20),
        "spo2_pct": ("secondary", 0.15), "steps": ("none", 0.08),
        "active_min": ("none", 0.08), "rem_min": ("none", 0.05),
        "screen_off_min": ("none", 0.03), "aqi_pm25": ("none", 0.02),
        "ambient_temp_c": ("none", 0.02),
    },
    "respiratory": {
        "spo2_pct": ("primary", 0.75), "aqi_pm25": ("primary", 0.80),
        "resting_hr_bpm": ("primary", 0.45),
        "hrv_rmssd_ms": ("secondary", 0.20), "sleep_efficiency_pct": ("secondary", 0.10),
        "deep_min": ("none", 0.05), "steps": ("none", 0.03),
        "active_min": ("none", 0.05), "rem_min": ("none", 0.02),
        "screen_off_min": ("none", 0.01), "ambient_temp_c": ("secondary", 0.15),
    },
    "hormonal": {
        "hrv_rmssd_ms": ("primary", 0.55), "rem_min": ("primary", 0.50),
        "ambient_temp_c": ("primary", 0.40),
        "sleep_efficiency_pct": ("secondary", 0.25), "deep_min": ("secondary", 0.20),
        "resting_hr_bpm": ("secondary", 0.15), "steps": ("none", 0.08),
        "active_min": ("none", 0.08), "spo2_pct": ("none", 0.02),
        "screen_off_min": ("secondary", 0.20), "aqi_pm25": ("none", 0.01),
    },
    "cog_fatigue": {
        "screen_off_min": ("primary", 0.75), "rem_min": ("primary", 0.65),
        "sleep_efficiency_pct": ("primary", 0.55),
        "deep_min": ("secondary", 0.25), "hrv_rmssd_ms": ("secondary", 0.20),
        "resting_hr_bpm": ("none", 0.08), "spo2_pct": ("none", 0.02),
        "steps": ("none", 0.05), "active_min": ("none", 0.05),
        "aqi_pm25": ("none", 0.01), "ambient_temp_c": ("none", 0.02),
    },
}

# Temporal patterns per agent (for ground truth dynamics)
AGENT_TEMPORAL = {
    "cardio": {
        "timescale": "slow",           # Weeks to months
        "half_life_days": 21,
        "volatility": 0.15,
    },
    "mental": {
        "timescale": "fast",           # Days
        "half_life_days": 3,
        "volatility": 0.35,
    },
    "metabolic": {
        "timescale": "medium",         # Weeks
        "half_life_days": 14,
        "volatility": 0.20,
    },
    "recovery": {
        "timescale": "exponential",    # Recovery curves
        "half_life_days": 10,
        "volatility": 0.25,
    },
    "immune": {
        "timescale": "acute",          # Sharp drops, gradual recovery
        "half_life_days": 5,
        "volatility": 0.40,
    },
    "respiratory": {
        "timescale": "event_driven",   # AQI spikes
        "half_life_days": 7,
        "volatility": 0.30,
    },
    "hormonal": {
        "timescale": "cyclical",       # 28-day cycles
        "half_life_days": 14,
        "volatility": 0.20,
        "cycle_period": 28,
    },
    "cog_fatigue": {
        "timescale": "cumulative",     # Sleep debt accumulation
        "half_life_days": 7,
        "volatility": 0.30,
    },
}


@dataclass
class AgentGroundTruth:
    """Ground truth risk curve for one agent."""
    name: str
    values: torch.Tensor          # (n_days,) risk values [0, 1]
    base: float
    drift_per_day: float
    events: List[Tuple[int, float, float]]  # [(day, magnitude, width), ...]
    primary_pattern: str          # "slow", "fast", "cyclical", etc.
    
    def to_tensor(self) -> torch.Tensor:
        return self.values


@dataclass  
class Sample8Agent:
    """Complete data sample with 8-agent ground truths and signals."""
    persona_name: str
    t: torch.Tensor                              # (n_days,)
    agent_gts: Dict[str, AgentGroundTruth]       # agent_name -> ground truth
    signals: Dict[str, torch.Tensor]             # signal_name -> values
    routing_targets: Dict[str, int]              # agent_name -> primary_expert_idx
    divergence_score: float                      # How independent are the agents?


def compute_independence_matrix(agent_gts: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise independence between agent ground truths.
    
    Returns matrix where 0 = perfectly correlated, 1 = perfectly independent.
    """
    n = len(agent_gts)
    agent_names = list(agent_gts.keys())
    
    independence = torch.zeros(n, n)
    
    for i, name_i in enumerate(agent_names):
        for j, name_j in enumerate(agent_names):
            if i == j:
                independence[i, j] = 0.0
            else:
                # Correlation-based independence
                x = agent_gts[name_i].float()
                y = agent_gts[name_j].float()
                
                x_centered = x - x.mean()
                y_centered = y - y.mean()
                
                corr = (x_centered * y_centered).sum() / (
                    x_centered.norm() * y_centered.norm() + 1e-8
                )
                
                # Independence = 1 - |correlation|
                independence[i, j] = 1.0 - abs(corr.item())
    
    return independence


def compute_divergence_score(agent_gts: Dict[str, torch.Tensor]) -> float:
    """
    Compute overall divergence score for a sample.
    
    Higher = agents more independent (better for specialization training).
    """
    independence = compute_independence_matrix(agent_gts)
    
    # Average off-diagonal independence
    n = independence.size(0)
    off_diag = independence.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    return off_diag.mean().item()
