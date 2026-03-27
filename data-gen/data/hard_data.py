"""
data/hard_data.py
═══════════════════════════════════════════════════════════════════════════════
Hard dataset generator — builds ON TOP of the existing pipeline.
Does NOT overwrite any existing file.

Imports:
    from data.personas    import Persona, RiskParams, load_personas
    from data.spawner     import spawn_all
    from data.generator   import generate, DataSample, DEVICE, DTYPE, _arange, _randn, _is
    from data.degradation import DegradedSample, degrade_sample, _gen, _rand, _randint, _choice
    from data.tokenizer   import tokenize

New things added here:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HARD PERSONAS        — 6 new archetypes in HARD_PERSONAS dict          │
    │  HARD GENERATOR       — ground_truth_* overrides for hard dynamics      │
    │  HARD DEGRADATION     — 3 new injectors (correlated, adversarial, burst)│
    │  HARD SPAWNER         — spawn_hard() with difficulty tagging            │
    │  HARD SHARD WRITER    — generate_hard_dataset() → outputs/hard_shards/  │
    └─────────────────────────────────────────────────────────────────────────┘

Curriculum mapping (matches spec §5.5):
    difficulty="easy"        → Phase 1  (clean joint)
    difficulty="medium"      → Phase 2  (noisy)
    difficulty="hard"        → Phase 3  (degraded)
    difficulty="adversarial" → Phase 4  (edge cases)

Usage:
    python -m data.hard_data                          # full 500-persona hard set
    python -m data.hard_data --single                 # debug, 1 worker
    python -m data.hard_data --out outputs/hard_shards --workers 2
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# ── Import existing pipeline ──────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.personas import Persona, RiskParams, load_personas
from data.spawner import spawn_all, _pure_variant, _crossbreed, _g, _jitter_risk, _jitter_dict, _jitter_scalar, _jclamp, _blend_risk, _blend_dict
from data.generator import (
    generate as _base_generate,
    DataSample,
    DEVICE,
    DTYPE,
    _arange,
    _randn,
    _is,
    ground_truth_cardio as _base_cardio,
    ground_truth_mental as _base_mental,
    sample_signals,
    _build_aqi_curve,
    _spike_days,
    _flare_days,
    _travel_days,
    _base_risk,
)
from data.degradation import (
    DegradedSample,
    degrade_sample as _base_degrade,
    _gen,
    _rand,
    _randint,
    _choice,
    _dropout_windows,
    _sensor_drift,
    _device_swap,
    _irregular_sampling,
    DRIFT_SIGNALS,
    SWAP_SIGNALS,
)
from data.tokenizer import tokenize, SIGNAL_KEYS, TokenizedSample


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HARD PERSONAS
# Six new archetypes targeting failure modes of DRIFT's specialist routing.
# ══════════════════════════════════════════════════════════════════════════════

def _rp(base, drift, event_day=-1, event_mag=0.0, event_width=20.0, seasonal=0.02) -> RiskParams:
    return RiskParams(
        base=base,
        drift_per_day=drift,
        event_day=event_day,
        event_magnitude=event_mag,
        event_width_days=event_width,
        seasonal_amplitude=seasonal,
    )


# Shared signal/noise templates
_HEALTHY_SIGS = dict(
    hrv_rmssd_ms=68.0, resting_hr_bpm=58.0, spo2_pct=98.6,
    sleep_efficiency_pct=90.0, rem_min=102.0, deep_min=88.0,
    steps=10000.0, active_min=50.0, screen_off_min=80.0,
    aqi_pm25=14.0, ambient_temp_c=22.0,
)
_HEALTHY_NOISE = dict(
    hrv_rmssd_ms=4.5, resting_hr_bpm=2.0, spo2_pct=0.3,
    sleep_efficiency_pct=0.03, rem_min=8.0, deep_min=7.0,
    steps=900.0, active_min=6.0, screen_off_min=12.0,
    aqi_pm25=3.0, ambient_temp_c=2.0,
)


HARD_PERSONAS: dict[str, Persona] = {

    # ── 1. silent_drifter ─────────────────────────────────────────────────────
    # Pure monotonic drift, no event spikes. Forces CardioLSTM to catch slow
    # velocity. Decoy: all signals look near-normal until day ~200.
    "silent_drifter": Persona(
        name="silent_drifter",
        description=(
            "Appears healthy but has slow compounding cardio risk from "
            "chronic low-grade inflammation. No acute events, no spikes. "
            "Cardio risk grows silently via drift alone."
        ),
        seed=2001,
        cardio=_rp(base=0.12, drift=0.00045, seasonal=0.01),   # 0.12→0.94 over 5yr
        mental=_rp(base=0.16, drift=0.00006, seasonal=0.02),
        signals={**_HEALTHY_SIGS, "hrv_rmssd_ms": 65.0, "resting_hr_bpm": 60.0},
        noise={**_HEALTHY_NOISE, "hrv_rmssd_ms": 3.5},          # low noise — harder to see
    ),

    # ── 2. contradictory_signals ──────────────────────────────────────────────
    # Steps INCREASE while HRV DROPS and HR RISES simultaneously.
    # Overreaching athlete pattern. Confuses naive signal-averaging models.
    # VSN must learn to down-weight steps and up-weight HRV+HR.
    "contradictory_signals": Persona(
        name="contradictory_signals",
        description=(
            "Overreaching: ramps activity aggressively (steps +60%) while "
            "HRV collapses and resting HR climbs. Cardio risk is HIGH even "
            "though activity signal looks great."
        ),
        seed=2002,
        cardio=_rp(base=0.20, drift=0.00030, event_day=120, event_mag=0.22, event_width=25.0),
        mental=_rp(base=0.24, drift=0.00010, event_day=120, event_mag=0.10, event_width=20.0),
        signals=dict(
            hrv_rmssd_ms=42.0,       # suppressed — overtraining
            resting_hr_bpm=78.0,     # elevated despite high fitness activity
            spo2_pct=98.2,
            sleep_efficiency_pct=74.0,
            rem_min=68.0,
            deep_min=52.0,
            steps=18000.0,           # HIGH — decoy signal
            active_min=110.0,        # HIGH — decoy signal
            screen_off_min=90.0,
            aqi_pm25=16.0,
            ambient_temp_c=21.0,
        ),
        noise=dict(
            hrv_rmssd_ms=7.0,
            resting_hr_bpm=4.0,
            spo2_pct=0.35,
            sleep_efficiency_pct=0.06,
            rem_min=12.0,
            deep_min=9.0,
            steps=2500.0,            # high variance — training load varies daily
            active_min=18.0,
            screen_off_min=15.0,
            aqi_pm25=4.0,
            ambient_temp_c=2.0,
        ),
        overtraining_period_days=60,
        overtraining_magnitude=0.25,
        overtraining_width_days=18,
    ),

    # ── 3. phantom_recovery ───────────────────────────────────────────────────
    # Signals recover to near-baseline BUT ground truth risk stays elevated
    # (biological debt). Trains the model to not trust surface recovery.
    # Quantile heads must learn right-skewed uncertainty (risk could resurge).
    "phantom_recovery": Persona(
        name="phantom_recovery",
        description=(
            "Post-illness: signals (HRV, sleep, steps) return to baseline "
            "by day 60 but ground truth cardio risk stays 0.35+ due to "
            "residual inflammation. Surface looks recovered; underlying risk is not."
        ),
        seed=2003,
        recovery_days=240,           # signals recover quickly; risk does not
        cardio=_rp(base=0.62, drift=-0.00020, seasonal=0.02),   # very slow recovery
        mental=_rp(base=0.38, drift=-0.00045, seasonal=0.03),   # mental recovers faster
        signals=dict(
            hrv_rmssd_ms=58.0,       # returns to reasonable range fast
            resting_hr_bpm=70.0,
            spo2_pct=97.8,
            sleep_efficiency_pct=82.0,
            rem_min=88.0,
            deep_min=72.0,
            steps=8500.0,
            active_min=40.0,
            screen_off_min=115.0,
            aqi_pm25=14.0,
            ambient_temp_c=22.0,
        ),
        noise={**_HEALTHY_NOISE, "hrv_rmssd_ms": 5.5, "resting_hr_bpm": 3.0},
    ),

    # ── 4. multi_domain_event ─────────────────────────────────────────────────
    # Single stressor (severe flu + AQI spike) shifts cardio + mental + immune
    # simultaneously with DIFFERENT decay rates.
    # Forces specialist routing — which LSTM owns this event?
    "multi_domain_event": Persona(
        name="multi_domain_event",
        description=(
            "Acute multi-system event (severe flu during pollution spike): "
            "cardio risk spikes at day 90, mental at day 85, respiratory at 88. "
            "Each domain recovers at a different rate. Routing stress test."
        ),
        seed=2004,
        aqi_spike_count=2,
        aqi_spike_magnitude=95.0,
        aqi_spike_width_days=14,
        cardio=_rp(base=0.22, drift=0.00008, event_day=90,  event_mag=0.28, event_width=22.0),
        mental=_rp(base=0.28, drift=0.00012, event_day=85,  event_mag=0.20, event_width=18.0),
        signals=dict(
            hrv_rmssd_ms=55.0,
            resting_hr_bpm=66.0,
            spo2_pct=98.0,
            sleep_efficiency_pct=80.0,
            rem_min=85.0,
            deep_min=70.0,
            steps=7000.0,
            active_min=35.0,
            screen_off_min=140.0,
            aqi_pm25=30.0,
            ambient_temp_c=18.0,
        ),
        noise=dict(
            hrv_rmssd_ms=6.0,
            resting_hr_bpm=3.5,
            spo2_pct=0.5,
            sleep_efficiency_pct=0.06,
            rem_min=12.0,
            deep_min=9.0,
            steps=1200.0,
            active_min=8.0,
            screen_off_min=22.0,
            aqi_pm25=8.0,
            ambient_temp_c=3.0,
        ),
    ),

    # ── 5. asymmetric_risk ────────────────────────────────────────────────────
    # Cardio risk follows a fast right-skewed trajectory (sudden jumps, slow
    # recoveries). Mental risk follows a left-skewed accumulation (slow build,
    # fast crash). Specifically designed to stress quantile regression heads.
    "asymmetric_risk": Persona(
        name="asymmetric_risk",
        description=(
            "Cardio: sudden spikes + slow recoveries (right-skewed). "
            "Mental: slow accumulation + fast crashes (left-skewed). "
            "Gaussian loss fails here. Tests quantile head calibration."
        ),
        seed=2005,
        flare_count=5,
        flare_magnitude=0.22,
        flare_width_days=8,          # sharp spikes, not gaussian bumps
        cardio=_rp(base=0.32, drift=0.00010, event_day=150, event_mag=0.18, event_width=10.0, seasonal=0.04),
        mental=_rp(base=0.44, drift=0.00018, event_day=200, event_mag=0.24, event_width=35.0, seasonal=0.05),
        signals=dict(
            hrv_rmssd_ms=44.0,
            resting_hr_bpm=73.0,
            spo2_pct=97.5,
            sleep_efficiency_pct=75.0,
            rem_min=72.0,
            deep_min=55.0,
            steps=4500.0,
            active_min=20.0,
            screen_off_min=220.0,
            aqi_pm25=32.0,
            ambient_temp_c=20.0,
        ),
        noise=dict(
            hrv_rmssd_ms=8.0,        # high noise — harder to separate signal from noise
            resting_hr_bpm=5.0,
            spo2_pct=0.6,
            sleep_efficiency_pct=0.08,
            rem_min=15.0,
            deep_min=12.0,
            steps=800.0,
            active_min=6.0,
            screen_off_min=38.0,
            aqi_pm25=7.0,
            ambient_temp_c=2.5,
        ),
    ),

    # ── 6. phase_shifted_recovery ─────────────────────────────────────────────
    # Each signal recovers at a different timescale after illness:
    #   HRV:   14 days
    #   Sleep: 45 days
    #   Steps: 90 days
    #   SpO2:  120 days
    # Forces multi-horizon heads to not conflate signal recovery with risk recovery.
    "phase_shifted_recovery": Persona(
        name="phase_shifted_recovery",
        description=(
            "Post-illness multi-speed recovery: HRV recovers in ~14d, "
            "sleep in ~45d, activity in ~90d, SpO2 in ~120d. "
            "Cross-signal inconsistency stress-tests VSN gating."
        ),
        seed=2006,
        recovery_days=120,
        cardio=_rp(base=0.55, drift=-0.00200, seasonal=0.025),
        mental=_rp(base=0.42, drift=-0.00300, seasonal=0.030),
        signals=dict(
            hrv_rmssd_ms=30.0,
            resting_hr_bpm=86.0,
            spo2_pct=95.8,           # low — slow to recover
            sleep_efficiency_pct=62.0,
            rem_min=48.0,
            deep_min=36.0,
            steps=600.0,
            active_min=4.0,
            screen_off_min=220.0,
            aqi_pm25=14.0,
            ambient_temp_c=22.0,
        ),
        noise=dict(
            hrv_rmssd_ms=5.0,
            resting_hr_bpm=4.0,
            spo2_pct=0.6,
            sleep_efficiency_pct=0.07,
            rem_min=10.0,
            deep_min=8.0,
            steps=150.0,
            active_min=2.0,
            screen_off_min=28.0,
            aqi_pm25=3.0,
            ambient_temp_c=2.0,
        ),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HARD GROUND TRUTH OVERRIDES
# Replaces or supplements generator.py's risk functions for hard archetypes.
# ══════════════════════════════════════════════════════════════════════════════

def _hard_cardio(t: torch.Tensor, p: Persona, seed: int) -> torch.Tensor:
    """
    Hard cardio ground truth. Handles 4 new patterns:
      silent_drifter       — pure drift, no event, accelerating late
      contradictory_signals — overtraining spikes triggered more frequently
      phantom_recovery      — fast signal improvement, slow risk descent
      phase_shifted_recovery — per-signal recovery at different timescales
    Falls back to base generator for anything not in HARD_PERSONAS.
    """
    name = p.name.split("_v")[0].split("X")[0]

    if "silent_drifter" in name:
        # Drift that accelerates slightly after day 180 (compounding)
        r = p.cardio.base + p.cardio.drift_per_day * t
        r = r + 0.00010 * torch.clamp(t - 180, min=0.0)   # late acceleration
        r = r + p.cardio.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
        return r.clamp(0.0, 1.0)

    if "contradictory_signals" in name:
        # Overtraining spikes every overtraining_period_days
        r      = _base_risk(t, p.cardio)
        period = p.overtraining_period_days or 60
        mag    = p.overtraining_magnitude   or 0.25
        width  = p.overtraining_width_days  or 18
        n      = len(t)
        for sc in range(period, n, period):
            r = r + mag * torch.exp(-0.5 * ((t - sc) / (width / 2.5)) ** 2)
        return r.clamp(0.0, 1.0)

    if "phantom_recovery" in name:
        # Cardio risk decays very slowly despite signals recovering fast
        rd = p.recovery_days or 240
        # Plateau is higher than normal — risk never fully clears
        floor = 0.35
        plateau = max(floor, p.cardio.base + p.cardio.drift_per_day * t[-1].item())
        r = floor + (p.cardio.base - floor) * torch.exp(-t / (rd * 1.8))
        r = r + p.cardio.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
        return r.clamp(0.0, 1.0)

    if "multi_domain_event" in name:
        r = _base_risk(t, p.cardio)
        for sc in _spike_days(len(t), p, seed):
            w = (p.aqi_spike_width_days or 14) / 2.5
            r = r + 0.15 * torch.exp(-0.5 * ((t - sc) / w) ** 2)
        return r.clamp(0.0, 1.0)

    if "asymmetric_risk" in name:
        r = _base_risk(t, p.cardio)
        # Sharp right-skewed spikes: fast rise, slow decay
        for sc in _flare_days(len(t), p, seed):
            rise  = (t >= sc).float()
            decay = torch.exp(-torch.clamp(t - sc, min=0.0) / 20.0)  # slow decay
            spike = (p.flare_magnitude or 0.22) * rise * decay
            r = r + spike
        return r.clamp(0.0, 1.0)

    if "phase_shifted" in name:
        rd = p.recovery_days or 120
        r  = p.cardio.base + p.cardio.drift_per_day * t
        r  = r + (p.cardio.base - r[-1].item()) * torch.exp(-t / rd)
        r  = r + p.cardio.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
        return r.clamp(0.0, 1.0)

    # Fallback — use base generator
    return _base_cardio(t, p, seed)


def _hard_mental(t: torch.Tensor, p: Persona, seed: int) -> torch.Tensor:
    """
    Hard mental ground truth. Mirrors _hard_cardio for the mental axis.
    Key addition: asymmetric_risk has LEFT-skewed mental (slow build, fast crash).
    """
    name = p.name.split("_v")[0].split("X")[0]

    if "silent_drifter" in name:
        r = _base_risk(t, p.mental, phase=torch.pi / 4)
        return r.clamp(0.0, 1.0)

    if "contradictory_signals" in name:
        # Mental spikes at same time as cardio overtraining
        r      = _base_risk(t, p.mental)
        period = p.overtraining_period_days or 60
        n      = len(t)
        for sc in range(period, n, period):
            r = r + 0.12 * torch.exp(-0.5 * ((t - sc) / 10.0) ** 2)
        return r.clamp(0.0, 1.0)

    if "phantom_recovery" in name:
        # Mental recovers faster — diverges from cardio
        rd = max(1, (p.recovery_days or 240) // 3)
        r  = 0.18 + (p.mental.base - 0.18) * torch.exp(-t / rd)
        r  = r + p.mental.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
        return r.clamp(0.0, 1.0)

    if "multi_domain_event" in name:
        # Mental peaks 5 days BEFORE cardio — different temporal signature
        r = _base_risk(t, p.mental, phase=torch.pi / 6)
        for sc in _spike_days(len(t), p, seed):
            early_sc = max(0, sc - 5)
            r = r + 0.10 * torch.exp(-0.5 * ((t - early_sc) / 10.0) ** 2)
        return r.clamp(0.0, 1.0)

    if "asymmetric_risk" in name:
        # Left-skewed: mental accumulates slowly then crashes fast
        r = _base_risk(t, p.mental, phase=torch.pi / 4)
        for sc in _flare_days(len(t), p, seed):
            buildup = torch.sigmoid((t - (sc - 30)).float() / 8.0)  # slow build
            crash   = torch.exp(-torch.clamp(t - sc, min=0.0) / 5.0)   # fast crash
            r = r + (p.flare_magnitude or 0.22) * 0.7 * buildup * (1 - crash + 0.1)
        return r.clamp(0.0, 1.0)

    if "phase_shifted" in name:
        rd = max(1, (p.recovery_days or 120) * 2 // 3)
        r  = p.mental.base + p.mental.drift_per_day * t
        r  = r + (p.mental.base - r[-1].item()) * torch.exp(-t / rd)
        r  = r + p.mental.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
        return r.clamp(0.0, 1.0)

    return _base_mental(t, p, seed)


def _hard_signals(
        p: Persona,
        cardio_gt: torch.Tensor,
        mental_gt: torch.Tensor,
        t: torch.Tensor,
        seed: int,
        aqi: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    Signal overrides for hard archetypes.
    Most hard personas use the base sample_signals() with targeted tweaks.
    """
    name = p.name.split("_v")[0].split("X")[0]
    n    = len(t)

    # Get base signals first
    sigs = sample_signals(p, cardio_gt, mental_gt, t, seed, aqi)

    if "contradictory_signals" in name:
        # Steps INCREASE as cardio worsens — the decoy
        overtraining_boost = 0.6 * cardio_gt   # more overtrained → more steps (paradox)
        sigs["steps"]      = (sigs["steps"] * (1 + overtraining_boost)).clamp(0, 30000)
        sigs["active_min"] = (sigs["active_min"] * (1 + 0.4 * cardio_gt)).clamp(0, 150)
        # HRV drops harder than base formula
        sigs["hrv_rmssd_ms"] = (sigs["hrv_rmssd_ms"] * (1 - 0.25 * cardio_gt)).clamp(8, 120)

    if "phase_shifted" in name:
        # Each signal has a unique recovery timescale
        rd          = p.recovery_days or 120
        hrv_tau     = rd / 8.0    # fast: 15-day half-life
        sleep_tau   = rd / 2.7    # medium: 45-day half-life
        steps_tau   = rd / 1.3    # slow: 90-day half-life
        spo2_tau    = rd / 1.0    # slowest: 120-day half-life

        base_hrv    = p.signals["hrv_rmssd_ms"]
        base_sleep  = p.signals["sleep_efficiency_pct"] / 100
        base_steps  = p.signals["steps"]
        base_spo2   = p.signals["spo2_pct"]

        low_hrv     = 22.0
        low_sleep   = 0.55
        low_steps   = 400.0
        low_spo2    = 95.0

        sigs["hrv_rmssd_ms"]         = (low_hrv  + (base_hrv   - low_hrv  ) * (1 - torch.exp(-t / hrv_tau))   + _randn(n, p.noise["hrv_rmssd_ms"],  11, seed)).clamp(8,  120)
        sigs["sleep_efficiency_pct"] = (low_sleep + (base_sleep - low_sleep) * (1 - torch.exp(-t / sleep_tau)) + _randn(n, p.noise["sleep_efficiency_pct"], 12, seed)).clamp(0.3, 1.0)
        sigs["steps"]                = (low_steps + (base_steps - low_steps) * (1 - torch.exp(-t / steps_tau)) + _randn(n, p.noise["steps"], 13, seed)).clamp(0, 30000)
        sigs["spo2_pct"]             = (low_spo2  + (base_spo2  - low_spo2 ) * (1 - torch.exp(-t / spo2_tau))  + _randn(n, p.noise["spo2_pct"],  14, seed)).clamp(90, 100)

    if "phantom_recovery" in name:
        # Signals recover quickly (surface looks fine) but risk stays high
        rd = p.recovery_days or 240
        fast_tau = rd / 17.0   # signals recover in ~14 days
        base_hrv = p.signals["hrv_rmssd_ms"]
        low_hrv  = 24.0
        sigs["hrv_rmssd_ms"]   = (low_hrv + (base_hrv - low_hrv) * (1 - torch.exp(-t / fast_tau)) + _randn(n, p.noise["hrv_rmssd_ms"], 15, seed)).clamp(8, 120)
        sigs["resting_hr_bpm"] = (p.signals["resting_hr_bpm"] + 18 * torch.exp(-t / fast_tau)      + _randn(n, p.noise["resting_hr_bpm"], 16, seed)).clamp(38, 115)
        sigs["steps"]          = (p.signals["steps"] * (1 - 0.7 * torch.exp(-t / (fast_tau * 2)))  + _randn(n, p.noise["steps"], 17, seed)).clamp(0, 30000)

    return sigs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HARD GENERATE
# Top-level generate() replacement that routes hard personas to overrides.
# ══════════════════════════════════════════════════════════════════════════════

def _is_hard(name: str) -> bool:
    return any(k in name for k in HARD_PERSONAS.keys())


def generate_hard(p: Persona, n_days: int = 1825) -> DataSample:
    """
    Drop-in replacement for generator.generate() that uses hard GT functions
    for hard archetypes, falls back to base generate() for everything else.
    """
    if not _is_hard(p.name):
        return _base_generate(p, n_days)

    seed = p.seed
    t    = _arange(n_days)
    aqi  = _build_aqi_curve(t, p, seed) if p.aqi_spike_count else None

    cardio_gt = _hard_cardio(t, p, seed)
    mental_gt = _hard_mental(t, p, seed)

    return DataSample(
        persona_name=p.name,
        t=t,
        cardio_gt=cardio_gt,
        mental_gt=mental_gt,
        signals=_hard_signals(p, cardio_gt, mental_gt, t, seed, aqi),
        aqi_curve=aqi,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HARD DEGRADATION
# Three new injectors on top of base degradation.py.
# ══════════════════════════════════════════════════════════════════════════════

def _correlated_dropout(
        signals: dict[str, torch.Tensor],
        g: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Device-loss dropout: when HRV is missing, HR goes missing too.
    When steps are missing, active_min goes missing too.
    Models a watch being taken off or out of battery — not random per-signal.
    Confidence is 0 for all co-missing signals.
    """
    out  = {k: v.clone() for k, v in signals.items()}
    conf = {k: torch.ones(len(v)) for k, v in signals.items()}

    n       = len(next(iter(signals.values())))
    n_drops = _randint(1, 4, g)

    CORRELATED_GROUPS = [
        ["hrv_rmssd_ms", "resting_hr_bpm"],      # same wrist sensor
        ["steps", "active_min"],                   # same motion sensor
        ["sleep_efficiency_pct", "rem_min", "deep_min"],  # sleep tracker off
        ["spo2_pct", "resting_hr_bpm"],            # pulse ox cluster
    ]

    for _ in range(n_drops):
        group = _choice(CORRELATED_GROUPS, g)
        wl    = _randint(4, 18, g)
        ws    = _randint(3, max(4, n - wl - 3), g)
        for sig in group:
            if sig in out:
                out [sig][ws:ws + wl] = float("nan")
                conf[sig][ws:ws + wl] = 0.0

    return out, conf


def _adversarial_drift(
        signals: dict[str, torch.Tensor],
        g: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Adversarial sensor drift that mimics genuine health improvement:
      - HRV drifts UP (looks like fitness gain)
      - HR drifts DOWN (looks like recovery)
    But it's a sensor artifact, not real improvement.
    Confidence = 0.35 in the affected window (uncertain, not zero).
    """
    out  = {k: v.clone() for k, v in signals.items()}
    conf = {k: torch.ones(len(v)) for k, v in signals.items()}

    n  = len(next(iter(signals.values())))
    wl = _randint(25, 55, g)
    ws = _randint(15, max(16, n - wl - 15), g)
    we = ws + wl

    if "hrv_rmssd_ms" in out:
        mean_hrv      = float(out["hrv_rmssd_ms"][~out["hrv_rmssd_ms"].isnan()].mean().item()) if not out["hrv_rmssd_ms"].isnan().all() else 50.0
        fake_gain_hrv = mean_hrv * (_rand(g) * 0.12 + 0.06)      # +6% to +18% fake HRV gain
        ramp          = torch.linspace(0, fake_gain_hrv, wl)
        out ["hrv_rmssd_ms"][ws:we] = (out["hrv_rmssd_ms"][ws:we] + ramp).clamp(8, 120)
        conf["hrv_rmssd_ms"][ws:we] = 0.35

    if "resting_hr_bpm" in out:
        mean_hr       = float(out["resting_hr_bpm"][~out["resting_hr_bpm"].isnan()].mean().item()) if not out["resting_hr_bpm"].isnan().all() else 70.0
        fake_drop_hr  = mean_hr * (_rand(g) * 0.08 + 0.04)       # -4% to -12% fake HR drop
        ramp          = torch.linspace(0, fake_drop_hr, wl)
        out ["resting_hr_bpm"][ws:we] = (out["resting_hr_bpm"][ws:we] - ramp).clamp(38, 115)
        conf["resting_hr_bpm"][ws:we] = 0.35

    return out, conf


def _burst_noise(
        arr: torch.Tensor,
        key: str,
        g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Short random windows of 4–6× amplified noise (not NaN).
    Confidence = 0.25 in burst windows.
    Unlike dropout, the signal is PRESENT but highly unreliable.
    Forces the model to use confidence weights, not just NaN masks.
    """
    n    = arr.shape[0]
    out  = arr.clone()
    conf = torch.ones(n)

    n_bursts = _randint(2, 5, g)
    for _ in range(n_bursts):
        wl   = _randint(2, 6, g)
        ws   = _randint(2, max(3, n - wl - 2), g)
        amp  = _rand(g) * 3.0 + 1.5     # 1.5–4.5× noise amplification

        # Estimate local std from clean neighbours
        lo   = max(0, ws - 10)
        hi   = min(n, ws + wl + 10)
        region = out[lo:hi]
        valid  = region[~region.isnan()]
        std    = float(valid.std().item()) if len(valid) > 2 else 5.0

        burst = torch.randn(wl) * std * amp
        out [ws:ws + wl] = out[ws:ws + wl] + burst
        conf[ws:ws + wl] = 0.25

    return out, conf


def _degrade_hard(
        signals: dict[str, torch.Tensor],
        persona_name: str,
        difficulty: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Full hard degradation pipeline.
    difficulty controls which injectors are active:
      easy        → base degradation only (no hard injectors)
      medium      → base + burst_noise
      hard        → base + burst_noise + correlated_dropout
      adversarial → all of the above + adversarial_drift
    """
    base_seed = abs(hash(persona_name)) % (2 ** 31) + 44444
    g_hard    = _gen(base_seed + 99999)

    # 1. Apply base per-signal degradation first
    degraded_sigs = {}
    conf_sigs     = {}
    for i, (key, arr) in enumerate(signals.items()):
        from data.degradation import _degrade_signal
        d, c = _degrade_signal(arr, key, base_seed + i * 1000)
        degraded_sigs[key] = d
        conf_sigs[key]     = c

    if difficulty == "easy":
        return degraded_sigs, conf_sigs

    # 2. Burst noise (medium+)
    g_burst = _gen(base_seed + 11111)
    for key in degraded_sigs:
        d, c = _burst_noise(degraded_sigs[key], key, g_burst)
        degraded_sigs[key] = d
        conf_sigs[key]     = torch.minimum(conf_sigs[key], c)

    if difficulty == "medium":
        return degraded_sigs, conf_sigs

    # 3. Correlated dropout (hard+)
    g_corr = _gen(base_seed + 22222)
    corr_out, corr_conf = _correlated_dropout(degraded_sigs, g_corr)
    degraded_sigs = corr_out
    for key in conf_sigs:
        conf_sigs[key] = torch.minimum(conf_sigs[key], corr_conf[key])

    if difficulty == "hard":
        return degraded_sigs, conf_sigs

    # 4. Adversarial drift (adversarial only)
    g_adv = _gen(base_seed + 33333)
    adv_out, adv_conf = _adversarial_drift(degraded_sigs, g_adv)
    degraded_sigs = adv_out
    for key in conf_sigs:
        conf_sigs[key] = torch.minimum(conf_sigs[key], adv_conf[key])

    return degraded_sigs, conf_sigs


def degrade_hard(sample: DataSample, difficulty: str = "hard") -> DegradedSample:
    """
    Drop-in for degradation.degrade_sample() with hard injectors.
    difficulty: "easy" | "medium" | "hard" | "adversarial"
    """
    assert difficulty in ("easy", "medium", "hard", "adversarial"), \
        f"Unknown difficulty: {difficulty}"

    degraded, confidence = _degrade_hard(sample.signals, sample.persona_name, difficulty)

    return DegradedSample(
        persona_name     = sample.persona_name,
        t                = sample.t.cpu(),
        cardio_gt        = sample.cardio_gt.cpu(),
        mental_gt        = sample.mental_gt.cpu(),
        clean_signals    = {k: v.cpu() for k, v in sample.signals.items()},
        degraded_signals = {k: v.cpu() for k, v in degraded.items()},
        confidence       = {k: v.cpu() for k, v in confidence.items()},
        aqi_curve        = sample.aqi_curve.cpu() if sample.aqi_curve is not None else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HARD SPAWNER
# Generates hard persona variants with difficulty tags.
# ══════════════════════════════════════════════════════════════════════════════

# Difficulty assignment rules:
#   hard archetype × hard archetype crossbreed → adversarial
#   hard archetype × base archetype crossbreed → hard
#   pure hard variant                          → hard
#   pure base variant                          → medium (already done by spawner.py)

HARD_CROSSBREEDS = [
    # (name_a, name_b, count, difficulty)
    ("silent_drifter",          "swe",                  20, "hard"),
    ("silent_drifter",          "elderly",              20, "hard"),
    ("contradictory_signals",   "athlete",              20, "hard"),
    ("contradictory_signals",   "recovering",           15, "adversarial"),
    ("phantom_recovery",        "chronic_illness",      20, "adversarial"),
    ("phantom_recovery",        "elderly",              15, "hard"),
    ("multi_domain_event",      "high_aqi",             20, "adversarial"),
    ("multi_domain_event",      "shift_worker",         15, "hard"),
    ("asymmetric_risk",         "hormonal",             20, "adversarial"),
    ("asymmetric_risk",         "college_student",      15, "hard"),
    ("phase_shifted_recovery",  "new_parent",           20, "adversarial"),
    ("phase_shifted_recovery",  "recovering",           20, "hard"),
    # Hard × Hard crossbreeds — most adversarial
    ("silent_drifter",          "phantom_recovery",     10, "adversarial"),
    ("contradictory_signals",   "asymmetric_risk",      10, "adversarial"),
    ("multi_domain_event",      "phase_shifted_recovery", 10, "adversarial"),
]

N_HARD_PURE_PER_ARCHETYPE = 30   # 6 archetypes × 30 = 180 pure hard
N_HARD_CROSSBREEDS        = sum(n for _, _, n, _ in HARD_CROSSBREEDS)  # 250
N_HARD_TOTAL              = N_HARD_PURE_PER_ARCHETYPE * len(HARD_PERSONAS) + N_HARD_CROSSBREEDS


@dataclass
class TaggedPersona:
    """Wraps a Persona with a curriculum difficulty tag."""
    persona:    Persona
    difficulty: str   # "easy" | "medium" | "hard" | "adversarial"

    # Forward all Persona attributes for drop-in compatibility
    def __getattr__(self, name):
        return getattr(self.persona, name)


def spawn_hard(
        base_personas: dict[str, Persona],
        seed_offset: int = 90000,
) -> list[TaggedPersona]:
    """
    Returns a list of TaggedPersona mixing hard archetypes and crossbreeds.
    base_personas: output of load_personas() — used for crossbreeding base × hard.
    """
    all_personas = {**base_personas, **HARD_PERSONAS}
    result: list[TaggedPersona] = []
    counter = seed_offset

    # Pure hard variants
    for name, hp in HARD_PERSONAS.items():
        for v in range(N_HARD_PURE_PER_ARCHETYPE):
            variant = _pure_variant(hp, v, counter)
            result.append(TaggedPersona(persona=variant, difficulty="hard"))
            counter += 1

    # Crossbreeds
    for name_a, name_b, count, difficulty in HARD_CROSSBREEDS:
        pa = all_personas.get(name_a)
        pb = all_personas.get(name_b)
        if pa is None or pb is None:
            print(f"  [warn] skipping crossbreed {name_a}×{name_b} — persona not found")
            continue
        for v in range(count):
            blended = _crossbreed(pa, pb, v, counter)
            result.append(TaggedPersona(persona=blended, difficulty=difficulty))
            counter += 1

    print(
        f"Spawned {len(result)} hard personas "
        f"({N_HARD_PURE_PER_ARCHETYPE * len(HARD_PERSONAS)} pure hard + "
        f"{N_HARD_CROSSBREEDS} crossbreeds)"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HARD SHARD WRITER
# Writes hard shards to outputs/hard_shards/.
# Each shard carries a difficulty field in metadata.
# ══════════════════════════════════════════════════════════════════════════════

N_HARD_SHARDS           = 43    # ~10 personas/shard; ceil(430/10)
HARD_PERSONAS_PER_SHARD = 10
N_DAYS_HARD             = 1825  # 5 years — longer sequences for drift detection
N_WORKERS_HARD          = 2


def _process_hard_shard(args: tuple) -> dict:
    shard_idx, tagged_slice, output_dir = args

    tokens_list = []
    conf_list   = []
    raw_list    = []
    cardio_list = []
    mental_list = []
    meta_list   = []

    for tagged in tagged_slice:
        p          = tagged.persona
        difficulty = tagged.difficulty

        sample   = generate_hard(p, n_days=N_DAYS_HARD)
        degraded = degrade_hard(sample, difficulty=difficulty)
        tok      = tokenize(degraded, n_days=N_DAYS_HARD)

        tokens_list.append(tok.tokens)
        conf_list.append(tok.confidence)
        raw_list.append(tok.raw_signals)
        cardio_list.append(tok.cardio_gt)
        mental_list.append(tok.mental_gt)

        meta_list.append({
            "name":        p.name,
            "description": p.description,
            "seed":        p.seed,
            "archetype":   p.name.split("_v")[0],
            "cardio_base": round(p.cardio.base, 4),
            "mental_base": round(p.mental.base, 4),
            "difficulty":  difficulty,                    # ← new field
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shard_data = {
        "tokens":      torch.stack(tokens_list, dim=0),
        "confidence":  torch.stack(conf_list,   dim=0),
        "raw_signals": torch.stack(raw_list,    dim=0),
        "cardio_gt":   torch.stack(cardio_list, dim=0),
        "mental_gt":   torch.stack(mental_list, dim=0),
        "metadata":    meta_list,
    }

    path = Path(output_dir) / f"hard_shard_{shard_idx:04d}.pt"
    torch.save(shard_data, str(path), _use_new_zipfile_serialization=False)

    if shard_idx == 0:
        print(f"\n  [hard shard 0] shapes:")
        for k, v in shard_data.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape} {v.dtype}", flush=True)

    size_mb  = path.stat().st_size / 1e6
    tok_std  = shard_data["tokens"].float().std().item()
    diff_str = ", ".join(set(m["difficulty"] for m in meta_list))
    print(f"  [hard shard {shard_idx:02d}]  {path.name}  {size_mb:.1f} MB  "
          f"std={tok_std:.4f}  [{diff_str}]", flush=True)

    return {"shard": shard_idx, "path": str(path), "size_mb": size_mb, "personas": meta_list}


def generate_hard_dataset(
        output_dir: str = "outputs/hard_shards",
        n_workers: int  = N_WORKERS_HARD,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading base personas...")
    base = load_personas()

    print(f"Spawning hard variants ({N_HARD_TOTAL} total)...")
    tagged = spawn_hard(base)

    # Pad to exact shard multiple
    n_shards = math.ceil(len(tagged) / HARD_PERSONAS_PER_SHARD)
    while len(tagged) < n_shards * HARD_PERSONAS_PER_SHARD:
        tagged.append(tagged[len(tagged) % len(tagged)])   # cycle fill

    shards = [
        tagged[i * HARD_PERSONAS_PER_SHARD:(i + 1) * HARD_PERSONAS_PER_SHARD]
        for i in range(n_shards)
    ]

    print(f"\nGenerating {n_shards} hard shards × {HARD_PERSONAS_PER_SHARD} personas "
          f"with {n_workers} workers...")
    print(f"Output: {out.resolve()}\n")

    t0   = time.time()
    args = [(i, shards[i], str(out)) for i in range(n_shards)]

    if n_workers <= 1:
        results = [_process_hard_shard(a) for a in args]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_process_hard_shard, args)

    elapsed  = time.time() - t0
    total_mb = sum(r["size_mb"] for r in results)

    # Count difficulty distribution
    diff_counts: dict[str, int] = {}
    for r in results:
        for m in r["personas"]:
            d = m["difficulty"]
            diff_counts[d] = diff_counts.get(d, 0) + 1

    manifest = {
        "dataset":              "hard",
        "n_personas":           len(tagged),
        "n_shards":             n_shards,
        "n_days":               N_DAYS_HARD,
        "token_dim":            104,
        "total_size_mb":        round(total_mb, 1),
        "generation_time_s":    round(elapsed, 1),
        "signal_keys":          SIGNAL_KEYS,
        "difficulty_counts":    diff_counts,
        "curriculum_mapping":   {
            "easy":        "Phase 1 — clean joint",
            "medium":      "Phase 2 — noisy",
            "hard":        "Phase 3 — degraded",
            "adversarial": "Phase 4 — edge cases",
        },
        "hard_archetypes":      list(HARD_PERSONAS.keys()),
        "shards":               sorted(results, key=lambda r: r["shard"]),
    }
    with open(out / "hard_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  Hard shards:  {n_shards}")
    print(f"  Personas:     {len(tagged)}")
    print(f"  Size:         {total_mb / 1024:.2f} GB  ({total_mb:.0f} MB)")
    print(f"  Time:         {elapsed:.1f}s")
    print(f"  Difficulty breakdown:")
    for d, cnt in sorted(diff_counts.items()):
        print(f"    {d:14s}: {cnt}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — QUICK SANITY CHECK
# Run as: python -m data.hard_data --check
# Generates 1 sample per hard archetype, prints shapes and GT stats.
# ══════════════════════════════════════════════════════════════════════════════

def _sanity_check() -> None:
    print("\n── Hard persona sanity check ─────────────────────────────────────")
    for name, p in HARD_PERSONAS.items():
        sample = generate_hard(p, n_days=365)
        c_mean = sample.cardio_gt.mean().item()
        c_max  = sample.cardio_gt.max().item()
        m_mean = sample.mental_gt.mean().item()

        degraded = degrade_hard(sample, difficulty="adversarial")
        tok      = tokenize(degraded, n_days=365)

        # Check NaN rate in degraded signals
        nan_rates = {
            k: float(v.isnan().float().mean().item())
            for k, v in degraded.degraded_signals.items()
        }
        max_nan = max(nan_rates.values())

        print(
            f"  {name:28s}  "
            f"cardio_mean={c_mean:.3f}  cardio_max={c_max:.3f}  "
            f"mental_mean={m_mean:.3f}  "
            f"tokens={tok.tokens.shape}  "
            f"max_nan_rate={max_nan:.2f}"
        )
    print("── OK ─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    mp.freeze_support()
    _sanity_check()
    generate_hard_dataset(
        output_dir="outputs/hard_shards",
        n_workers=N_WORKERS_HARD,
    )