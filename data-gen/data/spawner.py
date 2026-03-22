"""
data/spawner.py
Spawns ~500 Persona variants from 6 archetypes — pure torch RNG, no numpy.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from .personas import Persona, RiskParams, load_personas

JITTER_LO = 0.20
JITTER_HI = 0.30
N_TOTAL   = 5000

CROSSBREED_PAIRS = [
    # Original pairs
    ("healthy",        "swe",            30),
    ("healthy",        "recovering",     25),
    ("swe",            "sedentary",      25),
    ("swe",            "hormonal",       25),
    ("sedentary",      "hormonal",       25),
    ("recovering",     "high_aqi",       20),
    # New pairs — biologically meaningful blends
    ("athlete",        "swe",            30),   # high-performing SWE who trains
    ("athlete",        "recovering",     25),   # athlete post-injury
    ("shift_worker",   "sedentary",      25),   # night shift + no exercise
    ("shift_worker",   "chronic_illness",20),   # shift work worsening chronic condition
    ("chronic_illness","elderly",        25),   # elderly with chronic condition
    ("elderly",        "sedentary",      25),   # sedentary elderly
    ("college_student","swe",            25),   # junior dev / new grad
    ("college_student","remote_worker",  20),   # remote student
    ("new_parent",     "swe",            25),   # SWE with newborn
    ("new_parent",     "remote_worker",  20),   # remote worker + new parent
    ("remote_worker",  "sedentary",      25),   # extreme inactivity
    ("traveler",       "swe",            25),   # frequent-flyer SWE
    ("traveler",       "athlete",        20),   # traveling athlete
    ("hormonal",       "college_student",20),   # college woman with strong cycle
]
N_CROSSBREEDS = sum(n for _, _, n in CROSSBREED_PAIRS)
N_PURE        = N_TOTAL - N_CROSSBREEDS


# ─── Torch-seeded helpers ──────────────────────────────────────────────────────

def _g(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _jitter_scalar(v: float, g: torch.Generator) -> float:
    rate = float(torch.empty(1).uniform_(JITTER_LO, JITTER_HI, generator=g).item())
    sign = 1.0 if torch.rand(1, generator=g).item() > 0.5 else -1.0
    return v * (1 + sign * rate)


def _jclamp(v: float, lo: float, hi: float, g: torch.Generator) -> float:
    return max(lo, min(hi, _jitter_scalar(v, g)))


def _jitter_risk(rp: RiskParams, g: torch.Generator) -> RiskParams:
    return RiskParams(
        base               = _jclamp(rp.base,              0.01, 0.95, g),
        drift_per_day      = _jitter_scalar(rp.drift_per_day, g),
        event_day          = rp.event_day,
        event_magnitude    = _jclamp(rp.event_magnitude if rp.event_magnitude > 0 else 0.0, 0.0, 0.4, g),
        event_width_days   = _jclamp(rp.event_width_days, 5.0, 60.0, g),
        seasonal_amplitude = _jclamp(rp.seasonal_amplitude, 0.0, 0.1, g),
    )


def _jitter_dict(d: dict, g: torch.Generator) -> dict:
    return {k: max(0.001, _jitter_scalar(v, g)) for k, v in d.items()}


# ─── Pure variant ──────────────────────────────────────────────────────────────

def _pure_variant(base: Persona, idx: int, seed: int) -> Persona:
    g = _g(seed)
    return Persona(
        name              = f"{base.name}_v{idx:04d}",
        description       = f"{base.description} [v{idx}]",
        seed              = seed,
        cardio            = _jitter_risk(base.cardio, g),
        mental            = _jitter_risk(base.mental, g),
        signals           = _jitter_dict(base.signals, g),
        noise             = _jitter_dict(base.noise,   g),
        recovery_days     = base.recovery_days,
        cycle_period_days = base.cycle_period_days,
        aqi_spike_count   = base.aqi_spike_count,
        aqi_spike_magnitude  = max(0.001, _jitter_scalar(base.aqi_spike_magnitude, g)) if base.aqi_spike_magnitude else None,
        aqi_spike_width_days = base.aqi_spike_width_days,
        overtraining_period_days = base.overtraining_period_days,
        overtraining_magnitude   = max(0.001, _jitter_scalar(base.overtraining_magnitude, g)) if base.overtraining_magnitude else None,
        overtraining_width_days  = base.overtraining_width_days,
        shift_period_days        = base.shift_period_days,
        shift_mental_amplitude   = max(0.001, _jitter_scalar(base.shift_mental_amplitude, g)) if base.shift_mental_amplitude else None,
        flare_count              = base.flare_count,
        flare_magnitude          = max(0.001, _jitter_scalar(base.flare_magnitude, g)) if base.flare_magnitude else None,
        flare_width_days         = base.flare_width_days,
        semester_period_days     = base.semester_period_days,
        exam_mental_amplitude    = max(0.001, _jitter_scalar(base.exam_mental_amplitude, g)) if base.exam_mental_amplitude else None,
        travel_count             = base.travel_count,
        travel_width_days        = base.travel_width_days,
        travel_mental_magnitude  = max(0.001, _jitter_scalar(base.travel_mental_magnitude, g)) if base.travel_mental_magnitude else None,
        travel_cardio_magnitude  = max(0.001, _jitter_scalar(base.travel_cardio_magnitude, g)) if base.travel_cardio_magnitude else None,
    )


# ─── Cross-breed ───────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, w: float) -> float:
    return a * (1 - w) + b * w


def _blend_risk(a: RiskParams, b: RiskParams, w: float) -> RiskParams:
    return RiskParams(
        base               = max(0.01, min(0.95, _lerp(a.base, b.base, w))),
        drift_per_day      = _lerp(a.drift_per_day,      b.drift_per_day,      w),
        event_day          = a.event_day if w < 0.5 else b.event_day,
        event_magnitude    = max(0.0, min(0.4, _lerp(a.event_magnitude,   b.event_magnitude,   w))),
        event_width_days   = _lerp(a.event_width_days,   b.event_width_days,   w),
        seasonal_amplitude = max(0.0, min(0.1, _lerp(a.seasonal_amplitude, b.seasonal_amplitude, w))),
    )


def _blend_dict(a: dict, b: dict, w: float) -> dict:
    return {k: a[k] * (1 - w) + b[k] * w for k in a}


def _crossbreed(a: Persona, b: Persona, idx: int, seed: int) -> Persona:
    g = _g(seed)
    w = float(torch.empty(1).uniform_(0.25, 0.75, generator=g).item())

def _crossbreed(a: Persona, b: Persona, idx: int, seed: int) -> Persona:
    g = _g(seed)
    w = float(torch.empty(1).uniform_(0.25, 0.75, generator=g).item())

    def pick(va, vb):
        return va if va is not None else vb

    blended = Persona(
        name        = f"{a.name}X{b.name}_v{idx:04d}",
        description = f"Cross: {a.name} × {b.name} (w={w:.2f})",
        seed        = seed,
        cardio      = _blend_risk(a.cardio, b.cardio, w),
        mental      = _blend_risk(a.mental, b.mental, w),
        signals     = _blend_dict(a.signals, b.signals, w),
        noise       = _blend_dict(a.noise,   b.noise,   w),
        recovery_days            = pick(a.recovery_days,            b.recovery_days),
        cycle_period_days        = pick(a.cycle_period_days,        b.cycle_period_days),
        aqi_spike_count          = pick(a.aqi_spike_count,          b.aqi_spike_count),
        aqi_spike_magnitude      = pick(a.aqi_spike_magnitude,      b.aqi_spike_magnitude),
        aqi_spike_width_days     = pick(a.aqi_spike_width_days,     b.aqi_spike_width_days),
        overtraining_period_days = pick(a.overtraining_period_days, b.overtraining_period_days),
        overtraining_magnitude   = pick(a.overtraining_magnitude,   b.overtraining_magnitude),
        overtraining_width_days  = pick(a.overtraining_width_days,  b.overtraining_width_days),
        shift_period_days        = pick(a.shift_period_days,        b.shift_period_days),
        shift_mental_amplitude   = pick(a.shift_mental_amplitude,   b.shift_mental_amplitude),
        flare_count              = pick(a.flare_count,              b.flare_count),
        flare_magnitude          = pick(a.flare_magnitude,          b.flare_magnitude),
        flare_width_days         = pick(a.flare_width_days,         b.flare_width_days),
        semester_period_days     = pick(a.semester_period_days,     b.semester_period_days),
        exam_mental_amplitude    = pick(a.exam_mental_amplitude,    b.exam_mental_amplitude),
        travel_count             = pick(a.travel_count,             b.travel_count),
        travel_width_days        = pick(a.travel_width_days,        b.travel_width_days),
        travel_mental_magnitude  = pick(a.travel_mental_magnitude,  b.travel_mental_magnitude),
        travel_cardio_magnitude  = pick(a.travel_cardio_magnitude,  b.travel_cardio_magnitude),
    )

    # Light jitter pass on top of blend
    return Persona(
        name        = blended.name,
        description = blended.description,
        seed        = seed,
        cardio      = _jitter_risk(blended.cardio, g),
        mental      = _jitter_risk(blended.mental, g),
        signals     = _jitter_dict(blended.signals, g),
        noise       = _jitter_dict(blended.noise,   g),
        recovery_days            = blended.recovery_days,
        cycle_period_days        = blended.cycle_period_days,
        aqi_spike_count          = blended.aqi_spike_count,
        aqi_spike_magnitude      = blended.aqi_spike_magnitude,
        aqi_spike_width_days     = blended.aqi_spike_width_days,
        overtraining_period_days = blended.overtraining_period_days,
        overtraining_magnitude   = blended.overtraining_magnitude,
        overtraining_width_days  = blended.overtraining_width_days,
        shift_period_days        = blended.shift_period_days,
        shift_mental_amplitude   = blended.shift_mental_amplitude,
        flare_count              = blended.flare_count,
        flare_magnitude          = blended.flare_magnitude,
        flare_width_days         = blended.flare_width_days,
        semester_period_days     = blended.semester_period_days,
        exam_mental_amplitude    = blended.exam_mental_amplitude,
        travel_count             = blended.travel_count,
        travel_width_days        = blended.travel_width_days,
        travel_mental_magnitude  = blended.travel_mental_magnitude,
        travel_cardio_magnitude  = blended.travel_cardio_magnitude,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def spawn_all(base_personas: dict[str, Persona], seed_offset: int = 10000) -> list[Persona]:
    result  = []
    counter = seed_offset

    # Pure variants
    names       = list(base_personas.keys())
    per_arch    = N_PURE // len(names)
    remainder   = N_PURE  % len(names)
    for i, name in enumerate(names):
        count = per_arch + (1 if i < remainder else 0)
        for v in range(count):
            result.append(_pure_variant(base_personas[name], v, counter))
            counter += 1

    # Cross-breeds
    for name_a, name_b, count in CROSSBREED_PAIRS:
        for v in range(count):
            result.append(_crossbreed(base_personas[name_a], base_personas[name_b], v, counter))
            counter += 1

    print(f"Spawned {len(result)} personas ({N_PURE} pure + {N_CROSSBREEDS} cross-breeds)")
    return result
