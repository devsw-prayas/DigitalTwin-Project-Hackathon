"""
data/spawner.py
Spawns 5000 Persona variants from 19 archetypes — pure torch RNG, no numpy.

KEY CHANGE FROM ORIGINAL: Independent cardio/mental jitter seeds.
─────────────────────────────────────────────────────────────────────────────
The original spawner applied the same generator g to both cardio and mental
risk params in sequence. This meant the jitter was correlated: if cardio.base
got jittered up, mental.base tended to get jittered up too (same RNG state).

Fix: use independent generators for cardio vs mental jitter.
    g_cardio = seeded with seed
    g_mental  = seeded with seed + 999983  (large prime offset)

This decouples the jitter, so spawned variants have varied cardio/mental
correlation — some will have both high, some both low, some divergent.
This is critical for training the JSD divergence signal.

Also added new crossbreed pairs targeting the 6 divergent archetypes,
and increased N_TOTAL to 10000 to generate more data.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from .personas import Persona, RiskParams, load_personas

JITTER_LO = 0.20
JITTER_HI = 0.30
N_TOTAL   = 10000   # increased from 5000 for more data

# Crossbreed pairs: (archetype_a, archetype_b, count)
# Divergent archetypes included to generate hard mixed cases
CROSSBREED_PAIRS = [
    # Original pairs
    ("healthy",          "swe",               30),
    ("healthy",          "recovering",        25),
    ("swe",              "sedentary",         25),
    ("swe",              "hormonal",          25),
    ("sedentary",        "hormonal",          25),
    ("recovering",       "high_aqi",          20),
    # Original extended pairs
    ("athlete",          "swe",               30),
    ("athlete",          "recovering",        25),
    ("shift_worker",     "sedentary",         25),
    ("shift_worker",     "chronic_illness",   20),
    ("chronic_illness",  "elderly",           25),
    ("elderly",          "sedentary",         25),
    ("college_student",  "swe",               25),
    ("college_student",  "remote_worker",     20),
    ("new_parent",       "swe",               25),
    ("new_parent",       "remote_worker",     20),
    ("remote_worker",    "sedentary",         25),
    ("traveler",         "swe",               25),
    ("traveler",         "athlete",           20),
    ("hormonal",         "college_student",   20),
    # New pairs — divergent archetypes as anchors
    # These produce variants where cardio and mental genuinely diverge
    ("cardiac_stoic",    "healthy",           40),  # high cardio, low mental variants
    ("cardiac_stoic",    "swe",               35),  # high cardio + burnout mental
    ("cardiac_stoic",    "sedentary",         35),
    ("anxious_athlete",  "healthy",           40),  # low cardio, high mental variants
    ("anxious_athlete",  "college_student",   35),  # athlete anxiety + exam stress
    ("anxious_athlete",  "swe",               35),
    ("mindful_couch",    "sedentary",         35),  # opposite drift direction
    ("mindful_couch",    "remote_worker",     30),
    ("burned_out_runner","recovering",        35),  # cardio recovering, mental worsening
    ("burned_out_runner","athlete",           30),
    ("pollution_anxious","high_aqi",          35),  # pure cardio-AQI signal
    ("pollution_anxious","healthy",           30),
    ("grief",            "recovering",        30),  # acute mental, delayed cardio
    ("grief",            "remote_worker",     25),
    # Cross-divergent pairs — maximum asymmetry
    ("cardiac_stoic",    "anxious_athlete",   25),  # opposite dominant agents
    ("mindful_couch",    "burned_out_runner", 25),  # opposite drift directions
    ("pollution_anxious","anxious_athlete",   20),  # cardio-only vs mental-only
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
    """
    Jitter a single agent's risk params using the provided generator.
    Using separate generators for cardio vs mental breaks correlation.
    """
    return RiskParams(
        base               = _jclamp(rp.base,              0.01, 0.95, g),
        drift_per_day      = _jitter_scalar(rp.drift_per_day, g),
        event_day          = rp.event_day,
        event_magnitude    = _jclamp(rp.event_magnitude, -0.4, 0.4, g) if rp.event_magnitude != 0.0 else 0.0,
        event_width_days   = _jclamp(rp.event_width_days, 5.0, 60.0, g),
        seasonal_amplitude = _jclamp(rp.seasonal_amplitude, 0.0, 0.1, g),
    )


def _jitter_dict(d: dict, g: torch.Generator) -> dict:
    return {k: max(0.001, _jitter_scalar(v, g)) for k, v in d.items()}


# ─── Pure variant ──────────────────────────────────────────────────────────────

def _pure_variant(base: Persona, idx: int, seed: int) -> Persona:
    """
    Create a jittered variant of a base persona.

    CRITICAL: use independent generators for cardio vs mental.
    g_cardio and g_mental are seeded with different values so their
    jitter sequences are uncorrelated — breaking cardio/mental correlation
    in spawned variants.
    """
    g_cardio = _g(seed)
    g_mental  = _g(seed + 999983)   # large prime offset — ensures independence
    g_signals = _g(seed + 1999979)  # third independent generator for signals

    def _opt_jitter(v, g):
        if v is None:
            return None
        return max(0.001, _jitter_scalar(v, g))

    return Persona(
        name              = f"{base.name}_v{idx:04d}",
        description       = f"{base.description} [v{idx}]",
        seed              = seed,
        cardio            = _jitter_risk(base.cardio, g_cardio),
        mental            = _jitter_risk(base.mental, g_mental),
        signals           = _jitter_dict(base.signals, g_signals),
        noise             = _jitter_dict(base.noise,   g_signals),
        recovery_days     = base.recovery_days,
        cycle_period_days = base.cycle_period_days,
        aqi_spike_count   = base.aqi_spike_count,
        aqi_spike_magnitude  = _opt_jitter(base.aqi_spike_magnitude, g_cardio),
        aqi_spike_width_days = base.aqi_spike_width_days,
        overtraining_period_days = base.overtraining_period_days,
        overtraining_magnitude   = _opt_jitter(base.overtraining_magnitude, g_cardio),
        overtraining_width_days  = base.overtraining_width_days,
        shift_period_days        = base.shift_period_days,
        shift_mental_amplitude   = _opt_jitter(base.shift_mental_amplitude, g_mental),
        flare_count              = base.flare_count,
        flare_magnitude          = _opt_jitter(base.flare_magnitude, g_cardio),
        flare_width_days         = base.flare_width_days,
        semester_period_days     = base.semester_period_days,
        exam_mental_amplitude    = _opt_jitter(base.exam_mental_amplitude, g_mental),
        travel_count             = base.travel_count,
        travel_width_days        = base.travel_width_days,
        travel_mental_magnitude  = _opt_jitter(base.travel_mental_magnitude, g_mental),
        travel_cardio_magnitude  = _opt_jitter(base.travel_cardio_magnitude, g_cardio),
    )


# ─── Cross-breed ───────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, w: float) -> float:
    return a * (1 - w) + b * w


def _blend_risk(a: RiskParams, b: RiskParams, w: float) -> RiskParams:
    return RiskParams(
        base               = max(0.01, min(0.95, _lerp(a.base, b.base, w))),
        drift_per_day      = _lerp(a.drift_per_day,      b.drift_per_day,      w),
        event_day          = a.event_day if w < 0.5 else b.event_day,
        event_magnitude    = _lerp(a.event_magnitude,    b.event_magnitude,    w),
        event_width_days   = max(5.0, _lerp(a.event_width_days, b.event_width_days, w)),
        seasonal_amplitude = max(0.0, min(0.1, _lerp(a.seasonal_amplitude, b.seasonal_amplitude, w))),
    )


def _blend_dict(a: dict, b: dict, w: float) -> dict:
    return {k: a[k] * (1 - w) + b[k] * w for k in a}


def _crossbreed(a: Persona, b: Persona, idx: int, seed: int) -> Persona:
    """
    Blend two personas, then apply independent cardio/mental jitter.

    Blend weight w is random in [0.25, 0.75].
    After blending, cardio and mental params are jittered with independent
    generators — so the cross-breed can land anywhere in the space between
    the two archetypes with varied cardio/mental correlation.
    """
    g_blend  = _g(seed)
    g_cardio = _g(seed + 999983)
    g_mental  = _g(seed + 1999979)
    g_signals = _g(seed + 2999993)

    w = float(torch.empty(1).uniform_(0.25, 0.75, generator=g_blend).item())

    def pick(va, vb):
        return va if va is not None else vb

    def _opt_jitter_c(v):
        return max(0.001, _jitter_scalar(v, g_cardio)) if v is not None else None

    def _opt_jitter_m(v):
        return max(0.001, _jitter_scalar(v, g_mental)) if v is not None else None

    blended_cardio  = _blend_risk(a.cardio, b.cardio, w)
    blended_mental  = _blend_risk(a.mental, b.mental, w)
    blended_signals = _blend_dict(a.signals, b.signals, w)
    blended_noise   = _blend_dict(a.noise,   b.noise,   w)

    return Persona(
        name        = f"{a.name}X{b.name}_v{idx:04d}",
        description = f"Cross: {a.name} × {b.name} (w={w:.2f})",
        seed        = seed,
        cardio      = _jitter_risk(blended_cardio, g_cardio),
        mental      = _jitter_risk(blended_mental, g_mental),
        signals     = _jitter_dict(blended_signals, g_signals),
        noise       = _jitter_dict(blended_noise,   g_signals),
        recovery_days            = pick(a.recovery_days,            b.recovery_days),
        cycle_period_days        = pick(a.cycle_period_days,        b.cycle_period_days),
        aqi_spike_count          = pick(a.aqi_spike_count,          b.aqi_spike_count),
        aqi_spike_magnitude      = _opt_jitter_c(pick(a.aqi_spike_magnitude,  b.aqi_spike_magnitude)),
        aqi_spike_width_days     = pick(a.aqi_spike_width_days,     b.aqi_spike_width_days),
        overtraining_period_days = pick(a.overtraining_period_days, b.overtraining_period_days),
        overtraining_magnitude   = _opt_jitter_c(pick(a.overtraining_magnitude, b.overtraining_magnitude)),
        overtraining_width_days  = pick(a.overtraining_width_days,  b.overtraining_width_days),
        shift_period_days        = pick(a.shift_period_days,        b.shift_period_days),
        shift_mental_amplitude   = _opt_jitter_m(pick(a.shift_mental_amplitude, b.shift_mental_amplitude)),
        flare_count              = pick(a.flare_count,              b.flare_count),
        flare_magnitude          = _opt_jitter_c(pick(a.flare_magnitude, b.flare_magnitude)),
        flare_width_days         = pick(a.flare_width_days,         b.flare_width_days),
        semester_period_days     = pick(a.semester_period_days,     b.semester_period_days),
        exam_mental_amplitude    = _opt_jitter_m(pick(a.exam_mental_amplitude, b.exam_mental_amplitude)),
        travel_count             = pick(a.travel_count,             b.travel_count),
        travel_width_days        = pick(a.travel_width_days,        b.travel_width_days),
        travel_mental_magnitude  = _opt_jitter_m(pick(a.travel_mental_magnitude, b.travel_mental_magnitude)),
        travel_cardio_magnitude  = _opt_jitter_c(pick(a.travel_cardio_magnitude, b.travel_cardio_magnitude)),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def spawn_all(base_personas: dict[str, Persona], seed_offset: int = 10000) -> list[Persona]:
    result  = []
    counter = seed_offset

    # Pure variants distributed across all archetypes
    names     = list(base_personas.keys())
    per_arch  = N_PURE // len(names)
    remainder = N_PURE  % len(names)

    for i, name in enumerate(names):
        count = per_arch + (1 if i < remainder else 0)
        for v in range(count):
            result.append(_pure_variant(base_personas[name], v, counter))
            counter += 1

    # Cross-breeds
    for name_a, name_b, count in CROSSBREED_PAIRS:
        if name_a not in base_personas:
            print(f"  [spawner] Warning: archetype '{name_a}' not found, skipping crossbreed")
            continue
        if name_b not in base_personas:
            print(f"  [spawner] Warning: archetype '{name_b}' not found, skipping crossbreed")
            continue
        for v in range(count):
            result.append(_crossbreed(base_personas[name_a], base_personas[name_b], v, counter))
            counter += 1

    n_pure_actual = len([p for p in result if 'X' not in p.name])
    n_cross_actual = len([p for p in result if 'X' in p.name])
    print(f"Spawned {len(result)} personas ({n_pure_actual} pure + {n_cross_actual} cross-breeds)")
    return result
