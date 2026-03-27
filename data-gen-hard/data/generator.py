"""
data/generator.py
Ground truth risk curves + signal sampler — pure torch, no numpy.

KEY CHANGE FROM ORIGINAL: Physiological signal separation.
─────────────────────────────────────────────────────────────────────────────
The original sampler coupled every signal to BOTH cardio_gt and mental_gt
with similar coefficients, making every token a superposition of both agents.
This caused guaranteed JSD collapse — attending to any signal told you about
both agents equally.

The fix: each signal has a PRIMARY agent and a small SECONDARY coupling.
Some signals have near-zero coupling to one agent entirely.

Signal → Primary agent mapping:
    CARDIO-PRIMARY (mental coupling ≤ 0.10):
        resting_hr_bpm      — direct autonomic load marker
        spo2_pct            — respiratory/cardiovascular only
        active_min          — physical exertion output
        aqi_pm25            — environmental cardio stressor

    MENTAL-PRIMARY (cardio coupling ≤ 0.10):
        rem_min             — REM is the mental health sleep stage
        screen_off_min      — behavioral mental health proxy
        (sleep_efficiency is mental-primary but has moderate cardio coupling)

    SHARED (both agents, but different coefficients and lags):
        hrv_rmssd_ms        — responds to both; cardio effect is slow/sustained,
                              mental effect is fast/acute
        sleep_efficiency    — mental-primary but cardio has a secondary effect
        deep_min            — physical recovery (cardio-secondary, mental-primary)
        steps               — cardio-primary output, mental has small coupling

    ENVIRONMENT (neither agent directly — mediated through cardio):
        aqi_pm25            — feeds into SpO2 and resting HR, not mental directly
        ambient_temp_c      — background only

HRV dual-timescale model:
    HRV has two components:
        slow_cardio: responds to cardio_gt over a 14-day rolling window (slow)
        fast_mental: responds to mental_gt at the current timestep (fast)
    This means cardio and mental create DISTINGUISHABLE HRV patterns:
        - CardioAgent should learn to attend to the slow HRV trend
        - MentalAgent should learn to attend to the fast HRV dips
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os
import torch
from dataclasses import dataclass
from typing import Optional
from .personas import Persona

DEVICE = torch.device("cpu") if os.environ.get("FORCE_CPU") else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DTYPE = torch.float32


def _arange(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=DTYPE, device=DEVICE)


def _randn(n: int, std: float, offset: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed((seed * 31337 + offset * 1009) % (2 ** 31))
    return torch.randn(n, generator=g, dtype=DTYPE).to(DEVICE) * std


def _is(name: str, *fragments) -> bool:
    return any(f in name for f in fragments)


@dataclass
class DataSample:
    persona_name: str
    t:         torch.Tensor
    cardio_gt: torch.Tensor
    mental_gt: torch.Tensor
    signals:   dict[str, torch.Tensor]
    aqi_curve: Optional[torch.Tensor] = None


# ─── Slow EMA for HRV cardio component ────────────────────────────────────────

def _slow_ema(x: torch.Tensor, alpha: float = 0.07) -> torch.Tensor:
    """
    Causal exponential moving average — approximates a ~14-day rolling influence.
    alpha=0.07 -> half-life ~= ln(2)/0.07 ~= 9.9 days.
    Used to make HRV's cardio component slow-moving vs mental's fast response.
    """
    out = torch.zeros_like(x)
    val = x[0].item()
    for i in range(len(x)):
        val = alpha * x[i].item() + (1 - alpha) * val
        out[i] = val
    return out


# ─── Ground truth ─────────────────────────────────────────────────────────────

def _base_risk(t: torch.Tensor, p, phase: float = 0.0) -> torch.Tensor:
    r = p.base + p.drift_per_day * t
    r = r + p.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365 + phase)
    if p.event_day >= 0:
        # event_magnitude can be negative (improvement events, e.g. mindful_couch therapy)
        r = r + p.event_magnitude * torch.exp(
            -0.5 * ((t - p.event_day) / p.event_width_days) ** 2
        )
    return r


def _spike_days(n: int, p: Persona, seed: int) -> list[int]:
    count = p.aqi_spike_count or 0
    if count == 0:
        return []
    g = torch.Generator()
    g.manual_seed(seed + 9999)
    spacing = n // (count + 1)
    return [
        max(30, min(n - 30, i * spacing + int((torch.rand(1, generator=g).item() - 0.5) * spacing // 2)))
        for i in range(1, count + 1)
    ]


def _flare_days(n: int, p: Persona, seed: int) -> list[int]:
    count = p.flare_count or 0
    if count == 0:
        return []
    g = torch.Generator()
    g.manual_seed(seed + 8888)
    spacing = n // (count + 1)
    return [
        max(30, min(n - 30, i * spacing + int((torch.rand(1, generator=g).item() - 0.5) * spacing // 3)))
        for i in range(1, count + 1)
    ]


def _travel_days(n: int, p: Persona, seed: int) -> list[int]:
    count = p.travel_count or 0
    if count == 0:
        return []
    g = torch.Generator()
    g.manual_seed(seed + 7777)
    spacing = n // (count + 1)
    return [
        max(20, min(n - 20, i * spacing + int((torch.rand(1, generator=g).item() - 0.5) * spacing // 3)))
        for i in range(1, count + 1)
    ]


def _build_aqi_curve(t: torch.Tensor, p: Persona, seed: int) -> torch.Tensor:
    n   = len(t)
    aqi = p.signals["aqi_pm25"] + _randn(n, p.noise["aqi_pm25"], 1, seed)
    aqi = aqi + 8 * torch.sin(2 * torch.pi * t / 365 + torch.pi)
    for sc in _spike_days(n, p, seed):
        w   = (p.aqi_spike_width_days or 21) / 3.0
        aqi = aqi + (p.aqi_spike_magnitude or 120.0) * torch.exp(-0.5 * ((t - sc) / w) ** 2)
    return aqi.clamp(0, 500)


def ground_truth_cardio(t: torch.Tensor, p: Persona, seed: int) -> torch.Tensor:
    name = p.name

    if _is(name, "recovering", "burned_out_runner"):
        rd      = p.recovery_days or 120
        plateau = p.cardio.base + p.cardio.drift_per_day * t
        r       = plateau + (p.cardio.base - plateau) * torch.exp(-t / (rd / 3.0))
        r       = r + p.cardio.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
        # Standard event on top if present
        if p.cardio.event_day >= 0:
            r = r + p.cardio.event_magnitude * torch.exp(
                -0.5 * ((t - p.cardio.event_day) / p.cardio.event_width_days) ** 2
            )

    elif _is(name, "new_parent"):
        rd      = p.recovery_days or 365
        plateau = p.cardio.base + p.cardio.drift_per_day * t
        r       = plateau + (p.cardio.base - plateau) * torch.exp(-t / (rd / 3.0))
        r       = r + p.cardio.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)

    elif _is(name, "high_aqi", "pollution_anxious"):
        r = _base_risk(t, p.cardio)
        for sc in _spike_days(len(t), p, seed):
            w = (p.aqi_spike_width_days or 21) / 3.0
            r = r + 0.22 * torch.exp(-0.5 * ((t - sc) / w) ** 2)

    elif _is(name, "athlete"):
        r      = _base_risk(t, p.cardio)
        period = p.overtraining_period_days or 90
        mag    = p.overtraining_magnitude or 0.18
        width  = p.overtraining_width_days or 14
        n      = len(t)
        for sc in range(period, n, period):
            r = r + mag * torch.exp(-0.5 * ((t - sc) / (width / 2.5)) ** 2)

    elif _is(name, "chronic_illness"):
        r = _base_risk(t, p.cardio)
        for sc in _flare_days(len(t), p, seed):
            w = (p.flare_width_days or 21) / 3.0
            r = r + (p.flare_magnitude or 0.16) * torch.exp(-0.5 * ((t - sc) / w) ** 2)

    elif _is(name, "traveler"):
        r = _base_risk(t, p.cardio)
        for sc in _travel_days(len(t), p, seed):
            w = (p.travel_width_days or 10) / 2.5
            r = r + (p.travel_cardio_magnitude or 0.08) * torch.exp(-0.5 * ((t - sc) / w) ** 2)

    else:
        r = _base_risk(t, p.cardio)

    return r.clamp(0.0, 1.0)


def ground_truth_mental(t: torch.Tensor, p: Persona, seed: int) -> torch.Tensor:
    name = p.name

    if _is(name, "recovering"):
        rd      = p.recovery_days or 120
        plateau = p.mental.base + p.mental.drift_per_day * t
        r       = plateau + (p.mental.base - plateau) * torch.exp(-t / (rd / 2.5))
        r       = r + p.mental.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)

    elif _is(name, "new_parent"):
        rd      = p.recovery_days or 365
        plateau = p.mental.base + p.mental.drift_per_day * t
        r       = plateau + (p.mental.base - plateau) * torch.exp(-t / (rd / 2.0))
        r       = r + p.mental.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)

    elif _is(name, "hormonal"):
        period = p.cycle_period_days or 28
        r      = _base_risk(t, p.mental, phase=torch.pi / 4)
        r      = r + 0.18 * (0.5 - 0.5 * torch.cos(2 * torch.pi * t / period))

    elif _is(name, "high_aqi"):
        # AQI has minimal direct mental coupling for high_aqi persona
        r = _base_risk(t, p.mental, phase=torch.pi / 6)
        for sc in _spike_days(len(t), p, seed):
            # Only 0.04 mental effect vs 0.22 cardio — explicit asymmetry
            r = r + 0.04 * torch.exp(-0.5 * ((t - sc) / ((p.aqi_spike_width_days or 21) / 5.0)) ** 2)

    elif _is(name, "pollution_anxious"):
        # Mental is explicitly stable — near-zero AQI coupling
        r = _base_risk(t, p.mental, phase=torch.pi / 8)
        # No AQI spikes on mental at all

    elif _is(name, "shift_worker"):
        period = p.shift_period_days or 7
        amp    = p.shift_mental_amplitude or 0.22
        r      = _base_risk(t, p.mental)
        r      = r + amp * (0.5 - 0.5 * torch.cos(2 * torch.pi * t / period))

    elif _is(name, "college_student"):
        period = p.semester_period_days or 120
        amp    = p.exam_mental_amplitude or 0.20
        r      = _base_risk(t, p.mental, phase=torch.pi / 4)
        n      = len(t)
        for sc in range(period - 10, n, period):
            r = r + amp * torch.exp(-0.5 * ((t - sc) / 8.0) ** 2)

    elif _is(name, "chronic_illness"):
        r = _base_risk(t, p.mental)
        for sc in _flare_days(len(t), p, seed):
            w = (p.flare_width_days or 21) / 3.5
            # Mental flare is 60% of cardio flare magnitude
            r = r + (p.flare_magnitude or 0.16) * 0.60 * torch.exp(-0.5 * ((t - sc) / w) ** 2)

    elif _is(name, "traveler"):
        r = _base_risk(t, p.mental, phase=torch.pi / 6)
        for sc in _travel_days(len(t), p, seed):
            w = (p.travel_width_days or 10) / 2.0
            r = r + (p.travel_mental_magnitude or 0.14) * torch.exp(-0.5 * ((t - sc) / w) ** 2)

    elif _is(name, "burned_out_runner"):
        # Mental worsens while cardio recovers — opposite trajectories
        r = _base_risk(t, p.mental)

    elif _is(name, "mindful_couch"):
        # Therapy breakthrough: negative event_magnitude = improvement
        r = _base_risk(t, p.mental)

    else:
        r = _base_risk(t, p.mental, phase=torch.pi / 4)

    return r.clamp(0.0, 1.0)


# ─── Signal sampler — physiologically separated ───────────────────────────────

def sample_signals(
    p: Persona,
    cardio_gt: torch.Tensor,
    mental_gt: torch.Tensor,
    t: torch.Tensor,
    seed: int,
    aqi: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    Sample physiological signals with explicit primary/secondary agent coupling.

    Design principles:
        1. Cardio-primary signals have near-zero mental coupling coefficients.
        2. Mental-primary signals have near-zero cardio coupling coefficients.
        3. HRV uses a dual-timescale model: slow cardio component + fast mental component.
        4. AQI affects SpO2 and resting HR (cardio), NOT mental directly.
        5. Screen time and REM respond to mental only, not cardio.
        6. Noise is higher for shared signals and lower for primary-only signals
           to maintain signal clarity.
    """
    n  = len(t)
    s  = p.signals
    ns = p.noise

    def noise(key: str, off: int) -> torch.Tensor:
        return _randn(n, ns[key], off, seed)

    # ── HRV: dual timescale (cardio slow + mental fast) ──────────────────────
    # Slow component: cardio_gt filtered through ~14-day EMA
    # Fast component: mental_gt direct (same-day response)
    # This creates temporally distinguishable patterns per agent.
    cardio_slow = _slow_ema(cardio_gt, alpha=0.07)   # ~14-day half-life
    hrv = (
        s["hrv_rmssd_ms"]
        * (1 - 0.62 * cardio_slow)     # cardio: slow sustained suppression
        * (1 - 0.18 * mental_gt)       # mental: fast acute suppression
        + noise("hrv_rmssd_ms", 10)
    )

    # ── Resting HR: CARDIO-PRIMARY ────────────────────────────────────────────
    # Strong cardio coupling. Mental has only 3 bpm effect (sympathetic arousal).
    aqi_hr_effect = 0.0
    if aqi is not None:
        aqi_hr_effect = 0.04 * (aqi - 20).clamp(min=0)   # AQI elevates HR directly

    hr = (
        s["resting_hr_bpm"]
        + 28 * cardio_gt               # cardio: strong (28 bpm range)
        + 3  * mental_gt               # mental: weak (3 bpm — sympathetic only)
        + aqi_hr_effect
        + noise("resting_hr_bpm", 20)
    )

    # ── SpO2: CARDIO-PRIMARY ──────────────────────────────────────────────────
    # Only cardiovascular and respiratory effects. Mental has zero direct effect.
    aqi_spo2_effect = torch.zeros(n, device=DEVICE)
    if aqi is not None:
        aqi_spo2_effect = 0.010 * (aqi - 20).clamp(min=0)  # AQI suppresses SpO2

    spo2 = (
        s["spo2_pct"]
        - 1.8 * cardio_gt              # cardio: moderate suppression
        - aqi_spo2_effect              # AQI: direct suppression
        + 0.0 * mental_gt              # mental: zero coupling (explicit)
        + noise("spo2_pct", 30)
    )

    # ── Sleep efficiency: MENTAL-PRIMARY with moderate cardio secondary ───────
    # Poor cardiovascular health disrupts sleep but mental is the dominant driver.
    sleep = (
        s["sleep_efficiency_pct"] / 100
        - 0.32 * mental_gt             # mental: strong (32% range)
        - 0.06 * cardio_gt             # cardio: weak secondary
        + noise("sleep_efficiency_pct", 40)
    )

    # ── REM: MENTAL-PRIMARY ───────────────────────────────────────────────────
    # REM sleep is the mental health recovery stage. Cardio has near-zero effect.
    rem = (
        s["rem_min"]
        - 55 * mental_gt               # mental: strong suppression
        - 4  * cardio_gt               # cardio: minimal (< 10% of mental effect)
        + noise("rem_min", 50)
    )

    # ── Deep sleep: CARDIO-SECONDARY, MENTAL-PRIMARY ─────────────────────────
    # Deep sleep is physical recovery — moderate cardio effect, strong mental effect.
    deep = (
        s["deep_min"]
        - 22 * mental_gt               # mental: strong (physical recovery disrupted)
        - 18 * cardio_gt               # cardio: moderate (cardiovascular load)
        + noise("deep_min", 60)
    )

    # ── Steps: CARDIO-PRIMARY ─────────────────────────────────────────────────
    # Physical activity output is cardiovascular capacity. Mental has small effect
    # (motivation), but the primary driver is cardio fitness.
    if _is(p.name, "recovering", "burned_out_runner") and p.recovery_days:
        step_base = s["steps"] * (t / p.recovery_days).clamp(0, 1) ** 1.5
    else:
        step_base = torch.full((n,), s["steps"], dtype=DTYPE, device=DEVICE)

    steps = (
        step_base
        * (1 - 0.42 * cardio_gt)       # cardio: strong reduction
        * (1 - 0.06 * mental_gt)       # mental: weak reduction (motivation)
        + noise("steps", 70)
    )

    # ── Active minutes: CARDIO-PRIMARY ────────────────────────────────────────
    active = (
        s["active_min"]
        * (1 - 0.45 * cardio_gt)       # cardio: strong
        * (1 - 0.05 * mental_gt)       # mental: near-zero
        + noise("active_min", 80)
    )

    # ── Screen off time: MENTAL-PRIMARY ──────────────────────────────────────
    # Behavioral proxy for mental health. High mental risk → more screen time.
    # Cardio has essentially no effect on screen use behavior.
    screen = (
        s["screen_off_min"]
        + 110 * mental_gt              # mental: strong increase
        + 4   * cardio_gt              # cardio: near-zero (explicit)
        + noise("screen_off_min", 90)
    )

    # ── Hormonal cycle modulation (persona-specific) ──────────────────────────
    if _is(p.name, "hormonal"):
        period = p.cycle_period_days or 28
        # Luteal phase (days 15-28): higher progesterone suppresses HRV, disrupts sleep/REM
        luteal = 0.5 - 0.5 * torch.cos(2 * torch.pi * t / period)
        hrv    = hrv   - 9.0  * luteal
        sleep  = sleep - 0.07 * luteal
        rem    = rem   - 14.0 * luteal
        active = active - 9.0 * luteal
        # Screen time increases in luteal phase (fatigue, mood changes)
        screen = screen + 30.0 * luteal

    # ── AQI and temperature ───────────────────────────────────────────────────
    aqi_out = aqi if aqi is not None else (
        s["aqi_pm25"] + 6 * torch.sin(2 * torch.pi * t / 365 + torch.pi)
        + noise("aqi_pm25", 100)
    )
    temp = (
        s["ambient_temp_c"]
        + 8 * torch.sin(2 * torch.pi * t / 365)
        + noise("ambient_temp_c", 110)
    )

    return {
        "hrv_rmssd_ms":         hrv.clamp(8,    120),
        "resting_hr_bpm":       hr.clamp(38,    115),
        "spo2_pct":             spo2.clamp(90,  100),
        "sleep_efficiency_pct": sleep.clamp(0.3, 1.0),
        "rem_min":              rem.clamp(0,    180),
        "deep_min":             deep.clamp(0,   150),
        "steps":                steps.clamp(0, 30000),
        "active_min":           active.clamp(0, 150),
        "screen_off_min":       screen.clamp(0, 720),
        "aqi_pm25":             aqi_out.clamp(0, 500),
        "ambient_temp_c":       temp,
    }


# ─── Entry ────────────────────────────────────────────────────────────────────

def generate(p: Persona, n_days: int = 365) -> DataSample:
    seed = p.seed
    t    = _arange(n_days)
    aqi  = _build_aqi_curve(t, p, seed) if _is(p.name, "high_aqi", "pollution_anxious") else None

    cardio = ground_truth_cardio(t, p, seed)
    mental = ground_truth_mental(t, p, seed)

    return DataSample(
        persona_name=p.name,
        t=t,
        cardio_gt=cardio,
        mental_gt=mental,
        signals=sample_signals(p, cardio, mental, t, seed, aqi),
        aqi_curve=aqi,
    )


def generate_all(personas: dict, n_days: int = 365) -> dict[str, DataSample]:
    return {name: generate(p, n_days) for name, p in personas.items()}
