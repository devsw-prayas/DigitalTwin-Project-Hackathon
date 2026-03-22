"""
data/generator.py
Ground truth risk curves + signal sampler — pure torch, no numpy.
All tensors stay on DEVICE until shard_writer saves them.
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


# ─── Ground truth ──────────────────────────────────────────────────────────────

def _base_risk(t: torch.Tensor, p, phase: float = 0.0) -> torch.Tensor:
    r = p.base + p.drift_per_day * t
    r = r + p.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365 + phase)
    if p.event_day >= 0:
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
    if _is(p.name, "recovering", "new_parent"):
        rd      = p.recovery_days or 120
        plateau = p.cardio.base + p.cardio.drift_per_day * t
        r       = plateau + (p.cardio.base - plateau) * torch.exp(-t / (rd / 3.0))
        r       = r + p.cardio.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
    elif _is(p.name, "high_aqi", "Xhigh_aqi", "high_aqiX"):
        r = _base_risk(t, p.cardio)
        for sc in _spike_days(len(t), p, seed):
            r = r + 0.18 * torch.exp(-0.5 * ((t - sc) / ((p.aqi_spike_width_days or 21) / 3.0)) ** 2)
    elif _is(p.name, "athlete"):
        # Healthy baseline + periodic overtraining spikes
        r      = _base_risk(t, p.cardio)
        period = p.overtraining_period_days or 90
        mag    = p.overtraining_magnitude or 0.18
        width  = p.overtraining_width_days or 14
        n      = len(t)
        spike_days = range(period, n, period)
        for sc in spike_days:
            r = r + mag * torch.exp(-0.5 * ((t - sc) / (width / 2.5)) ** 2)
    elif _is(p.name, "chronic_illness"):
        r          = _base_risk(t, p.cardio)
        flare_days = _flare_days(len(t), p, seed)
        for sc in flare_days:
            w = (p.flare_width_days or 21) / 3.0
            r = r + (p.flare_magnitude or 0.16) * torch.exp(-0.5 * ((t - sc) / w) ** 2)
    elif _is(p.name, "traveler"):
        r          = _base_risk(t, p.cardio)
        travel_days = _travel_days(len(t), p, seed)
        for sc in travel_days:
            w = (p.travel_width_days or 10) / 2.5
            r = r + (p.travel_cardio_magnitude or 0.08) * torch.exp(-0.5 * ((t - sc) / w) ** 2)
    else:
        r = _base_risk(t, p.cardio)
    return r.clamp(0.0, 1.0)


def ground_truth_mental(t: torch.Tensor, p: Persona, seed: int) -> torch.Tensor:
    if _is(p.name, "recovering", "new_parent"):
        rd      = p.recovery_days or 120
        plateau = p.mental.base + p.mental.drift_per_day * t
        r       = plateau + (p.mental.base - plateau) * torch.exp(-t / (rd / 2.5))
        r       = r + p.mental.seasonal_amplitude * torch.sin(2 * torch.pi * t / 365)
    elif _is(p.name, "hormonal", "Xhormonal", "hormonalX"):
        period = p.cycle_period_days or 28
        r      = _base_risk(t, p.mental, phase=torch.pi / 4)
        r      = r + 0.18 * (0.5 - 0.5 * torch.cos(2 * torch.pi * t / period))
    elif _is(p.name, "high_aqi", "Xhigh_aqi", "high_aqiX"):
        r = _base_risk(t, p.mental, phase=torch.pi / 6)
        for sc in _spike_days(len(t), p, seed):
            r = r + 0.06 * torch.exp(-0.5 * ((t - sc) / ((p.aqi_spike_width_days or 21) / 4.0)) ** 2)
    elif _is(p.name, "shift_worker"):
        # 7-day cycle matching shift rotation
        period = p.shift_period_days or 7
        amp    = p.shift_mental_amplitude or 0.22
        r      = _base_risk(t, p.mental)
        r      = r + amp * (0.5 - 0.5 * torch.cos(2 * torch.pi * t / period))
    elif _is(p.name, "college_student"):
        # Semester cycle with exam spikes
        period = p.semester_period_days or 120
        amp    = p.exam_mental_amplitude or 0.20
        r      = _base_risk(t, p.mental, phase=torch.pi / 4)
        # Spike at end of each semester (exam week)
        n = len(t)
        for sc in range(period - 10, n, period):
            r = r + amp * torch.exp(-0.5 * ((t - sc) / 8.0) ** 2)
    elif _is(p.name, "chronic_illness"):
        r          = _base_risk(t, p.mental)
        flare_days = _flare_days(len(t), p, seed)
        for sc in flare_days:
            w = (p.flare_width_days or 21) / 3.5
            r = r + (p.flare_magnitude or 0.16) * 0.7 * torch.exp(-0.5 * ((t - sc) / w) ** 2)
    elif _is(p.name, "traveler"):
        r           = _base_risk(t, p.mental, phase=torch.pi / 6)
        travel_days = _travel_days(len(t), p, seed)
        for sc in travel_days:
            w = (p.travel_width_days or 10) / 2.0
            r = r + (p.travel_mental_magnitude or 0.14) * torch.exp(-0.5 * ((t - sc) / w) ** 2)
    else:
        r = _base_risk(t, p.mental, phase=torch.pi / 4)
    return r.clamp(0.0, 1.0)


# ─── Signal sampler ────────────────────────────────────────────────────────────

def sample_signals(
    p: Persona,
    cardio_gt: torch.Tensor,
    mental_gt: torch.Tensor,
    t: torch.Tensor,
    seed: int,
    aqi: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    n  = len(t)
    s  = p.signals
    ns = p.noise

    def noise(key: str, off: int) -> torch.Tensor:
        return _randn(n, ns[key], off, seed)

    hrv   = s["hrv_rmssd_ms"]  * (1 - 0.55 * cardio_gt) * (1 - 0.15 * mental_gt) + noise("hrv_rmssd_ms",  10)
    hr    = s["resting_hr_bpm"] + 22 * cardio_gt + 5 * mental_gt                  + noise("resting_hr_bpm",20)
    spo2  = (s["spo2_pct"]
             - (0.008 * (aqi - 20).clamp(min=0) if aqi is not None else torch.zeros(n, device=DEVICE))
             - 1.2 * cardio_gt - 0.3 * mental_gt + noise("spo2_pct", 30))
    sleep = s["sleep_efficiency_pct"] / 100 - 0.28 * mental_gt - 0.05 * cardio_gt + noise("sleep_efficiency_pct", 40)
    rem   = s["rem_min"]  - 45 * mental_gt - 10 * cardio_gt + noise("rem_min",  50)
    deep  = s["deep_min"] - 30 * mental_gt - 15 * cardio_gt + noise("deep_min", 60)

    if _is(p.name, "recovering") and p.recovery_days:
        step_base = s["steps"] * (t / p.recovery_days).clamp(0, 1) ** 1.5
    else:
        step_base = torch.full((n,), s["steps"], dtype=DTYPE, device=DEVICE)

    steps  = step_base * (1 - 0.35 * cardio_gt)               + noise("steps",          70)
    active = s["active_min"]     * (1 - 0.40 * cardio_gt)     + noise("active_min",     80)
    screen = s["screen_off_min"] + 80 * mental_gt + 20 * cardio_gt + noise("screen_off_min", 90)

    if _is(p.name, "hormonal", "Xhormonal", "hormonalX"):
        period = p.cycle_period_days or 28
        luteal = 0.5 - 0.5 * torch.cos(2 * torch.pi * t / period)
        hrv    = hrv   - 8.0  * luteal
        sleep  = sleep - 0.06 * luteal
        rem    = rem   - 12.0 * luteal
        active = active - 8.0 * luteal

    aqi_out = aqi if aqi is not None else (
        s["aqi_pm25"] + 6 * torch.sin(2 * torch.pi * t / 365 + torch.pi) + noise("aqi_pm25", 100)
    )
    temp = s["ambient_temp_c"] + 8 * torch.sin(2 * torch.pi * t / 365) + noise("ambient_temp_c", 110)

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
    aqi  = _build_aqi_curve(t, p, seed) if _is(p.name, "high_aqi", "Xhigh_aqi", "high_aqiX") else None
    return DataSample(
        persona_name=p.name, t=t,
        cardio_gt=ground_truth_cardio(t, p, seed),
        mental_gt=ground_truth_mental(t, p, seed),
        signals=sample_signals(p, ground_truth_cardio(t, p, seed), ground_truth_mental(t, p, seed), t, seed, aqi),
        aqi_curve=aqi,
    )


def generate_all(personas: dict, n_days: int = 365) -> dict[str, DataSample]:
    return {name: generate(p, n_days) for name, p in personas.items()}
