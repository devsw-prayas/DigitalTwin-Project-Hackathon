"""
data/personas.py
Loads personas.yaml and exposes typed Persona dataclasses.
Fails loudly on missing required fields.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

YAML_PATH = Path(__file__).parent / "personas.yaml"

REQUIRED_SIGNAL_KEYS = [
    "hrv_rmssd_ms", "resting_hr_bpm", "spo2_pct",
    "sleep_efficiency_pct", "rem_min", "deep_min",
    "steps", "active_min", "screen_off_min",
    "aqi_pm25", "ambient_temp_c",
]

REQUIRED_RISK_KEYS = [
    "base", "drift_per_day", "event_day", "event_magnitude",
    "event_width_days", "seasonal_amplitude",
]


@dataclass
class RiskParams:
    base: float
    drift_per_day: float
    event_day: int
    event_magnitude: float
    event_width_days: float
    seasonal_amplitude: float


@dataclass
class Persona:
    name: str
    description: str
    seed: int
    cardio: RiskParams
    mental: RiskParams
    signals: dict[str, float]
    noise: dict[str, float]
    # Recovery / cycle / environment specials
    recovery_days: Optional[int] = None
    cycle_period_days: Optional[int] = None
    aqi_spike_count: Optional[int] = None
    aqi_spike_magnitude: Optional[float] = None
    aqi_spike_width_days: Optional[int] = None
    # Athlete
    overtraining_period_days: Optional[int] = None
    overtraining_magnitude: Optional[float] = None
    overtraining_width_days: Optional[int] = None
    # Shift worker
    shift_period_days: Optional[int] = None
    shift_mental_amplitude: Optional[float] = None
    # Chronic illness
    flare_count: Optional[int] = None
    flare_magnitude: Optional[float] = None
    flare_width_days: Optional[int] = None
    # College student
    semester_period_days: Optional[int] = None
    exam_mental_amplitude: Optional[float] = None
    # Traveler
    travel_count: Optional[int] = None
    travel_width_days: Optional[int] = None
    travel_mental_magnitude: Optional[float] = None
    travel_cardio_magnitude: Optional[float] = None


def _parse_risk(name: str, block: str, raw: dict) -> RiskParams:
    for key in REQUIRED_RISK_KEYS:
        if key not in raw:
            raise ValueError(f"Persona '{name}' missing required risk key '{key}' in '{block}'")
    return RiskParams(**{k: raw[k] for k in REQUIRED_RISK_KEYS})


def _validate_signals(name: str, signals: dict, noise: dict) -> None:
    for key in REQUIRED_SIGNAL_KEYS:
        if key not in signals:
            raise ValueError(f"Persona '{name}' missing signal baseline '{key}'")
        if key not in noise:
            raise ValueError(f"Persona '{name}' missing noise value for '{key}'")


def load_personas(path: Path = YAML_PATH) -> dict[str, Persona]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    personas: dict[str, Persona] = {}
    for name, cfg in raw["personas"].items():
        cardio = _parse_risk(name, "cardio", cfg["cardio"])
        mental = _parse_risk(name, "mental", cfg["mental"])
        _validate_signals(name, cfg["signals"], cfg["noise"])

        personas[name] = Persona(
            name=name,
            description=cfg["description"],
            seed=cfg["seed"],
            cardio=cardio,
            mental=mental,
            signals=cfg["signals"],
            noise=cfg["noise"],
            recovery_days=cfg.get("recovery_days"),
            cycle_period_days=cfg.get("cycle_period_days"),
            aqi_spike_count=cfg.get("aqi_spike_count"),
            aqi_spike_magnitude=cfg.get("aqi_spike_magnitude"),
            aqi_spike_width_days=cfg.get("aqi_spike_width_days"),
            overtraining_period_days=cfg.get("overtraining_period_days"),
            overtraining_magnitude=cfg.get("overtraining_magnitude"),
            overtraining_width_days=cfg.get("overtraining_width_days"),
            shift_period_days=cfg.get("shift_period_days"),
            shift_mental_amplitude=cfg.get("shift_mental_amplitude"),
            flare_count=cfg.get("flare_count"),
            flare_magnitude=cfg.get("flare_magnitude"),
            flare_width_days=cfg.get("flare_width_days"),
            semester_period_days=cfg.get("semester_period_days"),
            exam_mental_amplitude=cfg.get("exam_mental_amplitude"),
            travel_count=cfg.get("travel_count"),
            travel_width_days=cfg.get("travel_width_days"),
            travel_mental_magnitude=cfg.get("travel_mental_magnitude"),
            travel_cardio_magnitude=cfg.get("travel_cardio_magnitude"),
        )

    return personas


if __name__ == "__main__":
    personas = load_personas()
    for name, p in personas.items():
        print(f"  {name:12s} seed={p.seed:4d}  cardio_base={p.cardio.base:.2f}  mental_base={p.mental.base:.2f}"
              f"  {'[recovery]' if p.recovery_days else ''}{'[cycle]' if p.cycle_period_days else ''}{'[aqi]' if p.aqi_spike_count else ''}")
    print(f"\n{len(personas)} personas loaded OK.")
