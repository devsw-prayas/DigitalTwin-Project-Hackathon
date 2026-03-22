"""
data/degradation.py
Full degradation pipeline — pure torch, no numpy.
CPU-only to avoid multi-worker VRAM contention.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional
from .generator import DataSample

DEVICE = torch.device("cpu")
DTYPE  = torch.float32

DRIFT_SIGNALS = {"hrv_rmssd_ms", "resting_hr_bpm"}
SWAP_SIGNALS  = {"steps", "resting_hr_bpm"}


@dataclass
class DegradedSample:
    persona_name:     str
    t:                torch.Tensor
    cardio_gt:        torch.Tensor
    mental_gt:        torch.Tensor
    clean_signals:    dict[str, torch.Tensor]
    degraded_signals: dict[str, torch.Tensor]
    confidence:       dict[str, torch.Tensor]
    aqi_curve:        Optional[torch.Tensor] = None


# ─── Seeded RNG helper ─────────────────────────────────────────────────────────

def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _randint(lo: int, hi: int, g: torch.Generator) -> int:
    return int(torch.randint(lo, hi, (1,), generator=g).item())


def _rand(g: torch.Generator) -> float:
    return float(torch.rand(1, generator=g).item())


def _choice(values: list, g: torch.Generator):
    idx = _randint(0, len(values), g)
    return values[idx]


# ─── Injectors ─────────────────────────────────────────────────────────────────

def _dropout_windows(arr: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n    = arr.shape[0]
    out  = arr.clone()
    conf = torch.ones(n, dtype=DTYPE)
    for _ in range(_randint(1, 3, g)):
        wl = _randint(3, 15, g)
        ws = _randint(5, max(6, n - wl - 5), g)
        out [ws:ws + wl] = float("nan")
        conf[ws:ws + wl] = 0.0
    return out, conf


def _sensor_drift(arr: torch.Tensor, key: str, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n    = arr.shape[0]
    out  = arr.clone()
    conf = torch.ones(n, dtype=DTYPE)
    if key not in DRIFT_SIGNALS:
        return out, conf
    wl   = _randint(20, 45, g)
    ws   = _randint(10, max(11, n - wl - 10), g)
    we   = ws + wl
    mean = float(out[~out.isnan()].mean().item()) if not out.isnan().all() else 1.0
    mag  = _choice([-1.0, 1.0], g) * mean * _rand(g) * 0.15 + mean * 0.10
    ramp = torch.linspace(0, mag, wl)
    out [ws:we] = out[ws:we] + ramp
    conf[ws:we] = 0.5
    return out, conf


def _device_swap(arr: torch.Tensor, key: str, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n    = arr.shape[0]
    out  = arr.clone()
    conf = torch.ones(n, dtype=DTYPE)
    if key not in SWAP_SIGNALS:
        return out, conf
    sd   = _randint(n // 4, 3 * n // 4, g)
    mean = float(out[~out.isnan()].mean().item()) if not out.isnan().all() else 1.0
    off  = _choice([-1.0, 1.0], g) * mean * (_rand(g) * 0.10 + 0.05)
    out[sd:] = out[sd:] + off
    gs = max(0, sd - 1)
    ge = min(n, sd + 3)
    out [gs:ge] = float("nan")
    conf[gs:ge] = 0.0
    conf[ge:min(n, sd + 10)] = conf[ge:min(n, sd + 10)].clamp(max=0.4)
    return out, conf


def _irregular_sampling(arr: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n    = arr.shape[0]
    out  = arr.clone()
    conf = torch.ones(n, dtype=DTYPE)
    rate = _rand(g) * 0.10 + 0.10   # 10–20%
    mask = torch.rand(n, generator=g) < rate
    out [mask] = float("nan")
    conf[mask] = 0.0
    return out, conf


def _degrade_signal(arr: torch.Tensor, key: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    arr = arr.cpu()   # degradation is CPU-only
    g   = _gen(seed)
    arr, c1 = _dropout_windows(arr, g)
    arr, c2 = _sensor_drift(arr, key, g)
    arr, c3 = _device_swap(arr, key, g)
    arr, c4 = _irregular_sampling(arr, g)
    conf = torch.stack([c1, c2, c3, c4]).min(dim=0).values
    return arr, conf


# ─── Full pipeline ─────────────────────────────────────────────────────────────

def degrade_sample(sample: DataSample) -> DegradedSample:
    base_seed = abs(hash(sample.persona_name)) % (2 ** 31) + 77777

    degraded   = {}
    confidence = {}
    for i, (key, arr) in enumerate(sample.signals.items()):
        d, c = _degrade_signal(arr, key, base_seed + i * 1000)
        degraded[key]   = d
        confidence[key] = c

    return DegradedSample(
        persona_name     = sample.persona_name,
        t                = sample.t.cpu(),
        cardio_gt        = sample.cardio_gt.cpu(),
        mental_gt        = sample.mental_gt.cpu(),
        clean_signals    = {k: v.cpu() for k, v in sample.signals.items()},
        degraded_signals = degraded,
        confidence       = confidence,
        aqi_curve        = sample.aqi_curve.cpu() if sample.aqi_curve is not None else None,
    )


def degrade_all(samples: dict) -> dict[str, DegradedSample]:
    return {name: degrade_sample(s) for name, s in samples.items()}
