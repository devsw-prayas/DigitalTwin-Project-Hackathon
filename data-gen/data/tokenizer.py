"""
data/tokenizer.py
Converts DegradedSample → R^104 health tokens — pure torch, CPU only.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from .degradation import DegradedSample

DEVICE    = torch.device("cpu")
DTYPE     = torch.float32
TOKEN_DIM = 104

SIGNAL_NORM = {
    "hrv_rmssd_ms":         (10.0,  110.0),
    "resting_hr_bpm":       (40.0,  110.0),
    "spo2_pct":             (90.0,  100.0),
    "sleep_efficiency_pct": (0.3,    1.0),
    "rem_min":              (0.0,   160.0),
    "deep_min":             (0.0,   140.0),
    "steps":                (0.0,  25000.0),
    "active_min":           (0.0,   120.0),
    "screen_off_min":       (0.0,   600.0),
    "aqi_pm25":             (0.0,   300.0),
    "ambient_temp_c":       (-10.0,  40.0),
}

SIGNAL_KEYS = list(SIGNAL_NORM.keys())  # 11 signals

# Pre-compute fixed projection matrices once at module load (CPU, deterministic)
def _make_proj(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g, dtype=DTYPE)
    return W / (in_dim ** 0.5)

_PROJ = {
    "physio": _make_proj(8,  32, seed=1001),
    "behav":  _make_proj(6,  24, seed=2002),
    "env":    _make_proj(4,  16, seed=3003),
    "sigma":  _make_proj(20, 16, seed=4004),
}


def _normalize(t: torch.Tensor, key: str) -> torch.Tensor:
    lo, hi = SIGNAL_NORM[key]
    n = (t - lo) / (hi - lo + 1e-8) * 2 - 1   # → [-1, 1]
    return torch.nan_to_num(n, nan=0.0).clamp(-3.0, 3.0)


@dataclass
class TokenizedSample:
    persona_name: str
    tokens:     torch.Tensor   # (N, 104) float32
    confidence: torch.Tensor   # (N, 11)  float32  compact
    cardio_gt:  torch.Tensor   # (N,)     float32
    mental_gt:  torch.Tensor   # (N,)     float32
    raw_signals:torch.Tensor   # (N, 11)  float32


def tokenize(sample: DegradedSample, n_days: int = 365) -> TokenizedSample:
    n    = n_days
    sigs = sample.degraded_signals
    conf = sample.confidence

    def sig(key: str) -> torch.Tensor:
        return _normalize(sigs[key].to(DTYPE), key)

    def sc(key: str) -> torch.Tensor:
        return conf[key].to(DTYPE)

    # ── Physio: 8 → 32 ──
    physio_raw  = torch.stack([sig(k) for k in [
        "hrv_rmssd_ms", "resting_hr_bpm", "spo2_pct", "sleep_efficiency_pct",
        "rem_min", "deep_min", "resting_hr_bpm", "spo2_pct",
    ]], dim=1)                                              # (N, 8)
    physio_proj = physio_raw @ _PROJ["physio"]             # (N, 32)

    # ── Behav: 6 → 24 ──
    behav_raw  = torch.stack([sig(k) for k in [
        "steps", "active_min", "screen_off_min",
        "rem_min", "deep_min", "sleep_efficiency_pct",
    ]], dim=1)                                              # (N, 6)
    behav_proj = behav_raw @ _PROJ["behav"]                # (N, 24)

    # ── Env: 4 → 16 ──
    env_raw  = torch.stack([sig(k) for k in [
        "aqi_pm25", "ambient_temp_c", "aqi_pm25", "ambient_temp_c",
    ]], dim=1)                                              # (N, 4)
    env_proj = env_raw @ _PROJ["env"]                      # (N, 16)

    # ── Context flags: 16 ──
    is_high_aqi   = (sig("aqi_pm25")             >  0.0).float()
    is_poor_sleep = (sig("sleep_efficiency_pct") < -0.3).float()
    is_low_hrv    = (sig("hrv_rmssd_ms")         < -0.3).float()
    is_low_steps  = (sig("steps")                < -0.3).float()
    is_hi_screen  = (sig("screen_off_min")       >  0.3).float()

    def flag(fragment: str) -> float:
        return 1.0 if fragment in sample.persona_name else 0.0

    static = torch.tensor([
        flag("high_aqi"), flag("hormonal"), flag("recovering"),
        flag("swe"), flag("sedentary"), flag("healthy"),
        0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=DTYPE).unsqueeze(0).expand(n, -1)             # (N, 11)

    ctx = torch.cat([
        static,
        torch.stack([is_high_aqi, is_poor_sleep, is_low_hrv, is_low_steps, is_hi_screen], dim=1),
    ], dim=1)                                              # (N, 16)

    # ── Uncertainty: 11 conf → pad 20 → 16 ──
    conf_stack = torch.stack([sc(k) for k in SIGNAL_KEYS], dim=1)  # (N, 11)
    conf_20    = torch.cat([conf_stack, torch.zeros(n, 9)], dim=1) # (N, 20)
    sigma_proj = conf_20 @ _PROJ["sigma"]                           # (N, 16)

    # ── Concat → R^104 ──
    tokens = torch.cat([physio_proj, behav_proj, env_proj, ctx, sigma_proj], dim=1)
    assert tokens.shape == (n, TOKEN_DIM)

    mean_conf = conf_stack.mean(dim=1, keepdim=True).expand(n, TOKEN_DIM)

    # ── Raw signals (N, 11) for shard storage ──
    raw = torch.stack([sigs[k].to(DTYPE) for k in SIGNAL_KEYS], dim=1)

    return TokenizedSample(
        persona_name = sample.persona_name,
        tokens       = tokens.to(torch.float32),
        confidence   = conf_stack.to(torch.float32),
        cardio_gt    = sample.cardio_gt.to(DTYPE),
        mental_gt    = sample.mental_gt.to(DTYPE),
        raw_signals  = raw.to(torch.float32),
    )
