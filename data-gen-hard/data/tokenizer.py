"""
data/tokenizer.py
Converts DegradedSample → R^104 health tokens — pure torch, CPU only.

KEY CHANGE FROM ORIGINAL: Subspace separation.
─────────────────────────────────────────────────────────────────────────────
The original tokenizer mixed cardio-primary and mental-primary signals into
the same projection subspaces. This destroyed the signal separation created
by the generator.

Fix: each projection subspace now maps to a coherent physiological domain.

    physio [0:32]  — CARDIO-PRIMARY signals
        hrv_rmssd_ms (slow-filtered cardio component preserved)
        resting_hr_bpm
        spo2_pct
        active_min
        resting_hr_bpm (repeated — HR is the most important cardio signal)
        spo2_pct       (repeated — SpO2 is critical for cardio/respiratory)
        active_min     (repeated)
        steps (normalized)

    behav  [32:56] — MENTAL-PRIMARY signals
        sleep_efficiency_pct
        rem_min
        screen_off_min
        sleep_efficiency_pct (repeated — sleep architecture is key mental signal)
        rem_min              (repeated)
        screen_off_min       (repeated)

    env    [56:72] — ENVIRONMENTAL (cardio-mediated, not mental-direct)
        aqi_pm25
        ambient_temp_c
        aqi_pm25       (repeated)
        ambient_temp_c (repeated)

    ctx    [72:88] — CONTEXT FLAGS (persona + behavioral flags)
        11 static persona flags
        5 dynamic behavioral flags
        = 16 total

    sigma  [88:104] — UNCERTAINTY (confidence encoding)
        11 signal confidences → padded to 20 → projected to 16

Output shape: (N, 104) — IDENTICAL TO ORIGINAL. No downstream changes.
─────────────────────────────────────────────────────────────────────────────
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

SIGNAL_KEYS = list(SIGNAL_NORM.keys())  # 11 signals — unchanged

# Pre-compute fixed projection matrices once at module load (CPU, deterministic)
def _make_proj(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g, dtype=DTYPE)
    return W / (in_dim ** 0.5)

_PROJ = {
    # 8 cardio-primary inputs -> 32-dim physio subspace
    "physio": _make_proj(8,  32, seed=1001),
    # 6 mental-primary inputs -> 24-dim behav subspace
    "behav":  _make_proj(6,  24, seed=2002),
    # 4 environmental inputs -> 16-dim env subspace
    "env":    _make_proj(4,  16, seed=3003),
    # 20 confidence values -> 16-dim sigma subspace
    "sigma":  _make_proj(20, 16, seed=4004),
}


def _normalize(t: torch.Tensor, key: str) -> torch.Tensor:
    lo, hi = SIGNAL_NORM[key]
    n = (t - lo) / (hi - lo + 1e-8) * 2 - 1   # → [-1, 1]
    return torch.nan_to_num(n, nan=0.0).clamp(-3.0, 3.0)


@dataclass
class TokenizedSample:
    persona_name: str
    tokens:      torch.Tensor   # (N, 104) float32  — UNCHANGED
    confidence:  torch.Tensor   # (N, 11)  float32  — UNCHANGED
    cardio_gt:   torch.Tensor   # (N,)     float32  — UNCHANGED
    mental_gt:   torch.Tensor   # (N,)     float32  — UNCHANGED
    raw_signals: torch.Tensor   # (N, 11)  float32  — UNCHANGED


def tokenize(sample: DegradedSample, n_days: int = 365) -> TokenizedSample:
    n    = n_days
    sigs = sample.degraded_signals
    conf = sample.confidence

    def sig(key: str) -> torch.Tensor:
        return _normalize(sigs[key].to(DTYPE), key)

    def sc(key: str) -> torch.Tensor:
        return conf[key].to(DTYPE)

    # ── Physio subspace [0:32]: CARDIO-PRIMARY signals ──────────────────────
    # 8 inputs, all cardiovascular/respiratory primary.
    # resting_hr_bpm and spo2_pct repeated — emphasize the two purest
    # cardio signals in this subspace. VSN can up-weight these for CardioAgent.
    physio_raw = torch.stack([
        sig("hrv_rmssd_ms"),           # dual-timescale (cardio-slow component)
        sig("resting_hr_bpm"),         # cardio-primary
        sig("spo2_pct"),               # cardio-primary
        sig("active_min"),             # cardio output
        sig("resting_hr_bpm"),         # repeated: most important cardio signal
        sig("spo2_pct"),               # repeated: critical respiratory marker
        sig("active_min"),             # repeated: physical capacity output
        sig("steps"),                  # cardio-primary (physical activity)
    ], dim=1)                          # (N, 8)
    physio_proj = physio_raw @ _PROJ["physio"]   # (N, 32)

    # ── Behav subspace [32:56]: MENTAL-PRIMARY signals ──────────────────────
    # 6 inputs, all mental health / behavioral primary.
    # sleep_efficiency and rem_min repeated — these are the clearest
    # mental health signals. VSN can up-weight these for MentalAgent.
    behav_raw = torch.stack([
        sig("sleep_efficiency_pct"),   # mental-primary (sleep quality)
        sig("rem_min"),                # mental-primary (REM = mental recovery)
        sig("screen_off_min"),         # mental-primary (behavioral proxy)
        sig("sleep_efficiency_pct"),   # repeated: key mental signal
        sig("rem_min"),                # repeated: key mental signal
        sig("screen_off_min"),         # repeated: key behavioral signal
    ], dim=1)                          # (N, 6)
    behav_proj = behav_raw @ _PROJ["behav"]      # (N, 24)

    # ── Env subspace [56:72]: environmental signals ──────────────────────────
    # AQI and temperature — mediate through cardio, not mental directly.
    # Both repeated to give the projection enough signal.
    env_raw = torch.stack([
        sig("aqi_pm25"),
        sig("ambient_temp_c"),
        sig("aqi_pm25"),
        sig("ambient_temp_c"),
    ], dim=1)                          # (N, 4)
    env_proj = env_raw @ _PROJ["env"]            # (N, 16)

    # ── Context subspace [72:88]: persona flags + dynamic behavioral flags ───
    # Static: which archetype is this persona? (11 flags)
    # Dynamic: current behavioral state flags (5 flags)
    is_high_aqi   = (sig("aqi_pm25")             >  0.0).float()
    is_poor_sleep = (sig("sleep_efficiency_pct") < -0.3).float()
    is_low_hrv    = (sig("hrv_rmssd_ms")         < -0.3).float()
    is_low_steps  = (sig("steps")                < -0.3).float()
    is_hi_screen  = (sig("screen_off_min")       >  0.3).float()

    def flag(fragment: str) -> float:
        return 1.0 if fragment in sample.persona_name else 0.0

    static = torch.tensor([
        flag("high_aqi"),
        flag("hormonal"),
        flag("recovering"),
        flag("swe"),
        flag("sedentary"),
        flag("healthy"),
        flag("cardiac_stoic"),
        flag("anxious_athlete"),
        flag("mindful_couch"),
        flag("burned_out_runner"),
        flag("pollution_anxious"),
    ], dtype=DTYPE).unsqueeze(0).expand(n, -1)   # (N, 11)

    ctx = torch.cat([
        static,
        torch.stack([is_high_aqi, is_poor_sleep, is_low_hrv, is_low_steps, is_hi_screen], dim=1),
    ], dim=1)                                      # (N, 16)

    # ── Sigma subspace [88:104]: uncertainty encoding ─────────────────────────
    conf_stack = torch.stack([sc(k) for k in SIGNAL_KEYS], dim=1)  # (N, 11)
    conf_20    = torch.cat([conf_stack, torch.zeros(n, 9)], dim=1) # (N, 20)
    sigma_proj = conf_20 @ _PROJ["sigma"]                           # (N, 16)

    # ── Concatenate → R^104 ──────────────────────────────────────────────────
    tokens = torch.cat([physio_proj, behav_proj, env_proj, ctx, sigma_proj], dim=1)
    assert tokens.shape == (n, TOKEN_DIM), f"Token shape mismatch: {tokens.shape}"

    # ── Raw signals (N, 11) — unchanged ──────────────────────────────────────
    raw = torch.stack([sigs[k].to(DTYPE) for k in SIGNAL_KEYS], dim=1)

    return TokenizedSample(
        persona_name = sample.persona_name,
        tokens       = tokens.to(torch.float32),
        confidence   = conf_stack.to(torch.float32),
        cardio_gt    = sample.cardio_gt.to(DTYPE),
        mental_gt    = sample.mental_gt.to(DTYPE),
        raw_signals  = raw.to(torch.float32),
    )
