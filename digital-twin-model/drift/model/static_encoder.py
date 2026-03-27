"""
static_encoder.py — Static Covariate Encoder

Encodes time-invariant user metadata into 4 learned context vectors that
condition every downstream stage of the model.

From the spec (Section 5, Stage 1):
    Input: [age_norm, sex_onehot, condition_flags, device_id] -> R^static_dim
    Output: c_s (selection), c_e (enrichment), c_h (hidden init), c_c (cross-attn)
    Method: 4 separate GRN(static_input) -> one context vector each

Why 4 separate vectors?
    Each downstream stage needs a different type of conditioning:
    - c_s: conditions VSN variable selection (which signals matter?)
    - c_e: conditions attention enrichment (what context enriches attention?)
    - c_h: initializes LSTM hidden states (what's your baseline dynamic?)
    - c_c: conditions Tier 2 FFN experts (what context shapes computation depth?)

    A single shared vector would entangle these different roles.
    Four independent GRNs let each specialize for its purpose.

Why not just concatenate static features to every token?
    Treating static metadata as temporal tokens is a category error (spec, §5.1).
    A 55-year-old's HRV baseline is categorically different from a 25-year-old's —
    this should condition the *interpretation* of all temporal signals, not compete
    with them for attention.

Static input schema:
    age_norm          : float in [0,1]  (age / 100)
    sex_onehot        : 2-dim one-hot   [male, female]
    condition_flags   : N-dim binary    [hypertension, diabetes, ...]  (up to 8 flags)
    device_type       : D-dim one-hot   [garmin, apple, fitbit, ...]   (up to 4 types)

    Total raw static dim = 1 + 2 + 8 + 4 = 15 (padded to d_static=32 by linear proj)
"""

import torch
import torch.nn as nn
from model.grn import GRN


# Raw static feature dimensions
AGE_DIM = 1
SEX_DIM = 2
CONDITION_DIM = 8   # up to 8 chronic condition flags
DEVICE_DIM = 4      # device type one-hot
RAW_STATIC_DIM = AGE_DIM + SEX_DIM + CONDITION_DIM + DEVICE_DIM  # = 15


class StaticCovariateEncoder(nn.Module):
    """
    Encodes static user metadata into 4 context vectors.

    All downstream model components receive one or more of these vectors
    as conditioning inputs.

    Args:
        d_raw:    raw static feature dimension (default: 15)
        d_static: projected/output context vector dimension (default: 32)
        dropout:  dropout in GRN layers
    """

    def __init__(self, d_raw: int = RAW_STATIC_DIM, d_static: int = 32, dropout: float = 0.0):
        super().__init__()

        self.d_static = d_static

        # Project raw static features to d_static before GRN processing
        self.input_proj = nn.Linear(d_raw, d_static)
        self.input_norm = nn.LayerNorm(d_static)

        # 4 independent GRNs — each produces one context vector
        # No external context here (context=0) — static is the root
        self.grn_cs = GRN(d_static, d_context=0, dropout=dropout)  # selection
        self.grn_ce = GRN(d_static, d_context=0, dropout=dropout)  # enrichment
        self.grn_ch = GRN(d_static, d_context=0, dropout=dropout)  # LSTM hidden init
        self.grn_cc = GRN(d_static, d_context=0, dropout=dropout)  # cross-attn / FFN

    def forward(self, static_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            static_input: (B, d_raw)  raw static features per user

        Returns:
            dict with keys:
                "c_s": (B, d_static)  selection context  -> VSN conditioning
                "c_e": (B, d_static)  enrichment context -> attention Q injection
                "c_h": (B, d_static)  hidden init context -> LSTM h0/c0 init
                "c_c": (B, d_static)  cross-attn context  -> Tier 2 FFN experts
        """
        # Project and normalize raw features
        s = self.input_norm(self.input_proj(static_input))  # (B, d_static)

        return {
            "c_s": self.grn_cs(s),   # (B, d_static)
            "c_e": self.grn_ce(s),   # (B, d_static)
            "c_h": self.grn_ch(s),   # (B, d_static)
            "c_c": self.grn_cc(s),   # (B, d_static)
        }


def build_static_input(
    age: float,
    sex: str,               # "male" or "female"
    conditions: list[str],  # e.g. ["hypertension", "diabetes"]
    device: str,            # e.g. "garmin", "apple", "fitbit", "other"
) -> torch.Tensor:
    """
    Helper: construct a raw static feature vector from human-readable inputs.

    Returns:
        (1, RAW_STATIC_DIM) tensor ready for StaticCovariateEncoder
    """
    CONDITION_MAP = {
        "hypertension": 0, "diabetes": 1, "asthma": 2,
        "depression": 3, "anxiety": 4, "sleep_apnea": 5,
        "heart_disease": 6, "copd": 7
    }
    DEVICE_MAP = {"garmin": 0, "apple": 1, "fitbit": 2, "other": 3}

    age_vec = torch.tensor([age / 100.0])

    sex_vec = torch.zeros(SEX_DIM)
    if sex.lower() == "male":
        sex_vec[0] = 1.0
    else:
        sex_vec[1] = 1.0

    cond_vec = torch.zeros(CONDITION_DIM)
    for c in conditions:
        if c in CONDITION_MAP:
            cond_vec[CONDITION_MAP[c]] = 1.0

    dev_vec = torch.zeros(DEVICE_DIM)
    dev_idx = DEVICE_MAP.get(device.lower(), 3)
    dev_vec[dev_idx] = 1.0

    raw = torch.cat([age_vec, sex_vec, cond_vec, dev_vec])  # (RAW_STATIC_DIM,)
    return raw.unsqueeze(0)  # (1, RAW_STATIC_DIM)
