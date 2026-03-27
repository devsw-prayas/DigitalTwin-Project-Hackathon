"""
vsn.py — Variable Selection Networks (VSN)

Per-specialist learned feature selection. Each specialist learns which input
signals are most relevant for its domain at each timestep.

From the spec (Section 5, Stage 2):
    VSN_i(x_t) = sum_j [ softmax(GRN_j(x_t, c_s))_j * GRN_shared(x_{t,j}) ]

Where:
    x_t      = input token at timestep t  (d_in,)
    c_s      = static context vector for selection (d_static,)
    GRN_j    = per-variable GRN that produces a scalar selection weight
    GRN_shared = shared GRN that transforms each variable's value

Why per-specialist VSN?
    A single shared projection weights all signals the same for all agents.
    CardioVSN should learn to up-weight HRV and resting HR.
    MentalVSN should learn to up-weight sleep disruption and screen time.
    The VSN makes this specialization explicit and interpretable —
    you can read out "which signals matter most to which specialist".

Intuition:
    At each timestep, the VSN first independently processes each of the
    n_vars input features through a shared transformation (GRN_shared).
    Then it computes a soft selection weight for each feature via a separate
    per-variable GRN. The final output is a weighted sum — a "selected"
    representation where irrelevant signals are gated near zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.grn import GRN


class VSN(nn.Module):
    """
    Variable Selection Network.

    Processes each input variable independently, then soft-selects
    across variables using a context-conditioned weighting.

    Args:
        d_in:      raw input dimension (total token size, e.g. 104)
        n_vars:    number of individual variables to select over
        d_model:   output dimension per variable (and final output)
        d_static:  dimension of static context vector c_s
        d_hidden:  GRN hidden dimension (defaults to d_model)
        dropout:   dropout in GRN layers
    """

    def __init__(
        self,
        d_in: int,
        n_vars: int,
        d_model: int,
        d_static: int = 32,
        d_hidden: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vars = n_vars
        self.d_model = d_model
        d_hidden = d_hidden or d_model

        # Per-variable input projection: each variable -> d_model embedding
        # Each variable is treated as a scalar, projected to d_model
        self.var_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_vars)
        ])

        # Shared GRN: transforms each projected variable embedding
        # One shared GRN, applied to each variable's embedding independently
        self.grn_shared = GRN(d_model, d_context=0, d_hidden=d_hidden, dropout=dropout)

        # Per-variable GRNs: produce scalar selection logit for each variable
        # Conditioned on static context c_s to allow persona-level specialization
        self.grn_vars = nn.ModuleList([
            GRN(d_model, d_context=d_static, d_hidden=d_hidden, dropout=dropout)
            for _ in range(n_vars)
        ])

        # Final linear to project per-variable logits to scalar for softmax
        self.selection_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,       # (B, N, d_in) — full token sequence
        c_s: torch.Tensor,     # (B, d_static) — static selection context
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:   (B, N, d_in)    input token sequence
            c_s: (B, d_static)   static context for selection conditioning

        Returns:
            out:     (B, N, d_model)  selected/weighted variable representation
            weights: (B, N, n_vars)  softmax selection weights (interpretable!)
        """
        B, N, D = x.shape

        # Split into individual variable scalars
        # We treat the first n_vars dims as the variables to select over
        # (in practice, the tokenizer has already structured these)
        vars_raw = x[..., :self.n_vars]  # (B, N, n_vars)

        # --- Step 1: Project each variable scalar to d_model embedding ---
        var_embeds = []
        for i, proj in enumerate(self.var_projections):
            vi = vars_raw[..., i:i+1]          # (B, N, 1)
            ve = proj(vi)                        # (B, N, d_model)
            ve = self.grn_shared(ve)             # (B, N, d_model) — shared transform
            var_embeds.append(ve)

        # Stack: (B, N, n_vars, d_model)
        var_embeds = torch.stack(var_embeds, dim=2)

        # --- Step 2: Compute selection weights ---
        # c_s: (B, d_static) -> expand to (B, N, d_static) for per-timestep conditioning
        c_s_exp = c_s.unsqueeze(1).expand(B, N, -1)  # (B, N, d_static)

        logits = []
        for i, grn in enumerate(self.grn_vars):
            ve = var_embeds[:, :, i, :]              # (B, N, d_model)
            out = grn(ve, c_s_exp)                   # (B, N, d_model)
            logit = self.selection_head(out)          # (B, N, 1)
            logits.append(logit)

        # (B, N, n_vars)
        logits = torch.cat(logits, dim=-1)
        weights = F.softmax(logits, dim=-1)          # (B, N, n_vars)

        # --- Step 3: Weighted sum of variable embeddings ---
        # weights: (B, N, n_vars, 1) * var_embeds: (B, N, n_vars, d_model)
        w = weights.unsqueeze(-1)                    # (B, N, n_vars, 1)
        out = (w * var_embeds).sum(dim=2)            # (B, N, d_model)

        return out, weights
