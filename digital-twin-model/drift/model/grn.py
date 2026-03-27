"""
grn.py — Gated Residual Network (GRN)

The fundamental nonlinear building block used throughout the MoE-TFT architecture.
Replaces plain linear layers wherever context-conditional gating is needed.

From the spec (Appendix A.4):
    GRN(a, c) = LayerNorm(a + sigmoid(W_g * (W_1*ELU(W_2*[a;c]) + b_1)) * (W_3*a + b_3))

Where:
    a = primary input
    c = optional context vector (static covariate, or None)
    gate = sigmoid(...) controls how much of the transformed input passes through
    residual connection ensures gradient flow

Used in:
    - Static covariate encoder (4x GRN -> context vectors)
    - Variable Selection Networks (per-specialist feature gating)
    - Expert FFN pool (Tier 2 MoE experts)
    - Agent attention gating
    - Prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    Gated Residual Network.

    Args:
        d_model:     primary input/output dimension
        d_context:   dimension of optional context vector (0 = no context)
        d_hidden:    hidden dimension (defaults to d_model)
        dropout:     dropout rate on hidden layer
    """

    def __init__(self, d_model: int, d_context: int = 0, d_hidden: int = None, dropout: float = 0.0):
        super().__init__()

        d_hidden = d_hidden or d_model
        d_input = d_model + d_context  # concatenated input size

        # W_2: maps [a; c] -> hidden
        self.fc1 = nn.Linear(d_input, d_hidden)

        # W_1: maps hidden -> hidden (after ELU)
        self.fc2 = nn.Linear(d_hidden, d_model)

        # Gate: sigmoid(W_g * hidden) -> scalar gate per feature
        self.gate = nn.Linear(d_hidden, d_model)

        # Residual projection: needed when d_input != d_model
        # W_3: maps a -> d_model for the gated residual
        self.skip = nn.Linear(d_model, d_model) if d_context > 0 else nn.Identity()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            a: (B, ..., d_model)  primary input
            c: (B, ..., d_context) optional context — broadcast if needed

        Returns:
            out: (B, ..., d_model)
        """
        # Concatenate context if provided
        if c is not None:
            # Handle shape mismatch: c may be (B, d_context), a may be (B, N, d_model)
            if c.dim() < a.dim():
                c = c.unsqueeze(1).expand(*a.shape[:-1], c.shape[-1])
            inp = torch.cat([a, c], dim=-1)
        else:
            inp = a

        # Hidden representation with ELU
        h = F.elu(self.fc1(inp))          # (B, ..., d_hidden)
        h = self.dropout(h)

        # Transformed output
        out = self.fc2(h)                  # (B, ..., d_model)

        # Gate in [0, 1]
        g = torch.sigmoid(self.gate(h))   # (B, ..., d_model)

        # Gated residual: gate controls blend of transform vs skip
        residual = self.skip(a)            # (B, ..., d_model)
        out = g * out + (1 - g) * residual

        return self.norm(out)
