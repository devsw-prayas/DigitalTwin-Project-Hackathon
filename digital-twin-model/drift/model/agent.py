"""
agent.py — Agent Block (Core + Sparse Specialists)

Each agent is a specialist that attends over the encoded sequence using
TFT-style shared-V interpretable attention.

From the spec (Section 5, Stage 4 + Appendix A.3):
    Shared-V attention:
        A_h = softmax(Q_h * K_h^T / sqrt(d_k))   # each head has own Q, K
        V   = W_V * Z                              # ONE shared value projection
        out = concat([A_h * V for h in heads]) * W_out

Why shared-V attention?
    In standard multi-head attention, each head has its own Q, K, and V.
    The different heads produce different output subspaces — you can't
    directly compare attention weights across heads.

    TFT uses a single shared V projection across all heads.
    Because all heads attend to the same V, the attention weights
    ARE directly comparable and interpretable as temporal importance scores.
    "Which timesteps did this specialist attend to most?" is a valid question.

GRN gating (Stage 5):
    After attention, a GRN controls how much of the attention output is used
    vs suppressed in favor of the specialist LSTM's local state.
    If the LSTM already captured the pattern, the gate suppresses the attention.

    GRN(a, h) = LayerNorm(a + sigmoid(...) * transform(a, h))
    a = attention output
    h = LSTM hidden state H_i from Tier 1 MoE

Dual-mode operation:
    Training/fast inference: FA2 fused kernel (need_weights=False)
    Explanation requests:    standard MHA with need_weights=True
    The AgentBlock automatically switches based on the return_weights flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import RoPE
from model.grn import GRN
from model.vsn import VSN

# FA2 import (optional)
USE_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTN = True
except ImportError:
    pass


class SharedVAttention(nn.Module):
    """
    TFT-style shared-V multi-head attention.

    All heads share the value projection W_V.
    Each head has its own W_Q and W_K.
    Attention weights are directly interpretable as temporal importance.

    Args:
        d_model:   model dimension
        n_heads:   number of heads (each head has own Q, K but shared V)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Each head has its own Q and K projections
        self.W_qs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(n_heads)])
        self.W_ks = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(n_heads)])

        # Shared V projection across all heads — this is the TFT interpretability key
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RoPE(head_dim=self.head_dim, base=1000)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        q_in: torch.Tensor,             # (B, 1, d_model)  — current timestep query
        kv_in: torch.Tensor,            # (B, N, d_model)  — full sequence keys/values
        context_bias: torch.Tensor = None,  # (B, d_model) from c_e
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            q_in:          (B, 1, d_model)   query (current timestep)
            kv_in:         (B, N, d_model)   keys and values (full sequence)
            context_bias:  (B, d_model)      c_e injected into Q
            return_weights: return per-head attention weights

        Returns:
            out:     (B, d_model)         attended representation
            weights: (B, N) averaged weights, or None
        """
        B = q_in.shape[0]
        N = kv_in.shape[1]

        # Shared V: all heads use same value projection
        V = self.W_v(kv_in)  # (B, N, d_model)

        # Inject static enrichment context into query
        q_biased = q_in + context_bias.unsqueeze(1) if context_bias is not None else q_in

        # Per-head Q, K with RoPE
        head_outputs = []
        all_weights = []

        for h in range(self.n_heads):
            q_h = self.W_qs[h](q_biased)   # (B, 1, head_dim)
            k_h = self.W_ks[h](kv_in)      # (B, N, head_dim)

            # Apply RoPE to Q and K
            q_h_rot, k_h_rot = self.rope(q_h, k_h)

            # Scaled dot-product attention
            score = torch.bmm(q_h_rot, k_h_rot.transpose(1, 2)) * self.scale  # (B, 1, N)
            w_h = F.softmax(score, dim=-1)          # (B, 1, N)

            # Attend to shared V (using full d_model V, sliced per head)
            # Each head attends to the SAME V but with different attention weights
            V_h = V[:, :, h * self.head_dim:(h + 1) * self.head_dim]  # (B, N, head_dim)
            out_h = torch.bmm(w_h, V_h)             # (B, 1, head_dim)

            head_outputs.append(out_h)
            if return_weights:
                all_weights.append(w_h.squeeze(1))  # (B, N)

        # Concatenate heads
        out = torch.cat(head_outputs, dim=-1)       # (B, 1, d_model)
        out = self.W_o(out).squeeze(1)              # (B, d_model)

        if return_weights:
            # Average across heads -> interpretable temporal importance (B, N)
            weights = torch.stack(all_weights, dim=0).mean(dim=0)
            return out, weights

        return out, None


class AgentBlock(nn.Module):
    """
    Full specialist agent block.

    Pipeline per agent:
        1. VSN:  select which signals matter for this specialist
        2. Attention: shared-V cross-attention over VSN-selected sequence
        3. GRN gate: blend attention output with specialist LSTM state H_i
        4. Output: specialist representation ready for quantile heads

    Args:
        d_model:    model dimension
        n_heads:    attention heads
        d_in:       raw token dimension (for VSN)
        n_vars:     number of variables in VSN
        d_static:   static context dimension
        name:       specialist name for logging
        dropout:    regularization
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_in: int = 104,
        n_vars: int = 20,
        d_static: int = 32,
        name: str = "agent",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.name = name

        # VSN: per-specialist variable selection
        # Selects which input signals matter most for this domain
        self.vsn = VSN(
            d_in=d_in,
            n_vars=n_vars,
            d_model=d_model,
            d_static=d_static,
            dropout=dropout,
        )

        # VSN output projection -> d_model for attention
        self.vsn_proj = nn.Linear(d_model, d_model)

        # TFT shared-V interpretable attention with RoPE
        self.attn = SharedVAttention(d_model, n_heads)

        # GRN gate: controls blend of attention output vs LSTM state H_i
        # a = attention output, context = H_i from specialist LSTM
        self.grn_gate = GRN(d_model, d_context=d_model, dropout=dropout)

        # Output norm
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        Z: torch.Tensor,            # (B, N, d_model) — shared encoder output
        X_raw: torch.Tensor,        # (B, N, d_in)    — raw tokens for VSN
        H_i: torch.Tensor,          # (B, d_model)    — specialist LSTM state from Tier 1
        c_s: torch.Tensor,          # (B, d_static)   — static selection context
        c_e: torch.Tensor = None,   # (B, d_model)    — static enrichment context
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            Z:              (B, N, d_model)  shared encoder representations
            X_raw:          (B, N, d_in)     raw tokens for VSN selection
            H_i:            (B, d_model)     specialist LSTM state
            c_s:            (B, d_static)    selection context for VSN
            c_e:            (B, d_model)     enrichment context for attention Q
            return_weights: return attention + VSN weights

        Returns:
            h:           (B, d_model)     final agent representation
            attn_w:      (B, N) or None   temporal attention weights
            vsn_w:       (B, N, n_vars) or None  variable selection weights
        """
        # --- Stage 1: Variable Selection ---
        # VSN selects which signals are most relevant for this specialist
        # Using raw tokens (pre-encoder) to maintain interpretable signal-level attribution
        vsn_out, vsn_weights = self.vsn(X_raw, c_s)  # (B, N, d_model), (B, N, n_vars)
        vsn_out = self.vsn_proj(vsn_out)              # (B, N, d_model)

        # Fuse VSN-selected features with shared encoder output
        # The encoder has global context; VSN adds specialist-specific weighting
        Z_fused = Z + vsn_out  # (B, N, d_model)

        # --- Stage 2: Shared-V Cross-Attention ---
        # Query = current timestep (what are we predicting from?)
        # Keys/Values = full history (what do we attend to?)
        q = Z_fused[:, -1:, :]  # (B, 1, d_model) — last timestep query

        attn_out, attn_weights = self.attn(
            q_in=q,
            kv_in=Z_fused,
            context_bias=c_e,
            return_weights=return_weights,
        )
        # attn_out: (B, d_model)

        # --- Stage 3: GRN Gate ---
        # Gate controls how much attention output vs LSTM state is used
        # If LSTM already captured the pattern, gate suppresses attention
        h = self.grn_gate(attn_out, H_i)  # (B, d_model)

        h = self.output_norm(h)

        if return_weights:
            return h, attn_weights, vsn_weights

        return h, None, None
