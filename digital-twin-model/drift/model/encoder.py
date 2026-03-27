"""
encoder.py — Shared Encoder with RoPE

Bidirectional transformer encoder that processes the full token sequence
before specialist routing. Uses RoPE for temporal position encoding.

From the spec (Section 5, Stage 4 / hyperparameters):
    Shared encoder layers: 2 (bidirectional; processes LSTM outputs)
    Attention: FlashAttention-2 with RoPE fused on Q and K
    n_heads: 4, head_dim: 32 (d_model=128 / 4 heads)

Why a shared encoder before specialists?
    The shared encoder produces a unified contextual representation Z
    that all specialists then read from via their VSNs and LSTMs.
    It captures cross-signal dependencies (e.g. HRV and sleep co-moving)
    before specialization begins.

Why bidirectional (not causal)?
    The encoder processes the past observation window (90 days of history).
    All of this is already observed — there's no future leakage risk.
    Bidirectional attention lets each day's token attend to all other days,
    capturing long-range dependencies across the window.
    Causal masking is only appropriate for the decoder (future generation).

RoPE vs sinusoidal:
    Sinusoidal encoding adds a fixed absolute position signal to each token.
    RoPE injects relative position information directly into Q/K dot products.
    This means the attention score between two tokens reflects their relative
    temporal distance, which is exactly what matters for health trajectories
    (how far apart were these two measurements?).

FA2 note:
    Production uses flash_attn_func for IO-aware tiled attention.
    This implementation uses standard nn.MultiheadAttention as fallback.
    To enable FA2: set USE_FLASH_ATTN=True and ensure flash-attn is installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import RoPE

# Set to True if flash-attn is installed: pip install flash-attn
# Falls back gracefully to standard MHA if False or unavailable
USE_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTN = True
except ImportError:
    pass


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with RoPE applied to Q and K before dot product.

    Supports both FA2 (if available) and standard PyTorch MHA fallback.

    Args:
        d_model:  total model dimension
        n_heads:  number of attention heads
        dropout:  attention dropout (not applied in FA2 mode)
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # QKV projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(head_dim=self.head_dim, base=1000)

    def forward(
        self,
        q_in: torch.Tensor,          # (B, seq_q, d_model)
        k_in: torch.Tensor,          # (B, seq_k, d_model)
        v_in: torch.Tensor,          # (B, seq_k, d_model)
        context_bias: torch.Tensor = None,   # (B, d_model) optional additive to Q
        need_weights: bool = False,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            q_in:          (B, seq_q, d_model)
            k_in:          (B, seq_k, d_model)
            v_in:          (B, seq_k, d_model)
            context_bias:  (B, d_model) — static context c_e injected into Q
            need_weights:  return attention weights (disables FA2)
            causal:        causal masking (for decoder use)

        Returns:
            out:     (B, seq_q, d_model)
            weights: (B, n_heads, seq_q, seq_k) or None
        """
        B, seq_q, _ = q_in.shape
        seq_k = k_in.shape[1]

        # Project
        Q = self.W_q(q_in)  # (B, seq_q, d_model)
        K = self.W_k(k_in)  # (B, seq_k, d_model)
        V = self.W_v(v_in)  # (B, seq_k, d_model)

        # Inject static enrichment context into Q (additive, from c_e)
        if context_bias is not None:
            Q = Q + context_bias.unsqueeze(1)  # broadcast over seq_q

        # Reshape to (B, seq, n_heads, head_dim) for RoPE
        Q = Q.view(B, seq_q, self.n_heads, self.head_dim)
        K = K.view(B, seq_k, self.n_heads, self.head_dim)
        V = V.view(B, seq_k, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K per head
        # Process each head's Q/K pair through RoPE
        Q_rot = torch.zeros_like(Q)
        K_rot = torch.zeros_like(K)
        for h in range(self.n_heads):
            q_h = Q[:, :, h, :]  # (B, seq_q, head_dim)
            k_h = K[:, :, h, :]  # (B, seq_k, head_dim)
            q_h_rot, k_h_rot = self.rope(q_h, k_h)
            Q_rot[:, :, h, :] = q_h_rot
            K_rot[:, :, h, :] = k_h_rot

        if USE_FLASH_ATTN and not need_weights:
            # FA2 path: expects (B, seq, n_heads, head_dim)
            out = flash_attn_func(
                Q_rot.to(torch.bfloat16),
                K_rot.to(torch.bfloat16),
                V.to(torch.bfloat16),
                dropout_p=0.0,
                causal=causal,
            ).to(Q.dtype)
            out = out.view(B, seq_q, self.d_model)
            out = self.W_o(out)
            return out, None

        else:
            # Standard PyTorch MHA fallback
            # Reshape to (B, n_heads, seq, head_dim) for scaled dot-product
            Q_rot = Q_rot.transpose(1, 2)  # (B, n_heads, seq_q, head_dim)
            K_rot = K_rot.transpose(1, 2)  # (B, n_heads, seq_k, head_dim)
            V = V.transpose(1, 2)           # (B, n_heads, seq_k, head_dim)

            scale = self.head_dim ** -0.5
            scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) * scale  # (B, H, sq, sk)

            if causal:
                mask = torch.triu(torch.ones(seq_q, seq_k, device=Q.device), diagonal=1).bool()
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            weights = F.softmax(scores, dim=-1)  # (B, H, sq, sk)
            weights = self.dropout(weights)

            out = torch.matmul(weights, V)         # (B, H, sq, head_dim)
            out = out.transpose(1, 2).contiguous().view(B, seq_q, self.d_model)
            out = self.W_o(out)

            return out, weights if need_weights else None


class SharedEncoderLayer(nn.Module):
    """
    One transformer encoder layer with RoPE attention.
    Pre-norm architecture (more stable than post-norm for health data scales).

    Optionally accepts static enrichment context c_e injected into Q.
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        d_ff = d_ff or d_model * 4

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = RoPEMultiheadAttention(d_model, n_heads, dropout=0.0)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        Z: torch.Tensor,           # (B, N, d_model)
        c_e: torch.Tensor = None,  # (B, d_model)  static enrichment context
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Pre-norm attention
        Z_norm = self.norm1(Z)
        attn_out, weights = self.attn(
            Z_norm, Z_norm, Z_norm,
            context_bias=c_e,
            need_weights=need_weights,
        )
        Z = Z + attn_out

        # Pre-norm FFN
        Z = Z + self.ffn(self.norm2(Z))

        return Z, weights


class SharedEncoder(nn.Module):
    """
    Shared bidirectional transformer encoder.

    Processes the full 90-day token sequence and produces contextual
    representations Z that all specialist LSTMs read from.

    From the spec:
        Shared encoder layers: 2
        Bidirectional (no causal mask — all past tokens are observed)
        Conditioned on static enrichment context c_e

    Args:
        d_in:     raw token dimension (104)
        d_model:  model dimension (128)
        n_heads:  attention heads (4)
        n_layers: encoder depth (2)
        dropout:  regularization
    """

    def __init__(
        self,
        d_in: int = 104,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection: raw token -> d_model
        self.input_proj = nn.Linear(d_in, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            SharedEncoderLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        X: torch.Tensor,           # (B, N, d_in)
        c_e: torch.Tensor = None,  # (B, d_model) static enrichment context
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            X:            (B, N, 104)  input token sequence
            c_e:          (B, d_model) optional static enrichment context
            need_weights: return per-layer attention weights

        Returns:
            Z:            (B, N, d_model)  contextual representations
            all_weights:  list of per-layer attention weight tensors (or empty)
        """
        # Project input tokens to model dimension
        Z = self.input_dropout(self.input_norm(self.input_proj(X)))

        all_weights = []
        for layer in self.layers:
            Z, w = layer(Z, c_e=c_e, need_weights=need_weights)
            if w is not None:
                all_weights.append(w)

        return self.output_norm(Z), all_weights
