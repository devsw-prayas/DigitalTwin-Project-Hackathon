"""
rope.py — Rotary Position Embedding (RoPE)

Encodes relative temporal position directly into Q/K vectors inside attention.
This replaces sinusoidal or learned absolute position embeddings.

From the spec (Appendix A.2):
    RoPE(x, t)_{2i}   = x_{2i}   * cos(t * theta_i) - x_{2i+1} * sin(t * theta_i)
    RoPE(x, t)_{2i+1} = x_{2i}   * sin(t * theta_i) + x_{2i+1} * cos(t * theta_i)
    theta_i = base^(-2i/d),  base = 1000  (health-timescale calibration)

Why base=1000 (not the standard 10000)?
    Standard transformers use base=10000, tuned for token-level text sequences.
    Health data uses daily tokens over 90-day windows. A lower base compresses
    the frequency range so the model can distinguish "7 days ago" vs "60 days ago"
    more cleanly at this shorter sequence length.

Used in:
    - SharedEncoder transformer layers (Q/K of each attention head)
    - AgentBlock cross-attention (Q/K)

Note on FA2:
    In production, flash_attn.layers.rotary.RotaryEmbedding fuses RoPE with FA2.
    This implementation is a pure-PyTorch fallback that is functionally identical
    and works on any hardware. Swap for the fused version when flash-attn is available.
"""

import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding.

    Args:
        head_dim:  dimension per attention head (must be even)
        base:      frequency base (1000 for health timescales, 10000 for NLP)
        max_seq:   maximum sequence length to cache (default: 512)
    """

    def __init__(self, head_dim: int, base: int = 1000, max_seq: int = 512):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.base = base

        # Precompute theta_i = base^(-2i/d) for i in [0, head_dim/2)
        # Shape: (head_dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache cos/sin for efficiency up to max_seq
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        """Precompute and cache cos/sin tables up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Outer product: (seq_len, head_dim/2)
        freqs = torch.outer(t, self.inv_freq)
        # Full embedding: (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cache_len = seq_len

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension: [-x2, x1]."""
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: (..., seq_len, head_dim)
            k: (..., seq_len, head_dim)

        Returns:
            q_rot, k_rot: same shapes as input
        """
        seq_len = q.shape[-2]

        # Extend cache if needed
        if seq_len > self._cache_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(q.dtype)  # (seq_len, head_dim)
        sin = self.sin_cached[:seq_len].to(q.dtype)

        # Broadcast to (..., seq_len, head_dim)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot


def apply_rope_to_qk(q: torch.Tensor, k: torch.Tensor, rope: RoPE) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function: reshape multihead q/k, apply RoPE, reshape back.

    Args:
        q: (B, seq_q, n_heads * head_dim)  or  (B, n_heads, seq_q, head_dim)
        k: (B, seq_k, n_heads * head_dim)  or  (B, n_heads, seq_k, head_dim)
        rope: RoPE module

    Returns:
        q_rot, k_rot with same shapes
    """
    return rope(q, k)
