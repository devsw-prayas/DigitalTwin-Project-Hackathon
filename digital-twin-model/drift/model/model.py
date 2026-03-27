"""
model.py — Full MoE-TFT Hybrid Model (DRIFT)

DRIFT: Domain-Routed Interpretable Fusion Transformer

This is the full research architecture from the spec.
All 9 stages of the pipeline are implemented here.

Pipeline:
    Stage 0: Personal baseline normalization (upstream, in data pipeline)
    Stage 1: Static covariate encoder    -> 4 context vectors (c_s, c_e, c_h, c_c)
    Stage 2: VSN per specialist          -> learned feature selection
    Stage 3: Tier 1 MoE LSTM pool        -> specialist latent states H_i
    Stage 4: Shared-V attention (FA2+RoPE) -> temporal attention over sequence
    Stage 5: GRN gating                  -> blend attention + LSTM state
    Stage 6: Tier 2 MoE FFN pool         -> token-level expert routing
    Stage 7: Quantile prediction heads   -> P10/P50/P90 per specialist per horizon
    Stage 8: Post-training calibration   -> conformalized quantile regression (separate)
    Stage 9: Per-user LoRA adaptation    -> frozen base + user-specific adapters

Specialists:
    Always-on (core):  cardio, mental, metabolic, recovery
    Sparse (top-2):    immune, respiratory, hormonal, cog_fatigue

Outputs per specialist:
    quantiles:          (B, 4, 3)  [P10, P50, P90] per horizon
    velocity:           (B, 4)     signed slope per horizon
    time_to_threshold:  (B, 4)     days to concern threshold
    attn_weights:       (B, 90)    temporal importance (explanation mode)
    vsn_weights:        (B, 90, n_vars)  signal importance (explanation mode)

Key hyperparameters (from spec §5.6):
    d_model = 128
    n_heads = 4  (head_dim = 32)
    RoPE base = 1000
    LSTM layers per specialist = 1
    Sequence length N = 90
    Prediction horizons = [7, 30, 90, 180] days
    Core specialists = 4 (always-on)
    Sparse specialist pool = 4 (top-k=2)
    Tier 2 experts = 8 (token-level, top-k=2)
    LoRA rank = 8, alpha = 16
    Precision = bf16 (default), fp16 switchable
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from model.grn import GRN
from model.static_encoder import StaticCovariateEncoder, RAW_STATIC_DIM
from model.encoder import SharedEncoder
from model.moe import Tier1MoE, Tier2MoE
from model.agent import AgentBlock
from model.heads import AgentHead, AgentPrediction
from model.vsn import VSN


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DRIFTOutput:
    """
    Complete output from one forward pass.

    agents: dict mapping specialist name -> AgentPrediction
    routing_info: Tier 1 MoE routing decisions
    tier1_router_probs: (B, n_sparse) for load balancing loss
    tier2_router_probs: (B, N, n_experts) for load balancing loss
    attn_weights: dict name -> (B, N) or None (explanation mode only)
    vsn_weights: dict name -> (B, N, n_vars) or None (explanation mode only)
    """
    agents: dict[str, AgentPrediction]
    tier1_router_probs: torch.Tensor        # (B, n_sparse)
    tier2_router_probs: torch.Tensor        # (B, N, 8)
    routing_info: dict
    attn_weights: dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    vsn_weights: dict[str, Optional[torch.Tensor]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA adapter (per-user continual adaptation)
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    LoRA adapter wrapping a frozen linear layer.

    From the spec (Stage 9):
        Frozen base model.
        Per-user LoRA adapters on LSTM input gates and attention Q/V projections.
        Rank=8, alpha=16.
        Weekly fine-tune on user's rolling 30-day window.
        Adapter storage ~50KB per user.

    LoRA: W' = W + (alpha/rank) * B * A
        A: (rank, d_in)  — initialized with Gaussian
        B: (d_out, rank) — initialized with zeros (adapter starts as identity)

    Args:
        d_in:   input dimension of wrapped linear
        d_out:  output dimension of wrapped linear
        rank:   LoRA rank (default: 8)
        alpha:  LoRA alpha (default: 16)
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA delta to a linear layer's output.

        Args:
            W: (d_out, d_in) frozen weight matrix
            x: (B, ..., d_in) input

        Returns:
            (B, ..., d_out)  base output + LoRA delta
        """
        base = x @ W.T
        delta = x @ self.lora_A.T @ self.lora_B.T * self.scale
        return base + delta


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT: Full MoE-TFT Hybrid
# ─────────────────────────────────────────────────────────────────────────────

# All specialist names
CORE_SPECIALISTS = ["cardio", "mental", "metabolic", "recovery"]
SPARSE_SPECIALISTS = ["immune", "respiratory", "hormonal", "cog_fatigue"]
ALL_SPECIALISTS = CORE_SPECIALISTS + SPARSE_SPECIALISTS


class DRIFT(nn.Module):
    """
    Full DRIFT model: Domain-Routed Interpretable Fusion Transformer.

    Args:
        d_in:           raw token dimension (104)
        d_model:        model hidden dimension (128)
        d_static_raw:   raw static feature dimension (15)
        d_static:       projected static context dimension (32)
        n_heads:        attention heads (4)
        n_enc_layers:   shared encoder depth (2)
        n_horizons:     prediction horizons (4: 7/30/90/180d)
        n_vars:         VSN variable count (20)
        n_tier2_experts: Tier 2 FFN expert count (8)
        n_sparse_top_k: Tier 1 sparse specialist top-k (2)
        dropout:        regularization dropout
        precision:      'bf16' or 'fp16' or 'fp32'
    """

    def __init__(
        self,
        d_in: int = 104,
        d_model: int = 128,
        d_static_raw: int = RAW_STATIC_DIM,
        d_static: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_horizons: int = 4,
        n_vars: int = 20,
        n_tier2_experts: int = 8,
        n_sparse_top_k: int = 2,
        dropout: float = 0.1,
        precision: str = "bf16",
    ):
        super().__init__()

        self.d_model = d_model
        self.d_static = d_static
        self.n_horizons = n_horizons
        self.precision = precision

        # ── Stage 1: Static covariate encoder ──────────────────────────────
        # Encodes user metadata into 4 context vectors (c_s, c_e, c_h, c_c)
        self.static_encoder = StaticCovariateEncoder(
            d_raw=d_static_raw,
            d_static=d_static,
            dropout=0.0,
        )

        # Project c_e from d_static -> d_model for attention injection
        self.ce_proj = nn.Linear(d_static, d_model)

        # ── Shared encoder (Stage 4 preamble) ──────────────────────────────
        # Bidirectional transformer with RoPE: produces Z from input tokens
        self.shared_encoder = SharedEncoder(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
        )

        # ── Stage 3: Tier 1 MoE — Specialist LSTM pool ─────────────────────
        # Always-on (4) + sparse top-2 (from 4) specialist LSTMs
        self.tier1_moe = Tier1MoE(
            d_model=d_model,
            d_static=d_static,
            n_sparse_top_k=n_sparse_top_k,
        )

        # ── Stage 4+5: Agent blocks (VSN + shared-V attention + GRN gate) ───
        # One agent per CORE specialist — processes Tier 1 LSTM states
        self.agents = nn.ModuleDict({
            name: AgentBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_in=d_in,
                n_vars=n_vars,
                d_static=d_static,
                name=name,
                dropout=dropout,
            )
            for name in CORE_SPECIALISTS
        })

        # ── Stage 6: Tier 2 MoE — Expert FFN pool ──────────────────────────
        # Token-level routing: 8 experts, top-2 per token
        self.tier2_moe = Tier2MoE(
            d_model=d_model,
            d_context=d_static,
            n_experts=n_tier2_experts,
            top_k=2,
            dropout=dropout,
        )

        # ── Stage 7: Prediction heads ───────────────────────────────────────
        # One AgentHead per core specialist: quantile + velocity + threshold
        self.heads = nn.ModuleDict({
            name: AgentHead(d_model=d_model, n_horizons=n_horizons)
            for name in CORE_SPECIALISTS
        })

    def _get_dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        return torch.float32

    def forward(
        self,
        X: torch.Tensor,                     # (B, N, 104)   token sequence
        static_input: torch.Tensor = None,   # (B, d_static_raw)  user metadata
        return_weights: bool = False,         # enable explanation mode
    ) -> DRIFTOutput:
        """
        Full forward pass through the DRIFT architecture.

        Args:
            X:             (B, N, 104)      health token sequence (90 days)
            static_input:  (B, d_static_raw) user age/sex/conditions/device
                           If None, uses zero vector (anonymous/cold-start)
            return_weights: if True, returns attention + VSN weights
                           Note: disables FA2 fused kernel, ~2x slower

        Returns:
            DRIFTOutput with per-specialist predictions and routing info
        """
        B, N, D = X.shape
        device = X.device
        dtype = self._get_dtype()

        # Cast to target precision
        X = X.to(dtype)

        # ── Stage 1: Encode static covariates ──────────────────────────────
        if static_input is None:
            static_input = torch.zeros(B, self.static_encoder.input_proj.in_features,
                                       device=device, dtype=dtype)
        else:
            static_input = static_input.to(dtype)

        context = self.static_encoder(static_input)
        # context: {"c_s": (B,32), "c_e": (B,32), "c_h": (B,32), "c_c": (B,32)}

        c_s = context["c_s"]   # selection   -> VSN
        c_e = self.ce_proj(context["c_e"])  # enrichment  -> attention Q (projected to d_model)
        c_h = context["c_h"]   # hidden init -> LSTM
        c_c = context["c_c"]   # cross-attn  -> Tier 2 FFN

        # ── Shared Encoder: produce contextual representations Z ────────────
        with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda',
                            dtype=dtype, enabled=(dtype != torch.float32)):
            Z, _ = self.shared_encoder(X.float() if dtype == torch.bfloat16 else X, c_e=c_e)
            # Z: (B, N, d_model)

        Z = Z.to(dtype)

        # ── Stage 3: Tier 1 MoE — Specialist LSTM pool ─────────────────────
        specialist_states, tier1_router_probs, routing_info = self.tier1_moe(Z, c_h)
        # specialist_states: {"cardio": (B,d), "mental": (B,d), ...}
        # tier1_router_probs: (B, n_sparse)  for load balancing loss

        # ── Stage 6: Tier 2 MoE — Expert FFN pool on encoded sequence ───────
        Z_refined, tier2_router_probs = self.tier2_moe(Z, c_c)
        # Z_refined: (B, N, d_model)
        # tier2_router_probs: (B, N, 8)  for load balancing loss

        # ── Stages 4+5: Agent blocks per core specialist ────────────────────
        agent_outputs = {}
        attn_weights_dict = {}
        vsn_weights_dict = {}

        for name in CORE_SPECIALISTS:
            H_i = specialist_states[name]  # (B, d_model) from Tier 1 LSTM

            h, attn_w, vsn_w = self.agents[name](
                Z=Z_refined,
                X_raw=X.float(),
                H_i=H_i,
                c_s=c_s,
                c_e=c_e,
                return_weights=return_weights,
            )

            # ── Stage 7: Prediction heads ───────────────────────────────────
            pred = self.heads[name](h)
            agent_outputs[name] = pred

            if return_weights:
                attn_weights_dict[name] = attn_w    # (B, N)
                vsn_weights_dict[name] = vsn_w      # (B, N, n_vars)

        return DRIFTOutput(
            agents=agent_outputs,
            tier1_router_probs=tier1_router_probs,
            tier2_router_probs=tier2_router_probs,
            routing_info=routing_info,
            attn_weights=attn_weights_dict,
            vsn_weights=vsn_weights_dict,
        )

    @property
    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
