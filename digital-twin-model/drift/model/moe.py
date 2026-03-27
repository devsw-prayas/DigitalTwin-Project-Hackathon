"""
moe.py — Mixture of Experts: Tier 1 (Specialist LSTMs) + Tier 2 (Expert FFNs)

This is the core research contribution of the DRIFT architecture.
Two tiers of MoE routing, each at a different level of the temporal hierarchy.

═══════════════════════════════════════════════════════════════════════════════
TIER 1 — SPECIALIST LSTM POOL (sequence-level routing)
═══════════════════════════════════════════════════════════════════════════════

From the spec (Section 5, Stage 3):
    Router: gate(Z_summary) -> softmax -> top-k selection
    Z_summary = mean(projected_tokens) over sequence
    Routing is FIXED for the entire forward pass (one decision per sequence)

Why sequence-level routing for LSTMs?
    LSTMs process sequences step-by-step, maintaining state across timesteps.
    If you routed per-token, different timesteps of the same sequence would go
    to different LSTMs — breaking the internal state coherence.
    Routing once per sequence keeps each specialist LSTM coherent.

Specialists:
    Always-on (core): CardioLSTM, MentalLSTM, MetabolicLSTM, RecoveryLSTM
    Sparse (top-k=2): ImmuneLSTM, RespiratoryLSTM, HormonalLSTM, CognitiveFatigueLSTM

Each specialist LSTM:
    - Encoder: processes past 90 tokens, initialized from c_h
    - Decoder: generates future K tokens, cross-attends to encoder output
    - Output: H_i — specialist latent state (d_model,)

═══════════════════════════════════════════════════════════════════════════════
TIER 2 — EXPERT FFN POOL (token-level routing)
═══════════════════════════════════════════════════════════════════════════════

From the spec (Section 5, Stage 6):
    Router: gate(z_t) -> softmax -> top-2 experts
    Expert_i: GRN(z_t, c_c)   # GRN not plain linear
    Output: sum_i [ weight_i * Expert_i(z_t) ] for selected experts

Why token-level routing for FFNs?
    The FFN operates on enriched representations after attention. At this stage
    the sequence dimension has been compressed — we're processing individual
    enriched tokens. Different tokens (days) may benefit from different
    computational paths depending on their context.

8 expert GRN-style FFNs, top-2 activated per token.
Conditioned on static context vector c_c.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.grn import GRN


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1: Specialist LSTM
# ─────────────────────────────────────────────────────────────────────────────

class SpecialistLSTM(nn.Module):
    """
    One specialist LSTM encoder-decoder.

    Each specialist captures a distinct health dynamic:
        CardioLSTM:         slow accumulation, weeks-to-months timescale
        MentalLSTM:         rapid volatility, days timescale
        MetabolicLSTM:      medium-term drift, weeks timescale
        RecoveryLSTM:       cyclical load/recovery, bi-weekly timescale
        ImmuneLSTM:         illness events, AQI spikes, HRV suppression
        RespiratoryLSTM:    sustained AQI exposure, SpO2 depression
        HormonalLSTM:       menstrual phase, cyclical volatility
        CognitiveFatigueLSTM: screen time, sleep debt, workload

    Args:
        d_model:    input/output dimension
        d_static:   static context dimension (for c_h LSTM init)
        name:       specialist name (for interpretability / logging)
        n_layers:   LSTM depth (1 for MVP/stability, 2 for v2)
    """

    def __init__(self, d_model: int, d_static: int = 32, name: str = "specialist", n_layers: int = 1):
        super().__init__()
        self.name = name
        self.d_model = d_model

        # Encoder LSTM: processes past N tokens
        self.encoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
        )

        # Decoder LSTM: generates future K step representations
        # Cross-attends to encoder output to incorporate past context
        self.decoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
        )

        # Cross-attention: decoder queries encoder output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True,
        )

        # Project c_h (static context) to LSTM h0/c0 initialization
        # Allows static user info to shape the specialist's initial hidden state
        self.h0_proj = nn.Linear(d_static, d_model)
        self.c0_proj = nn.Linear(d_static, d_model)

        # Pool encoder outputs -> single latent state H_i
        self.output_pool = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def _init_hidden(self, c_h: torch.Tensor, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from static context vector c_h.

        Args:
            c_h: (B, d_static)

        Returns:
            h0: (n_layers, B, d_model)
            c0: (n_layers, B, d_model)
        """
        h0 = self.h0_proj(c_h).unsqueeze(0).repeat(n_layers, 1, 1)  # (n_layers, B, d_model)
        c0 = self.c0_proj(c_h).unsqueeze(0).repeat(n_layers, 1, 1)
        return h0, c0

    def forward(
        self,
        Z: torch.Tensor,        # (B, N, d_model) — past token sequence (from VSN)
        c_h: torch.Tensor,      # (B, d_static)   — LSTM init context
        n_future: int = 4,      # number of future steps to decode
    ) -> torch.Tensor:
        """
        Args:
            Z:        (B, N, d_model)  encoded past sequence
            c_h:      (B, d_static)    static hidden init context
            n_future: number of future decoder steps

        Returns:
            H_i: (B, d_model)  specialist latent state
        """
        B, N, D = Z.shape
        n_layers = self.encoder_lstm.num_layers

        # Initialize hidden from static context
        h0, c0 = self._init_hidden(c_h, n_layers)

        # Encode past sequence
        enc_out, (h_n, c_n) = self.encoder_lstm(Z, (h0, c0))
        # enc_out: (B, N, d_model), h_n: (n_layers, B, d_model)

        # Decode future steps using encoder's final state
        # Decoder input: repeat last encoded token n_future times
        dec_input = enc_out[:, -1:, :].expand(B, n_future, D)  # (B, n_future, d_model)
        dec_out, _ = self.decoder_lstm(dec_input, (h_n, c_n))
        # dec_out: (B, n_future, d_model)

        # Cross-attention: decoder attends to encoder output
        # Decoder queries what it needs from the past
        dec_enriched, _ = self.cross_attn(
            query=dec_out,    # (B, n_future, d_model)
            key=enc_out,      # (B, N, d_model)
            value=enc_out,    # (B, N, d_model)
        )

        # Pool: mean of decoder outputs -> single specialist state H_i
        H_i = self.output_norm(self.output_pool(dec_enriched.mean(dim=1)))  # (B, d_model)

        return H_i


class Tier1MoE(nn.Module):
    """
    Tier 1 MoE: Specialist LSTM Pool with sequence-level routing.

    Always-on specialists (4) run every forward pass.
    Sparse specialists (4) are gated by the router: top-k=2 activate.

    Router: gate(Z_summary) -> softmax over sparse specialists -> top-k
    Z_summary = mean over sequence of a projected summary.

    Args:
        d_model:          model dimension
        d_static:         static context dimension
        n_sparse_top_k:   how many sparse specialists activate (default: 2)
        n_future:         future decoder steps per specialist
        dropout:          router dropout
    """

    ALWAYS_ON = ["cardio", "mental", "metabolic", "recovery"]
    SPARSE    = ["immune", "respiratory", "hormonal", "cog_fatigue"]

    def __init__(
        self,
        d_model: int,
        d_static: int = 32,
        n_sparse_top_k: int = 2,
        n_future: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sparse_top_k = n_sparse_top_k

        # Always-on specialist LSTMs
        self.core_specialists = nn.ModuleDict({
            name: SpecialistLSTM(d_model, d_static, name=name)
            for name in self.ALWAYS_ON
        })

        # Sparse specialist LSTMs
        self.sparse_specialists = nn.ModuleDict({
            name: SpecialistLSTM(d_model, d_static, name=name)
            for name in self.SPARSE
        })

        # Sequence-level router: maps sequence summary -> routing weights over sparse experts
        self.router_proj = nn.Linear(d_model, d_model)
        self.router_gate = nn.Linear(d_model, len(self.SPARSE))
        self.router_dropout = nn.Dropout(dropout)

    def _route(self, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights over sparse specialists.

        Args:
            Z: (B, N, d_model) — full sequence

        Returns:
            top_indices: (B, n_sparse_top_k)  which specialists activate
            top_weights: (B, n_sparse_top_k)  their routing weights (softmax)
        """
        # Summarize sequence: mean pooling -> router
        Z_summary = Z.mean(dim=1)                        # (B, d_model)
        Z_summary = self.router_proj(Z_summary)           # (B, d_model)
        Z_summary = self.router_dropout(Z_summary)

        logits = self.router_gate(Z_summary)              # (B, n_sparse)
        probs = F.softmax(logits, dim=-1)                 # (B, n_sparse)

        # Top-k selection
        top_weights, top_indices = torch.topk(probs, self.n_sparse_top_k, dim=-1)

        # Re-normalize selected weights so they sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return top_indices, top_weights, probs  # return full probs for load balancing loss

    def forward(
        self,
        Z: torch.Tensor,    # (B, N, d_model)
        c_h: torch.Tensor,  # (B, d_static)
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
        """
        Args:
            Z:   (B, N, d_model)
            c_h: (B, d_static)

        Returns:
            specialist_states: dict { specialist_name: (B, d_model) }
            router_probs:      (B, n_sparse)  for load balancing loss
            routing_info:      dict with routing metadata for interpretability
        """
        specialist_states = {}

        # --- Always-on: run all 4 core specialists ---
        for name, specialist in self.core_specialists.items():
            specialist_states[name] = specialist(Z, c_h)

        # --- Sparse: route and run top-k ---
        top_indices, top_weights, router_probs = self._route(Z)
        B = Z.shape[0]

        sparse_names = self.SPARSE
        routing_info = {"top_indices": top_indices, "top_weights": top_weights}

        # We need to run each sparse specialist only for the batches that selected it
        # For simplicity in training, run all sparse and mask by router weight
        # (In production, you'd only run the selected ones for efficiency)
        all_sparse_outputs = {}
        for name, specialist in self.sparse_specialists.items():
            all_sparse_outputs[name] = specialist(Z, c_h)  # (B, d_model)

        # Accumulate weighted sparse specialist outputs
        sparse_combined = torch.zeros(B, self.d_model, device=Z.device, dtype=Z.dtype)
        for i, name in enumerate(sparse_names):
            # Get the routing weight for this specialist across the batch
            # top_indices: (B, n_top_k), top_weights: (B, n_top_k)
            is_selected = (top_indices == i).any(dim=-1).float()   # (B,)
            weight_for_specialist = torch.zeros(B, device=Z.device, dtype=Z.dtype)
            for k in range(self.n_sparse_top_k):
                mask = (top_indices[:, k] == i).float()
                weight_for_specialist += mask * top_weights[:, k]
            sparse_combined += weight_for_specialist.unsqueeze(-1) * all_sparse_outputs[name]

        # Add combined sparse output as a pseudo-specialist
        specialist_states["sparse_combined"] = sparse_combined

        # Also expose individual sparse states for interpretability
        for name, h in all_sparse_outputs.items():
            specialist_states[f"sparse_{name}"] = h

        return specialist_states, router_probs, routing_info


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2: Expert FFN Pool
# ─────────────────────────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    """
    One expert in the Tier 2 FFN pool.

    Uses a GRN (not plain linear) to maintain gating behavior.
    Conditioned on static context c_c.

    From the spec:
        Expert_i: GRN(z_t, c_c)
    """

    def __init__(self, d_model: int, d_context: int = 32, d_hidden: int = None, dropout: float = 0.1):
        super().__init__()
        self.grn = GRN(d_model, d_context=d_context, d_hidden=d_hidden or d_model * 2, dropout=dropout)

    def forward(self, z: torch.Tensor, c_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:   (B, ..., d_model)  enriched token representation
            c_c: (B, d_context)     static cross-attn conditioning

        Returns:
            (B, ..., d_model)
        """
        return self.grn(z, c_c)


class Tier2MoE(nn.Module):
    """
    Tier 2 MoE: Expert FFN Pool with token-level routing.

    8 expert GRN-style FFNs, top-2 activated per token.
    Conditioned on static context vector c_c.

    From the spec (Section 5, Stage 6):
        Router: gate(z_t) -> softmax -> top-2 experts
        Output: sum_i [ weight_i * Expert_i(z_t) ] for selected experts

    Args:
        d_model:    model dimension
        d_context:  static context dimension (c_c)
        n_experts:  total number of expert FFNs (default: 8)
        top_k:      how many experts activate per token (default: 2)
        dropout:    expert and router dropout
    """

    def __init__(
        self,
        d_model: int,
        d_context: int = 32,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # 8 expert GRN FFNs
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_context=d_context, dropout=dropout)
            for _ in range(n_experts)
        ])

        # Token-level router: per-token routing decision
        self.router = nn.Linear(d_model, n_experts)
        self.router_dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,    # (B, N, d_model) — sequence of enriched tokens
        c_c: torch.Tensor,  # (B, d_context)  — static conditioning
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z:   (B, N, d_model)
            c_c: (B, d_context)

        Returns:
            out:          (B, N, d_model)  expert-processed representations
            router_probs: (B, N, n_experts) for load balancing loss
        """
        B, N, D = z.shape

        # Token-level routing: each of the N tokens independently selects top-k experts
        router_logits = self.router(self.router_dropout(z))   # (B, N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)        # (B, N, n_experts)

        # Top-k selection per token
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_weights: (B, N, top_k),  top_indices: (B, N, top_k)

        # Re-normalize
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Expand c_c to (B, N, d_context) for per-token expert conditioning
        c_c_exp = c_c.unsqueeze(1).expand(B, N, -1)

        # Compute output: weighted sum of selected expert outputs
        out = torch.zeros_like(z)

        for i in range(self.n_experts):
            # Which (batch, token) positions selected expert i?
            is_selected = (top_indices == i)  # (B, N, top_k) bool

            if not is_selected.any():
                continue  # skip experts not selected by anyone this batch

            # Compute expert output for all tokens (efficient: can be gated in prod)
            expert_out = self.experts[i](z, c_c_exp)  # (B, N, d_model)

            # Aggregate weights for this expert: sum over top_k positions
            # For each (batch, token), the weight is the routing weight if selected, else 0
            weight = (is_selected.float() * top_weights).sum(dim=-1)  # (B, N)
            out = out + weight.unsqueeze(-1) * expert_out

        return out, router_probs
