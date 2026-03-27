"""
model_8.py — 8-Agent DRIFT Model

Extends the base DRIFT model to output predictions for all 8 agents:
- cardio, mental, metabolic, recovery (always-on core specialists)
- immune, respiratory, hormonal, cog_fatigue (sparse specialists)

Key changes from base model:
1. All 8 specialists have prediction heads
2. Stronger orthogonality enforcement
3. Routing supervision inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .model import DRIFT, DRIFTOutput, CORE_SPECIALISTS, SPARSE_SPECIALISTS
from .heads import AgentHead, AgentPrediction
from ..data.agents_8 import ALL_AGENTS, N_AGENTS


@dataclass
class DRIFT8Output:
    """Output from 8-agent DRIFT model."""
    agents: Dict[str, AgentPrediction]      # All 8 agents
    tier1_router_probs: torch.Tensor        # (B, n_sparse)
    tier2_router_probs: torch.Tensor        # (B, N, 8)
    routing_info: dict
    specialist_states: Dict[str, torch.Tensor]  # All 8 specialist hidden states
    attn_weights: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    vsn_weights: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    
    def get_agent_predictions(self) -> torch.Tensor:
        """Get all agent predictions as tensor (B, N_AGENTS, N_HORIZONS, 3)."""
        batch_size = list(self.agents.values())[0].quantiles.size(0)
        n_horizons = list(self.agents.values())[0].quantiles.size(1)
        
        preds = torch.zeros(batch_size, N_AGENTS, n_horizons, 3)
        for i, name in enumerate(ALL_AGENTS):
            if name in self.agents:
                preds[:, i] = self.agents[name].quantiles
        return preds


class DRIFT8(DRIFT):
    """
    Extended DRIFT model with 8-agent output.
    
    Inherits from base DRIFT and adds:
    1. Prediction heads for sparse specialists (immune, respiratory, hormonal, cog_fatigue)
    2. Orthogonality regularization
    3. Routing supervision support
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add prediction heads for sparse specialists
        self.sparse_heads = nn.ModuleDict({
            name: AgentHead(d_model=self.d_model, n_horizons=self.n_horizons)
            for name in SPARSE_SPECIALISTS
        })
        
        # Agent embedding for routing supervision
        self.agent_embeddings = nn.Parameter(
            torch.randn(N_AGENTS, self.d_model) * 0.02
        )
        
    def forward(
        self,
        X: torch.Tensor,
        static_input: torch.Tensor = None,
        return_weights: bool = False,
        routing_targets: Optional[torch.Tensor] = None,  # (B,) index of primary agent
    ) -> DRIFT8Output:
        """
        Forward pass returning all 8 agent predictions.
        
        Args:
            X: (B, N, 104) token sequence
            static_input: (B, d_static_raw) user metadata
            return_weights: if True, return attention and VSN weights
            routing_targets: (B,) indices of which agent should be primary for routing supervision
        
        Returns:
            DRIFT8Output with all 8 agent predictions
        """
        B, N, D = X.shape
        device = X.device
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        
        # Cast input
        X = X.to(dtype)
        
        # Get base model outputs
        base_output = super().forward(X, static_input, return_weights)
        
        # Get specialist states from Tier 1 MoE
        specialist_states, tier1_router_probs, routing_info = self.tier1_moe(
            base_output.routing_info.get('Z', X),  # Use encoded sequence
            base_output.routing_info.get('c_h', torch.zeros(B, self.d_static, device=device))
        )
        
        # Collect all agent predictions
        all_agents = {}
        
        # Core specialists (from base model)
        for name in CORE_SPECIALISTS:
            if name in base_output.agents:
                all_agents[name] = base_output.agents[name]
        
        # Sparse specialists (need to compute predictions)
        for name in SPARSE_SPECIALISTS:
            if f"sparse_{name}" in specialist_states:
                H_i = specialist_states[f"sparse_{name}"]
            elif name in specialist_states:
                H_i = specialist_states[name]
            else:
                # Use combined sparse output
                H_i = specialist_states.get("sparse_combined", torch.zeros(B, self.d_model, device=device, dtype=dtype))
            
            # Get context for this agent
            c_c = torch.zeros(B, self.d_static, device=device, dtype=dtype)
            
            # Compute prediction
            pred = self.sparse_heads[name](H_i)
            all_agents[name] = pred
        
        return DRIFT8Output(
            agents=all_agents,
            tier1_router_probs=tier1_router_probs,
            tier2_router_probs=base_output.tier2_router_probs,
            routing_info=routing_info,
            specialist_states=specialist_states,
            attn_weights=base_output.attn_weights,
            vsn_weights=base_output.vsn_weights,
        )
    
    def compute_orthogonality_loss(self, specialist_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute orthogonality loss between specialist outputs.
        
        Forces specialists to produce diverse representations.
        """
        # Get all specialist outputs
        states = []
        names = []
        for name in ALL_AGENTS:
            if name in specialist_states:
                states.append(specialist_states[name])
                names.append(name)
        
        if len(states) < 2:
            return torch.tensor(0.0, device=states[0].device if states else 'cuda')
        
        # Stack and compute similarity
        stacked = torch.stack(states, dim=0)  # (n_specialists, B, d_model)
        
        # Mean over batch
        means = stacked.mean(dim=1)  # (n_specialists, d_model)
        
        # Normalize
        normalized = F.normalize(means, dim=-1)  # (n_specialists, d_model)
        
        # Similarity matrix
        sim_matrix = normalized @ normalized.T  # (n_specialists, n_specialists)
        
        # Penalize off-diagonal elements
        n = sim_matrix.size(0)
        identity = torch.eye(n, device=sim_matrix.device)
        
        # Want diagonal = 1, off-diagonal = 0
        off_diag_mask = 1 - identity
        off_diag_loss = (sim_matrix * off_diag_mask).pow(2).sum()
        
        return off_diag_loss
    
    def compute_routing_supervision_loss(
        self,
        router_probs: torch.Tensor,
        routing_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised routing: encourage router to select the correct expert.
        
        Args:
            router_probs: (B, n_sparse) routing probabilities
            routing_targets: (B,) which sparse expert should be selected
        """
        # Create target distribution
        B = router_probs.size(0)
        n_experts = router_probs.size(-1)
        
        # One-hot targets
        targets = torch.zeros_like(router_probs)
        targets.scatter_(1, routing_targets.unsqueeze(1), 1.0)
        
        # KL divergence
        log_probs = router_probs.clamp(1e-8).log()
        loss = F.kl_div(log_probs, targets, reduction='batchmean')
        
        return loss


# Convenience function
def build_drift_8(
    d_model: int = 128,
    d_static: int = 32,
    n_enc_layers: int = 2,
    n_heads: int = 4,
    precision: str = "bf16",
) -> DRIFT8:
    """Build 8-agent DRIFT model with sensible defaults."""
    return DRIFT8(
        d_in=104,
        d_model=d_model,
        d_static=d_static,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        n_horizons=4,
        n_vars=20,
        n_tier2_experts=8,
        n_sparse_top_k=2,
        dropout=0.1,
        precision=precision,
    )
