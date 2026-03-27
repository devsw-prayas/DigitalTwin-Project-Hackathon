"""
training_8.py — Divergence-Aware 8-Agent Training

Implements comprehensive training with:
1. Multi-agent quantile loss
2. Orthogonality loss (force diverse specialist outputs)
3. Routing supervision loss (guide router to correct experts)
4. VSN divergence loss (force different attention patterns)
5. Entropy regularization (encourage confident routing)
6. Load balancing (prevent expert collapse)

Loss = L_quantile + α_orth * L_orth + α_route * L_route + α_vsn * L_vsn + α_ent * L_ent + α_lb * L_lb

Key innovations:
- Persona-aware routing supervision
- Dynamic loss weighting based on divergence score
- Early stopping on orthogonality metrics
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .model_8 import DRIFT8, DRIFT8Output
from .agents_8 import ALL_AGENTS, N_AGENTS
from .heads import QUANTILES, N_HORIZONS


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Horizon weights: down-weight far horizons
HORIZON_WEIGHTS = torch.tensor([1.0, 0.8, 0.6, 0.4])


def pinball_loss(
    y: torch.Tensor,        # (B, N_HORIZONS)
    q_hat: torch.Tensor,    # (B, N_HORIZONS, 3)
    taus: List[float] = QUANTILES,
) -> torch.Tensor:
    """Quantile (pinball) loss."""
    device = y.device
    horizon_weights = HORIZON_WEIGHTS.to(device)
    
    total = 0.0
    for i, tau in enumerate(taus):
        q = q_hat[..., i]
        diff = y - q
        loss = torch.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
        loss = (loss * horizon_weights.unsqueeze(0)).mean()
        total = total + loss
    
    return total


def multi_agent_quantile_loss(
    outputs: DRIFT8Output,
    y: torch.Tensor,           # (B, N_AGENTS, N_HORIZONS)
    agent_mask: Optional[torch.Tensor] = None,  # (B, N_AGENTS) - which agents to train
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Quantile loss across all 8 agents.
    
    Args:
        outputs: Model output with all agent predictions
        y: Ground truth values for all agents
        agent_mask: Optional mask for which agents to include in loss
    
    Returns:
        total_loss: scalar
        per_agent_losses: dict of individual agent losses
    """
    device = y.device
    total_loss = torch.tensor(0.0, device=device)
    per_agent = {}
    
    for i, agent_name in enumerate(ALL_AGENTS):
        if agent_name not in outputs.agents:
            continue
        
        if agent_mask is not None:
            # Only compute loss for masked agents
            if not agent_mask[:, i].any():
                continue
        
        pred = outputs.agents[agent_name]
        y_agent = y[:, i, :].float()
        
        loss = pinball_loss(y_agent, pred.quantiles.float())
        
        per_agent[agent_name] = loss.item()
        total_loss = total_loss + loss
    
    return total_loss, per_agent


def orthogonality_loss(specialist_states: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Force specialist outputs to be orthogonal.
    
    Orthogonal representations = each specialist captures different information.
    """
    states = []
    for name in ALL_AGENTS:
        if name in specialist_states:
            states.append(specialist_states[name])
    
    if len(states) < 2:
        return torch.tensor(0.0, device=states[0].device if states else 'cuda')
    
    # Stack: (n_specialists, B, d_model)
    stacked = torch.stack(states, dim=0)
    
    # Mean over batch: (n_specialists, d_model)
    means = stacked.mean(dim=1)
    
    # Normalize
    normalized = F.normalize(means, dim=-1)
    
    # Similarity matrix
    sim = normalized @ normalized.T
    
    # Penalize off-diagonal similarity
    n = sim.size(0)
    mask = 1 - torch.eye(n, device=sim.device)
    
    return (sim * mask).pow(2).sum()


def vsn_divergence_loss(
    vsn_weights: Dict[str, torch.Tensor],
    target_agents: List[str] = None,
) -> torch.Tensor:
    """
    Force VSN attention patterns to be different between agents.
    
    If cardio and mental attend to the same features, they're not specializing.
    """
    if target_agents is None:
        target_agents = ["cardio", "mental"]  # Start with these two
    
    weights = []
    for name in target_agents:
        if name in vsn_weights and vsn_weights[name] is not None:
            w = vsn_weights[name].float()  # (B, N, n_vars)
            weights.append(w.mean(dim=1))  # (B, n_vars) - avg over sequence
    
    if len(weights) < 2:
        return torch.tensor(0.0, device=weights[0].device if weights else 'cuda')
    
    # Normalize to probability distributions
    probs = [F.softmax(w, dim=-1) for w in weights]
    
    # Compute pairwise JS divergence
    total_jsd = torch.tensor(0.0, device=probs[0].device)
    n_pairs = 0
    
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            p = probs[i]
            q = probs[j]
            
            # JS divergence
            eps = 1e-8
            p = p.clamp(min=eps)
            q = q.clamp(min=eps)
            m = 0.5 * (p + q)
            
            kl_pm = (p * (p / m).log()).sum(dim=-1)
            kl_qm = (q * (q / m).log()).sum(dim=-1)
            jsd = 0.5 * (kl_pm + kl_qm)
            
            # We want HIGH JSD = different attention patterns
            # So we penalize LOW JSD
            total_jsd = total_jsd + torch.relu(0.05 - jsd.mean())  # Target JSD > 0.05
            n_pairs += 1
    
    return total_jsd / max(1, n_pairs)


def routing_entropy_loss(
    router_probs: torch.Tensor,
    target: str = "confident",  # "confident" or "uniform"
) -> torch.Tensor:
    """
    Control routing entropy.
    
    "confident": encourage low entropy (router should be decisive)
    "uniform": encourage high entropy (load balancing)
    """
    # Entropy per sample
    entropy = -(router_probs * router_probs.clamp(1e-8).log()).sum(dim=-1)
    
    if target == "confident":
        # We want LOW entropy
        # Penalize entropy > threshold
        threshold = 0.5 * math.log(router_probs.size(-1))
        return torch.relu(entropy - threshold).mean()
    else:
        # We want HIGH entropy (uniform)
        max_entropy = math.log(router_probs.size(-1))
        return (max_entropy - entropy).mean()


def load_balancing_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """
    Prevent router collapse by encouraging uniform expert usage.
    
    L_lb = n * sum(f_i * p_i)
    
    where f_i = fraction of samples routed to expert i
          p_i = mean routing probability for expert i
    """
    if router_probs.dim() == 3:
        probs_flat = router_probs.reshape(-1, router_probs.size(-1))
    else:
        probs_flat = router_probs
    
    n = probs_flat.size(-1)
    
    # Fraction routed to each expert
    top_idx = probs_flat.argmax(dim=-1)
    f = torch.zeros(n, device=probs_flat.device)
    for i in range(n):
        f[i] = (top_idx == i).float().mean()
    
    # Mean probability per expert
    p = probs_flat.mean(dim=0)
    
    return n * (f * p).sum()


def routing_supervision_loss(
    router_probs: torch.Tensor,
    routing_targets: torch.Tensor,
    expert_mapping: Dict[int, int] = None,  # agent_idx -> expert_idx
) -> torch.Tensor:
    """
    Supervised routing: guide router based on persona type.
    
    Args:
        router_probs: (B, n_experts)
        routing_targets: (B,) agent indices that should be primary
        expert_mapping: which expert handles which agent
    """
    B = router_probs.size(0)
    n_experts = router_probs.size(-1)
    
    # Default: expert i handles agent i
    if expert_mapping is None:
        expert_mapping = {i: i % n_experts for i in range(N_AGENTS)}
    
    # Convert agent targets to expert targets
    expert_targets = torch.zeros(B, dtype=torch.long, device=router_probs.device)
    for i in range(B):
        agent_idx = routing_targets[i].item()
        expert_targets[i] = expert_mapping.get(agent_idx, agent_idx % n_experts)
    
    # One-hot targets
    targets = torch.zeros_like(router_probs)
    targets.scatter_(1, expert_targets.unsqueeze(1), 1.0)
    
    # KL divergence
    log_probs = router_probs.clamp(1e-8).log()
    return F.kl_div(log_probs, targets, reduction='batchmean')


# ═══════════════════════════════════════════════════════════════════════════════
# TOTAL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LossWeights:
    """Weights for different loss components."""
    quantile: float = 1.0
    orthogonality: float = 0.1
    vsn_divergence: float = 0.05
    routing_entropy: float = 0.01
    load_balance: float = 0.01
    routing_supervision: float = 0.1


@dataclass
class LossBreakdown:
    """Detailed loss breakdown for logging."""
    total: float
    quantile: float
    orthogonality: float
    vsn_divergence: float
    routing_entropy: float
    load_balance_t1: float
    load_balance_t2: float
    routing_supervision: float
    per_agent: Dict[str, float]


def compute_total_loss_8(
    outputs: DRIFT8Output,
    y: torch.Tensor,
    weights: LossWeights = None,
    routing_targets: Optional[torch.Tensor] = None,
    agent_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, LossBreakdown]:
    """
    Compute total training loss for 8-agent model.
    
    Args:
        outputs: Model outputs
        y: Ground truth (B, N_AGENTS, N_HORIZONS)
        weights: Loss component weights
        routing_targets: Optional supervised routing targets (B,)
        agent_mask: Optional mask for which agents to train
    
    Returns:
        total_loss: scalar tensor
        breakdown: detailed loss breakdown
    """
    if weights is None:
        weights = LossWeights()
    
    device = y.device
    
    # 1. Multi-agent quantile loss
    loss_quantile, per_agent = multi_agent_quantile_loss(outputs, y, agent_mask)
    
    # 2. Orthogonality loss
    loss_orth = orthogonality_loss(outputs.specialist_states)
    
    # 3. VSN divergence loss
    loss_vsn = vsn_divergence_loss(outputs.vsn_weights)
    
    # 4. Routing entropy loss (encourage confident routing)
    loss_ent_t1 = routing_entropy_loss(outputs.tier1_router_probs, target="confident")
    loss_ent_t2 = routing_entropy_loss(outputs.tier2_router_probs, target="confident")
    loss_ent = loss_ent_t1 + loss_ent_t2
    
    # 5. Load balancing loss (prevent collapse)
    loss_lb_t1 = load_balancing_loss(outputs.tier1_router_probs)
    loss_lb_t2 = load_balancing_loss(outputs.tier2_router_probs)
    
    # 6. Routing supervision (if targets provided)
    loss_route = torch.tensor(0.0, device=device)
    if routing_targets is not None:
        loss_route = routing_supervision_loss(
            outputs.tier1_router_probs,
            routing_targets
        )
    
    # Total
    total = (
        weights.quantile * loss_quantile +
        weights.orthogonality * loss_orth +
        weights.vsn_divergence * loss_vsn +
        weights.routing_entropy * loss_ent +
        weights.load_balance * (loss_lb_t1 + loss_lb_t2) +
        weights.routing_supervision * loss_route
    )
    
    breakdown = LossBreakdown(
        total=total.item(),
        quantile=loss_quantile.item(),
        orthogonality=loss_orth.item(),
        vsn_divergence=loss_vsn.item(),
        routing_entropy=loss_ent.item(),
        load_balance_t1=loss_lb_t1.item(),
        load_balance_t2=loss_lb_t2.item(),
        routing_supervision=loss_route.item(),
        per_agent=per_agent,
    )
    
    return total, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer8:
    """
    8-Agent Divergence-Aware Trainer.
    
    Features:
    - Dynamic loss weighting based on training phase
    - Orthogonality monitoring
    - Router supervision for personas with clear primary agents
    - Early stopping on collapse detection
    """
    
    def __init__(
        self,
        model: DRIFT8,
        device: str = "cuda",
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        total_epochs: int = 150,
        warmup_steps: int = 1000,
        checkpoint_dir: str = "checkpoints",
        precision: str = "bf16",
    ):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.checkpoint_dir = checkpoint_dir
        self.precision = precision
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # AMP
        self.use_amp = precision in ("bf16", "fp16") and device == "cuda"
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Scheduler (set up after knowing steps_per_epoch)
        self.scheduler = None
        self.warmup_steps = warmup_steps
        
        # Tracking
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.collapse_history = []
    
    def _build_scheduler(self, steps_per_epoch: int):
        """Build LR scheduler."""
        total_steps = self.total_epochs * steps_per_epoch
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - self.warmup_steps),
            eta_min=self.optimizer.param_groups[0]["lr"] * 0.01,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )
    
    def _get_loss_weights(self, epoch: int, orth_history: List[float]) -> LossWeights:
        """Dynamic loss weighting based on training phase."""
        
        # Phase 1 (0-30): Focus on learning the task
        if epoch < 30:
            return LossWeights(
                quantile=1.0,
                orthogonality=0.05,
                vsn_divergence=0.02,
                routing_entropy=0.005,
                load_balance=0.01,
                routing_supervision=0.05,
            )
        
        # Phase 2 (30-80): Increase specialization pressure
        elif epoch < 80:
            # Increase orthogonality if specialists are too similar
            recent_orth = np.mean(orth_history[-10:]) if orth_history else 0.5
            
            orth_weight = 0.1
            if recent_orth > 0.8:  # High similarity = need more pressure
                orth_weight = 0.2
            elif recent_orth < 0.3:  # Already diverse, can reduce
                orth_weight = 0.05
            
            return LossWeights(
                quantile=1.0,
                orthogonality=orth_weight,
                vsn_divergence=0.08,
                routing_entropy=0.01,
                load_balance=0.01,
                routing_supervision=0.1,
            )
        
        # Phase 3 (80+): Fine-tune with strong supervision
        else:
            return LossWeights(
                quantile=1.0,
                orthogonality=0.15,
                vsn_divergence=0.1,
                routing_entropy=0.01,
                load_balance=0.005,
                routing_supervision=0.15,
            )
    
    def train_epoch(
        self,
        loader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_losses = defaultdict(float)
        n_batches = 0
        
        for xb, yb, routing_targets in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            routing_targets = routing_targets.to(self.device)
            
            with torch.autocast(
                device_type=self.device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    xb, 
                    return_weights=True,
                    routing_targets=routing_targets,
                )
                
                weights = self._get_loss_weights(epoch, self.collapse_history)
                loss, breakdown = compute_total_loss_8(
                    outputs, yb, weights, routing_targets
                )
            
            self.optimizer.zero_grad()
            
            if self.precision == "fp16":
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.global_step += 1
            
            # Track orthogonality for dynamic weighting
            self.collapse_history.append(breakdown.orthogonality)
            if len(self.collapse_history) > 100:
                self.collapse_history.pop(0)
            
            # Accumulate losses
            total_losses["total"] += breakdown.total
            total_losses["quantile"] += breakdown.quantile
            total_losses["orthogonality"] += breakdown.orthogonality
            total_losses["vsn_divergence"] += breakdown.vsn_divergence
            total_losses["load_balance"] += breakdown.load_balance_t1 + breakdown.load_balance_t2
            n_batches += 1
        
        return {k: v / max(1, n_batches) for k, v in total_losses.items()}
    
    @torch.no_grad()
    def validate(self, loader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_losses = defaultdict(float)
        n_batches = 0
        
        for xb, yb, routing_targets in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            routing_targets = routing_targets.to(self.device)
            
            with torch.autocast(
                device_type=self.device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(xb, return_weights=True)
                loss, breakdown = compute_total_loss_8(outputs, yb, routing_targets=routing_targets)
            
            total_losses["total"] += breakdown.total
            total_losses["quantile"] += breakdown.quantile
            total_losses["orthogonality"] += breakdown.orthogonality
            n_batches += 1
        
        return {k: v / max(1, n_batches) for k, v in total_losses.items()}
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
        }
        
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
        torch.save(state, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            print(f"  [checkpoint] New best: val_loss={val_loss:.4f}")
    
    def train(
        self,
        train_loader,
        val_loader,
        resume_from: Optional[str] = None,
    ) -> Dict:
        """Full training run."""
        
        start_epoch = 0
        if resume_from:
            state = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            if self.scheduler and "scheduler_state" in state:
                self.scheduler.load_state_dict(state["scheduler_state"])
            start_epoch = state["epoch"] + 1
            print(f"[resume] From epoch {state['epoch']}, val_loss={state['val_loss']:.4f}")
        
        # Build scheduler
        if self.scheduler is None:
            self._build_scheduler(len(train_loader))
        
        print(f"[trainer] Starting: {self.total_epochs} epochs | {self.device} | {self.precision}")
        print(f"[trainer] Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        history = {"train": [], "val": []}
        
        for epoch in range(start_epoch, self.total_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Check for improvement
            val_loss = val_losses["total"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | "
                f"train={train_losses['total']:.4f} "
                f"(q={train_losses['quantile']:.4f} "
                f"orth={train_losses['orthogonality']:.4f} "
                f"lb={train_losses['load_balance']:.4f}) | "
                f"val={val_loss:.4f} | lr={lr:.2e}"
            )
            
            history["train"].append(train_losses)
            history["val"].append(val_losses)
        
        print(f"\n[trainer] Done. Best val_loss={self.best_val_loss:.4f}")
        return history
