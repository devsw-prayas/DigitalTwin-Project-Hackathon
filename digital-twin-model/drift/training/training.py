"""
training.py — Full Training Loop with 5-Phase Curriculum

Implements the complete training objective from the spec:

    L = L_quantile + 0.1*L_vel + 0.01*L_lb1 + 0.01*L_lb2

Loss components (from spec §5.4):

    L_quantile  — Pinball (quantile) loss per specialist per horizon
        L_q(y, q_hat, tau) = tau * max(y - q_hat, 0) + (1-tau) * max(q_hat - y, 0)
        Summed over tau in {0.1, 0.5, 0.9}, horizons weighted [1.0, 0.8, 0.6, 0.4]

    L_vel — Velocity consistency loss
        Forces predicted velocity to match the slope of consecutive P50 predictions
        L_vel = || velocity_h - (P50_{h+1} - P50_h) / delta_t ||^2

    L_lb1 — Tier 1 load balancing loss (sequence-level, sparse specialists)
        Prevents router collapse: all sparse experts should be used roughly equally
        L_lb1 = n_specialists * sum_i (f_i * p_i)
        alpha_1 = 0.01

    L_lb2 — Tier 2 load balancing loss (token-level, FFN experts)
        Same form, token-level
        L_lb2 = n_experts * sum_i (f_i * p_i)
        alpha_2 = 0.01

Curriculum phases (from spec §5.5):
    Phase 0 (epochs  0-15):  Pre-train each specialist LSTM independently
    Phase 1 (epochs 15-35):  Clean joint training
    Phase 2 (epochs 35-55):  Noisy (10% signal dropout)
    Phase 3 (epochs 55-75):  Degraded (30% dropout, high missing)
    Phase 4 (epochs 75-90):  Edge cases
    Phase 5 (epochs 90-100): Full distribution + load balancing verification

Optimizer:
    AdamW, lr=3e-4, cosine decay, 1000 warmup steps
    Gradient checkpointing enabled
    bf16 AMP (fp16 optional)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional

from drift.model.model import DRIFT, DRIFTOutput, CORE_SPECIALISTS
from drift.model.heads import HORIZONS, N_HORIZONS, QUANTILES


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

# Horizon weighting: down-weight far horizons (spec §5.4)
# [7d, 30d, 90d, 180d] -> [1.0, 0.8, 0.6, 0.4]
HORIZON_WEIGHTS = torch.tensor([1.0, 0.8, 0.6, 0.4])

# Delta-t between consecutive horizons (for velocity consistency)
# HORIZONS = [7, 30, 90, 180] -> deltas = [23, 60, 90]
HORIZON_DELTAS = torch.tensor([
    HORIZONS[i+1] - HORIZONS[i] for i in range(len(HORIZONS)-1)
], dtype=torch.float32)


def pinball_loss(
    y: torch.Tensor,        # (B, N_HORIZONS)  ground truth
    q_hat: torch.Tensor,    # (B, N_HORIZONS, 3)  [P10, P50, P90]
    taus: list[float] = QUANTILES,
    horizon_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Quantile (pinball) loss across all quantile levels and horizons.

    L_q(y, q_hat, tau) = tau * max(y - q_hat, 0) + (1-tau) * max(q_hat - y, 0)

    Args:
        y:               (B, H)    ground truth values
        q_hat:           (B, H, 3) predicted [P10, P50, P90]
        taus:            quantile levels [0.1, 0.5, 0.9]
        horizon_weights: (H,)  per-horizon loss weights

    Returns:
        scalar loss
    """
    if horizon_weights is None:
        horizon_weights = HORIZON_WEIGHTS.to(y.device)

    total = 0.0
    for i, tau in enumerate(taus):
        q = q_hat[..., i]           # (B, H)
        diff = y - q                # (B, H)
        loss = torch.where(diff >= 0, tau * diff, (tau - 1.0) * diff)  # (B, H)
        # Weight by horizon importance
        loss = (loss * horizon_weights.unsqueeze(0)).mean()
        total = total + loss

    return total


def velocity_consistency_loss(
    velocity: torch.Tensor,    # (B, N_HORIZONS)  predicted velocity
    p50: torch.Tensor,         # (B, N_HORIZONS)  predicted median
    deltas: torch.Tensor = None,
) -> torch.Tensor:
    """
    Velocity consistency loss: predicted velocity should match slope of P50 predictions.

    L_vel = || velocity_h - (P50_{h+1} - P50_h) / delta_t ||^2

    Args:
        velocity: (B, H)   predicted signed velocity per horizon
        p50:      (B, H)   predicted P50 per horizon
        deltas:   (H-1,)   time gaps between consecutive horizons

    Returns:
        scalar loss
    """
    if deltas is None:
        deltas = HORIZON_DELTAS.to(velocity.device)

    # Slope between consecutive horizon P50 predictions
    implied_velocity = (p50[:, 1:] - p50[:, :-1]) / deltas.unsqueeze(0)  # (B, H-1)

    # Match velocity predictions at non-final horizons
    vel_pred = velocity[:, :-1]  # (B, H-1)

    return F.mse_loss(vel_pred, implied_velocity.detach())


def load_balancing_loss(
    router_probs: torch.Tensor,  # (B, n_experts) or (B, N, n_experts)
) -> torch.Tensor:
    """
    Auxiliary load balancing loss to prevent router collapse.

    From the spec (§5.4):
        L_lb = n_experts * sum_i (f_i * p_i)
        f_i = fraction of tokens/sequences routed to expert i
        p_i = mean routing probability for expert i

    This penalizes high correlation between utilization frequency and
    routing probability — it pushes the router toward uniform usage.

    Args:
        router_probs: routing probabilities from softmax
                      Tier 1: (B, n_sparse) — one prob vector per sequence
                      Tier 2: (B, N, n_experts) — one per token

    Returns:
        scalar auxiliary loss
    """
    # Flatten batch/token dimensions
    if router_probs.dim() == 3:
        B, N, E = router_probs.shape
        probs_flat = router_probs.reshape(B * N, E)  # (B*N, E)
    else:
        probs_flat = router_probs  # (B, E)

    n_experts = probs_flat.shape[-1]

    # f_i: fraction of sequences routed to expert i (based on argmax)
    top_idx = probs_flat.argmax(dim=-1)  # (B*N,) or (B,)
    f = torch.zeros(n_experts, device=router_probs.device)
    for i in range(n_experts):
        f[i] = (top_idx == i).float().mean()

    # p_i: mean routing probability for expert i
    p = probs_flat.mean(dim=0)  # (n_experts,)

    return n_experts * (f * p).sum()


def compute_total_loss(
    outputs: DRIFTOutput,
    y: torch.Tensor,             # (B, n_agents, N_HORIZONS)
    agent_names: list[str],
    alpha_vel: float = 0.1,
    alpha_lb1: float = 0.01,
    alpha_lb2: float = 0.01,
    alpha_jsd: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total training loss.

    L = L_quantile + alpha_vel*L_vel + alpha_lb1*L_lb1 + alpha_lb2*L_lb2

    Args:
        outputs:     DRIFTOutput from model forward pass
        y:           (B, n_agents, N_HORIZONS)  ground truth risk values
        agent_names: ordered list of agent names (must match y dim 1)
        alpha_vel:   velocity loss weight
        alpha_lb1:   Tier 1 load balancing weight
        alpha_lb2:   Tier 2 load balancing weight
        alpha_jsd:   Tier 1 JSD loss weight

    Returns:
        total_loss:  scalar tensor (backpropagatable)
        loss_dict:   dict of component loss values for logging
    """
    device = y.device
    loss_quantile = torch.tensor(0.0, device=device)
    loss_vel = torch.tensor(0.0, device=device)

    for i, name in enumerate(agent_names):
        if name not in outputs.agents:
            continue

        pred = outputs.agents[name]
        y_agent = y[:, i, :].float()  # (B, N_HORIZONS)

        # Quantile loss
        loss_quantile = loss_quantile + pinball_loss(y_agent, pred.quantiles.float())

        # Velocity consistency loss
        loss_vel = loss_vel + velocity_consistency_loss(
            pred.velocity.float(), pred.p50.float()
        )

    # Average over agents
    n_agents = max(1, len(agent_names))
    loss_quantile = loss_quantile / n_agents
    loss_vel = loss_vel / n_agents

    # Load balancing losses
    loss_lb1 = load_balancing_loss(outputs.tier1_router_probs.float())
    loss_lb2 = load_balancing_loss(outputs.tier2_router_probs.float())

    # ── VSN divergence loss (cardio vs mental) ─────────────────────────────
    loss_jsd = torch.tensor(0.0, device=device)

    if "cardio" in outputs.vsn_weights and "mental" in outputs.vsn_weights:
        w_cardio = outputs.vsn_weights["cardio"]   # (B, N, V)
        w_mental = outputs.vsn_weights["mental"]   # (B, N, V)

        # Normalize defensively (should already be softmax)
        w_cardio = w_cardio / (w_cardio.sum(dim=-1, keepdim=True) + 1e-8)
        w_mental = w_mental / (w_mental.sum(dim=-1, keepdim=True) + 1e-8)

        jsd = js_divergence(w_cardio, w_mental)  # (B, N)
        loss_jsd = jsd.mean()

    # Total
    total = (
            loss_quantile
            + alpha_vel * loss_vel
            + alpha_lb1 * loss_lb1
            + alpha_lb2 * loss_lb2
            + alpha_jsd * loss_jsd
    )
    return total, {
        "loss": total.item(),
        "loss_quantile": loss_quantile.item(),
        "loss_vel": loss_vel.item(),
        "loss_lb1": loss_lb1.item(),
        "loss_lb2": loss_lb2.item(),
        "loss_jsd": loss_jsd.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum phase schedule
# ─────────────────────────────────────────────────────────────────────────────

# Map epoch -> curriculum phase (from spec §5.5)
CURRICULUM_SCHEDULE = [
    (0,   5,   0),   # stabilization
    (5,  30,   1),   # clean learning
    (30, 60,   2),   # mild noise
    (60, 95,   3),   # moderate degradation
    (95, 125,  4),   # edge cases
    (125,150,  5),   # final distribution
]

def epoch_to_phase(epoch: int) -> int:
    """Return curriculum phase for a given epoch."""
    for start, end, phase in CURRICULUM_SCHEDULE:
        if start <= epoch < end:
            return phase
    return 5  # beyond 100 epochs -> phase 5

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two distributions.

    p, q: (..., D) probability distributions (sum to 1)
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    m = 0.5 * (p + q)

    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)

    return 0.5 * (kl_pm + kl_qm)

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer + scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model: DRIFT,
    total_epochs: int = 100,
    lr: float = 3e-4,
    warmup_steps: int = 1000,
    steps_per_epoch: int = 100,
) -> tuple:
    """
    Build AdamW optimizer with linear warmup + cosine decay.

    From the spec:
        Optimizer: AdamW
        LR: 3e-4, cosine decay
        Warmup: 1000 steps

    Returns:
        optimizer, scheduler
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )

    total_steps = total_epochs * steps_per_epoch

    # Linear warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Cosine decay after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=lr * 0.01,
    )

    # Combined: warmup then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full DRIFT training manager.

    Handles:
        - 5-phase curriculum (DataLoader switching per phase change)
        - bf16 AMP training
        - Gradient checkpointing
        - Checkpoint saving/loading
        - Per-epoch logging

    Args:
        model:         DRIFT model instance
        shard_path:    path to shard file or directory
        device:        "cuda" or "cpu"
        total_epochs:  total training epochs (100 per spec)
        batch_size:    training batch size (32 per spec)
        lr:            learning rate (3e-4 per spec)
        checkpoint_dir: where to save checkpoints
        agent_names:   which agents to train / evaluate
        precision:     "bf16" or "fp16" or "fp32"
    """

    def __init__(
        self,
        model: DRIFT,
        shard_path: str,
        device: str = "cuda",
        total_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 3e-4,
        checkpoint_dir: str = "checkpoints",
        agent_names: list[str] = None,
        precision: str = "bf16",
    ):
        self.model = model.to(device)
        self.shard_path = shard_path
        self.device = device
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.agent_names = agent_names or ["cardio", "mental"]
        self.precision = precision

        os.makedirs(checkpoint_dir, exist_ok=True)

        # AMP scaler
        self.use_amp = precision in ("bf16", "fp16") and device == "cuda"
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.amp_dtype = amp_dtype
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

        # Gradient checkpointing (spec: required at this model size)
        if hasattr(model.shared_encoder, "layers"):
            for layer in model.shared_encoder.layers:
                # Enable gradient checkpointing on encoder layers
                layer.attn._use_checkpoint = True

        # Current phase and loaders
        self.current_phase = -1
        self.train_loader = None
        self.val_loader = None

        # Build optimizer (steps_per_epoch estimated; updated after first loader build)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model,
            total_epochs=total_epochs,
            lr=lr,
            warmup_steps=1000,
            steps_per_epoch=200,  # rough estimate; fine-tuned after loader creation
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _maybe_switch_phase(self, epoch: int):
        """Switch DataLoader if curriculum phase has changed."""
        from drift.data.loader import make_loaders

        phase = epoch_to_phase(epoch)
        if phase == self.current_phase:
            return

        print(f"\n[curriculum] Epoch {epoch}: switching to Phase {phase}")
        self.current_phase = phase

        self.train_loader, self.val_loader = make_loaders(
            shard_path=self.shard_path,
            phase=phase,
            batch_size=self.batch_size,
            agent_names=self.agent_names,
            pin_memory=(self.device == "cuda"),
            num_workers=2 if self.device == "cuda" else 0,
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns loss dict."""
        self.model.train()
        total_losses = {}
        n_batches = 0

        for xb, yb in self.train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            with torch.autocast(
                device_type=self.device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(xb, return_weights=True)
                loss, loss_dict = compute_total_loss(outputs, yb, self.agent_names)

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

            self.scheduler.step()
            self.global_step += 1

            # Accumulate losses for logging
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in total_losses.items()}

    @torch.no_grad()
    def val_epoch(self) -> dict[str, float]:
        """Run validation. Returns loss dict."""
        self.model.eval()
        total_losses = {}
        n_batches = 0

        for xb, yb in self.val_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            with torch.autocast(
                device_type=self.device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(xb, return_weights=True)
                _, loss_dict = compute_total_loss(outputs, yb, self.agent_names)

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in total_losses.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
        torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            print(f"  [checkpoint] New best: val_loss={val_loss:.4f} -> {best_path}")

    def load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.global_step = state["global_step"]
        print(f"[checkpoint] Resumed from epoch {state['epoch']}, val_loss={state['val_loss']:.4f}")
        return state["epoch"] + 1

    def train(self, resume_from: Optional[str] = None) -> dict:
        """
        Full training run.

        Args:
            resume_from: optional path to checkpoint to resume from

        Returns:
            history: dict of per-epoch train/val losses
        """
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        history = {"train": [], "val": []}

        print(f"[trainer] Starting training: {self.total_epochs} epochs | "
              f"device={self.device} | precision={self.precision}")
        print(f"[trainer] Model parameters: {self.model.n_parameters:,}")

        for epoch in range(start_epoch, self.total_epochs):
            # Switch curriculum phase if needed
            self._maybe_switch_phase(epoch)

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.val_epoch()

            val_loss = val_losses["loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint every 5 epochs and when best
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # Log
            lr_now = self.optimizer.param_groups[0]["lr"]
            phase = epoch_to_phase(epoch)
            print(
                f"Epoch {epoch:3d} | Phase {phase} | "
                f"train={train_losses['loss']:.4f} "
                f"(q={train_losses['loss_quantile']:.4f} "
                f"v={train_losses['loss_vel']:.4f} "
                f"lb1={train_losses['loss_lb1']:.4f} "
                f"lb2={train_losses['loss_lb2']:.4f}) | "
                f"val={val_loss:.4f} | lr={lr_now:.2e}"
            )

            history["train"].append(train_losses)
            history["val"].append(val_losses)

        print(f"\n[trainer] Training complete. Best val_loss={self.best_val_loss:.4f}")
        return history
