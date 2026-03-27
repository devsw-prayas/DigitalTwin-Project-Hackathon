"""
loader_8.py — Data Loader for 8-Agent Training

Creates batches with:
- X: (B, N, 104) token sequences
- y: (B, 8, 4) ground truth for all 8 agents and 4 horizons
- routing_targets: (B,) which agent is primary for this sample

Supports curriculum phases with increasing difficulty.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np

from .agents_8 import ALL_AGENTS, N_AGENTS, N_SIGNALS
from .generator_8 import generate_sample, get_divergence_personas, Persona8, Sample8Agent


# Token dimensions
TOKEN_DIM = 104  # 11 signals + 93 derived features


def sample_to_tokens(sample: Sample8Agent, start_idx: int = 0, seq_len: int = 90) -> torch.Tensor:
    """
    Convert a sample to token sequence.
    
    Returns: (seq_len, TOKEN_DIM)
    """
    n_days = len(sample.t)
    end_idx = min(start_idx + seq_len, n_days)
    actual_len = end_idx - start_idx
    
    # Build token features
    tokens = torch.zeros(seq_len, TOKEN_DIM)
    
    # Signals (first 11 dims)
    signal_list = list(sample.signals.keys())
    for i, sig_name in enumerate(signal_list):
        if i < 11:
            tokens[:actual_len, i] = sample.signals[sig_name][start_idx:end_idx]
    
    # Derived features (dims 11-103)
    # Rolling statistics, differences, etc.
    for i in range(actual_len):
        idx = start_idx + i
        
        # Rolling means (7-day)
        for j, sig_name in enumerate(signal_list[:5]):
            if idx >= 7:
                tokens[i, 11 + j] = sample.signals[sig_name][idx-7:idx].mean()
        
        # Differences from previous day
        for j, sig_name in enumerate(signal_list[:5]):
            if idx > 0:
                tokens[i, 16 + j] = sample.signals[sig_name][idx] - sample.signals[sig_name][idx-1]
        
        # Rolling stds (7-day)
        for j, sig_name in enumerate(signal_list[:4]):
            if idx >= 7:
                tokens[i, 21 + j] = sample.signals[sig_name][idx-7:idx].std()
        
        # Day of year (normalized)
        tokens[i, 25] = (idx % 365) / 365.0
        
        # Day of week (one-hot)
        dow = idx % 7
        tokens[i, 26 + dow] = 1.0
    
    return tokens


def get_horizon_targets(
    sample: Sample8Agent,
    start_idx: int,
    horizons: List[int] = [7, 30, 90, 180],
) -> torch.Tensor:
    """
    Get ground truth targets for all agents at specified horizons.
    
    Returns: (N_AGENTS, len(horizons))
    """
    n_days = len(sample.t)
    targets = torch.zeros(N_AGENTS, len(horizons))
    
    for agent_idx, agent_name in enumerate(ALL_AGENTS):
        if agent_name in sample.agent_gts:
            gt = sample.agent_gts[agent_name].values
            for h_idx, h in enumerate(horizons):
                target_idx = start_idx + h
                if target_idx < n_days:
                    targets[agent_idx, h_idx] = gt[target_idx]
                else:
                    # Extrapolate
                    targets[agent_idx, h_idx] = gt[-1]
    
    return targets


class Dataset8(Dataset):
    """Dataset for 8-agent training."""
    
    def __init__(
        self,
        personas: Dict[str, Persona8],
        n_days: int = 365,
        seq_len: int = 90,
        horizons: List[int] = [7, 30, 90, 180],
        n_samples_per_persona: int = 50,
        degradation: Dict[str, float] = None,
    ):
        self.personas = personas
        self.n_days = n_days
        self.seq_len = seq_len
        self.horizons = horizons
        self.n_samples_per_persona = n_samples_per_persona
        self.degradation = degradation or {"dropout_prob": 0.0, "missing_prob": 0.0}
        
        # Pre-generate samples
        self.samples = self._generate_samples()
        
    def _generate_samples(self) -> List[Tuple[Sample8Agent, int, str]]:
        """Generate all samples with random start positions."""
        samples = []
        
        for persona_name, persona in self.personas.items():
            # Generate full sample once
            full_sample = generate_sample(persona, self.n_days)
            
            # Create multiple windows
            for i in range(self.n_samples_per_persona):
                max_start = self.n_days - max(self.horizons) - 1
                start_idx = np.random.randint(0, max(1, max_start))
                samples.append((full_sample, start_idx, persona_name))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, start_idx, persona_name = self.samples[idx]
        
        # Get tokens
        X = sample_to_tokens(sample, start_idx, self.seq_len)
        
        # Get targets
        y = get_horizon_targets(sample, start_idx, self.horizons)
        
        # Get routing target (which agent should be primary)
        persona = self.personas[persona_name]
        routing_target = ALL_AGENTS.index(persona.primary_agent)
        
        # Apply degradation
        if self.degradation["dropout_prob"] > 0:
            mask = torch.rand(self.seq_len) > self.degradation["dropout_prob"]
            X = X * mask.unsqueeze(1)
        
        return X, y, routing_target, persona_name


def make_loaders_8(
    personas: Optional[Dict[str, Persona8]] = None,
    batch_size: int = 16,
    seq_len: int = 90,
    n_days: int = 365,
    train_samples_per_persona: int = 80,
    val_samples_per_persona: int = 20,
    phase: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation loaders for 8-agent training.
    
    Phase determines degradation level:
        Phase 1: Clean data
        Phase 2: 10% dropout
        Phase 3: 20% dropout, 10% missing
        Phase 4: 30% dropout, 15% missing
        Phase 5: 40% dropout, 20% missing
    """
    if personas is None:
        personas = get_divergence_personas()
    
    # Degradation by phase
    degradation_schedule = {
        1: {"dropout_prob": 0.0, "missing_prob": 0.0},
        2: {"dropout_prob": 0.10, "missing_prob": 0.05},
        3: {"dropout_prob": 0.20, "missing_prob": 0.10},
        4: {"dropout_prob": 0.30, "missing_prob": 0.15},
        5: {"dropout_prob": 0.40, "missing_prob": 0.20},
    }
    
    degradation = degradation_schedule.get(phase, degradation_schedule[1])
    
    # Split personas into train/val
    persona_names = list(personas.keys())
    n_train = int(len(persona_names) * 0.8)
    
    train_personas = {k: personas[k] for k in persona_names[:n_train]}
    val_personas = {k: personas[k] for k in persona_names[n_train:]}
    
    # Also split samples for same personas
    train_dataset = Dataset8(
        train_personas,
        n_days=n_days,
        seq_len=seq_len,
        n_samples_per_persona=train_samples_per_persona,
        degradation=degradation,
    )
    
    val_dataset = Dataset8(
        val_personas,
        n_days=n_days,
        seq_len=seq_len,
        n_samples_per_persona=val_samples_per_persona,
        degradation={"dropout_prob": 0.0, "missing_prob": 0.0},  # Clean validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the loader."""
    print("Testing 8-agent data loader...")
    
    train_loader, val_loader = make_loaders_8(
        batch_size=8,
        train_samples_per_persona=10,
        val_samples_per_persona=5,
        phase=1,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    for X, y, routing, persona in train_loader:
        print(f"\nBatch shapes:")
        print(f"  X: {X.shape}")         # (B, 90, 104)
        print(f"  y: {y.shape}")         # (B, 8, 4)
        print(f"  routing: {routing.shape}")  # (B,)
        print(f"  personas: {persona}")
        
        # Check values
        print(f"\n  y[0] (first sample, all agents, first horizon):")
        print(f"    {y[0, :, 0]}")
        
        break
    
    print("\nLoader test passed!")


if __name__ == "__main__":
    main()
