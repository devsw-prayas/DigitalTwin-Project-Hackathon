"""
train_8.py — Main Training Script for 8-Agent DRIFT

Run: python train_8.py --epochs 150 --batch_size 16

This script:
1. Loads divergence-maximizing personas
2. Builds 8-agent DRIFT model
3. Trains with divergence-aware losses
4. Monitors orthogonality and routing health
5. Saves checkpoints
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift.model.model_8 import DRIFT8, build_drift_8
from drift.training.training_8 import Trainer8, LossWeights
from drift.data.loader_8 import make_loaders_8
from drift.data.generator_8 import get_divergence_personas


def parse_args():
    parser = argparse.ArgumentParser(description="Train 8-Agent DRIFT")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_8")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("8-AGENT DRIFT TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model dim: {args.d_model}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    print("="*60)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"
    
    # Build model
    print("\n[1/4] Building model...")
    model = build_drift_8(
        d_model=args.d_model,
        precision=args.precision,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Load personas
    print("\n[2/4] Loading personas...")
    personas = get_divergence_personas()
    print(f"  Personas: {len(personas)}")
    for name, p in personas.items():
        print(f"    - {name}: primary={p.primary_agent}, divergence={p.divergence_type}")
    
    # Create data loaders
    print("\n[3/4] Creating data loaders...")
    train_loader, val_loader = make_loaders_8(
        personas=personas,
        batch_size=args.batch_size,
        train_samples_per_persona=100,
        val_samples_per_persona=25,
        phase=1,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n[4/4] Setting up trainer...")
    trainer = Trainer8(
        model=model,
        device=args.device,
        lr=args.lr,
        total_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
    )
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    
    return history


if __name__ == "__main__":
    main()
