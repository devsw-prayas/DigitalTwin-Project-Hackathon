"""
main.py — Entry point for DRIFT training and evaluation

Usage:
    # Train from scratch
    python main.py --shard outputs/shards/shard_0000.pt --epochs 100

    # Train with multiple shards
    python main.py --shard outputs/shards/ --epochs 100

    # Resume from checkpoint
    python main.py --shard outputs/shards/ --resume checkpoints/best_model.pt

    # Evaluate only
    python main.py --shard outputs/shards/ --eval-only --resume checkpoints/best_model.pt

    # Quick sanity check (2 epochs, fast)
    python main.py --shard outputs/shards/shard_0000.pt --epochs 2 --batch-size 8

GPU memory guide:
    d_model=128, batch=32, N=90:  ~2-4GB VRAM  (fits RTX 3060+)
    d_model=256, batch=32, N=90:  ~6-8GB VRAM  (fits RTX 3080+)
"""

import argparse
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model import DRIFT
from training.training import Trainer
from training.eval import evaluate
from data.loader import make_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="DRIFT training")

    # Data
    parser.add_argument("--shard", type=str, default="outputs/hard/shards/shard_0000.pt",
                        help="Path to shard file or directory of shards")
    parser.add_argument("--agents", nargs="+", default=["cardio", "mental"],
                        help="Agent names to train (e.g. cardio mental metabolic recovery)")

    # Model
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-enc-layers", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="checkpoints/best_model.pt",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights (does NOT resume training)")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Device: {device}")
    if device == "cuda":
        print(f"[main] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[main] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build model
    model = DRIFT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers,
        precision=args.precision,
    )
    print(f"[main] Model parameters: {model.n_parameters:,}")

    if args.eval_only:
        # Load checkpoint and evaluate
        if not args.resume:
            raise ValueError("--eval-only requires --resume")

        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state"])
        model = model.to(device)

        _, val_loader = make_loaders(
            shard_path=args.shard,
            phase=5,
            batch_size=args.batch_size,
            agent_names=args.agents,
        )

        report = evaluate(model, val_loader, device=device, agent_names=args.agents)
        print(f"\nMean JS Divergence: {report.mean_js_divergence:.4f}")
        print(f"Router collapsed: {report.router_collapsed}")
        return

    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state["model_state"])

    # Train
    trainer = Trainer(
        model=model,
        shard_path=args.shard,
        device=device,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        agent_names=args.agents,
        precision=args.precision,
    )

    history = trainer.train(resume_from=args.resume)

    # Final evaluation
    print("\n[main] Running final evaluation...")
    _, val_loader = make_loaders(
        shard_path=args.shard,
        phase=5,
        batch_size=args.batch_size,
        agent_names=args.agents,
    )
    report = evaluate(model, val_loader, device=device, agent_names=args.agents)

    print(f"\nFinal mean JS divergence: {report.mean_js_divergence:.4f} (target: >0.20)")
    for name, cov in report.coverage.items():
        print(f"{name}: mean coverage={cov.mean_coverage:.3f} (target: 0.80), ECE={cov.mean_ece:.4f} (target: <0.05)")


if __name__ == "__main__":
    main()
