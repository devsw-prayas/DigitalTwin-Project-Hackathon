"""
run.py — Direct runner for IntelliJ/IDE

Just run this file directly in your IDE. No conda/terminal needed.

Usage in IntelliJ:
    1. Open this file
    2. Right-click -> Run 'run'
    3. Or use the green play button
"""

import os
import sys
import json
import torch

# Add current dir to path (so imports work in IDE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: GENERATE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(
        n_personas: int = 10000,
        n_days: int = 1825,
        n_shards: int = 50,
        output_dir: str = "data_shards",
):
    """Generate and save personas to sharded files."""
    from spawner_8 import generate_random_persona

    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("GENERATING & SAVING DATA")
    print("="*60)
    print(f"  Personas: {n_personas:,}")
    print(f"  Days:     {n_days:,}")
    print(f"  Shards:   {n_shards}")
    print(f"  Output:   {output_dir}/")
    print("="*60)

    samples_per_shard = n_personas // n_shards
    manifest = {
        "n_personas": n_personas,
        "n_days": n_days,
        "n_shards": n_shards,
        "agents": ["cardio", "mental", "metabolic", "recovery",
                   "immune", "respiratory", "hormonal", "cog_fatigue"],
        "shards": [],
    }

    for shard_idx in range(n_shards):
        shard_samples = []
        start_idx = shard_idx * samples_per_shard
        end_idx = start_idx + samples_per_shard

        for i in range(start_idx, end_idx):
            if (i - start_idx) % 100 == 0:
                print(f"  Shard {shard_idx+1}/{n_shards} | Sample {i-start_idx}/{samples_per_shard}")

            persona = generate_random_persona(seed=i)

            # Import here to avoid circular imports
            from data.generator_8 import generate_sample
            sample = generate_sample(persona, n_days=n_days)

            # Convert to serializable dict
            shard_samples.append({
                "persona_name": sample.persona_name,
                "t": sample.t.cpu(),
                "signals": {k: v.cpu() for k, v in sample.signals.items()},
                "agent_gts": {
                    name: {
                        "values": gt.values.cpu(),
                        "base": gt.base,
                        "drift_per_day": gt.drift_per_day,
                    }
                    for name, gt in sample.agent_gts.items()
                },
                "divergence_score": sample.divergence_score,
                "primary_agent": persona.primary_agent,
            })

        # Save shard
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(shard_samples, shard_path)

        size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        manifest["shards"].append({
            "path": f"shard_{shard_idx:04d}.pt",
            "n_samples": len(shard_samples),
            "size_mb": round(size_mb, 2),
        })

        print(f"  Saved shard_{shard_idx:04d}.pt ({size_mb:.1f} MB)")

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"  Total samples: {n_personas:,}")
    print(f"  Total size:    {sum(s['size_mb'] for s in manifest['shards']):.0f} MB")
    print(f"  Manifest:      {manifest_path}")

    return manifest


def generate_small_test(
        n_personas: int = 10,
        n_days: int = 365,
        output_dir: str = "data_test",
):
    """Generate small test dataset and save to disk."""
    from data.generator_8 import get_divergence_personas, generate_sample

    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("GENERATING TEST DATA")
    print("="*60)
    print(f"  Personas: {n_personas}")
    print(f"  Days:     {n_days}")
    print(f"  Output:   {output_dir}/")
    print("="*60)

    personas = get_divergence_personas()
    samples = []

    for i, (name, persona) in enumerate(personas.items()):
        if i >= n_personas:
            break
        print(f"  Generating {name}...")
        sample = generate_sample(persona, n_days=n_days)

        samples.append({
            "persona_name": sample.persona_name,
            "t": sample.t.cpu(),
            "signals": {k: v.cpu() for k, v in sample.signals.items()},
            "agent_gts": {
                agent_name: {
                    "values": gt.values.cpu(),
                    "base": gt.base,
                    "drift_per_day": gt.drift_per_day,
                }
                for agent_name, gt in sample.agent_gts.items()
            },
            "divergence_score": sample.divergence_score,
            "primary_agent": persona.primary_agent,
        })

    # Save
    shard_path = os.path.join(output_dir, "shard_0000.pt")
    torch.save(samples, shard_path)

    manifest = {
        "n_personas": len(samples),
        "n_days": n_days,
        "n_shards": 1,
        "agents": ["cardio", "mental", "metabolic", "recovery",
                   "immune", "respiratory", "hormonal", "cog_fatigue"],
        "shards": [{"path": "shard_0000.pt", "n_samples": len(samples)}],
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_dir}/")
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
        shard_dir: str = "data_test",
        epochs: int = 50,
        batch_size: int = 16,
        checkpoint_dir: str = "checkpoints_8",
):
    """Train the 8-agent model from saved shards."""
    from model.model_8 import build_drift_8
    from training.training_8 import Trainer8
    from data.shard_loader import ShardLoader

    print("="*60)
    print("TRAINING MODEL")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model
    model = build_drift_8(d_model=128, precision="bf16")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load shards
    loader = ShardLoader(
        shard_dir=shard_dir,
        batch_size=batch_size,
        n_windows=10,
        val_ratio=0.1,
    )

    train_loader, val_loader = loader.get_loaders(phase=1)

    # Train
    trainer = Trainer8(
        model=model,
        device=device,
        lr=3e-4,
        total_epochs=epochs,
        precision="bf16",
        checkpoint_dir=checkpoint_dir,
    )

    history = trainer.train(train_loader, val_loader)
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: DIAGNOSE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_model(checkpoint_dir: str = "checkpoints_8"):
    """Run diagnostic on trained model."""
    from probe_8 import diagnose_8

    print("="*60)
    print("DIAGNOSING MODEL")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        # Try other common locations
        candidates = [
            "best_model.pt",
            "checkpoints/best_model.pt",
            "checkpoints_8/best_model.pt",
        ]
        for path in candidates:
            if os.path.exists(path):
                checkpoint_path = path
                break

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found! Train first with STEP='train'")
        return

    print(f"Loading: {checkpoint_path}")

    from model.model_8 import build_drift_8

    model = build_drift_8()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    diagnose_8(model, device)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - Just change STEP to run different steps
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ════════════════════════════════════════════════════════════
    # CHANGE THIS TO RUN DIFFERENT STEPS
    # ════════════════════════════════════════════════════════════
    STEP = "generate"        # Options: "test", "train", "diagnose", "generate"

    # For "generate" step:
    N_PERSONAS = 10000   # Total personas to generate
    N_DAYS = 1825        # Days per persona (5 years)
    N_SHARDS = 50        # Number of shard files
    OUTPUT_DIR = "data_shards"

    # For "train" step:
    TRAIN_DIR = "data_test"  # or "data_shards" for full data
    EPOCHS = 50
    # ════════════════════════════════════════════════════════════

    if STEP == "test":
        # Quick test: small data + short train + diagnose
        print("Running quick test...")
        generate_small_test(n_personas=10, n_days=365)
        train_model(shard_dir="data_test", epochs=10)
        diagnose_model()

    elif STEP == "generate":
        # Generate full dataset
        generate_data(
            n_personas=N_PERSONAS,
            n_days=N_DAYS,
            n_shards=N_SHARDS,
            output_dir=OUTPUT_DIR,
        )

    elif STEP == "train":
        # Train model
        train_model(shard_dir=TRAIN_DIR, epochs=EPOCHS)

    elif STEP == "diagnose":
        # Diagnose existing model
        diagnose_model()

    else:
        print(f"Unknown step: {STEP}")
        print("Options: 'test', 'generate', 'train', 'diagnose'")
