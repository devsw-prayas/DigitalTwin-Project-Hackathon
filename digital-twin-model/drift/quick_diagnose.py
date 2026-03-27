"""
quick_diagnose.py — Quick MoE collapse diagnostic without CLI args.

Just run: python quick_diagnose.py

It will auto-detect the best checkpoint and data shard.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model import DRIFT, CORE_SPECIALISTS


def find_checkpoint():
    """Find the best checkpoint."""
    candidates = [
        "checkpoints/best_model.pt",
        "../checkpoints/best_model.pt", 
        "../../checkpoints/best_model.pt",
        "best_model.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def find_shard():
    """Find data shard."""
    candidates = [
        "data.shard",
        "../data.shard",
        "shards/data.shard",
        "../shards/data.shard",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_model(checkpoint_path: str, device: str = "cuda") -> DRIFT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = DRIFT(
        d_in=104,
        d_model=128,
        d_static=32,
        n_heads=4,
        n_enc_layers=2,
        n_horizons=4,
        n_vars=20,
        n_tier2_experts=8,
        n_sparse_top_k=2,
        dropout=0.1,
        precision="bf16",
    )
    
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    print(f"[load] Epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    return model


def generate_test_batch(n_samples: int = 32, seq_len: int = 90, device: str = "cuda"):
    """Generate synthetic test batch if no data available."""
    X = torch.randn(n_samples, seq_len, 104, device=device) * 0.5 + 0.5
    # Add some structure
    X[:, :, 0] = torch.linspace(0.1, 0.8, seq_len).unsqueeze(0).expand(n_samples, -1)  # HRV
    X[:, :, 1] = torch.linspace(60, 80, seq_len).unsqueeze(0).expand(n_samples, -1)   # HR
    return X


@torch.no_grad()
def quick_diagnose(model, device):
    """Run quick diagnostic on model."""
    
    # Generate test batch
    X = generate_test_batch(64, device=device)
    
    # Forward pass
    outputs = model(X, return_weights=True)
    
    print("\n" + "="*60)
    print("QUICK MoE DIAGNOSTIC")
    print("="*60)
    
    # 1. Tier 1 Router Analysis
    print("\n[TIER 1 ROUTER - Sparse Specialists]")
    t1_probs = outputs.tier1_router_probs  # (B, 4)
    mean_probs = t1_probs.mean(dim=0).cpu().numpy()
    
    print(f"  Mean routing probs: {np.round(mean_probs, 4)}")
    print(f"  Uniform baseline:   [0.25, 0.25, 0.25, 0.25]")
    
    # Entropy
    entropy = -(t1_probs * t1_probs.clamp(1e-8).log()).sum(dim=-1).mean().item()
    max_entropy = np.log(4)
    normalized_entropy = entropy / max_entropy
    
    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} = {normalized_entropy:.2%}")
    
    if normalized_entropy > 0.95:
        print("  ⚠️  COLLAPSED: Router is uniform!")
    elif normalized_entropy > 0.85:
        print("  ⚠️  NEAR-COLLAPSE: Router almost uniform")
    else:
        print("  ✓ Router shows specialization")
    
    # Argmax distribution
    argmax = t1_probs.argmax(dim=-1)
    counts = torch.bincount(argmax, minlength=4).float()
    pct = counts / counts.sum()
    print(f"  Argmax dist: {pct.cpu().numpy().round(4)}")
    
    # 2. Tier 2 Router Analysis
    print("\n[TIER 2 ROUTER - Token-Level FFN]")
    t2_probs = outputs.tier2_router_probs  # (B, seq, 8)
    t2_mean = t2_probs.mean(dim=[0, 1]).cpu().numpy()
    
    print(f"  Mean routing probs: {np.round(t2_mean, 4)}")
    
    t2_entropy = -(t2_probs * t2_probs.clamp(1e-8).log()).sum(dim=-1).mean().item()
    t2_max_entropy = np.log(8)
    t2_norm = t2_entropy / t2_max_entropy
    
    print(f"  Entropy: {t2_entropy:.4f} / {t2_max_entropy:.4f} = {t2_norm:.2%}")
    
    if t2_norm > 0.95:
        print("  ⚠️  COLLAPSED: Router is uniform!")
    else:
        print("  ✓ Shows some specialization")
    
    # 3. VSN Weights
    print("\n[VSN WEIGHTS - Cardio vs Mental]")
    
    if outputs.vsn_weights.get("cardio") is not None:
        w_cardio = outputs.vsn_weights["cardio"]  # (B, seq, n_vars)
        w_mental = outputs.vsn_weights["mental"]  # (B, seq, n_vars)
        
        # Mean over sequence
        w_cardio_mean = w_cardio.mean(dim=1)
        w_mental_mean = w_mental.mean(dim=1)
        
        # JSD
        def jsd(p, q, eps=1e-8):
            p = p.clamp(min=eps)
            q = q.clamp(min=eps)
            m = 0.5 * (p + q)
            return 0.5 * ((p * (p/m).log()).sum(-1) + (q * (q/m).log()).sum(-1))
        
        jsd_val = jsd(w_cardio_mean, w_mental_mean).mean().item()
        cos_sim = F.cosine_similarity(w_cardio_mean, w_mental_mean, dim=-1).mean().item()
        
        print(f"  JSD (cardio vs mental): {jsd_val:.6f}")
        print(f"  Cosine similarity:      {cos_sim:.4f}")
        
        if jsd_val < 0.01:
            print("  ⚠️  COLLAPSED: VSN weights identical!")
        elif jsd_val < 0.05:
            print("  ⚠️  LOW: VSN weights very similar")
        else:
            print("  ✓ VSN shows differentiation")
    else:
        print("  (VSN weights not available)")
    
    # 4. Expert Output Similarity (via specialist states)
    print("\n[EXPERT OUTPUT SIMILARITY]")
    
    # Check specialist states
    specialist_names = list(outputs.routing_info.get("top_indices", []).__class__.__bases__[0]().__class__.__dict__.keys()) if hasattr(outputs, 'routing_info') else []
    
    # Get specialist states from model's tier1_moe
    # Re-run to capture individual expert outputs
    expert_outputs = {}
    
    # Hook to capture outputs
    handles = []
    
    def make_hook(name):
        def hook(module, inp, out):
            expert_outputs[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return hook
    
    # Register hooks on core specialists
    for name, specialist in model.tier1_moe.core_specialists.items():
        h = specialist.register_forward_hook(make_hook(name))
        handles.append(h)
    
    # Re-run forward
    _ = model(X[:8], return_weights=False)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    if len(expert_outputs) >= 2:
        # Compare cardio vs mental expert outputs
        names = list(expert_outputs.keys())
        
        # Compute similarity matrix
        print(f"  Comparing {len(names)} core specialists:")
        
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i < j:
                    out_i = expert_outputs[name_i].mean(dim=0)  # (d_model,)
                    out_j = expert_outputs[name_j].mean(dim=0)
                    
                    sim = F.cosine_similarity(out_i.unsqueeze(0), out_j.unsqueeze(0)).item()
                    print(f"    {name_i} vs {name_j}: {sim:.4f}")
        
        # Average pairwise similarity
        sims = []
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i < j:
                    out_i = expert_outputs[name_i].mean(dim=0)
                    out_j = expert_outputs[name_j].mean(dim=0)
                    sim = F.cosine_similarity(out_i.unsqueeze(0), out_j.unsqueeze(0)).item()
                    sims.append(sim)
        
        avg_sim = np.mean(sims)
        print(f"\n  Average pairwise similarity: {avg_sim:.4f}")
        
        if avg_sim > 0.95:
            print("  ⚠️  COLLAPSED: Experts produce near-identical outputs!")
        elif avg_sim > 0.85:
            print("  ⚠️  HIGH similarity: Partial collapse")
        else:
            print("  ✓ Experts produce diverse outputs")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    issues = []
    
    if normalized_entropy > 0.9:
        issues.append("Tier 1 router collapsed (uniform)")
    if t2_norm > 0.95:
        issues.append("Tier 2 router collapsed (uniform)")
    if outputs.vsn_weights.get("cardio") is not None:
        if jsd_val < 0.02:
            issues.append("VSN weights collapsed (identical)")
    
    if not issues:
        print("✓ No major collapse indicators detected")
    else:
        print("⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
    
    print("\n" + "="*60)
    
    return {
        "t1_entropy": normalized_entropy,
        "t2_entropy": t2_norm,
        "vsn_jsd": jsd_val if outputs.vsn_weights.get("cardio") is not None else None,
    }


def main():
    print("DRIFT MoE Quick Diagnostic")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Find checkpoint
    ckpt = find_checkpoint()
    if ckpt:
        print(f"Found checkpoint: {ckpt}")
    else:
        print("No checkpoint found. Please specify path.")
        return
    
    # Load model
    model = load_model(ckpt, device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run diagnostic
    results = quick_diagnose(model, device)


if __name__ == "__main__":
    main()
