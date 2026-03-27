"""
probe.py — Quick MoE collapse diagnostic.

Run: python probe.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model import DRIFT


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


def generate_test_batch(n_samples: int = 32, seq_len: int = 90, device: str = "cuda", dtype=torch.bfloat16):
    """Generate synthetic test batch."""
    X = torch.randn(n_samples, seq_len, 104, device=device, dtype=dtype) * 0.5 + 0.5
    # Add some structure
    X[:, :, 0] = torch.linspace(0.1, 0.8, seq_len, device=device, dtype=dtype).unsqueeze(0).expand(n_samples, -1)
    X[:, :, 1] = torch.linspace(60, 80, seq_len, device=device, dtype=dtype).unsqueeze(0).expand(n_samples, -1)
    return X


@torch.no_grad()
def quick_diagnose(model, device):
    """Run quick diagnostic on model."""
    
    dtype = torch.bfloat16  # Match model precision
    
    # Generate test batch with correct dtype
    X = generate_test_batch(64, device=device, dtype=dtype)
    
    # Forward pass (need to handle autocast for bf16)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        outputs = model(X, return_weights=True)
    
    print("\n" + "="*60)
    print("QUICK MoE DIAGNOSTIC")
    print("="*60)
    
    # 1. Tier 1 Router Analysis
    print("\n[TIER 1 ROUTER - Sparse Specialists]")
    t1_probs = outputs.tier1_router_probs.float()  # Convert to float for analysis
    mean_probs = t1_probs.mean(dim=0).cpu().numpy()
    
    print(f"  Mean routing probs: {np.round(mean_probs, 4)}")
    print(f"  Uniform baseline:   [0.25, 0.25, 0.25, 0.25]")
    
    # Entropy
    entropy = -(t1_probs * t1_probs.clamp(1e-8).log()).sum(dim=-1).mean().item()
    max_entropy = np.log(4)
    normalized_entropy = entropy / max_entropy
    
    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} = {normalized_entropy:.2%}")
    
    if normalized_entropy > 0.95:
        print("  [!!!] COLLAPSED: Router is uniform!")
    elif normalized_entropy > 0.85:
        print("  [!!]  NEAR-COLLAPSE: Router almost uniform")
    elif normalized_entropy > 0.75:
        print("  [!]   WEAK specialization")
    else:
        print("  [OK]  Router shows specialization")
    
    # Argmax distribution
    argmax = t1_probs.argmax(dim=-1)
    counts = torch.bincount(argmax, minlength=4).float()
    pct = counts / counts.sum()
    print(f"  Argmax dist: {pct.cpu().numpy().round(4)}")
    
    # 2. Tier 2 Router Analysis
    print("\n[TIER 2 ROUTER - Token-Level FFN]")
    t2_probs = outputs.tier2_router_probs.float()  # (B, seq, 8)
    t2_mean = t2_probs.mean(dim=[0, 1]).cpu().numpy()
    
    print(f"  Mean routing probs: {np.round(t2_mean, 4)}")
    
    t2_entropy = -(t2_probs * t2_probs.clamp(1e-8).log()).sum(dim=-1).mean().item()
    t2_max_entropy = np.log(8)
    t2_norm = t2_entropy / t2_max_entropy
    
    print(f"  Entropy: {t2_entropy:.4f} / {t2_max_entropy:.4f} = {t2_norm:.2%}")
    
    if t2_norm > 0.95:
        print("  [!!!] COLLAPSED: Router is uniform!")
    elif t2_norm > 0.85:
        print("  [!!]  NEAR-COLLAPSE")
    else:
        print("  [OK]  Shows some specialization")
    
    # 3. VSN Weights
    print("\n[VSN WEIGHTS - Cardio vs Mental]")
    jsd_val = None
    
    if outputs.vsn_weights.get("cardio") is not None:
        w_cardio = outputs.vsn_weights["cardio"].float()  # (B, seq, n_vars)
        w_mental = outputs.vsn_weights["mental"].float()
        
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
        
        if jsd_val < 0.005:
            print("  [!!!] COLLAPSED: VSN weights identical!")
        elif jsd_val < 0.02:
            print("  [!!]  LOW: VSN weights very similar")
        elif jsd_val < 0.05:
            print("  [!]   MODERATE: Some differentiation")
        else:
            print("  [OK]  VSN shows good differentiation")
        
        # Show top features
        feature_names = ["hrv", "hr", "spo2", "sleep", "rem", "deep", 
                        "steps", "active", "screen", "aqi", "temp"]
        
        cardio_imp = w_cardio_mean.mean(dim=0).cpu().numpy()
        mental_imp = w_mental_mean.mean(dim=0).cpu().numpy()
        
        print(f"\n  Top 3 features for CARDIO:")
        for i in np.argsort(cardio_imp)[-3:][::-1]:
            name = feature_names[i] if i < len(feature_names) else f"var_{i}"
            print(f"    {name}: {cardio_imp[i]:.4f}")
        
        print(f"  Top 3 features for MENTAL:")
        for i in np.argsort(mental_imp)[-3:][::-1]:
            name = feature_names[i] if i < len(feature_names) else f"var_{i}"
            print(f"    {name}: {mental_imp[i]:.4f}")
    else:
        print("  (VSN weights not available)")
    
    # 4. Expert Output Similarity
    print("\n[EXPERT OUTPUT SIMILARITY]")
    
    expert_outputs = {}
    handles = []
    
    def make_hook(name):
        def hook(module, inp, out):
            expert_outputs[name] = out[0].detach().float() if isinstance(out, tuple) else out.detach().float()
        return hook
    
    # Register hooks on core specialists
    for name, specialist in model.tier1_moe.core_specialists.items():
        h = specialist.register_forward_hook(make_hook(name))
        handles.append(h)
    
    # Re-run forward
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        _ = model(X[:8], return_weights=False)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    if len(expert_outputs) >= 2:
        names = list(expert_outputs.keys())
        
        print(f"  Core specialists: {names}")
        print(f"  Pairwise cosine similarity:")
        
        sims = []
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i < j:
                    out_i = expert_outputs[name_i].mean(dim=0)
                    out_j = expert_outputs[name_j].mean(dim=0)
                    
                    sim = F.cosine_similarity(out_i.unsqueeze(0), out_j.unsqueeze(0)).item()
                    sims.append(sim)
                    print(f"    {name_i} vs {name_j}: {sim:.4f}")
        
        avg_sim = np.mean(sims)
        print(f"\n  Average pairwise similarity: {avg_sim:.4f}")
        
        if avg_sim > 0.95:
            print("  [!!!] COLLAPSED: Experts produce near-identical outputs!")
        elif avg_sim > 0.90:
            print("  [!!]  HIGH: Partial collapse")
        elif avg_sim > 0.80:
            print("  [!]   MODERATE: Some similarity")
        else:
            print("  [OK]  Experts produce diverse outputs")
    
    # 5. Load Balancing Loss Check
    print("\n[LOAD BALANCING LOSS VALUES]")
    
    # These are computed during training, but we can compute them now
    def load_balance_loss(router_probs):
        """L_lb = n * sum(f_i * p_i)"""
        if router_probs.dim() == 3:
            probs_flat = router_probs.reshape(-1, router_probs.size(-1))
        else:
            probs_flat = router_probs
        
        n = probs_flat.size(-1)
        top_idx = probs_flat.argmax(dim=-1)
        f = torch.zeros(n, device=probs_flat.device)  # Same device as input
        for i in range(n):
            f[i] = (top_idx == i).float().mean()
        
        p = probs_flat.mean(dim=0)
        return n * (f * p).sum().item()
    
    lb1 = load_balance_loss(t1_probs)
    lb2 = load_balance_loss(t2_probs)
    
    print(f"  Tier 1 LB loss: {lb1:.4f} (ideal for uniform = 0.25)")
    print(f"  Tier 2 LB loss: {lb2:.4f} (ideal for uniform = 0.125)")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    issues = []
    
    if normalized_entropy > 0.90:
        issues.append(("CRITICAL", "Tier 1 router collapsed (uniform)"))
    elif normalized_entropy > 0.80:
        issues.append(("WARNING", "Tier 1 router near-uniform"))
    
    if t2_norm > 0.95:
        issues.append(("CRITICAL", "Tier 2 router collapsed (uniform)"))
    
    if jsd_val is not None and jsd_val < 0.02:
        issues.append(("CRITICAL", "VSN weights collapsed (identical)"))
    
    if len(expert_outputs) >= 2:
        if avg_sim > 0.90:
            issues.append(("CRITICAL", "Expert outputs near-identical"))
        elif avg_sim > 0.80:
            issues.append(("WARNING", "Expert outputs too similar"))
    
    print()
    if not issues:
        print("[OK] No major collapse indicators")
    else:
        for severity, issue in issues:
            print(f"  [{severity}] {issue}")
    
    print("\n" + "="*60)
    
    return {
        "t1_entropy_normalized": normalized_entropy,
        "t2_entropy_normalized": t2_norm,
        "vsn_jsd": jsd_val,
        "expert_similarity": avg_sim if len(expert_outputs) >= 2 else None,
    }


def main():
    print("="*60)
    print("DRIFT MoE COLLAPSE DIAGNOSTIC")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Find checkpoint
    ckpt = find_checkpoint()
    if ckpt:
        print(f"Checkpoint: {ckpt}")
    else:
        print("ERROR: No checkpoint found!")
        print("Place best_model.pt in one of:")
        print("  - checkpoints/best_model.pt")
        print("  - ./best_model.pt")
        return
    
    # Load model
    model = load_model(ckpt, device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run diagnostic
    results = quick_diagnose(model, device)
    
    return results


if __name__ == "__main__":
    main()
