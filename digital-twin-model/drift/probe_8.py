"""
probe_8.py — Diagnostic for 8-Agent DRIFT Model

Run after training to check:
1. Are all 8 agents producing different predictions?
2. Is the router selecting appropriate experts?
3. Are VSN weights diverged across agents?
4. Is there orthogonality between specialist outputs?

Run: python probe_8.py --checkpoint checkpoints_8/best_model.pt
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model_8 import DRIFT8, build_drift_8
from agents_8 import ALL_AGENTS, N_AGENTS


def find_checkpoint():
    candidates = [
        "checkpoints_8/best_model.pt",
        "checkpoints/best_model.pt",
        "../checkpoints_8/best_model.pt",
        "best_model.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = build_drift_8()
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    print(f"[load] Epoch {checkpoint['epoch']}, val_loss={checkpoint.get('val_loss', 'N/A')}")
    return model


def generate_test_batch(n_samples=32, seq_len=90, device="cuda", dtype=torch.bfloat16):
    X = torch.randn(n_samples, seq_len, 104, device=device, dtype=dtype) * 0.3 + 0.5
    return X


@torch.no_grad()
def diagnose_8(model, device):
    """Run comprehensive 8-agent diagnostic."""
    
    dtype = torch.bfloat16
    X = generate_test_batch(64, device=device, dtype=dtype)
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        outputs = model(X, return_weights=True)
    
    print("\n" + "="*70)
    print("8-AGENT DRIFT DIAGNOSTIC")
    print("="*70)
    
    # ═══ 1. TIER 1 ROUTER ═══
    print("\n[1] TIER 1 ROUTER (Sparse Specialists)")
    t1_probs = outputs.tier1_router_probs.float()
    mean_probs = t1_probs.mean(dim=0).cpu().numpy()
    
    print(f"    Mean routing probs: {np.round(mean_probs, 4)}")
    print(f"    Uniform baseline:   [0.25, 0.25, 0.25, 0.25]")
    
    entropy = -(t1_probs * t1_probs.clamp(1e-8).log()).sum(-1).mean().item()
    max_ent = np.log(4)
    norm_ent = entropy / max_ent
    
    status = "✗ COLLAPSED" if norm_ent > 0.95 else "✓ DIVERSE" if norm_ent < 0.75 else "△ PARTIAL"
    print(f"    Entropy: {entropy:.4f}/{max_ent:.4f} = {norm_ent:.1%} [{status}]")
    
    # Argmax distribution
    argmax = t1_probs.argmax(dim=-1)
    counts = torch.bincount(argmax, minlength=4).float()
    print(f"    Argmax dist: {(counts/counts.sum()).cpu().numpy().round(4)}")
    
    # ═══ 2. TIER 2 ROUTER ═══
    print("\n[2] TIER 2 ROUTER (Token-Level FFN)")
    t2_probs = outputs.tier2_router_probs.float()
    t2_mean = t2_probs.mean([0,1]).cpu().numpy()
    
    print(f"    Mean routing probs: {np.round(t2_mean, 4)}")
    
    t2_ent = -(t2_probs * t2_probs.clamp(1e-8).log()).sum(-1).mean().item()
    t2_max = np.log(8)
    t2_norm = t2_ent / t2_max
    
    status = "✗ COLLAPSED" if t2_norm > 0.95 else "✓ DIVERSE" if t2_norm < 0.75 else "△ PARTIAL"
    print(f"    Entropy: {t2_ent:.4f}/{t2_max:.4f} = {t2_norm:.1%} [{status}]")
    
    # ═══ 3. AGENT PREDICTIONS ═══
    print("\n[3] AGENT PREDICTIONS")
    print("    Checking if agents produce different outputs...")
    
    preds = {}
    for name in ALL_AGENTS:
        if name in outputs.agents:
            pred = outputs.agents[name].p50[:, 0]  # 7-day horizon
            preds[name] = pred.float()
    
    if len(preds) >= 2:
        # Pairwise correlation
        print(f"\n    Pairwise correlations (P50, 7-day horizon):")
        names = list(preds.keys())
        
        high_corr_pairs = []
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if i < j:
                    corr = torch.corrcoef(torch.stack([preds[n1], preds[n2]]))[0,1].item()
                    status = "!" if abs(corr) > 0.9 else " " if abs(corr) < 0.5 else "~"
                    print(f"      {status} {n1:12} vs {n2:12}: r={corr:+.3f}")
                    if abs(corr) > 0.9:
                        high_corr_pairs.append((n1, n2, corr))
        
        if high_corr_pairs:
            print(f"\n    ⚠️  HIGH CORRELATION WARNING:")
            for n1, n2, r in high_corr_pairs:
                print(f"       {n1} vs {n2}: r={r:.3f} (predictions nearly identical!)")
        else:
            print(f"\n    ✓ All agent predictions are sufficiently different")
    
    # ═══ 4. SPECIALIST STATE ORTHOGONALITY ═══
    print("\n[4] SPECIALIST STATE ORTHOGONALITY")
    
    states = outputs.specialist_states
    core_states = {k: v.float() for k, v in states.items() if k in ALL_AGENTS}
    
    if len(core_states) >= 2:
        names = list(core_states.keys())
        
        # Similarity matrix
        print(f"\n    Cosine similarity matrix:")
        
        sims = []
        for i, n1 in enumerate(names):
            row = []
            for j, n2 in enumerate(names):
                s1 = core_states[n1].mean(0)
                s2 = core_states[n2].mean(0)
                sim = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
                row.append(sim)
                if i < j:
                    sims.append(sim)
            print(f"    {n1:12}: " + " ".join([f"{s:+.2f}" for s in row]))
        
        avg_sim = np.mean(sims)
        status = "✗ COLLAPSED" if avg_sim > 0.9 else "✓ DIVERSE" if avg_sim < 0.5 else "△ PARTIAL"
        print(f"\n    Average off-diagonal similarity: {avg_sim:.4f} [{status}]")
    
    # ═══ 5. VSN WEIGHTS ═══
    print("\n[5] VSN WEIGHT DIVERGENCE")
    
    if outputs.vsn_weights:
        vsn = {k: v.float() for k, v in outputs.vsn_weights.items() if v is not None}
        
        if len(vsn) >= 2:
            # Mean over sequence
            vsn_means = {k: v.mean(1) for k, v in vsn.items()}  # (B, n_vars)
            
            print(f"    Analyzing {len(vsn_means)} agent VSN patterns...")
            
            # Pairwise JSD
            names = list(vsn_means.keys())
            jsds = []
            
            for i, n1 in enumerate(names):
                for j, n2 in enumerate(names):
                    if i < j:
                        p = F.softmax(vsn_means[n1], dim=-1).clamp(1e-8)
                        q = F.softmax(vsn_means[n2], dim=-1).clamp(1e-8)
                        m = 0.5 * (p + q)
                        jsd = 0.5 * ((p * (p/m).log()).sum(-1) + (q * (q/m).log()).sum(-1)).mean().item()
                        jsds.append(jsd)
                        status = "!" if jsd < 0.01 else "✓" if jsd > 0.05 else "~"
                        print(f"      {status} {n1:12} vs {n2:12}: JSD={jsd:.4f}")
            
            avg_jsd = np.mean(jsds)
            status = "✗ COLLAPSED" if avg_jsd < 0.01 else "✓ DIVERSE" if avg_jsd > 0.05 else "△ PARTIAL"
            print(f"\n    Average JSD: {avg_jsd:.4f} [{status}]")
    else:
        print("    (VSN weights not available)")
    
    # ═══ 6. TOP FEATURES PER AGENT ═══
    print("\n[6] TOP FEATURES PER AGENT (VSN)")
    
    feature_names = ["hrv", "hr", "spo2", "sleep", "rem", "deep", 
                    "steps", "active", "screen", "aqi", "temp"]
    
    if outputs.vsn_weights:
        for agent_name, weights in outputs.vsn_weights.items():
            if weights is not None:
                w = weights.float().mean([0, 1]).cpu().numpy()  # Mean over batch and seq
                top3 = np.argsort(w)[-3:][::-1]
                top_names = [feature_names[i] if i < len(feature_names) else f"var_{i}" for i in top3]
                top_vals = [w[i] for i in top3]
                print(f"    {agent_name:12}: {', '.join([f'{n}={v:.3f}' for n,v in zip(top_names, top_vals)])}")
    
    # ═══ SUMMARY ═══
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    issues = []
    
    if norm_ent > 0.9:
        issues.append("Tier 1 router collapsed (uniform)")
    if t2_norm > 0.95:
        issues.append("Tier 2 router collapsed (uniform)")
    if 'avg_sim' in dir() and avg_sim > 0.85:
        issues.append("Specialist states too similar")
    if 'avg_jsd' in dir() and avg_jsd < 0.02:
        issues.append("VSN weights collapsed (identical)")
    if high_corr_pairs:
        issues.append(f"{len(high_corr_pairs)} agent pairs with r>0.9")
    
    if not issues:
        print("✓ Model shows healthy specialization!")
    else:
        print("⚠ ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n" + "="*70)


def main():
    print("="*70)
    print("8-AGENT DRIFT DIAGNOSTIC")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    ckpt = find_checkpoint()
    if not ckpt:
        print("ERROR: No checkpoint found!")
        return
    
    print(f"Checkpoint: {ckpt}")
    
    model = load_model(ckpt, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    diagnose_8(model, device)


if __name__ == "__main__":
    main()
