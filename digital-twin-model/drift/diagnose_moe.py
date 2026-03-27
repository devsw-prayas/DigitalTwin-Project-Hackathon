"""
diagnose_moe.py — Analyze MoE collapse in trained DRIFT model.

Run after training to diagnose:
    1. Router collapse (uniform vs specialized routing)
    2. Expert similarity (are experts producing identical outputs?)
    3. VSN weight divergence (cardio vs mental attending to same features?)
    4. Per-expert task performance (can all experts solve both tasks?)

Usage:
    python diagnose_moe.py --checkpoint checkpoints/best_model.pt --shard data.shard
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift.model.model import DRIFT, CORE_SPECIALISTS
from drift.data.loader import make_loaders


def load_model(checkpoint_path: str, device: str = "cuda") -> DRIFT:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    print(f"[load] Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    return model


@torch.no_grad()
def analyze_router_distribution(model: DRIFT, loader, device: str, n_batches: int = 20):
    """
    Analyze router behavior.
    
    Healthy MoE: routers show non-uniform distribution, some experts preferred.
    Collapsed MoE: routers are uniform (~0.25 each for 4 experts).
    """
    model.eval()
    
    tier1_probs_all = []
    tier2_probs_all = []
    
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        
        outputs = model(xb, return_weights=True)
        tier1_probs_all.append(outputs.tier1_router_probs.cpu())
        tier2_probs_all.append(outputs.tier2_router_probs.cpu())
    
    # Tier 1: (B*n_batches, n_sparse)
    tier1_probs = torch.cat(tier1_probs_all, dim=0)  # (N, 4)
    tier2_probs = torch.cat(tier2_probs_all, dim=0)  # (N, seq_len, 8)
    
    print("\n" + "="*60)
    print("ROUTER DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Tier 1 analysis
    print("\n[Tier 1 - Sparse Specialist Router]")
    print(f"  Mean routing probs: {tier1_probs.mean(dim=0).numpy()}")
    print(f"  Std routing probs:  {tier1_probs.std(dim=0).numpy()}")
    
    # Entropy (uniform = log(n), specialized = lower)
    entropy = -(tier1_probs * tier1_probs.clamp(1e-8).log()).sum(dim=-1).mean()
    max_entropy = np.log(tier1_probs.size(-1))
    print(f"  Mean entropy: {entropy:.4f} (uniform={max_entropy:.4f}, specialized < {max_entropy:.2f})")
    
    # How often is each expert the argmax?
    argmax_counts = tier1_probs.argmax(dim=-1).bincount(minlength=tier1_probs.size(-1))
    argmax_pct = argmax_counts.float() / argmax_counts.sum()
    print(f"  Argmax distribution: {argmax_pct.numpy()}")
    
    # Tier 2 analysis
    print("\n[Tier 2 - Token-Level FFN Router]")
    tier2_mean = tier2_probs.mean(dim=[0, 1])  # Average over batch and tokens
    print(f"  Mean routing probs: {tier2_mean.numpy()}")
    print(f"  Std routing probs:  {tier2_probs.std(dim=[0,1]).numpy()}")
    
    tier2_entropy = -(tier2_probs * tier2_probs.clamp(1e-8).log()).sum(dim=-1).mean()
    max_entropy_t2 = np.log(tier2_probs.size(-1))
    print(f"  Mean entropy: {tier2_entropy:.4f} (uniform={max_entropy_t2:.4f})")
    
    t2_argmax = tier2_probs.argmax(dim=-1).flatten()
    t2_argmax_counts = t2_argmax.bincount(minlength=tier2_probs.size(-1))
    t2_argmax_pct = t2_argmax_counts.float() / t2_argmax_counts.sum()
    print(f"  Argmax distribution: {t2_argmax_pct.numpy()}")
    
    return {
        "tier1_probs": tier1_probs,
        "tier2_probs": tier2_probs,
        "tier1_entropy": entropy.item(),
        "tier2_entropy": tier2_entropy.item(),
    }


@torch.no_grad()
def analyze_expert_similarity(model: DRIFT, loader, device: str, n_batches: int = 10):
    """
    Analyze if experts are producing similar outputs.
    
    Collapsed MoE: all expert outputs are highly correlated.
    Healthy MoE: expert outputs are diverse.
    """
    model.eval()
    
    # Collect expert outputs
    expert_outputs = defaultdict(list)  # expert_name -> list of outputs
    
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        
        # We need to hook into the model to get individual expert outputs
        # The specialist_states dict has individual expert outputs
        outputs = model(xb, return_weights=True)
        
        # Get specialist states from the forward pass
        # We'll need to re-run with hooks
        pass
    
    # Re-run with hooks to capture expert outputs
    specialist_outputs = {"cardio": [], "mental": [], "metabolic": [], "recovery": [],
                         "immune": [], "respiratory": [], "hormonal": [], "cog_fatigue": []}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                specialist_outputs[name].append(output[0].detach().cpu())
            else:
                specialist_outputs[name].append(output.detach().cpu())
        return hook
    
    # Register hooks on each specialist LSTM
    hooks = []
    for name in model.tier1_moe.core_specialists:
        h = model.tier1_moe.core_specialists[name].register_forward_hook(hook_fn(name))
        hooks.append(h)
    for name in model.tier1_moe.sparse_specialists:
        h = model.tier1_moe.sparse_specialists[name].register_forward_hook(hook_fn(name))
        hooks.append(h)
    
    # Run forward passes
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        _ = model(xb, return_weights=False)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    print("\n" + "="*60)
    print("EXPERT OUTPUT SIMILARITY ANALYSIS")
    print("="*60)
    
    # Stack and compute similarity
    results = {}
    
    for name, outputs in specialist_outputs.items():
        if len(outputs) > 0:
            results[name] = torch.cat(outputs, dim=0)  # (N, d_model)
    
    # Compute pairwise cosine similarity between experts
    expert_names = list(results.keys())
    n_experts = len(expert_names)
    
    print("\n[Cosine Similarity Between Expert Outputs]")
    print(f"  (Higher = more similar, 1.0 = identical)")
    
    similarity_matrix = torch.zeros(n_experts, n_experts)
    
    for i, name_i in enumerate(expert_names):
        for j, name_j in enumerate(expert_names):
            if i <= j:
                # Mean over batch, then compute similarity
                mean_i = results[name_i].mean(dim=0)
                mean_j = results[name_j].mean(dim=0)
                
                cos_sim = F.cosine_similarity(
                    mean_i.unsqueeze(0), 
                    mean_j.unsqueeze(0)
                ).item()
                
                similarity_matrix[i, j] = cos_sim
                similarity_matrix[j, i] = cos_sim
    
    # Print matrix
    print("\n  Expert Output Cosine Similarity Matrix:")
    header = "  " + "".join([f"{n[:6]:>8}" for n in expert_names])
    print(header)
    for i, name in enumerate(expert_names):
        row = f"  {name[:6]:<6}" + "".join([f"{similarity_matrix[i,j]:>8.3f}" for j in range(n_experts)])
        print(row)
    
    # Average off-diagonal similarity
    off_diag = similarity_matrix.flatten()[:-1].view(n_experts-1, n_experts+1)[:, 1:].flatten()
    avg_similarity = off_diag.mean().item()
    
    print(f"\n  Average off-diagonal similarity: {avg_similarity:.4f}")
    print(f"  (0.0 = orthogonal, 1.0 = identical)")
    
    if avg_similarity > 0.9:
        print("  ⚠️  HIGH SIMILARITY — Experts are collapsed!")
    elif avg_similarity > 0.7:
        print("  ⚠️  MODERATE SIMILARITY — Partial collapse")
    else:
        print("  ✓ LOW SIMILARITY — Experts are diverse")
    
    return {
        "similarity_matrix": similarity_matrix,
        "expert_names": expert_names,
        "avg_similarity": avg_similarity,
    }


@torch.no_grad()
def analyze_vsn_divergence(model: DRIFT, loader, device: str, n_batches: int = 20):
    """
    Analyze VSN weight divergence between cardio and mental agents.
    
    Healthy: cardio VSN weights differ from mental VSN weights.
    Collapsed: both agents attend to the same features.
    """
    model.eval()
    
    cardio_weights = []
    mental_weights = []
    
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        
        outputs = model(xb, return_weights=True)
        
        if outputs.vsn_weights.get("cardio") is not None:
            cardio_weights.append(outputs.vsn_weights["cardio"].cpu())
        if outputs.vsn_weights.get("mental") is not None:
            mental_weights.append(outputs.vsn_weights["mental"].cpu())
    
    print("\n" + "="*60)
    print("VSN WEIGHT DIVERGENCE ANALYSIS")
    print("="*60)
    
    if len(cardio_weights) == 0 or len(mental_weights) == 0:
        print("  No VSN weights collected (return_weights may not be working)")
        return None
    
    cardio_weights = torch.cat(cardio_weights, dim=0)  # (N, seq, n_vars)
    mental_weights = torch.cat(mental_weights, dim=0)  # (N, seq, n_vars)
    
    # Average over sequence dimension
    cardio_mean = cardio_weights.mean(dim=1)  # (N, n_vars)
    mental_mean = mental_weights.mean(dim=1)  # (N, n_vars)
    
    # Jensen-Shannon Divergence
    def js_divergence(p, q, eps=1e-8):
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum(dim=-1)
        kl_qm = (q * (q / m).log()).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)
    
    jsd = js_divergence(cardio_mean, mental_mean)  # (N,)
    mean_jsd = jsd.mean().item()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(cardio_mean, mental_mean, dim=-1).mean().item()
    
    # Feature-level analysis
    cardio_feature_importance = cardio_mean.mean(dim=0)  # (n_vars,)
    mental_feature_importance = mental_mean.mean(dim=0)  # (n_vars,)
    
    print(f"\n  Mean JSD (cardio vs mental VSN): {mean_jsd:.6f}")
    print(f"  Mean Cosine Similarity:          {cos_sim:.4f}")
    
    print(f"\n  (JSD: 0 = identical, 0.5 = orthogonal, 0.693 = max diverse)")
    
    if mean_jsd < 0.01:
        print("  ⚠️  VERY LOW JSD — Agents attend to identical features!")
    elif mean_jsd < 0.05:
        print("  ⚠️  LOW JSD — Agents have similar attention patterns")
    else:
        print("  ✓ MODERATE JSD — Agents have different attention patterns")
    
    # Top features per agent
    feature_names = ["hrv", "hr", "spo2", "sleep_eff", "rem", "deep", 
                     "steps", "active", "screen", "aqi", "temp",
                     "var_12", "var_13", "var_14", "var_15", "var_16",
                     "var_17", "var_18", "var_19", "var_20"]
    
    print(f"\n  Top 5 features for CARDIO agent:")
    _, top_idx = cardio_feature_importance.topk(5)
    for idx in top_idx:
        print(f"    {feature_names[idx] if idx < len(feature_names) else f'var_{idx}'}: {cardio_feature_importance[idx]:.4f}")
    
    print(f"\n  Top 5 features for MENTAL agent:")
    _, top_idx = mental_feature_importance.topk(5)
    for idx in top_idx:
        print(f"    {feature_names[idx] if idx < len(feature_names) else f'var_{idx}'}: {mental_feature_importance[idx]:.4f}")
    
    return {
        "jsd": mean_jsd,
        "cosine_similarity": cos_sim,
        "cardio_feature_importance": cardio_feature_importance,
        "mental_feature_importance": mental_feature_importance,
    }


@torch.no_grad()
def analyze_expert_task_performance(model: DRIFT, loader, device: str, n_batches: int = 30):
    """
    Test if individual experts can solve both tasks.
    
    If every expert can predict both cardio and mental well,
    there's no pressure to specialize.
    """
    model.eval()
    
    # We'll test each core specialist's predictions
    cardio_preds = defaultdict(list)
    mental_preds = defaultdict(list)
    cardio_gt = []
    mental_gt = []
    
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        yb = yb.to(device)
        
        outputs = model(xb, return_weights=False)
        
        # Store predictions per agent
        for agent_name in ["cardio", "mental"]:
            if agent_name in outputs.agents:
                pred = outputs.agents[agent_name].p50  # (B, n_horizons)
                
                if agent_name == "cardio":
                    cardio_preds["combined"].append(pred[:, 0].cpu())  # 7-day horizon
                else:
                    mental_preds["combined"].append(pred[:, 0].cpu())
        
        # Ground truth
        cardio_gt.append(yb[:, 0, 0].cpu())  # cardio, 7-day horizon
        mental_gt.append(yb[:, 1, 0].cpu())  # mental, 7-day horizon
    
    print("\n" + "="*60)
    print("PREDICTION PERFORMANCE ANALYSIS")
    print("="*60)
    
    cardio_gt = torch.cat(cardio_gt)
    mental_gt = torch.cat(mental_gt)
    
    # Combined predictions
    cardio_combined = torch.cat(cardio_preds["combined"])
    mental_combined = torch.cat(mental_preds["combined"])
    
    # MSE and correlation
    cardio_mse = F.mse_loss(cardio_combined, cardio_gt.float()).item()
    mental_mse = F.mse_loss(mental_combined, mental_gt.float()).item()
    
    cardio_corr = torch.corrcoef(torch.stack([cardio_combined.flatten(), cardio_gt.flatten()]))[0, 1].item()
    mental_corr = torch.corrcoef(torch.stack([mental_combined.flatten(), mental_gt.flatten()]))[0, 1].item()
    
    print(f"\n[Combined Model Performance]")
    print(f"  Cardio MSE: {cardio_mse:.6f}, Correlation: {cardio_corr:.4f}")
    print(f"  Mental MSE: {mental_mse:.6f}, Correlation: {mental_corr:.4f}")
    
    # Cross-task correlation (are cardio and mental predictions correlated?)
    cross_corr = torch.corrcoef(torch.stack([cardio_combined.flatten(), mental_combined.flatten()]))[0, 1].item()
    print(f"\n  Cross-task correlation (cardio_pred vs mental_pred): {cross_corr:.4f}")
    
    # Are ground truths correlated?
    gt_corr = torch.corrcoef(torch.stack([cardio_gt.float().flatten(), mental_gt.float().flatten()]))[0, 1].item()
    print(f"  Ground truth correlation (cardio_gt vs mental_gt): {gt_corr:.4f}")
    
    if abs(cross_corr - gt_corr) < 0.1:
        print("  ℹ️  Predictions mirror ground truth correlation — expected behavior")
    elif cross_corr > gt_corr + 0.15:
        print("  ⚠️  Predictions MORE correlated than ground truth — potential collapse!")
    
    return {
        "cardio_mse": cardio_mse,
        "mental_mse": mental_mse,
        "cardio_corr": cardio_corr,
        "mental_corr": mental_corr,
        "cross_pred_corr": cross_corr,
        "gt_corr": gt_corr,
    }


@torch.no_grad()
def analyze_routing_vs_task(model: DRIFT, loader, device: str, n_batches: int = 30):
    """
    Analyze whether routing decisions correlate with task type.
    
    Healthy: different routing for cardio-heavy vs mental-heavy samples.
    Collapsed: same routing regardless of sample type.
    """
    model.eval()
    
    all_router_probs = []
    cardio_values = []
    mental_values = []
    
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        
        outputs = model(xb, return_weights=False)
        all_router_probs.append(outputs.tier1_router_probs.cpu())
        
        # Mean ground truth for this sample
        cardio_values.append(yb[:, 0, :].mean(dim=-1).cpu())  # avg over horizons
        mental_values.append(yb[:, 1, :].mean(dim=-1).cpu())
    
    router_probs = torch.cat(all_router_probs, dim=0)  # (N, n_sparse)
    cardio_values = torch.cat(cardio_values, dim=0)    # (N,)
    mental_values = torch.cat(mental_values, dim=0)    # (N,)
    
    print("\n" + "="*60)
    print("ROUTING VS TASK CORRELATION")
    print("="*60)
    
    # Correlation between each expert's routing prob and cardio/mental gt
    n_experts = router_probs.size(-1)
    
    print("\n[Correlation: Expert Routing vs Ground Truth]")
    print(f"  {'Expert':<12} {'Cardio r':>12} {'Mental r':>12}")
    print("  " + "-"*38)
    
    for i in range(n_experts):
        cardio_r = torch.corrcoef(torch.stack([router_probs[:, i], cardio_values]))[0, 1].item()
        mental_r = torch.corrcoef(torch.stack([router_probs[:, i], mental_values]))[0, 1].item()
        print(f"  {f'expert_{i}':<12} {cardio_r:>12.4f} {mental_r:>12.4f}")
    
    # Routing entropy vs task dominance
    entropy = -(router_probs * router_probs.clamp(1e-8).log()).sum(dim=-1)
    task_diff = (cardio_values - mental_values).abs()
    
    # In a healthy MoE: higher task difference → lower entropy (more confident routing)
    entropy_task_corr = torch.corrcoef(torch.stack([entropy, task_diff]))[0, 1].item()
    
    print(f"\n  Routing entropy vs |cardio - mental| correlation: {entropy_task_corr:.4f}")
    print(f"  (Negative = model routes more confidently when one task dominates)")
    
    if entropy_task_corr > 0.1:
        print("  ⚠️  POSITIVE correlation — routing doesn't adapt to task!")
    elif entropy_task_corr < -0.1:
        print("  ✓ NEGATIVE correlation — routing adapts to task type")
    else:
        print("  ℹ️  Near-zero correlation — routing independent of task")
    
    return {
        "entropy_task_corr": entropy_task_corr,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose MoE collapse in DRIFT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--shard", type=str, required=True, help="Path to data shard")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--n_batches", type=int, default=30, help="Batches to analyze")
    args = parser.parse_args()
    
    print("="*60)
    print("DRIFT MoE COLLAPSE DIAGNOSTIC")
    print("="*60)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Create loader
    train_loader, val_loader = make_loaders(
        shard_path=args.shard,
        phase=1,
        batch_size=16,
        agent_names=["cardio", "mental"],
        pin_memory=(args.device == "cuda"),
        num_workers=0,
    )
    
    # Run diagnostics
    results = {}
    
    # 1. Router distribution
    results["router"] = analyze_router_distribution(model, val_loader, args.device, args.n_batches)
    
    # 2. Expert similarity
    results["expert_similarity"] = analyze_expert_similarity(model, val_loader, args.device, args.n_batches // 2)
    
    # 3. VSN divergence
    results["vsn"] = analyze_vsn_divergence(model, val_loader, args.device, args.n_batches)
    
    # 4. Task performance
    results["performance"] = analyze_expert_task_performance(model, val_loader, args.device, args.n_batches)
    
    # 5. Routing vs task
    results["routing_task"] = analyze_routing_vs_task(model, val_loader, args.device, args.n_batches)
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    collapse_indicators = []
    
    # Check router entropy
    tier1_ent = results["router"]["tier1_entropy"]
    max_ent = np.log(4)  # 4 sparse experts
    if tier1_ent > max_ent * 0.9:
        collapse_indicators.append("Tier1 router near-uniform (collapsed)")
    
    # Check expert similarity
    if results["expert_similarity"]["avg_similarity"] > 0.85:
        collapse_indicators.append("Expert outputs highly similar (collapsed)")
    
    # Check VSN JSD
    if results["vsn"] and results["vsn"]["jsd"] < 0.02:
        collapse_indicators.append("VSN weights nearly identical (collapsed)")
    
    # Check routing-task correlation
    if results["routing_task"]["entropy_task_corr"] > 0.1:
        collapse_indicators.append("Routing doesn't adapt to task type")
    
    print()
    if len(collapse_indicators) == 0:
        print("✓ No strong indicators of collapse detected")
    else:
        print("⚠️  COLLAPSE INDICATORS:")
        for indicator in collapse_indicators:
            print(f"   - {indicator}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
