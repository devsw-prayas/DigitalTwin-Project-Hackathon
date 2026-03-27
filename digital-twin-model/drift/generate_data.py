"""
generate_data.py — Standalone 8-Agent Data Generator

Run: python generate_data.py

Generates samples from all 10 divergence personas and displays:
- Ground truth curves for all 8 agents
- Signal statistics
- Divergence scores
- Inter-agent correlation matrix

Options:
    --save        Save samples to .pt file
    --plot        Show plots (requires matplotlib)
    --n_days      Number of days (default: 365)
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.agents_8 import ALL_AGENTS, N_AGENTS, ALL_SIGNALS
from data.generator_8 import (
    generate_sample, generate_all_personas,
    get_divergence_personas, Persona8,
    compute_divergence_score,
)


def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_agent_gt_summary(sample):
    """Print summary of agent ground truths."""
    print("\nAgent Ground Truth Summary:")
    print(f"{'Agent':<15} {'Base':>8} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-"*55)
    
    for name in ALL_AGENTS:
        if name in sample.agent_gts:
            gt = sample.agent_gts[name]
            v = gt.values
            print(f"{name:<15} {gt.base:>8.3f} {v.mean():>8.3f} {v.min():>8.3f} {v.max():>8.3f} {v.std():>8.3f}")


def print_signal_summary(sample):
    """Print summary of generated signals."""
    print("\nSignal Summary:")
    print(f"{'Signal':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*60)
    
    for name in ALL_SIGNALS:
        if name in sample.signals:
            v = sample.signals[name]
            print(f"{name:<20} {v.mean():>10.2f} {v.std():>10.2f} {v.min():>10.2f} {v.max():>10.2f}")


def compute_correlation_matrix(samples):
    """Compute correlation matrix between agent GTs across all samples."""
    # Collect all agent values
    agent_values = {name: [] for name in ALL_AGENTS}
    
    for sample in samples.values():
        for name in ALL_AGENTS:
            if name in sample.agent_gts:
                agent_values[name].append(sample.agent_gts[name].values)
    
    # Stack and compute correlations
    corr_matrix = np.zeros((N_AGENTS, N_AGENTS))
    
    for i, name_i in enumerate(ALL_AGENTS):
        for j, name_j in enumerate(ALL_AGENTS):
            if len(agent_values[name_i]) > 0 and len(agent_values[name_j]) > 0:
                x = torch.cat(agent_values[name_i]).float()
                y = torch.cat(agent_values[name_j]).float()
                
                x_c = x - x.mean()
                y_c = y - y.mean()
                corr = (x_c * y_c).sum() / (x_c.norm() * y_c.norm() + 1e-8)
                corr_matrix[i, j] = corr.item()
    
    return corr_matrix


def print_correlation_matrix(corr_matrix):
    """Pretty print correlation matrix."""
    print("\nInter-Agent Correlation Matrix:")
    
    # Header
    header = "           " + " ".join([f"{n[:5]:>7}" for n in ALL_AGENTS])
    print(header)
    
    # Rows
    for i, name in enumerate(ALL_AGENTS):
        row = f"{name[:8]:<10}" + " ".join([f"{corr_matrix[i,j]:>7.2f}" for j in range(N_AGENTS)])
        print(row)


def plot_samples(samples, n_plot=3):
    """Plot ground truth curves for first n_plot samples."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib not installed, skipping plots")
        return
    
    persona_names = list(samples.keys())[:n_plot]
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 4*n_plot))
    if n_plot == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, N_AGENTS))
    
    for idx, name in enumerate(persona_names):
        sample = samples[name]
        ax = axes[idx]
        
        for i, agent_name in enumerate(ALL_AGENTS):
            if agent_name in sample.agent_gts:
                gt = sample.agent_gts[agent_name]
                ax.plot(gt.values.cpu().numpy(), label=agent_name, color=colors[i], alpha=0.8)
        
        ax.set_title(f"{name} (divergence={sample.divergence_score:.3f})")
        ax.set_xlabel("Day")
        ax.set_ylabel("Risk")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("generated_data_preview.png", dpi=150)
    print(f"\n[✓] Saved plot to: generated_data_preview.png")
    plt.show()


def save_samples(samples, output_path):
    """Save samples to torch file."""
    data = {}
    
    for name, sample in samples.items():
        data[name] = {
            "t": sample.t.cpu(),
            "signals": {k: v.cpu() for k, v in sample.signals.items()},
            "agent_gts": {
                k: {
                    "values": v.values.cpu(),
                    "base": v.base,
                    "drift_per_day": v.drift_per_day,
                    "events": v.events,
                }
                for k, v in sample.agent_gts.items()
            },
            "divergence_score": sample.divergence_score,
        }
    
    torch.save(data, output_path)
    print(f"\n[✓] Saved {len(samples)} samples to: {output_path}")
    
    # File size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"    File size: {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate 8-Agent Training Data")
    parser.add_argument("--save", action="store_true", help="Save to .pt file")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--n_days", type=int, default=365, help="Days per sample")
    parser.add_argument("--output", type=str, default="data_8agent.pt", help="Output file")
    args = parser.parse_args()
    
    print_header("8-AGENT DATA GENERATOR")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Days per sample: {args.n_days}")
    
    # Load personas
    print_header("LOADING PERSONAS")
    personas = get_divergence_personas()
    print(f"Loaded {len(personas)} divergence personas:\n")
    
    for name, p in personas.items():
        print(f"  {name:<25} | primary={p.primary_agent:<12} | type={p.divergence_type}")
    
    # Generate samples
    print_header("GENERATING SAMPLES")
    samples = {}
    
    for name, persona in personas.items():
        print(f"  Generating {name}...", end=" ", flush=True)
        sample = generate_sample(persona, n_days=args.n_days)
        samples[name] = sample
        print(f"divergence={sample.divergence_score:.3f}")
    
    # Summary statistics
    print_header("SAMPLE STATISTICS")
    
    # Show first sample in detail
    first_name = list(samples.keys())[0]
    first_sample = samples[first_name]
    
    print(f"\n--- First sample: {first_name} ---")
    print_agent_gt_summary(first_sample)
    print_signal_summary(first_sample)
    
    # Divergence scores
    print("\nDivergence Scores (higher = more independent agents):")
    div_scores = [(name, s.divergence_score) for name, s in samples.items()]
    div_scores.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in div_scores:
        bar = "█" * int(score * 50)
        print(f"  {name:<25} {score:.3f} {bar}")
    
    avg_div = np.mean([s[1] for s in div_scores])
    print(f"\n  Average divergence: {avg_div:.3f}")
    
    # Correlation matrix
    print_header("INTER-AGENT CORRELATIONS")
    corr_matrix = compute_correlation_matrix(samples)
    print_correlation_matrix(corr_matrix)
    
    # High correlation pairs
    high_corr = []
    for i in range(N_AGENTS):
        for j in range(i+1, N_AGENTS):
            if abs(corr_matrix[i,j]) > 0.7:
                high_corr.append((ALL_AGENTS[i], ALL_AGENTS[j], corr_matrix[i,j]))
    
    if high_corr:
        print("\n⚠️  High correlation pairs (|r| > 0.7):")
        for n1, n2, r in high_corr:
            print(f"    {n1} <-> {n2}: r={r:.3f}")
    else:
        print("\n✓ No high-correlation pairs (all agents sufficiently independent)")
    
    # Save
    if args.save:
        print_header("SAVING DATA")
        save_samples(samples, args.output)
    
    # Plot
    if args.plot:
        print_header("PLOTTING")
        plot_samples(samples)
    
    # Final summary
    print_header("GENERATION COMPLETE")
    print(f"  Samples generated: {len(samples)}")
    print(f"  Days per sample:   {args.n_days}")
    print(f"  Total data points: {len(samples) * args.n_days * N_AGENTS:,}")
    print(f"  Average divergence: {avg_div:.3f}")
    
    if not args.save:
        print(f"\n  Tip: Use --save to save to {args.output}")
    if not args.plot:
        print(f"  Tip: Use --plot to visualize ground truth curves")
    
    print("\n" + "="*70 + "\n")
    
    return samples


if __name__ == "__main__":
    samples = main()
