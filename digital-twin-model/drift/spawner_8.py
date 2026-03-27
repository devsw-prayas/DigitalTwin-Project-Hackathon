"""
spawner_8.py — Large-Scale 8-Agent Data Spawner

Generates massive datasets with:
- N personas (default: 10,000)
- 1825 days (5 years) per persona
- Sharded output for efficient loading
- Parallel generation

Usage:
    python spawner_8.py --n_personas 10000 --n_days 1825 --shards 50

Output:
    shard_000.pt, shard_001.pt, ... (each ~100MB)
    manifest.json (metadata)
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.agents_8 import ALL_AGENTS, N_AGENTS, ALL_SIGNALS
from data.generator_8 import (
    generate_sample, Persona8, AgentParams,
    AGENT_TEMPORAL, AGENT_SIGNAL_COUPLING,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERGENCE PATTERN TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-defined divergence patterns for variety
DIVERGENCE_PATTERNS = {
    # Pattern name: (divergence_type, primary_agent, drift_directions per agent category)
    "cardio_up": ("opposite", "cardio", {"cardio": 1, "mental": 0, "metabolic": 0.5, "recovery": 0.3, "immune": 0.2, "respiratory": 0.2, "hormonal": 0, "cog_fatigue": -0.3}),
    "mental_up": ("opposite", "mental", {"cardio": 0, "mental": 1, "metabolic": 0.2, "recovery": -0.2, "immune": 0.3, "respiratory": 0, "hormonal": 0.5, "cog_fatigue": 0.8}),
    "metabolic_up": ("independent", "metabolic", {"cardio": 0.3, "mental": 0.1, "metabolic": 1, "recovery": 0.4, "immune": 0, "respiratory": 0, "hormonal": 0, "cog_fatigue": 0.2}),
    "recovery_curves": ("mixed", "recovery", {"cardio": -0.8, "mental": 0.4, "metabolic": -0.5, "recovery": -1, "immune": -0.7, "respiratory": -0.3, "hormonal": 0, "cog_fatigue": 0.5}),
    "immune_storm": ("independent", "immune", {"cardio": 0.2, "mental": 0.3, "metabolic": 0, "recovery": 0.3, "immune": 1, "respiratory": 0.5, "hormonal": 0, "cog_fatigue": 0.4}),
    "respiratory_aqi": ("independent", "respiratory", {"cardio": 0.3, "mental": 0, "metabolic": 0, "recovery": 0, "immune": 0.2, "respiratory": 1, "hormonal": 0, "cog_fatigue": 0}),
    "hormonal_cycle": ("mixed", "hormonal", {"cardio": 0.1, "mental": 0.3, "metabolic": 0, "recovery": 0.2, "immune": 0, "respiratory": 0, "hormonal": 1, "cog_fatigue": 0.4}),
    "cog_burnout": ("opposite", "cog_fatigue", {"cardio": -0.2, "mental": 0.6, "metabolic": -0.1, "recovery": -0.3, "immune": 0.2, "respiratory": 0, "hormonal": 0.3, "cog_fatigue": 1}),
    "all_improving": ("coupled", "cardio", {"cardio": -1, "mental": -1, "metabolic": -1, "recovery": -1, "immune": -1, "respiratory": -1, "hormonal": -0.5, "cog_fatigue": -1}),
    "all_declining": ("coupled", "cardio", {"cardio": 1, "mental": 1, "metabolic": 1, "recovery": 1, "immune": 1, "respiratory": 1, "hormonal": 0.5, "cog_fatigue": 1}),
    "chaotic": ("independent", "mixed", None),  # Random per agent
    "bipolar_cardio": ("opposite", "cardio", {"cardio": 1, "mental": -0.8, "metabolic": 0.3, "recovery": -0.5, "immune": 0, "respiratory": 0, "hormonal": -0.2, "cog_fatigue": -0.6}),
    "bipolar_mental": ("opposite", "mental", {"cardio": -0.6, "mental": 1, "metabolic": 0, "recovery": -0.3, "immune": 0, "respiratory": 0, "hormonal": 0.4, "cog_fatigue": 0.8}),
    "post_viral": ("mixed", "recovery", {"cardio": -0.6, "mental": 0.5, "metabolic": -0.4, "recovery": -0.8, "immune": -0.9, "respiratory": -0.5, "hormonal": 0, "cog_fatigue": 0.3}),
    "executive_stress": ("mixed", "cog_fatigue", {"cardio": 0.3, "mental": 0.7, "metabolic": 0.5, "recovery": 0.4, "immune": 0.3, "respiratory": 0, "hormonal": 0.5, "cog_fatigue": 0.9}),
    "athlete_peak": ("coupled", "cardio", {"cardio": -0.8, "mental": -0.5, "metabolic": -0.7, "recovery": -0.6, "immune": -0.3, "respiratory": -0.2, "hormonal": -0.1, "cog_fatigue": -0.4}),
    "chronic_illness": ("coupled", "cardio", {"cardio": 0.6, "mental": 0.5, "metabolic": 0.7, "recovery": 0.6, "immune": 0.8, "respiratory": 0.5, "hormonal": 0.3, "cog_fatigue": 0.6}),
    "healthy_maintenance": ("independent", "cardio", {a: 0 for a in ALL_AGENTS}),
}

# Signal baseline ranges for variety
SIGNAL_BASELINES = {
    "hrv_rmssd_ms": (25, 95),
    "resting_hr_bpm": (45, 95),
    "spo2_pct": (95.5, 99.5),
    "sleep_efficiency_pct": (55, 95),
    "rem_min": (40, 120),
    "deep_min": (25, 105),
    "steps": (1500, 22000),
    "active_min": (5, 140),
    "screen_off_min": (50, 500),
    "aqi_pm25": (8, 55),
    "ambient_temp_c": (18, 26),
}

NOISE_LEVELS = {
    "hrv_rmssd_ms": (3, 10),
    "resting_hr_bpm": (1.5, 5),
    "spo2_pct": (0.2, 0.6),
    "sleep_efficiency_pct": (0.02, 0.1),
    "rem_min": (5, 15),
    "deep_min": (4, 12),
    "steps": (300, 2500),
    "active_min": (2, 18),
    "screen_off_min": (8, 65),
    "aqi_pm25": (2, 12),
    "ambient_temp_c": (1.5, 4),
}


def sample_from_range(ranges: Dict[str, tuple], key: str) -> float:
    """Sample a value from range."""
    low, high = ranges[key]
    return np.random.uniform(low, high)


def generate_random_persona(seed: int, pattern_name: Optional[str] = None) -> Persona8:
    """Generate a random persona with diverse properties."""
    np.random.seed(seed)
    
    # Select pattern (random if not specified)
    if pattern_name is None:
        pattern_name = np.random.choice(list(DIVERGENCE_PATTERNS.keys()))
    
    divergence_type, primary_agent, drifts = DIVERGENCE_PATTERNS[pattern_name]
    
    # Generate agent parameters
    agents = {}
    for agent_name in ALL_AGENTS:
        temporal = AGENT_TEMPORAL[agent_name]
        
        # Base risk (0.05 to 0.8)
        base = np.random.uniform(0.05, 0.8)
        
        # Drift direction from pattern or random
        if drifts is not None and agent_name in drifts:
            drift_dir = drifts[agent_name]
        else:
            drift_dir = np.random.uniform(-1, 1)
        
        # Drift magnitude (scaled by temporal timescale)
        drift_mag = np.random.uniform(0.00005, 0.0003) * abs(drift_dir)
        drift_per_day = drift_mag * (1 if drift_dir >= 0 else -1)
        
        # Event (30% chance)
        has_event = np.random.random() < 0.3
        event_day = np.random.randint(50, 1700) if has_event else -1
        event_magnitude = np.random.uniform(0.08, 0.25) * (1 if np.random.random() > 0.3 else -1)
        event_width = np.random.uniform(10, 40)
        
        # Seasonal amplitude
        seasonal = np.random.uniform(0.01, 0.06)
        
        agents[agent_name] = AgentParams(
            base=base,
            drift_per_day=drift_per_day,
            event_day=event_day,
            event_magnitude=event_magnitude,
            event_width_days=event_width,
            seasonal_amplitude=seasonal,
        )
    
    # Generate signals
    signals = {name: sample_from_range(SIGNAL_BASELINES, name) for name in ALL_SIGNALS}
    
    # Generate noise levels
    noise = {name: sample_from_range(NOISE_LEVELS, name) for name in ALL_SIGNALS}
    
    # Special parameters
    special = {}
    if primary_agent == "respiratory":
        special["aqi_spike_count"] = np.random.randint(2, 8)
        special["aqi_spike_magnitude"] = np.random.uniform(80, 200)
    if primary_agent == "immune":
        special["flare_count"] = np.random.randint(2, 10)
        special["flare_magnitude"] = np.random.uniform(0.1, 0.25)
    if primary_agent == "recovery":
        special["recovery_days"] = np.random.randint(90, 365)
    if primary_agent == "hormonal":
        special["cycle_period_days"] = np.random.randint(24, 35)
    
    # Random travel for some
    if np.random.random() < 0.3:
        special["travel_count"] = np.random.randint(1, 8)
    
    return Persona8(
        name=f"persona_{seed:06d}",
        description=f"Auto-generated {pattern_name} persona",
        seed=seed,
        primary_agent=primary_agent,
        divergence_type=divergence_type,
        agents=agents,
        signals=signals,
        noise=noise,
        **special
    )


def generate_persona_batch(args):
    """Generate a batch of personas and samples (for parallel processing)."""
    start_idx, n_personas, n_days, pattern = args
    
    samples = []
    for i in range(n_personas):
        seed = start_idx + i
        persona = generate_random_persona(seed, pattern)
        sample = generate_sample(persona, n_days=n_days)
        samples.append((seed, sample))
    
    return samples


def sample_to_dict(sample) -> dict:
    """Convert sample to serializable dict."""
    return {
        "persona_name": sample.persona_name,
        "t": sample.t.cpu(),
        "signals": {k: v.cpu() for k, v in sample.signals.items()},
        "agent_gts": {
            name: {
                "values": gt.values.cpu(),
                "base": gt.base,
                "drift_per_day": gt.drift_per_day,
                "events": gt.events,
                "pattern": gt.primary_pattern,
            }
            for name, gt in sample.agent_gts.items()
        },
        "routing_targets": sample.routing_targets,
        "divergence_score": sample.divergence_score,
    }


def write_shard(samples: List, shard_path: str):
    """Write samples to a shard file."""
    data = [sample_to_dict(s) for _, s in samples]
    torch.save(data, shard_path)


def main():
    parser = argparse.ArgumentParser(description="Large-Scale 8-Agent Data Spawner")
    parser.add_argument("--n_personas", type=int, default=10000, help="Number of personas")
    parser.add_argument("--n_days", type=int, default=1825, help="Days per persona (default: 5 years)")
    parser.add_argument("--n_shards", type=int, default=50, help="Number of output shards")
    parser.add_argument("--output_dir", type=str, default="data_shards", help="Output directory")
    parser.add_argument("--n_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pattern", type=str, default=None, help="Force specific divergence pattern")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print(" 8-AGENT DATA SPAWNER")
    print("="*70)
    print(f" Personas:      {args.n_personas:,}")
    print(f" Days/persona:  {args.n_days:,}")
    print(f" Shards:        {args.n_shards}")
    print(f" Workers:       {args.n_workers}")
    print(f" Output:        {args.output_dir}/")
    print("="*70)
    
    # Calculate samples per shard
    samples_per_shard = args.n_personas // args.n_shards
    remaining = args.n_personas % args.n_shards
    
    # Manifest
    manifest = {
        "n_personas": args.n_personas,
        "n_days": args.n_days,
        "n_shards": args.n_shards,
        "agents": ALL_AGENTS,
        "signals": ALL_SIGNALS,
        "created": datetime.now().isoformat(),
        "shards": [],
    }
    
    # Generate in parallel batches
    start_time = time.time()
    generated = 0
    
    print(f"\nGenerating {args.n_personas:,} samples...")
    
    with tqdm(total=args.n_personas, unit="samples") as pbar:
        for shard_idx in range(args.n_shards):
            n_this_shard = samples_per_shard + (1 if shard_idx < remaining else 0)
            
            # Generate samples for this shard
            shard_samples = []
            
            if args.n_workers > 1:
                # Parallel generation
                batch_size = n_this_shard // args.n_workers
                tasks = [
                    (generated + i * batch_size, batch_size, args.n_days, args.pattern)
                    for i in range(args.n_workers)
                ]
                # Last batch gets remainder
                tasks[-1] = (generated + (args.n_workers - 1) * batch_size, 
                            n_this_shard - (args.n_workers - 1) * batch_size,
                            args.n_days, args.pattern)
                
                with mp.Pool(args.n_workers) as pool:
                    results = pool.map(generate_persona_batch, tasks)
                
                for batch in results:
                    shard_samples.extend(batch)
            else:
                # Sequential
                for i in range(n_this_shard):
                    seed = generated + i
                    persona = generate_random_persona(seed, args.pattern)
                    sample = generate_sample(persona, n_days=args.n_days)
                    shard_samples.append((seed, sample))
            
            # Write shard
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:04d}.pt")
            write_shard(shard_samples, shard_path)
            
            # Update manifest
            manifest["shards"].append({
                "path": f"shard_{shard_idx:04d}.pt",
                "n_samples": len(shard_samples),
                "size_mb": os.path.getsize(shard_path) / (1024 * 1024),
            })
            
            generated += n_this_shard
            pbar.update(n_this_shard)
    
    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    elapsed = time.time() - start_time
    total_size = sum(s["size_mb"] for s in manifest["shards"])
    
    print("\n" + "="*70)
    print(" GENERATION COMPLETE")
    print("="*70)
    print(f" Total samples:  {generated:,}")
    print(f" Total size:     {total_size:.1f} MB")
    print(f" Time elapsed:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f" Rate:           {generated/elapsed:.0f} samples/sec")
    print(f" Output:         {args.output_dir}/")
    print("="*70)
    
    return manifest


if __name__ == "__main__":
    main()
