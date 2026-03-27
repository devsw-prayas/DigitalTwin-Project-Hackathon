"""
data/shard_writer.py
Orchestrates 500-persona pipeline → 50 sharded .npz files.

Each shard contains torch tensors saved via numpy bridge (.numpy()) only
at the final save step — the only place numpy is used in the entire pipeline.

Workers: 2 (safe for 6GB VRAM machines).
Windows: freeze_support() included.
"""

from __future__ import annotations
import json
import multiprocessing as mp
import os
import time
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.personas    import load_personas
from data.spawner     import spawn_all
from data.generator   import generate
from data.degradation import degrade_sample
from data.tokenizer   import tokenize

N_PERSONAS         = 10000   # matches N_TOTAL in spawner.py
N_SHARDS           = 1000
PERSONAS_PER_SHARD = N_PERSONAS // N_SHARDS  # 10
N_DAYS             = 1825
N_WORKERS          = 2

SIGNAL_KEYS = [
    "hrv_rmssd_ms", "resting_hr_bpm", "spo2_pct",
    "sleep_efficiency_pct", "rem_min", "deep_min",
    "steps", "active_min", "screen_off_min",
    "aqi_pm25", "ambient_temp_c",
]


# ─── Worker ────────────────────────────────────────────────────────────────────

def _process_shard(args: tuple) -> dict:
    shard_idx, personas_slice, output_dir = args

    tokens_list = []
    conf_list   = []
    raw_list    = []
    cardio_list = []
    mental_list = []
    meta_list   = []

    for persona in personas_slice:
        sample   = generate(persona, n_days=N_DAYS)
        degraded = degrade_sample(sample)
        tok      = tokenize(degraded, n_days=N_DAYS)

        tokens_list.append(tok.tokens)       # (730, 104) float16
        conf_list.append(tok.confidence)     # (730, 11)  float16  ← compact
        raw_list.append(tok.raw_signals)     # (730, 11)  float16
        cardio_list.append(tok.cardio_gt)    # (730,)     float32
        mental_list.append(tok.mental_gt)    # (730,)     float32

        meta_list.append({
            "name":        persona.name,
            "description": persona.description,
            "seed":        persona.seed,
            "archetype":   persona.name.split("_v")[0],
            "cardio_base": round(persona.cardio.base, 4),
            "mental_base": round(persona.mental.base, 4),
        })

        # Free GPU memory after each persona
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stack into shard dict — all pure torch tensors
    shard_data = {
        "tokens":      torch.stack(tokens_list, dim=0),   # (10, 730, 104) float16
        "confidence":  torch.stack(conf_list,   dim=0),   # (10, 730, 11)  float16
        "raw_signals": torch.stack(raw_list,    dim=0),   # (10, 730, 11)  float16
        "cardio_gt":   torch.stack(cardio_list, dim=0),   # (10, 730)      float32
        "mental_gt":   torch.stack(mental_list, dim=0),   # (10, 730)      float32
        "metadata":    meta_list,
    }

    path = Path(output_dir) / f"shard_{shard_idx:04d}.pt"
    torch.save(shard_data, str(path), _use_new_zipfile_serialization=False)

    # Print shapes on first shard to verify
    if shard_idx == 0:
        for k, v in shard_data.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape} {v.dtype}", flush=True)

    size_mb = path.stat().st_size / 1e6
    tok_std = shard_data["tokens"].float().std().item()
    print(f"  [shard {shard_idx:02d}]  {path.name}  {size_mb:.1f} MB  std={tok_std:.4f}", flush=True)

    return {"shard": shard_idx, "path": str(path), "size_mb": size_mb, "personas": meta_list}


# ─── Orchestrator ──────────────────────────────────────────────────────────────

def generate_dataset(output_dir: str = "outputs/shards", n_workers: int = N_WORKERS) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading base personas...")
    base = load_personas()

    print("Spawning 500 variants...")
    all_personas = spawn_all(base)

    shards = [
        all_personas[i * PERSONAS_PER_SHARD:(i + 1) * PERSONAS_PER_SHARD]
        for i in range(N_SHARDS)
    ]

    print(f"\nGenerating {N_SHARDS} shards × {PERSONAS_PER_SHARD} personas with {n_workers} workers...")
    print(f"Output: {out.resolve()}\n")

    t0   = time.time()
    args = [(i, shards[i], str(out)) for i in range(N_SHARDS)]

    if n_workers <= 1:
        results = [_process_shard(a) for a in args]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_process_shard, args)

    elapsed  = time.time() - t0
    total_mb = sum(r["size_mb"] for r in results)

    manifest = {
        "n_personas":        N_PERSONAS,
        "n_shards":          N_SHARDS,
        "n_days":            N_DAYS,
        "token_dim":         104,
        "total_size_mb":     round(total_mb, 1),
        "generation_time_s": round(elapsed, 1),
        "signal_keys":       SIGNAL_KEYS,
        "shards":            sorted(results, key=lambda r: r["shard"]),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'═' * 50}")
    print(f"  Shards:    {N_SHARDS}")
    print(f"  Personas:  {N_PERSONAS}")
    print(f"  Size:      {total_mb / 1024:.2f} GB  ({total_mb:.0f} MB)")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"{'═' * 50}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.freeze_support()  # required on Windows
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="outputs/shards")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--single",  action="store_true", help="Single-process debug mode")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.out,
        n_workers=1 if args.single else args.workers,
    )
