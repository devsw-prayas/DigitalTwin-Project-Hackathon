"""
loader.py — Data Loading, Dataset, and Curriculum Sampling

Handles loading synthetic shards, building windowed datasets, and
serving curriculum-aware data loaders for the 5-phase training schedule.

Bug fixes vs original:
    - `from sympy import false` removed — use Python's built-in False
    - `weights_only=false` -> `weights_only=False`
    - DataLoader `num_workers` made configurable (0 on Windows, 2+ on Linux)

Curriculum phases (from spec §5.5):
    Phase 0 (epochs  0-15):  Pre-train each specialist independently on domain subsets
    Phase 1 (epochs 15-35):  Clean joint training — all personas, no degradation
    Phase 2 (epochs 35-55):  Noisy — 10% dropout on signals
    Phase 3 (epochs 55-75):  Degraded — 30% dropout, high missing rates
    Phase 4 (epochs 75-90):  Edge cases — rare events, illness, travel
    Phase 5 (epochs 90-100): Full distribution — generalization + LB verification

Each phase needs a different DataLoader configuration:
    - Different degradation probability applied to tokens at load time
    - Phase 0: persona-specific subsets per specialist
    - Phases 1-5: full dataset with increasing degradation

Multi-shard loading:
    The generator may produce multiple shard files (shard_0000.pt, shard_0001.pt, ...)
    This loader handles both single-shard (dev) and multi-shard (full training).
"""

import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from typing import Optional

# Default prediction horizons
HORIZONS = [7, 30, 90, 180]
WINDOW = 180  # 90-day observation window


# ─────────────────────────────────────────────────────────────────────────────
# Shard loading
# ─────────────────────────────────────────────────────────────────────────────

def load_shard(path: str) -> dict:
    """
    Load a single .pt shard file.

    Expected keys:
        tokens:    (P, T, 104)  tokenized health signals
        cardio_gt: (P, T)       ground truth cardio risk
        mental_gt: (P, T)       ground truth mental risk

    Optional keys (present in full generator output):
        metabolic_gt:  (P, T)
        recovery_gt:   (P, T)
        persona_ids:   list of persona name strings
        baseline_slow: (P, T, n_signals)  EMA slow baselines (if pre-computed)
        baseline_fast: (P, T, n_signals)  EMA fast baselines
    """
    data = torch.load(path, weights_only=False)
    return data


def load_all_shards(shard_dir: str, pattern: str = "shard_*.pt") -> dict:
    """
    Load and concatenate all shards from a directory.

    Args:
        shard_dir: directory containing shard files
        pattern:   glob pattern for shard files

    Returns:
        Merged dict with all tensors concatenated along persona dimension (dim 0)
    """
    paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No shards found at {shard_dir}/{pattern}")

    shards = [load_shard(p) for p in paths]
    print(f"[loader] Loaded {len(shards)} shard(s) from {shard_dir}")

    # Concatenate along persona dimension
    merged = {}
    for key in shards[0].keys():
        if isinstance(shards[0][key], torch.Tensor):
            merged[key] = torch.cat([s[key] for s in shards], dim=0)
        elif isinstance(shards[0][key], list):
            merged[key] = sum([s[key] for s in shards], [])
        else:
            merged[key] = shards[0][key]  # take first shard's value for metadata

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Degradation augmentation (curriculum phases 2-4)
# ─────────────────────────────────────────────────────────────────────────────

def apply_degradation(
    x: torch.Tensor,   # (N, 104) single sample
    dropout_prob: float = 0.0,
    missing_prob: float = 0.0,
) -> torch.Tensor:
    """
    Apply signal degradation for curriculum training phases.

    Phase 2: 10% dropout (random feature zeroing per timestep)
    Phase 3: 30% dropout + high missing rate (contiguous blocks)
    Phase 4: varied (edge cases handled by persona selection)

    Args:
        x:            (N, 104)  token sequence
        dropout_prob: probability of zeroing any individual feature-timestep
        missing_prob: probability of zeroing an entire timestep (simulates device dropout)

    Returns:
        degraded x with same shape
    """
    if dropout_prob > 0:
        # Per-feature dropout: random mask over (N, 104)
        mask = torch.bernoulli(torch.full_like(x, 1.0 - dropout_prob))
        x = x * mask

    if missing_prob > 0:
        # Contiguous missing block: pick a random start, zero out 3-14 days
        N = x.shape[0]
        if torch.rand(1).item() < missing_prob:
            block_len = torch.randint(3, 15, (1,)).item()
            start = torch.randint(0, max(1, N - block_len), (1,)).item()
            x = x.clone()
            x[start:start + block_len] = 0.0  # missing -> zero (confidence=0 encoding)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HealthDataset(Dataset):
    """
    Sliding window dataset over persona timeseries.

    From each persona's timeseries, we extract overlapping 90-day windows
    with multi-horizon targets at {7, 30, 90, 180} days after the window end.

    For each window starting at t:
        X: tokens[p, t : t+90]         shape (90, 104)
        y: risk values at t+89+h        shape (n_agents, 4)  for h in HORIZONS

    Args:
        tokens:        (P, T, 104)  all persona token sequences
        ground_truths: dict of {agent_name: (P, T)} risk curves
                       e.g. {"cardio": ..., "mental": ..., "metabolic": ..., "recovery": ...}
        horizons:      list of future horizons in days
        window:        observation window length in days
        dropout_prob:  online degradation for curriculum (set per-phase by loader)
        missing_prob:  online missing data injection
        persona_filter: optional list of persona indices to include (Phase 0 subsets)
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        ground_truths: dict[str, torch.Tensor],
        horizons: list[int] = HORIZONS,
        window: int = WINDOW,
        dropout_prob: float = 0.0,
        missing_prob: float = 0.0,
        persona_filter: Optional[list[int]] = None,
    ):
        self.horizons = horizons
        self.window = window
        self.dropout_prob = dropout_prob
        self.missing_prob = missing_prob

        P, T, D = tokens.shape
        max_h = max(horizons)

        # Apply persona filter for Phase 0 specialist pre-training
        if persona_filter is not None:
            tokens = tokens[persona_filter]
            ground_truths = {k: v[persona_filter] for k, v in ground_truths.items()}
            P = len(persona_filter)

        self.agent_names = list(ground_truths.keys())
        n_agents = len(self.agent_names)

        # Build all windows upfront and store as tensors (memory-efficient for <10GB)
        X_list, y_list = [], []

        for p in range(P):
            for t in range(0, T - window - max_h):
                x = tokens[p, t : t + window]  # (90, 104)

                anchor = t + window - 1  # last observed timestep

                # Targets: (n_agents, n_horizons)
                targets = torch.stack([
                    torch.stack([ground_truths[name][p, anchor + h] for h in horizons])
                    for name in self.agent_names
                ])  # (n_agents, n_horizons)

                X_list.append(x)
                y_list.append(targets)

        self.X = torch.stack(X_list).float()  # (N_samples, 90, 104)
        self.y = torch.stack(y_list).float()  # (N_samples, n_agents, 4)

        print(f"[dataset] {len(self.X)} samples | agents: {self.agent_names} | "
              f"horizons: {horizons}d | degradation: dropout={dropout_prob:.0%}")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (90, 104)
        y = self.y[idx]  # (n_agents, 4)

        # Online degradation (curriculum phases 2-4)
        if self.dropout_prob > 0 or self.missing_prob > 0:
            x = apply_degradation(x, self.dropout_prob, self.missing_prob)

        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum loader factory
# ─────────────────────────────────────────────────────────────────────────────

# Degradation settings per phase
PHASE_CONFIGS = {
    0: {"dropout_prob": 0.00, "missing_prob": 0.00},
    1: {"dropout_prob": 0.00, "missing_prob": 0.00},
    2: {"dropout_prob": 0.12, "missing_prob": 0.06},
    3: {"dropout_prob": 0.25, "missing_prob": 0.12},
    4: {"dropout_prob": 0.30, "missing_prob": 0.15},
    5: {"dropout_prob": 0.05, "missing_prob": 0.05},
}


def make_loaders(
    shard_path: str,
    phase: int = 1,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
    pin_memory: bool = True,
    agent_names: list[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train + val DataLoaders for a given curriculum phase.

    Args:
        shard_path:  path to shard file or directory of shard files
        phase:       curriculum phase (0-5)
        batch_size:  training batch size
        val_split:   fraction of data held out for validation
        num_workers: DataLoader workers (0 for Windows, 2+ for Linux/Mac)
        pin_memory:  pin to CUDA memory (set False if CPU-only)
        agent_names: which ground truth curves to include as targets
                     defaults to ["cardio", "mental"] for Phase 0

    Returns:
        train_loader, val_loader
    """
    if agent_names is None:
        agent_names = ["cardio", "mental"]  # minimal for Phase 0

    # Load data
    if os.path.isdir(shard_path):
        data = load_all_shards(shard_path)
    else:
        data = load_shard(shard_path)

    tokens = data["tokens"].float()

    # Build ground truth dict from available keys
    ground_truths = {}
    gt_key_map = {
        "cardio": "cardio_gt",
        "mental": "mental_gt",
        "metabolic": "metabolic_gt",
        "recovery": "recovery_gt",
    }
    for name in agent_names:
        key = gt_key_map.get(name, f"{name}_gt")
        if key in data:
            ground_truths[name] = data[key].float()
        else:
            print(f"[loader] Warning: {key} not found in shard, skipping {name}")

    if not ground_truths:
        raise ValueError("No ground truth curves found. Check shard keys.")

    cfg = PHASE_CONFIGS.get(phase, PHASE_CONFIGS[1])

    dataset = HealthDataset(
        tokens=tokens,
        ground_truths=ground_truths,
        dropout_prob=cfg["dropout_prob"],
        missing_prob=cfg["missing_prob"],
    )

    # Train/val split
    n = len(dataset)
    val_size = int(val_split * n)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[loader] Phase {phase} | train={train_size} | val={val_size} | "
          f"batch={batch_size} | degradation={cfg}")

    return train_loader, val_loader
