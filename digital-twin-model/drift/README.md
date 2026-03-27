# DRIFT — Domain-Routed Interpretable Fusion Transformer

**Full research architecture for the Preventive Health Digital Twin.**

Spec: MoE-TFT Hybrid | FlashAttention-2 | RoPE (base=1000) | LoRA adaptation  
Paper target: CHIL / ML4H / NeurIPS

---

## Project Structure

```
drift/
├── model/
│   ├── grn.py              Gated Residual Network (primitive used everywhere)
│   ├── rope.py             Rotary Position Embedding (base=1000, health timescales)
│   ├── baseline.py         EMA BaselineTracker (personal signal normalization)
│   ├── vsn.py              Variable Selection Networks (per-specialist)
│   ├── static_encoder.py   Static covariate encoder → 4 context vectors
│   ├── encoder.py          Shared bidirectional encoder with RoPE
│   ├── agent.py            Agent block: VSN + shared-V attention + GRN gate
│   ├── moe.py              Tier 1 (specialist LSTM pool) + Tier 2 (FFN experts)
│   ├── heads.py            Quantile heads + velocity + threshold + calibrator
│   └── model.py            Full DRIFT model assembly
│
├── data/
│   └── loader.py           Shard loading, HealthDataset, curriculum DataLoaders
│
├── training/
│   ├── training.py         Training loop, losses, 5-phase curriculum
│   ├── simulate.py         Counterfactual simulator with adherence profiles
│   └── eval.py             Coverage, ECE, JS divergence, VSN stability
│
├── main.py                 Entry point
└── requirements.txt
```

---

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Optional: install flash-attn for FA2 (recommended for GPU)
pip install flash-attn --no-build-isolation

# Train (2 agents: cardio + mental, using your existing shard)
python main.py \
    --shard outputs/shards/shard_0000.pt \
    --epochs 100 \
    --agents cardio mental

# Train all 4 core agents
python main.py \
    --shard outputs/shards/ \
    --epochs 100 \
    --agents cardio mental metabolic recovery

# Quick sanity check
python main.py --shard outputs/shards/shard_0000.pt --epochs 2 --batch-size 8

# Evaluate a checkpoint
python main.py \
    --shard outputs/shards/shard_0000.pt \
    --eval-only \
    --resume checkpoints/best_model.pt
```

---

## Architecture

```
Input (B, 90, 104)
        │
        ▼
[Stage 1] StaticCovariateEncoder → c_s, c_e, c_h, c_c
        │
        ▼
[Stage 2] SharedEncoder (2-layer bidirectional, RoPE base=1000) → Z (B, 90, 128)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[Stage 3] Tier 1 MoE: Specialist LSTM Pool      [Stage 6] Tier 2 MoE: Expert FFN Pool
  • 4 always-on: cardio/mental/metabolic/recovery    • 8 GRN experts, top-2 per token
  • 4 sparse (top-2): immune/respiratory/hormonal/cog • Token-level routing
  • Sequence-level routing (1 decision/sequence)     → Z_refined (B, 90, 128)
  → H_i per specialist (B, 128)
        │
        ▼
[Stages 4+5] AgentBlock per core specialist
  • VSN: per-specialist variable selection
  • Shared-V cross-attention (TFT-style interpretable)
  • GRN gate: blend attention + LSTM state H_i
        │
        ▼
[Stage 7] Prediction Heads per specialist
  • QuantileHead → P10, P50, P90 × 4 horizons
  • VelocityHead → signed slope × 4 horizons
  • ThresholdHead → days to concern × 4 horizons
```

---

## Training Curriculum

| Phase | Epochs | Data               | Degradation    |
|-------|--------|--------------------|----------------|
| 0     | 0–15   | Per-specialist subsets | None       |
| 1     | 15–35  | All personas, clean | None          |
| 2     | 35–55  | All personas        | 10% dropout   |
| 3     | 55–75  | All personas        | 30% dropout   |
| 4     | 75–90  | Edge cases          | Varied        |
| 5     | 90–100 | Full distribution   | 5% dropout    |

---

## Loss Function

```
L = L_quantile + 0.1 * L_vel + 0.01 * L_lb1 + 0.01 * L_lb2
```

- **L_quantile**: Pinball loss, τ ∈ {0.1, 0.5, 0.9}, horizons weighted [1.0, 0.8, 0.6, 0.4]
- **L_vel**: Velocity consistency — predicted slope must match consecutive P50 differences
- **L_lb1**: Tier 1 load balancing — prevents sparse specialist router collapse
- **L_lb2**: Tier 2 load balancing — prevents FFN expert collapse

---

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128 | MVP; expand to 256 for v2 |
| n_heads | 4 | head_dim=32; shared-V |
| RoPE base | 1000 | Health timescales (not 10000) |
| Sequence length | 90 | 90-day window |
| Core specialists | 4 | Always-on |
| Sparse specialists | 4 | Top-2 by router |
| Tier 2 experts | 8 | Token-level top-2 |
| LoRA rank | 8 | alpha=16; ~50KB/user |
| Batch size | 32 | |
| LR | 3e-4 | AdamW, cosine decay |
| Precision | bf16 | fp16 switchable |

---

## Dropping in your existing data

Your generator already produces `(P, T, 104)` tokens, `cardio_gt`, and `mental_gt`.
The loader expects exactly this format. No changes needed — just point `--shard` at your file.

If your shard also has `metabolic_gt` and `recovery_gt`, pass `--agents cardio mental metabolic recovery`.

---

## Flash Attention

FA2 is used automatically if `flash-attn` is installed:
```bash
pip install flash-attn --no-build-isolation
```
Falls back to standard PyTorch MHA if unavailable (slightly slower, same results).
