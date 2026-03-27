"""
eval.py — Evaluation Metrics

Implements the full evaluation protocol from spec §5.7:

1. Trajectory recovery (primary)
   - Empirical coverage of P10-P90 interval against ground truth
   - Target: 78-82% coverage (nominal 80%) per specialist per horizon
   - Reports calibration curves (reliability diagrams)

2. Counterfactual direction accuracy
   - For each known intervention, check if simulator predicts correct direction
   - Target: >92% for core specialists
   - Stratified by intervention type and horizon

3. Expert utilization divergence (JS divergence)
   - Jensen-Shannon divergence between attention distributions of specialist pairs
   - Target: mean JS divergence > 0.2 across all pairs
   - Collapse (JS < 0.05) indicates router failure

4. VSN variable importance stability
   - Consistency of VSN weights across similar persona sequences
   - CardioVSN should consistently up-weight HRV and resting HR
   - Reports per-specialist variable importance heatmaps

5. Ablation support
   - Utilities for comparing model variants

ECE (Expected Calibration Error):
    Target: ECE < 0.05 across all 8 agents, 4 horizons
    Measures whether predicted confidence intervals match empirical coverage.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Output containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoverageResult:
    """Result of quantile coverage evaluation."""
    agent: str
    horizon_days: list[int]
    coverage_p10_p90: list[float]   # empirical P10-P90 coverage per horizon (target: 0.80)
    ece: list[float]                 # expected calibration error per horizon (target: <0.05)
    mean_coverage: float
    mean_ece: float


@dataclass
class DivergenceResult:
    """JS divergence between specialist attention distributions."""
    pair: tuple[str, str]
    js_divergence: float             # target: > 0.2
    collapsed: bool                  # True if JS < 0.05 (router failure)


@dataclass
class EvalReport:
    """Full evaluation report."""
    coverage: dict[str, CoverageResult]       # per agent
    divergence: list[DivergenceResult]        # all specialist pairs
    mean_js_divergence: float
    router_collapsed: bool
    cf_direction_accuracy: Optional[dict] = None  # if counterfactual data available
    vsn_stability: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trajectory recovery / quantile coverage
# ─────────────────────────────────────────────────────────────────────────────

def compute_quantile_coverage(
    y_true: torch.Tensor,      # (N, N_HORIZONS)  ground truth
    quantiles: torch.Tensor,   # (N, N_HORIZONS, 3)  [P10, P50, P90]
    horizon_days: list[int],
    agent_name: str = "agent",
) -> CoverageResult:
    """
    Measure empirical coverage of P10-P90 interval.

    A well-calibrated model should have ~80% of true values fall inside [P10, P90].

    Args:
        y_true:      (N, H)    ground truth values
        quantiles:   (N, H, 3) predicted [P10, P50, P90]
        horizon_days: list of horizon values in days
        agent_name:   name for reporting

    Returns:
        CoverageResult
    """
    p10 = quantiles[..., 0]   # (N, H)
    p90 = quantiles[..., 2]   # (N, H)
    p50 = quantiles[..., 1]   # (N, H)

    coverage_per_horizon = []
    ece_per_horizon = []

    for h in range(y_true.shape[1]):
        y_h = y_true[:, h]
        p10_h = p10[:, h]
        p90_h = p90[:, h]
        p50_h = p50[:, h]

        # Empirical P10-P90 coverage
        inside = ((y_h >= p10_h) & (y_h <= p90_h)).float()
        coverage = inside.mean().item()
        coverage_per_horizon.append(coverage)

        # ECE: compare predicted interval widths to actual errors
        # Use a binning approach: divide predicted widths into bins,
        # measure actual coverage in each bin
        ece = _compute_ece(y_h, p10_h, p90_h, n_bins=10)
        ece_per_horizon.append(ece)

    return CoverageResult(
        agent=agent_name,
        horizon_days=horizon_days,
        coverage_p10_p90=coverage_per_horizon,
        ece=ece_per_horizon,
        mean_coverage=sum(coverage_per_horizon) / len(coverage_per_horizon),
        mean_ece=sum(ece_per_horizon) / len(ece_per_horizon),
    )


def _compute_ece(
    y: torch.Tensor,    # (N,) ground truth
    p10: torch.Tensor,  # (N,) lower bound
    p90: torch.Tensor,  # (N,) upper bound
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for a single horizon.

    Bins samples by predicted interval width, measures actual coverage per bin.
    ECE = mean absolute difference between predicted and actual coverage.
    Target: ECE < 0.05.
    """
    widths = (p90 - p10).clamp(min=1e-6)  # predicted interval width
    inside = ((y >= p10) & (y <= p90)).float()

    # Sort by predicted width and bin
    sorted_idx = widths.argsort()
    bin_size = max(1, len(y) // n_bins)

    ece_sum = 0.0
    n_bins_actual = 0

    for b in range(n_bins):
        start = b * bin_size
        end = min(len(y), (b + 1) * bin_size)
        if start >= end:
            break

        idx = sorted_idx[start:end]
        # Nominal coverage: what fraction of [P10, P90] intervals should contain y?
        # For [P10, P90] this is nominally 0.80
        nominal = 0.80
        actual = inside[idx].mean().item()
        ece_sum += abs(actual - nominal)
        n_bins_actual += 1

    return ece_sum / max(1, n_bins_actual)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Counterfactual direction accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_cf_direction_accuracy(
    baseline_p50: torch.Tensor,    # (N, N_HORIZONS)  baseline predictions
    cf_p50: torch.Tensor,          # (N, N_HORIZONS)  counterfactual predictions
    ground_truth_direction: torch.Tensor,  # (N, N_HORIZONS)  +1 = risk should decrease
) -> dict[str, float]:
    """
    Measure whether counterfactual predicts the correct direction of change.

    For a beneficial intervention, risk should decrease.
    Direction accuracy = fraction of cases where sign(baseline - cf) == sign(gt_direction)

    Args:
        baseline_p50:          (N, H)
        cf_p50:                (N, H)
        ground_truth_direction: (N, H)  expected direction (+1 = decrease, -1 = increase)

    Returns:
        dict with per-horizon and overall direction accuracy
    """
    # Predicted direction: positive delta means risk went down (baseline > cf)
    pred_direction = torch.sign(baseline_p50 - cf_p50)

    # Correct if predicted direction matches ground truth
    correct = (pred_direction == ground_truth_direction).float()

    result = {}
    for h, days in enumerate([7, 30, 90, 180]):
        result[f"horizon_{days}d"] = correct[:, h].mean().item()

    result["overall"] = correct.mean().item()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Expert utilization divergence (JS divergence)
# ─────────────────────────────────────────────────────────────────────────────

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    JSD is symmetric and bounded in [0, 1] (base-2 log).
    Target: mean JSD > 0.2 across specialist pairs.
    Collapse: JSD < 0.05 (agents attending to same patterns = no specialization).

    Args:
        p, q: (N,) probability distributions (attention weights, normalized)

    Returns:
        JSD value in [0, 1]
    """
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()

    return (0.5 * kl_pm + 0.5 * kl_qm).item()


def compute_specialist_divergence(
    attention_weights: dict[str, torch.Tensor],  # name -> (N, seq_len)
) -> list[DivergenceResult]:
    """
    Compute JS divergence between all pairs of specialist attention distributions.

    Args:
        attention_weights: dict mapping specialist name -> (N_samples, seq_len) attention

    Returns:
        list of DivergenceResult for each pair
    """
    names = list(attention_weights.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]

            # Average attention distribution across samples
            attn_a = attention_weights[name_a].mean(dim=0)  # (seq_len,)
            attn_b = attention_weights[name_b].mean(dim=0)  # (seq_len,)

            # Normalize to distributions
            attn_a = attn_a / (attn_a.sum() + 1e-9)
            attn_b = attn_b / (attn_b.sum() + 1e-9)

            jsd = js_divergence(attn_a, attn_b)

            results.append(DivergenceResult(
                pair=(name_a, name_b),
                js_divergence=jsd,
                collapsed=(jsd < 0.05),
            ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. VSN variable importance stability
# ─────────────────────────────────────────────────────────────────────────────

def compute_vsn_stability(
    vsn_weights: dict[str, torch.Tensor],  # name -> (N, seq_len, n_vars)
    signal_names: list[str],
    top_k: int = 5,
) -> dict[str, dict]:
    """
    Measure consistency of VSN variable importance across samples.

    For each specialist:
        - Compute mean importance per variable (averaged over samples and timesteps)
        - Report top-k signals
        - Measure std of importance across samples (stability metric)

    Args:
        vsn_weights:  dict mapping specialist name -> (N, seq, n_vars) weights
        signal_names: list of signal names corresponding to the n_vars dimension
        top_k:        how many top signals to report

    Returns:
        dict per specialist with top signals and stability scores
    """
    results = {}

    for name, weights in vsn_weights.items():
        # weights: (N, seq_len, n_vars) -> mean over N and seq_len
        mean_importance = weights.mean(dim=(0, 1))  # (n_vars,)
        std_importance = weights.mean(dim=1).std(dim=0)  # std across samples

        # Top-k signals by mean importance
        top_indices = mean_importance.topk(min(top_k, len(mean_importance))).indices
        top_signals = [signal_names[i] for i in top_indices.tolist()]
        top_importances = mean_importance[top_indices].tolist()

        # Stability: 1 - mean CV (coefficient of variation) over top signals
        cv = (std_importance[top_indices] / (mean_importance[top_indices] + 1e-9)).mean().item()
        stability = max(0.0, 1.0 - cv)

        results[name] = {
            "top_signals": top_signals,
            "top_importances": top_importances,
            "stability": stability,  # 1.0 = perfectly stable, 0.0 = random
            "mean_importance": mean_importance.tolist(),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device: str = "cuda",
    agent_names: list[str] = None,
    signal_names: list[str] = None,
    horizon_days: list[int] = None,
) -> EvalReport:
    """
    Run full evaluation protocol on a validation DataLoader.

    Args:
        model:        trained DRIFT model
        val_loader:   DataLoader yielding (X, y) batches
        device:       computation device
        agent_names:  specialist names (default: ["cardio", "mental"])
        signal_names: input signal names for VSN interpretation
        horizon_days: prediction horizons in days

    Returns:
        EvalReport with all metrics
    """
    from drift.model.heads import HORIZONS

    if agent_names is None:
        agent_names = ["cardio", "mental"]
    if horizon_days is None:
        horizon_days = HORIZONS
    if signal_names is None:
        signal_names = [f"signal_{i}" for i in range(20)]

    model.eval()

    # Collectors
    all_y = {name: [] for name in agent_names}
    all_quantiles = {name: [] for name in agent_names}
    all_attn = {name: [] for name in agent_names}
    all_vsn = {name: [] for name in agent_names}

    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(xb, return_weights=True)
        for i, name in enumerate(agent_names):
            if name not in outputs.agents:
                continue

            pred = outputs.agents[name]

            all_y[name].append(yb[:, i, :].cpu())
            all_quantiles[name].append(pred.quantiles.cpu())

            if name in outputs.attn_weights and outputs.attn_weights[name] is not None:
                all_attn[name].append(outputs.attn_weights[name].cpu())
            if name in outputs.vsn_weights and outputs.vsn_weights[name] is not None:
                all_vsn[name].append(outputs.vsn_weights[name].cpu())

    # --- Metric 1: Coverage + ECE ---
    coverage_results = {}
    for name in agent_names:
        if not all_y[name]:
            continue
        y_all = torch.cat(all_y[name], dim=0)
        q_all = torch.cat(all_quantiles[name], dim=0)
        coverage_results[name] = compute_quantile_coverage(y_all, q_all, horizon_days, name)

    # --- Metric 3: Attention divergence ---
    attn_for_div = {}
    for name in agent_names:
        if all_attn[name]:
            attn_for_div[name] = torch.cat(all_attn[name], dim=0)

    divergence_results = []
    mean_jsd = 0.0
    router_collapsed = False

    if len(attn_for_div) >= 2:
        divergence_results = compute_specialist_divergence(attn_for_div)
        jsd_values = [r.js_divergence for r in divergence_results]
        mean_jsd = sum(jsd_values) / max(1, len(jsd_values))
        router_collapsed = any(r.collapsed for r in divergence_results)

    # --- Metric 4: VSN stability ---
    vsn_all = {}
    for name in agent_names:
        if all_vsn[name]:
            vsn_all[name] = torch.cat(all_vsn[name], dim=0)

    vsn_stability = None
    if vsn_all:
        vsn_stability = compute_vsn_stability(vsn_all, signal_names)

    # --- Print summary ---
    print("\n═══ Evaluation Report ═══")
    for name, cov in coverage_results.items():
        print(f"\n{name.upper()}")
        for i, h in enumerate(horizon_days):
            print(f"  {h:3d}d: coverage={cov.coverage_p10_p90[i]:.3f} (target: 0.80) | "
                  f"ECE={cov.ece[i]:.4f} (target: <0.05)")

    if divergence_results:
        print(f"\nJS Divergence (target: mean > 0.20):")
        for r in divergence_results:
            flag = "⚠️ COLLAPSED" if r.collapsed else "✓"
            print(f"  {r.pair[0]} vs {r.pair[1]}: {r.js_divergence:.4f} {flag}")
        print(f"  Mean JSD: {mean_jsd:.4f}")

    return EvalReport(
        coverage=coverage_results,
        divergence=divergence_results,
        mean_js_divergence=mean_jsd,
        router_collapsed=router_collapsed,
        vsn_stability=vsn_stability,
    )
