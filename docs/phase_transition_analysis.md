# Phase Transition Analysis Guide

## Overview

This analysis sweeps the attention sparsity parameter **alpha** (the `topk_percent`
fed to `SparseAttentionWrapper`) from near-zero (highly sparse, few active paths) to
1.0 (dense, all attention weights kept) on a **frozen** LLM.  No model parameters
change.  We only observe how six pathway metrics respond to alpha, whether that
response is nonlinear, and whether the circuitry retains task-relevant information
(vividness prediction, condition classification) as sparsity increases.

The core hypothesis:
> *Neural measurements (EEG) reflect projections of computational pathways.*
> *If attention sparsity induces a phase transition in routing structure, and*
> *EEG tracks that structure, EEG-based decoding should degrade non-uniformly*
> *at the same critical alpha.*

---

## Interpreting the Metric Curves

### Routing Sparsity vs Alpha
- **What it measures**: Rényi-2 diversity of all attention weights. Higher = more
  evenly spread (dense); lower = concentrated on a few paths.
- **Expected shape**: Increases monotonically from low alpha (sparse, low diversity)
  to high alpha (dense, high diversity).
- **Phase transition signal**: A sudden jump or S-curve rather than a straight line.

### Path Competition Index (PCI) vs Alpha
- **What it measures**: max-weight / mean-weight across non-zero entries.
  Higher = stronger winner-take-all routing.
- **Expected shape**: Decreases as alpha increases (fewer dominant paths when dense).
- **Phase transition signal**: Nonmonotone behavior — a plateau followed by rapid
  drop — indicates a shift between routing regimes.  This is the **primary indicator**.

### Routing Entropy vs Alpha
- **What it measures**: Mean Shannon entropy per attention row.
  Higher = more uniform attention distribution.
- **Expected shape**: Should increase with alpha (denser → more uniform).
- **Phase transition signal**: A sudden entropy drop even as alpha increases
  = routing collapse (attention suddenly concentrates on very few tokens).

### Path Efficiency vs Alpha
- **What it measures**: Fraction of total attention energy in the top-20% of weights.
  Higher = more concentrated.
- **Expected shape**: Decreases with alpha (as dense attention distributes energy).
- **Phase transition signal**: Flat plateau then sharp drop = two distinct regimes.

### Inter-head Divergence vs Alpha
- **What it measures**: Mean pairwise KL divergence between attention heads.
  Higher = heads attending to very different positions.
- **Expected shape**: May increase at low alpha (sparse heads all latch onto
  different tokens) and decrease as alpha increases (heads converge on similar patterns).
- **Phase transition signal**: Non-monotone curve; may peak near alpha*.

### Layer Stability vs Alpha
- **What it measures**: Mean cosine similarity between consecutive layers'
  attention maps. Higher = attention patterns change little across depth.
- **Expected shape**: Should be relatively flat (layer patterns are architecturally
  determined), but may dip when sparsity forces each layer to a unique set of tokens.
- **Phase transition signal**: Sudden spike or dip at alpha*.

---

## What to Look For

### Strong Phase Transition Result
```
PCI curve:    flat at low alpha → sharp drop → plateau at high alpha
Entropy:      step-function increase around alpha*
Sparsity:     S-curve, inflection at alpha*
Multiple metrics change together at the same alpha*
```
This indicates the network is operating in qualitatively different computational
regimes below and above alpha*.

### Weak / No Transition
```
All curves smooth and monotonic
Gradients small and consistent across the full alpha range
```
This means sparsity is acting as a simple continuous modulator rather than
inducing a regime change.

---

## Identifying Alpha*

Three methods are combined; the median gives the final estimate.

### Method 1 — Maximum Gradient
```
alpha* = argmax |d(PCI)/d(alpha)|
```
Finds the point of steepest change in the primary indicator.

### Method 2 — Inflection Point
```
alpha* = argmax |d²(PCI)/d(alpha²)|
```
Finds the curvature peak — where the curve changes from concave to convex.
This is the *transition midpoint*, not necessarily where change is fastest.

### Method 3 — Multi-metric Consensus
For each alpha, count the number of metrics whose absolute first derivative
is in the top-25% of its own range.  The consensus peak is where the most
metrics are simultaneously changing rapidly.  This is the most robust method
because it does not depend on any single metric behaving well.

**Confidence score**: `1 - std(three_estimates) / alpha_range`.
High confidence (> 0.8) = all three methods agree; low confidence (< 0.4)
= no clear transition exists.

---

## Statistical Tests

### 1. Spearman Correlation (Monotonicity)
Tests whether each metric changes monotonically with alpha.

- **|r| > 0.8**: Strong monotonic relationship — no phase transition.
- **|r| < 0.4**: Non-monotone — consistent with a phase transition.
- **Intermediate**: Partial monotonicity; check the derivative plots.

### 2. Nonlinearity Test (ΔR²)
Fits a linear and a quadratic model to each metric curve.

```
ΔR² = R²_quadratic − R²_linear
```

- **ΔR² > 0.1**: Strong nonlinearity — the metric has a clear bend.
- **ΔR² > 0.3**: Very strong — almost certainly a phase transition.
- **ΔR² ≈ 0**: Linear relationship — simple modulator effect.

### 3. Pre/Post Transition (Mann-Whitney U)
Splits the alpha grid at alpha* and tests whether the metric distributions
before and after are significantly different.

- **p < 0.05**: The two regimes are statistically distinguishable.
- **p > 0.1**: No significant split — either alpha* was wrong or there is
  no transition.

### 4. Vividness Predictability Sweep
At each alpha, compute Spearman r between each pathway metric and the
participants' vividness ratings.

- **High |r| across all alpha**: The metric always correlates with vividness;
  sparsity is not critical.
- **|r| peaks near alpha***: The metric is *most* informative about vividness
  right at the phase transition — strong support for the hypothesis.
- **|r| collapses above alpha***: Dense attention destroys the vividness signal.

### 5. Task AUROC Sweep (if condition labels provided)
At each alpha, trains a logistic regression (5-fold CV) using pathway features
to classify task/condition.

- **AUROC > 0.7 at low alpha, drops sharply**: Sparse routing encodes task
  information better than dense routing.
- **AUROC relatively flat**: Routing structure is task-informative regardless
  of sparsity level.
- **AUROC ≈ 0.5 throughout**: Features do not encode task information (null result).

---

## EEG Plots (when `--eeg-projector` is used)

The `EEGProjector` maps the 6-dimensional pathway feature vector to a
105-channel EEG-like signal.  Because it is a linear projection (after
BatchNorm), its output directly tracks pathway feature structure.

If the EEG projection is meaningful:
- EEG mean vs alpha curves should mirror the pathway metric curves.
- EEG predictions should correlate with vividness at the same alpha values
  where pathway metrics do.

If EEG projection adds only noise:
- EEG curves will be noisier but follow the same trend.
- The EEG vividness correlation will be systematically lower than the raw
  pathway metric correlation.

---

## Running the Analysis

### Quick check (synthetic prompts)
```bash
python scripts/run_phase_transition.py \
    --alpha-steps 30 \
    --num-prompts 20
```

### With real narrative data and W&B
```bash
python scripts/run_phase_transition.py \
    --data-dir /data/EEG \
    --alpha-steps 40 \
    --batch-size 64 \
    --wandb-project eeg-phase-transition
```

### Using the cache (LLM runs once)
On the first run, the sweep is cached to `--cache-dir`.
Subsequent runs (e.g. different analysis parameters) reload the cache and
skip the LLM forward passes entirely:
```bash
# First run — LLM forward passes happen here
python scripts/run_phase_transition.py \
    --data-dir /data/EEG \
    --cache-dir ./cache/phase \
    --alpha-steps 40

# Re-analysis — instant (LLM skipped)
python scripts/run_phase_transition.py \
    --data-dir /data/EEG \
    --cache-dir ./cache/phase \
    --alpha-steps 40 \
    --wandb-project eeg-phase-transition
```

---

## Output Files

```
results/
  phase_transition_results.json   # alpha_star, confidence, all stat values
  phase_metric_curves.png         # 2×3 grid of metric vs alpha curves
  phase_derivatives.png           # first and second derivatives
  phase_consensus.png             # multi-metric consensus score
  phase_vividness.png             # Spearman r with vividness (if data available)
  phase_task_auroc.png            # task AUROC vs alpha (if conditions available)
```

W&B dashboard (when `--wandb-project` is set):
- All figures logged as images under `plots/`
- `phase_transition/alpha_star` and `phase_transition/confidence` as scalars
- All statistical test values under `stats/`

---

## Success Criteria

### Strong support for the hypothesis
1. PCI shows a clear nonlinear curve (ΔR² > 0.2)
2. Alpha* is consistently estimated by all three methods (confidence > 0.7)
3. Vividness Spearman r peaks near alpha* for at least two metrics
4. Task AUROC is above chance and varies non-monotonically with alpha
5. Mann-Whitney p < 0.05 for at least three metrics

### Weak / null result
- All curves smooth and monotonic
- Confidence < 0.4 (methods disagree on alpha*)
- Vividness Spearman r uniformly low (< 0.1) across all alpha
- Task AUROC ≈ 0.5 throughout

A null result is also interpretable: it would suggest that GPT-2's attention
routing (at the scale of prompts and sparsity levels tested) does not exhibit
phase-transition behavior, meaning EEG modulation by vividness may require a
different computational account.

---

## Interpretation

A phase transition at alpha* would mean:
- Below alpha*: routing is highly selective; a few dominant paths carry all information.
  This is closer to how biological neural circuits behave during focused attention.
- Above alpha*: routing becomes diffuse; many paths contribute equally.
  This is more consistent with resting or unfocused states.

If EEG measures track this transition, it supports the claim that what we measure
in EEG during imagery tasks reflects the *structure* of information routing, not
just the *amount* of neural activity.
