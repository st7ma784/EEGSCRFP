# EEGSCRFP — W&B Metric Guide

This guide explains every metric logged to Weights & Biases, what values to
expect, and how to interpret the dashboard when reviewing experimental results.

Set up a W&B project called `eegscrfp`.  All experiment scripts accept
`--wandb-project eegscrfp`.

---

## Dashboard organisation

Create the following W&B report sections (manually in the UI after runs complete):

```
Section 1: Pre-flight diagnostics      (audit_feature_variance)
Section 2: Phase transition            (run_phase_transition)
Section 3: Claim A — circuit ceiling   (claim_a_baseline)
Section 4: Claim B — sensor count      (sensor_count)
Section 5: Encoder comparison          (encoder sweep)
Section 6: Participant generalisation  (LoRA sweep)
Section 7: Debug — pathway scalars     (logged in all experiments)
```

---

## Section 1: Pre-flight diagnostics

Run once before any training. Outputs only to terminal — not logged to W&B.

| What to check | Target | Meaning if outside target |
|--------------|--------|--------------------------|
| CKA eta²_prompt (median) | > 0.3 | If < 0.1: features don't vary with prompt content — use real narrative data |
| CKA % dims > 0.3 | > 50% | If < 20%: CKA is not working properly — check hidden_states are coming through |
| 6-scalar eta²_prompt (median) | < 0.05 | If > 0.2: synthetic prompts are suspiciously diverse (unexpected) |
| Pathway participation ratio | 1.0–1.5 | This is expected (1D variance) — not a failure |
| CKA participation ratio | > 3.0 | If < 2: CKA space is near 1D — real data needed |

---

## Section 2: Phase transition

### Sweep metrics (logged for each alpha step)

| W&B key | Type | Expected range | Notes |
|---------|------|---------------|-------|
| `phase_transition/alpha` | scalar | [0.05, 1.0] | X-axis for all curves |
| `phase_transition/routing_sparsity` | scalar | [1, S] | Increases with alpha |
| `phase_transition/pci` | scalar | [1, ∞] | Key phase-transition indicator |
| `phase_transition/routing_entropy` | scalar | [0, log(S)] | Decreases with alpha |
| `phase_transition/inter_head_divergence` | scalar | [0, ~5] | Increases with alpha |
| `phase_transition/layer_stability` | scalar | [0, 1] | Should be high (>0.7) for alpha > 0.3 |
| `phase_transition/task_auroc` | scalar | [0.5, 1.0] | Should show knee at alpha* |
| `phase_transition/vividness_r2` | scalar | [0, 0.5] | Weaker signal, noisier |
| `phase_transition/cka_median` | scalar | [0, 1] | Median CKA across layer pairs |

### Summary metrics (logged once per run)

| W&B key | Type | Target | Notes |
|---------|------|--------|-------|
| `phase_transition/alpha_star` | scalar | 0.1–0.3 | Phase transition point |
| `phase_transition/confidence` | scalar | > 0.6 | Agreement across 3 detection methods |
| `phase_transition/auroc_above_star` | scalar | > 0.70 | Task decodability above threshold |
| `phase_transition/auroc_below_star` | scalar | < 0.55 | Should be near chance below threshold |

### Key W&B chart to create

Line chart: X = `phase_transition/alpha`, Y = `phase_transition/pci` and
`phase_transition/task_auroc` (dual Y-axis).  Add a vertical line at
`phase_transition/alpha_star`.

---

## Section 3: Claim A — circuit sufficiency baseline

### CV metrics

| W&B key | Type | Target | Notes |
|---------|------|--------|-------|
| `claim_a/task_auroc_cv` | scalar | > 0.70 | 5-fold stratified CV |
| `claim_a/task_auroc_std` | scalar | < 0.05 | CV stability |
| `claim_a/vividness_r2_cv` | scalar | > 0.15 | R² on vividness ratings |
| `claim_a/vividness_spearman` | scalar | > 0.3 | Rank correlation |
| `claim_a/task_auroc_cross_sub` | scalar | within 10% of CV | Cross-subject generalisation |

### Per-condition breakdown

| W&B key | Notes |
|---------|-------|
| `claim_a/auroc_Self_Relive` | |
| `claim_a/auroc_Non_Self_Relive` | |
| `claim_a/auroc_Self_Evaluate` | |
| `claim_a/auroc_Self_Define` | |

Bar chart in W&B: per-condition AUROC.  Conditions with AUROC < 0.60 indicate
the circuit features do not encode that condition well.

---

## Section 4: Sensor count degradation

### Per-M metrics (logged as a step for each M value)

| W&B key | Type | Notes |
|---------|------|-------|
| `sensor_count/n_patches` | int | X-axis |
| `sensor_count/task_auroc` | scalar | Main degradation curve |
| `sensor_count/auroc_std` | scalar | Std over 5 random patch subsets |
| `sensor_count/vividness_r2` | scalar | |
| `sensor_count/r2_std` | scalar | |

### Summary metrics

| W&B key | Target | Notes |
|---------|--------|-------|
| `sensor_count/full_coverage_count` | 12 (GPT-2) | Reference point |
| `sensor_count/max_auroc` | Matches claim_a | Validates patch extraction |
| `sensor_count/elbow_m` | 2–8 | Estimated minimum sensor count |
| `sensor_count/auroc_at_half_coverage` | > 0.80 × max_auroc | Half the sensors = 80% performance? |

### Key W&B chart to create

Line chart with error bars: X = `sensor_count/n_patches`, Y =
`sensor_count/task_auroc` ± `sensor_count/auroc_std`.
Add horizontal dashed lines at: max_auroc, 0.80 × max_auroc, 0.5 (chance).

### W&B table: `sensor_count/degradation_table`

Downloadable table of (n_patches, task_auroc, auroc_std, vividness_r2, r2_std).
Download and inspect for the paper.

---

## Section 5: Encoder comparison sweep

One run per encoder configuration.  Compare across runs in W&B parallel
coordinates view.

### Per-run metrics

| W&B key | Type | Target | Notes |
|---------|------|--------|-------|
| `encoder/task_auroc` | scalar | > 0.70 | Primary metric |
| `encoder/recon_cosine_sim` | scalar | > 0.60 | Prompt reconstruction quality |
| `encoder/alpha_r2` | scalar | > 0.80 | Should be easy (control objective) |
| `encoder/task_loss` | scalar | Decreasing | Training stability |
| `encoder/recon_loss` | scalar | Decreasing | |
| `encoder/alpha_loss` | scalar | Decreasing | |
| `encoder/param_count` | int | — | For efficiency comparison |
| `encoder/train_time_s` | scalar | — | Wall time |

### Hyperparameters logged (for parallel coordinates)

`encoder_type`, `hidden_dim`, `n_layers`, `dropout`, `lr`, `batch_size`

### Sweep metric to optimise

Primary: `encoder/task_auroc` (maximize)
Secondary: `encoder/recon_cosine_sim` (if multi-objective)

---

## Section 6: Participant generalisation (LoRA)

### Per-run metrics

| W&B key | Notes |
|---------|-------|
| `lora/lora_rank` | Configuration |
| `lora/n_train_participants` | |
| `lora/n_test_participants` | |
| `lora/within_auroc` | Train and test on same LoRA variant |
| `lora/cross_auroc` | Test on held-out LoRA variant |
| `lora/auroc_drop` | = within_auroc - cross_auroc; should be < 0.10 |
| `lora/routing_divergence` | Mean KL divergence between base and LoRA routing |

### Parallel coordinates

X-axes: `lora_rank`, `routing_divergence`  
Y-axis: `lora/cross_auroc`

The key question: does `cross_auroc` decrease as `lora_rank` increases?
If yes → harder transfer for more different participants.
If flat → encoder has learned participant-invariant features (ideal for EEG).

---

## Section 7: Debug — pathway scalars

Logged as histograms for every experiment at reference alpha=1.0 and debug alpha=0.2.
These are diagnostic only — not used for prediction.

| W&B key | Expected (alpha=1.0) | Expected (alpha=0.2) | Warning |
|---------|---------------------|---------------------|---------|
| `debug/routing_sparsity_hist` | Wide distribution | Narrow, low values | If identical → sparsity injection broken |
| `debug/pci_hist` | Mean ≈ 3–6 | Mean < 1.5 | If pci low at alpha=1 → model is very sparse by default |
| `debug/entropy_hist` | High | Low | |
| `debug/layer_stability_hist` | Concentrated near 0.8–1.0 | More spread | |
| `debug/floating_score_hist` | Per-patch distributions | | High variance = patches are active |
| `debug/floating_score_by_patch` | Bar chart, 12 bars | | Uniform = good coverage; one dominant = bad |

### Key diagnostic plot

Bar chart: `debug/floating_score_by_patch` at alpha=1.0.  Each bar is the
mean floating score for one patch.  This should be relatively uniform if
patches are providing complementary information.  If one patch dominates,
consider re-running with different random seeds or adjusting patch_depth.

---

## Interpreting a new run quickly

**Step 1:** Check `debug/routing_sparsity_hist` at alpha=0.2 vs alpha=1.0.
If they are identical, sparsity injection failed — stop here.

**Step 2:** Check `phase_transition/confidence`.  If < 0.5, the phase
transition is ambiguous — alpha* is unreliable.

**Step 3:** Check `claim_a/task_auroc_cv`.  If < 0.60 on real narrative data,
either the data is low-quality or CKA features are not working.

**Step 4:** Check `sensor_count/task_auroc` at M=12 vs `claim_a/task_auroc_cv`.
They should be within 5 percentage points (patches should approximate CKA).
If patches perform much worse, the floating score is losing information — try
increasing patch_depth or heads_per_patch.

**Step 5:** Look at `sensor_count/task_auroc` at M=1.
If this is already above 0.65, the signal is highly concentrated in a single
patch — investigate which patch dominates using `debug/floating_score_by_patch`.

---

## Creating the metric guide in W&B

1. Open the `eegscrfp` W&B project.
2. Create a new Report titled "Metric Guide".
3. Add a Table of Contents block.
4. For each section above, add a Markdown block with the table.
5. For key charts (degradation curve, per-condition bars, parallel coordinates),
   embed them directly in the report.
6. Pin the report to the project.

The report URL should be shared with all collaborators before running sweeps
so they know what to look for in the dashboard.
