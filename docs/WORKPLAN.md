# EEGSCRFP — Formal Research Work Plan

**Repository:** `EEGSCRFP` (EEG Sparse Causal Representation from Pathways)
**Last updated:** 2026-04-09

See `docs/EXPERIMENT_DESIGN.md` for full diagrams and algorithm descriptions.
See `docs/WANDB_GUIDE.md` for the W&B metric reference.

---

## Scientific Hypothesis

> *A frozen language model processing inner-speech prompts is a tractable
> proxy for the human brain during inner speech.  Topk attention sparsity
> (alpha) controls the richness of the simulated routing, analogous to
> inner speech vividness.  Network patches — local (layer, head) clusters
> pooled into floating scores — model EEG electrodes integrating from a
> spatial neighbourhood of neurons.  We train encoders on these patch signals
> to answer three questions: can the circuitry predict the task?  Can it
> reconstruct the prompt?  Can we infer the sparsity level?  The degradation
> of these three objectives as we reduce patch count M gives the minimum
> sensor count and validates the encoder for real EEG data.*

### Three encoder objectives (sweep axis: training_objectives)

**Objective A — Task / semantic prediction:**
Given M floating scores, predict the condition label (e.g. "Self_Relive").
Validates that patches encode *what* the LLM is computing.
Target: macro AUROC > 0.70 at M = full_coverage.

**Objective B — Prompt reconstruction:**
Given M floating scores, reconstruct the input prompt tokens (cosine similarity
to LLM embedding space).  Validates that patches carry semantic content.
Target: cosine similarity > 0.60 at M = full_coverage.

**Objective C — Alpha (IS intensity) prediction:**
Given M floating scores, predict the alpha value.
This is the *control* objective — should be easy (high R²) because alpha
dominates global variance.  If task AUROC is low but alpha R² is high, the
encoder is measuring activity level, not content.
Target: R² > 0.80 at M = full_coverage.

### Three axes of experimental variation

| Axis | Sweep config | Key metric |
|------|-------------|------------|
| **Encoder architecture** | `encoder_architecture.yaml` | `encoder/task_auroc` |
| **Patch count M** | `patch_simulation.yaml` | `sensor_count/task_auroc` vs M |
| **Participants (LoRA)** | `lora_participants.yaml` | `lora/cross_auroc` |

### Two structural claims

**Claim A — Circuit sufficiency:**
Complete circuit information (CKA features at alpha=1.0) predicts task and
vividness with AUROC > 0.70 / R² > 0.15.  This is the performance ceiling.

**Claim B — Graceful degradation:**
As M decreases from full_coverage to 1, prediction accuracy degrades
smoothly.  The elbow of the degradation curve is the minimum sensor count.
The curve shape informs how many real EEG electrodes are needed.

---

## Current State Assessment

### What works
- Frozen GPT-2 with topk sparsity injection
- 6 global pathway metrics computed from attention maps (`[B, 6]`)
- Linear EEG projector: `[B, 6] → [B, 105]`
- MLP predictors trained on (pathway features, sparsity level) pairs
- Phase transition sweep and analysis scripts
- Narrative data loader (prompt_with_condition, vividness_rating)
- Feature caching (LLM runs once, training is cheap)

### Critical gap: feature informativeness
Diagnostic experiment (20 synthetic prompts, GPT-2, alpha=0.5):

| Metric | Std across prompts | Std across alpha | Ratio |
|---|---|---|---|
| PCI | 0.34 | 1.53 | **0.22** |
| Routing Entropy | 0.04 | 0.25 | **0.17** |
| Inter-head Divergence | 0.17 | 1.45 | **0.12** |
| Layer Stability | 0.007 | 0.028 | **0.25** |

Alpha variation dominates prompt variation by 4–9×.
The current 6 summary statistics are primarily measuring *sparsity level*,
not *what the model is computing*.  They lack positional and semantic
sensitivity.

**Root causes:**
1. Global averages wash out prompt-specific routing patterns.
2. Synthetic prompts are semantically homogeneous.
3. No token-level or layer-pairwise features.

---

## Phase 0: Data Foundation (immediate)

**Goal:** Ensure experiments see genuine circuit variation driven by content,
not just by alpha.

### 0.1 Use real narrative prompts as the primary dataset
- 45 subjects × 6 tasks × 15 trials ≈ 4,050 prompts
- Each prompt is `prompt_with_condition` from the trial-info CSVs
- Each has a real `vividness_rating` label
- Task `condition` provides a multi-class classification target
- Path: `/data/EEG` → `NarrativeSparsityDataset`

**Acceptance criterion:** Across-prompt std / across-alpha std > 0.5 for at
least 4 of 6 metrics (after switching to content-sensitive features — Phase 1).

### 0.2 Prompt diversity diagnostic
Add a pre-flight check to the sweep scripts: compute and log the
"prompt signal ratio" (between-prompt variance / within-prompt variance)
for each metric.  Gate downstream analysis on this ratio being > 0.3.

---

## Phase 1: Richer Circuit Features

**Goal:** Replace or augment the 6 global scalars with features that are
sensitive to *what* is being computed, not just *how sparse* the routing is.

### 1.1 CKA-based representational similarity features

**Centered Kernel Alignment (CKA)** measures how similar two
representation matrices are, independently of affine transformations.
Unlike scalar summaries, it captures the *geometry* of the representation
space.

Feature set for a single forward pass through an L-layer model:

```
(a) Layer-pairwise CKA matrix: CKA(hidden_i, hidden_j) for i,j in [0,L]
    → L×L symmetric matrix; vectorise upper triangle → L(L+1)/2 values
    For GPT-2 (L=12): 78 values

(b) Input-to-layer alignment: CKA(embedding, hidden_i) for i in [1,L]
    → [L] values; measures how much each layer preserves the input

(c) Cross-prompt CKA (batch-level): CKA of representation matrix across
    N prompts at layer i → [L] per-layer alignment scores
    This captures whether the batch is being processed similarly

Total: up to ~100 CKA values per forward pass
```

**Why CKA is better than scatter statistics:**
- Invariant to orthogonal transformation (measures structure, not raw values)
- Captures cross-layer information flow, not just within-layer distributions
- Sensitive to which inputs are treated similarly — i.e., task semantics
- Directly comparable across models (Claim B: LLM circuits ↔ EEG)

**Implementation:**
- `src/metrics/cka_metrics.py`: linear and kernel CKA, batched
- Add to `compute_pathway_features()` as an optional extended mode
- Cache separately (CKA features are deterministic given the same prompts)

### 1.2 Token-selective attention statistics (light alternative)

For each of K "anchor" token positions (e.g., the subject noun, the verb,
the condition word), compute per-position versions of the 6 existing metrics.
K=5 anchors × 6 metrics = 30-dimensional feature vector.
More sensitive to content than the global averages.

**Implementation:**
- `src/metrics/token_selective.py`
- Anchor positions identified by the tokeniser (e.g., first noun/verb via POS tags)

### 1.3 Feature selection / dimensionality audit

After collecting real narrative features:
- Compute participation ratio (target: > 2.5 on narrative data)
- Compute prompt-signal ratio (target: > 0.5 for CKA features)
- Run PCA; use elbow criterion to choose the feature dimensionality D
  for downstream analysis (expect D ≈ 8–20 for CKA features)

**Deliverable:** `src/metrics/` contains CKA and token-selective implementations;
`scripts/audit_feature_variance.py` runs the diagnostic and prints the signal ratios.

---

## Phase 2: Claim A — Circuit Sufficiency Baseline

**Goal:** Establish that complete circuit information predicts task outcome.
This is the ceiling against which all degraded-readout experiments are measured.

### 2.1 Task prediction from full circuit features

For each task in the narrative dataset (condition labels):
```
Input:  CKA features from frozen LLM at alpha=1.0 (dense, complete)
Target: condition label  (multi-class)
Method: Logistic regression with 5-fold stratified CV
Metric: macro-AUROC
```
Report AUROC per task and mean across tasks.

**Success criterion:** Mean AUROC > 0.70 across conditions.
This would confirm that the circuit representation encodes task-discriminative
information.

### 2.2 Vividness prediction from full circuit features

```
Input:  CKA features at alpha=1.0
Target: vividness_rating (continuous)
Method: Ridge regression with 5-fold CV
Metric: R², Spearman r
```

**Success criterion:** R² > 0.15 (weak but significant signal is acceptable;
vividness is an inherently noisy subjective measure).

### 2.3 Cross-subject generalisation

Train predictor on circuits from subjects 1–35, test on 36–45.
If AUROC stays above chance: the circuit structure is subject-independent
(as it should be, since the LLM is shared).

**Deliverable:** `experiments/claim_a_baseline.py`; results logged to W&B.

---

## Phase 3: Claim B — Sensor Count Experiment

**Goal:** Show that prediction accuracy degrades gracefully as the number of
network patches (sensors) M decreases, and identify the minimum M.

### 3.1 Sensor model: network patches

A real EEG electrode integrates activity from a local spatial cluster of
neurons — it does not read out a mathematically precise global statistic.
We model this with **network patches**: each patch covers a contiguous window
of adjacent transformer layers and a random subset of attention heads.

Within each patch, attention weights are pooled to produce a single scalar
per input — the **floating score**: how much does this input's routing through
this patch deviate from the population-mean routing?

```
Fixed path   → low floating score   (patch routes identically for all inputs)
Floating path → high floating score (patch responds to this input specifically)
```

Task-relevant computation concentrates in patches with high floating scores.

### 3.2 Experiment design

For GPT-2 (12 layers, 12 heads, depth=3, heads_per_patch=4):
- Full coverage: 4 layer windows × 3 head groups = **12 patches** — every
  (layer, head) sampled exactly once.
- M values to sweep: `[1, 2, 4, 8, 12]`

```
1. Run all N prompts through frozen LLM → collect attention maps
2. Calibrate PatchFeatureExtractor on full prompt set (sets population baseline)
3. For M in [1, 2, 4, 8, 12]:
   a. Sample M patches from the full-coverage set (repeat K=5 times)
   b. Extract [N, M] floating-score feature matrix
   c. Predict task (logistic regression, 5-fold macro-AUROC)
   d. Predict vividness (ridge regression, 5-fold R²)
   e. Record mean ± std over repeats
4. Plot degradation curves: AUROC / R² vs M
5. The elbow of the curve = minimum useful sensor count
```

Produces two degradation curves:
- Task AUROC vs M
- Vividness R² vs M

The curve shape answers:
- **Flat until small M, then sharp drop:** task information is concentrated
  in a few patches — a sparse sensor array would work.
- **Gradual linear decay:** information is distributed; more sensors always help.
- **Plateau at M ≈ full_coverage:** reading the full network is necessary.

### 3.3 Patch parameters and full coverage

| Model | n_layers | n_heads | depth | heads/patch | Full coverage M |
|-------|----------|---------|-------|-------------|-----------------|
| GPT-2 small | 12 | 12 | 3 | 4 | 12 |
| GPT-2 medium | 24 | 16 | 3 | 4 | 32 |
| GPT-2 large | 36 | 20 | 3 | 5 | 48 |

Minimum sensor count analogy: M patches ≈ K EEG sensors where each sensor
integrates from a local cluster. M=12 for GPT-2 corresponds to 12 "electrode
positions" covering the full "scalp".

### 3.4 Connection to real EEG

The 91-channel real EEG dataset has a fixed spatial resolution.  If the
degradation curve shows that M ≥ 6 retains > 80% of full-coverage
performance, then roughly half the spatial resolution is sufficient — this
sets a lower bound on the EEG sensor density needed.

**Deliverable:** `experiments/sensor_count.py`; `src/metrics/network_patches.py`;
produces degradation curves and plots them; results logged to W&B under
`sensor_count/`.

---

## Phase 4: The Circuit → EEG Mapping Architecture

**Goal:** Design and validate the mapping from circuit features to
EEG-like signals in a principled, learnable way.

This is the core technical contribution.  The mapping must:
1. Accept circuit features of varying dimensionality
2. Produce K-dimensional output (one value per "sensor")
3. Preserve task-relevant information (validated by Phases 2–3)
4. Be learnable from the circuit data alone (no real EEG required)

### 4.1 Why the current mapping is insufficient

Current: `6 scalar statistics → BatchNorm → Linear → 105D`

Problems:
- 6D input has participation ratio ≈ 1.0 (only sparsity varies)
- Linear projection cannot recover information lost by global averaging
- No architectural prior about spatial structure

### 4.2 Proposed mapping architectures

#### Architecture A: CKA matrix → Spatial EEG map

```
Input:    [B, L×L]  CKA similarity matrix (vectorised)
Network:  MLP: L² → 256 → 128 → K
Output:   [B, K]  simulated EEG amplitudes

Rationale: CKA matrix captures inter-layer representational geometry.
           Each output unit = one "electrode" = a weighted readout of
           how similar two circuit stages are.
```

#### Architecture B: Attention maps → Temporal EEG (full fidelity)

```
Input:    [B, L, H, S, S]  full attention tensors
Network:  - Pool over sequence positions: [B, L, H, H] -> [B, L, H]
          - 1D conv across layers (temporal axis): [B, L, H] -> [B, K]
Output:   [B, K]

Rationale: Attention patterns across layers = "temporal" dynamics.
           Convolution extracts how routing evolves through depth.
           K output channels = K sensors.
```

#### Architecture C: Hidden states → EEG (richest signal)

```
Input:    [B, S, hidden_dim]  last-layer hidden states
Network:  - Learnable query vectors: [K, hidden_dim]
          - Cross-attention: each query attends to the sequence
          - Output: [B, K, hidden_dim] -> linear -> [B, K]
Output:   [B, K]

Rationale: Each query = one "electrode".  The model learns which token
           positions and features each electrode is sensitive to.
           This is the analogue of a spatial filter in EEG source modelling.
```

#### Architecture D: Multi-scale (recommended starting point)

Combines CKA matrix (global, structural) with token-selective statistics
(local, content-sensitive):

```
Branch 1: CKA features [B, L²] → MLP → [B, 64]
Branch 2: Token stats  [B, K_tok × 6] → MLP → [B, 64]
Merge:    concat → Linear → [B, 105]
```

### 4.3 Training objective for the mapping

**Without real EEG (synthetic training):**
```
Loss = task_prediction_loss(decoder(mapping(circuit_features)))
     + reconstruction_loss(mapping(circuit_features), mapping(circuit_features_augmented))
```
The mapping is trained to produce representations that:
(a) predict the known task label (supervised signal from the LLM's own output)
(b) are stable under alpha augmentation (same prompt at different alpha)

**With real EEG (fine-tuning):**
```
Loss = alignment_loss(mapping(circuit_features), normalised_EEG)
     + CKA_alignment(circuit_embedding, EEG_embedding)
```
The mapping is fine-tuned to match actual EEG recordings using CKA-based
alignment (rather than a pixel-level MSE, which is too sensitive to
electrode ordering).

### 4.4 Evaluation protocol for the mapping

```
1. Extract circuit features for N prompts (known task labels)
2. Apply mapping → [N, K] simulated EEG
3. Train EEG predictor: simulated EEG → task
4. Compare AUROC to:
   (a) AUROC from raw circuit features (Phase 2)
   (b) AUROC from random K-dim projection
   (c) AUROC from real EEG recordings (if available)
```

**Deliverable:** `src/mapping/` containing all four architectures as
`nn.Module` subclasses with a unified interface; `experiments/mapping_eval.py`
for systematic comparison.

---

## Phase 5: Synthetic Participant Generation (LoRA variants)

**Goal:** Simulate multiple participants by using different LLM variants,
each representing a slightly different "cognitive routing style".
Test whether the encoder generalises across participants.

### 5.1 Why LoRA for synthetic participants

Real inter-individual differences in brain function manifest as:
- Different attention patterns for the same stimulus
- Different relative weights of processing stages
- Different amounts of task-specificity

LoRA (Low-Rank Adaptation) injects small rank-r matrices into the attention
layers: `W' = W + AB` where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d}.
Different LoRA initialisations produce models with:
- The same overall capability (base model unchanged)
- Different routing micro-structure (attention patterns differ)
- Systematic, controllable variation (LoRA rank controls amount of deviation)

This is a principled way to generate "participant diversity" without real data.

### 5.2 Experimental design

```
1. Create N_participants LoRA variants of GPT-2 (N=20–50)
   - Rank r ∈ {2, 4, 8} (controls deviation magnitude)
   - Applied to: q_proj, v_proj (attention routing layers only)

2. For each variant:
   - Extract circuit features on the narrative prompt set
   - Store with participant ID label

3. Participant-agnostic encoder training:
   - Train encoder on variants 1–(N-10)
   - Test generalisation on variants (N-9)–N
   - Metric: does task AUROC stay above chance on held-out "participants"?

4. Participant-specific fine-tuning:
   - Take the general encoder
   - Fine-tune with K labelled trials per held-out participant
   - Show learning curve: how many trials needed for good personalisation?
```

### 5.3 Connection to real EEG participants

The LoRA experiment is a proxy for the key EEG question:
> Can an encoder trained on participant A's brain (circuits) be transferred
> to participant B with minimal adaptation?

If the answer is yes from the LoRA experiment, it motivates trying the same
transfer with real EEG:
- Train on subjects 1–35
- Fine-tune on 5 trials from subject 36
- Test on remaining trials of subject 36

**Deliverable:** `src/synthetic_participants/lora_variants.py`;
`experiments/cross_participant.py`; W&B run comparing generalisation
across LoRA rank values.

---

## Phase 6: Real EEG Alignment (closing the loop)

**Goal:** Connect the synthetic circuit–EEG mapping to real human EEG data.

### 6.1 Data alignment

For each narrative EEG trial:
- LLM input: `prompt_with_condition` string
- LLM output: circuit features (CKA + token-selective) at alpha=1.0
- Real EEG: 91-channel, 1900 timepoints at 100 Hz, averaged over trial

**Feature alignment:**
```
Circuit features  [D]  ←→  EEG features  [91 channels × n_timepoints]
```
The mapping target is not the raw EEG waveform, but a learned EEG embedding:
```
EEG encoder (existing PatchEEGEncoder) → pooled [embed_dim] vector
Circuit features → mapping → [embed_dim] vector
Loss: CKA alignment or contrastive (same trial = positive pair)
```

### 6.2 Validation: does the mapping preserve task information?

```
For each trial:
  - Mapped circuit embedding → task predictor → AUROC
  - Real EEG embedding → task predictor → AUROC

Test: mapped_AUROC ≈ real_AUROC ?
```
If yes: the circuit-derived EEG simulation is behaviourally equivalent to
the real recording — the mapping correctly captures how EEG encodes circuits.

### 6.3 The sensor count result in the context of real data

The degradation curve from Phase 3 (AUROC vs K) can now be compared against
the actual 91-channel EEG performance.
- If 91-channel AUROC ≈ K=91 on the degradation curve: real EEG is
  well-modelled by the uniform sensor-count model.
- If 91-channel AUROC > K=91: real EEG contains information that the
  uniform projection misses (e.g., temporal dynamics, spatial structure).
- If 91-channel AUROC < K=91: there is real measurement noise in EEG that
  the simulation does not model — motivates adding noise augmentation.

---

## Implementation Roadmap

```
Phase 0  [now]        Data foundation, signal ratio diagnostic
Phase 1  [week 1–2]   CKA features + token-selective features
Phase 2  [week 2–3]   Claim A baseline (circuit sufficiency)
Phase 3  [week 3–4]   Sensor count degradation experiment
Phase 4  [week 4–6]   Mapping architectures (A–D)
Phase 5  [week 6–8]   LoRA synthetic participants
Phase 6  [week 8–10]  Real EEG alignment
```

### File structure additions

```
src/
  metrics/
    cka_metrics.py          # Linear and kernel CKA, batched
    token_selective.py      # Per-anchor-token attention statistics
  mapping/
    __init__.py
    architectures.py        # Architectures A–D as nn.Module
    training.py             # Training loops (synthetic + real-EEG fine-tune)
  synthetic_participants/
    lora_variants.py        # LoRA variant generation
    cross_participant.py    # Cross-participant evaluation utilities
experiments/
  claim_a_baseline.py       # Phase 2
  sensor_count.py           # Phase 3
  mapping_eval.py           # Phase 4
  cross_participant.py      # Phase 5
scripts/
  audit_feature_variance.py # Signal ratio diagnostic
  run_claim_a.py
  run_sensor_count.py
  run_mapping_eval.py
  run_lora_participants.py
docs/
  WORKPLAN.md               # This document
  phase_transition_analysis.md
  circuit_eeg_mapping.md    # Detailed mapping architecture notes
```

---

## Key Open Questions

| Question | Addressed in |
|---|---|
| How many CKA dimensions are sufficient? | Phase 1.3 feature audit |
| Is alpha=1.0 the right operating point for Claim A? | Phase 2.1 (test at multiple alpha) |
| Does the K-dim degradation curve plateau at K=105? | Phase 3 |
| Which mapping architecture is most sample-efficient? | Phase 4.4 |
| Does cross-participant generalisation work with LoRA rank-2? | Phase 5.2 |
| Is the simulation noise consistent with real EEG noise? | Phase 6.3 |

---

## Success Criteria Summary

| Milestone | Criterion |
|---|---|
| Phase 1 complete | Prompt-signal ratio > 0.5 for CKA features on narrative data |
| Claim A established | Task AUROC > 0.70 from full circuit features |
| Sensor count result | Degradation curve shape identified; minimum K for AUROC > 0.6 |
| Mapping validated | Mapped-circuit AUROC within 10% of raw-circuit AUROC |
| LoRA generalisation | Held-out participant AUROC > 0.60 without fine-tuning |
| Real EEG alignment | Real-EEG AUROC ≈ K=91 on degradation curve (within 15%) |
