# EEGSCRFP — Experiment Design Reference

**Last updated:** 2026-04-09  
**Status:** Active development — Phases 0–3 complete, Phases 4–6 in progress

---

## 1. Scientific Background

### 1.1 The inner-speech hypothesis

Voluntary inner speech — the experience of silently "talking to yourself" —
is associated with coordinated but spatially diffuse neural activity.  EEG
can record the gross electrical consequence of this activity with high temporal
resolution but poor spatial precision: 91 electrodes each average signal from
a cortical patch of several square centimetres.

We do not know:
- How many electrodes are needed before the signal becomes decodable
- Which computational features of inner speech are captured by scalp EEG
- Whether an encoder trained on one person's EEG generalises to another

These questions are hard to answer directly because we have no ground truth
for what "perfect" inner-speech circuitry looks like.

### 1.2 The LLM as a tractable proxy

A transformer language model processing inner-speech prompts has:
- **Known ground truth**: we know exactly what it is computing (attention maps)
- **Controllable fidelity**: topk sparsity (alpha) can simulate degradation
- **Diverse stimuli**: we can use the exact narrative prompts from the EEG study
- **Multiple "participants"**: LoRA variants produce individuals with different routing styles

This gives us a closed-loop experimental system where every quantity is
measurable, unlike real EEG where the true circuit is hidden.

### 1.3 The sparsity–inner speech analogy

| LLM parameter | Inner speech analogue |
|--------------|----------------------|
| alpha = 1.0 (dense) | Vivid, elaborated inner speech (rich routing) |
| alpha → 0.1 (sparse) | Degraded, minimal inner speech (few active paths) |
| vividness_rating | Observed correlation with EEG amplitude |
| topk mask | EEG amplitude threshold (only strong signals propagate) |

Phase transition in routing metrics at critical alpha* corresponds to the
collapse of coherent inner speech structure.

---

## 2. System Architecture

### 2.1 Full pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT                                                               │
│  prompt_with_condition  (from narrative EEG trial-info CSVs)        │
│  e.g. "Imagine reliving the time you felt proud [Self_Relive]"      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  tokenised → [B, S]
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FROZEN LLM  (GPT-2 / GPT-2-medium)                                 │
│                                                                      │
│  ┌─────────────┐   topk(alpha)   ┌───────────────────────────────┐  │
│  │  Layer 0    │ ─────────────►  │  attention [B, H, S, S]       │  │
│  │  Layer 1    │    mask         │  alpha = topk percent          │  │
│  │  ...        │                 │  1.0 → dense                   │  │
│  │  Layer L-1  │                 │  0.1 → only top 10% weights   │  │
│  └─────────────┘                 └───────────────────────────────┘  │
│                                                                      │
│  NO TRAINING — only inference under torch.no_grad()                 │
└────────────────────┬─────────────────────────────────────────────────┘
                     │  attention_maps: List[L × Tensor[B,H,S,S]]
                     │  hidden_states:  Tuple[L+1 × Tensor[B,S,d]]
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION  (three parallel streams)                        │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Network Patches   │  │   CKA Features   │  │ Pathway Metrics  │  │
│  │                   │  │                  │  │   (debug only)   │  │
│  │ M patches, each:  │  │ pairwise CKA of  │  │ 6 scalar stats   │  │
│  │ • layer window    │  │ hidden states    │  │ routing_sparsity │  │
│  │ • head subset     │  │ [B, L*(L+1)/2+L] │  │ PCI, entropy...  │  │
│  │ → floating score  │  │                  │  │                  │  │
│  │ [B, M]           │  │                  │  │ [B, 6]           │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │ Primary              │ Phase 1-2            │ Diagnostic │
└───────────┼──────────────────────┼──────────────────────┼────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌──────────────────────────────────────────────────┐  W&B debug
│  ENCODER  (what we train — 3 heads)               │  plots only
│                                                   │
│  Architecture A: Linear probe (baseline)          │
│  Architecture B: MLP [M → 256 → 128 → head]       │
│  Architecture C: Transformer over patch tokens    │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  Head A: task_label prediction             │  │
│  │          Softmax → [n_conditions]          │  │
│  │          Loss: CrossEntropy                │  │
│  │          Metric: macro AUROC              │  │
│  ├────────────────────────────────────────────┤  │
│  │  Head B: prompt token reconstruction       │  │
│  │          Autoregressive or bag-of-words    │  │
│  │          Loss: CrossEntropy over vocab     │  │
│  │          Metric: BLEU, semantic similarity │  │
│  ├────────────────────────────────────────────┤  │
│  │  Head C: alpha regression                  │  │
│  │          Linear → scalar                   │  │
│  │          Loss: MSE                         │  │
│  │          Metric: R², Spearman r            │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 2.2 Network patch detail

```
Transformer layers (depth = 12 for GPT-2)

Layer  0  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
          │ H0 │ H1 │ H2 │ H3 │ H4 │ H5 │ H6 │ H7 │ H8 │ H9 │H10 │H11 │
Layer  1  ├────┴────┴────┴────┼────┴────┴────┴────┼────┴────┴────┴────┤
          │   PATCH 0         │   PATCH 1          │   PATCH 2         │
Layer  2  │   layers 0-2      │   layers 0-2       │   layers 0-2      │
          │   heads 0-3       │   heads 4-7        │   heads 8-11      │
          ├───────────────────┼────────────────────┼───────────────────┤
Layer  3  │   PATCH 3         │   PATCH 4          │   PATCH 5         │
Layer  4  │   layers 3-5      │   layers 3-5       │   layers 3-5      │
Layer  5  │   heads 0-3       │   heads 4-7        │   heads 8-11      │
          ├───────────────────┼────────────────────┼───────────────────┤
Layer  6  │   PATCH 6         │   PATCH 7          │   PATCH 8         │
Layer  7  │   layers 6-8      │   layers 6-8       │   layers 6-8      │
Layer  8  │   heads 0-3       │   heads 4-7        │   heads 8-11      │
          ├───────────────────┼────────────────────┼───────────────────┤
Layer  9  │   PATCH 9         │   PATCH 10         │   PATCH 11        │
Layer 10  │   layers 9-11     │   layers 9-11      │   layers 9-11     │
Layer 11  │   heads 0-3       │   heads 4-7        │   heads 8-11      │
          └───────────────────┴────────────────────┴───────────────────┘

Full coverage (GPT-2): 4 layer windows × 3 head groups = 12 patches
```

### 2.3 Floating score algorithm

For a patch covering layers L_p and heads H_p:

```
1. Extract [n_l × n_h, B, S, S] attention weights from all (layer, head) pairs
2. Mean-pool → [B, S, S]  (patch_attn)
3. Calibration baseline: run all N reference prompts once → store mean routing [S, S]
4. Deviation:
   z[b, ij] = (patch_attn[b, ij] - ref[ij]) / ref_std[ij]   ∀ position (i,j)
   floating_score[b] = mean(z[b]²)  over all S² positions
```

High floating score = this input's routing through this patch deviates from
the population average → patch is responding to input-specific content.

---

## 3. Three Encoder Objectives

### 3.1 Objective A — Task / semantic prediction

**What:** Given [B, M] patch features, predict the condition label
(e.g. "Self_Relive" vs "Non-Self_Relive" vs "Self_Evaluate" etc.)

**Why:** Confirms that the patch readout encodes *what* the model is computing,
not just *how much* it is computing.  If AUROC is high at M=12 and degrades
gracefully as M decreases, the signal is genuine.

**Success criterion:** AUROC > 0.70 at M=full_coverage (matching Phase 2 CKA baseline)

**W&B key:** `task/auroc`, `task/loss`

### 3.2 Objective B — Prompt reconstruction

**What:** Decode the patch features back to the input text, either as:
- Bag-of-words (simpler): predict which vocabulary tokens are present
- Full autoregressive: predict the token sequence

**Why:** Tests whether the patch features contain enough information to
reconstruct the semantic content of the prompt.  A high reconstruction score
would confirm that the floating-score readout is genuinely content-sensitive
(not just detecting activity level).

**Implementation:** Use the frozen LLM's token embedding matrix as the
reconstruction target — we align the encoder output to the embedding of the
correct token sequence (cosine similarity loss in embedding space).

**Success criterion:** Cosine similarity to LLM embedding > 0.60 at full coverage

**W&B key:** `recon/cosine_sim`, `recon/loss`

### 3.3 Objective C — Alpha (sparsity) prediction

**What:** Given the [B, M] patch features, predict the alpha value that
generated them.

**Why:** This is a *control* objective.  We expect this to be easy (high R²
even at low M) because the 6 pathway metrics already show that alpha explains
most of the variance in global statistics.  Comparing Objective C to Objectives
A and B tells us:
- If alpha-R² is high but task-AUROC is low → patches measure activity level,
  not content (bad — we need more patches or a better encoder)
- If task-AUROC is high → patches encode content (good)

**Success criterion:** R² > 0.80 at M=full_coverage (this should be easy to meet)

**W&B key:** `alpha/r2`, `alpha/loss`

---

## 4. Three Axes of Variation

### 4.1 Axis 1 — Encoder architecture

| Encoder | Input | Parameters | Expected to win on |
|---------|-------|------------|-------------------|
| Linear probe | [B, M] | M × n_class | Baseline — if this works, signal is linearly separable |
| MLP-small | [B, M] | ~10k | Good for task prediction |
| MLP-large | [B, M] | ~100k | May overfit if N is small |
| Transformer | [B, M, 1] (M "tokens") | ~50k | Should capture patch interactions |

The sweep (`configs/sweep/encoder_architecture.yaml`) holds M constant at
full_coverage and varies the encoder.

### 4.2 Axis 2 — Patch count M (EEG simulation fidelity)

For GPT-2 full_coverage = 12 patches.  We sweep M ∈ {1, 2, 4, 8, 12}.

The degradation curve shape predicts the minimum real EEG sensor count:
```
AUROC
 1.0 ├──────────────────────────────────────────●
     │                                      ●
     │                                  ●
 0.7 │                         ●──────
     │                 ●
     │       ●
 0.5 ├──────
     └─────────────────────────────────────────
         1    2    4    8   12     M (patches)
              ↑
          Elbow = minimum useful sensor count
```

Flat curves (elbow at M=2–3) → inner-speech signal is concentrated,
few EEG electrodes would suffice.  Steep curves → spread signal, many sensors needed.

### 4.3 Axis 3 — Participant variation (LoRA variants)

Different LoRA initialisations of GPT-2 produce models with the same
base capability but different attention micro-structure — analogous to
inter-individual differences in neural routing.

| LoRA rank | Routing deviation | Analogue |
|-----------|------------------|---------|
| r=0 (base GPT-2) | None | "Average participant" |
| r=2 | Small | Similar to population |
| r=8 | Moderate | Different cognitive style |
| r=32 | Large | Highly idiosyncratic routing |

The cross-participant sweep tests whether an encoder trained on one LoRA
variant (participant) generalises to another.  Success here motivates the
real EEG cross-subject experiment in Phase 6.

---

## 5. Experiments

### Experiment 0: Pre-flight — feature variance audit

**Script:** `scripts/audit_feature_variance.py`

**Question:** Are the features content-sensitive (eta²_prompt > 0.3)?

**Run:**
```bash
python scripts/audit_feature_variance.py --data-dir /data/EEG
```

**Gate:** If eta²_prompt < 0.1 for CKA features, do not proceed — investigate
data quality or prompt diversity first.

**W&B:** Not logged — diagnostic only, check terminal output.

---

### Experiment 1: Phase transition analysis

**Script:** `scripts/run_phase_transition.py`

**Question:** At what alpha* does routing structure collapse?  Is there a
phase transition in the quality of circuit information?

**Pipeline:**
```
Sweep alpha ∈ [0.05, 1.0] (30-50 steps)
  For each alpha:
    Run N prompts through frozen LLM
    Compute 6 pathway metrics + CKA features
    Compute task AUROC and vividness R² via CV probes
  Detect phase transition in PCI / routing entropy curves
  Report alpha* (median of 3 detection methods) + confidence
```

**Expected result:** Sharp inflection in PCI at alpha* ≈ 0.15–0.25.  Task AUROC
should stay high above alpha* and drop sharply below.

**W&B panels to check:**
```
phase_transition/pci_vs_alpha          (should show an S-curve or knee)
phase_transition/task_auroc_vs_alpha   (should show plateau then drop)
phase_transition/alpha_star            (summary statistic)
```

---

### Experiment 2: Claim A — circuit sufficiency baseline

**Script:** `experiments/claim_a_baseline.py`

**Question:** Does complete circuit information (CKA features at alpha=1.0)
predict task condition and vividness rating?  This is the performance ceiling.

**Pipeline:**
```
Extract CKA features for all N prompts at alpha=1.0
Stratified 5-fold CV:
  → Logistic regression → task AUROC
  → Ridge regression   → vividness R²
Cross-subject split (train on subjects 1-35, test 36-45)
```

**Success criterion:**
- Task macro-AUROC > 0.70
- Vividness R² > 0.15 (signal present, even if weak)
- Cross-subject AUROC within 10% of within-subject

**W&B panels:**
```
claim_a/task_auroc_cv        (should be > 0.70)
claim_a/vividness_r2_cv      (should be > 0.15)
claim_a/task_auroc_cross_sub (generalisation check)
claim_a/per_condition_auroc  (which tasks are most decodable?)
```

---

### Experiment 3: Claim B — sensor count degradation

**Script:** `experiments/sensor_count.py`

**Question:** How quickly does decoding accuracy degrade as we reduce the
number of patches from full_coverage to 1?

**Pipeline:**
```
Calibrate PatchFeatureExtractor on full prompt set
For M in [1, 2, 4, 8, 12]:
  Repeat 5 times with different random patch subsets:
    Extract [N, M] floating scores
    CV probe → task AUROC
    CV probe → vividness R²
  Record mean ± std
Plot degradation curves
```

**Success criterion:**
- At M=12 (full coverage): AUROC matches Claim A CKA baseline
- At M=6: AUROC retains > 80% of full-coverage value
- At M=1: AUROC is above chance (>0.5) — some signal even in one patch

**W&B panels:**
```
sensor_count/task_auroc_vs_m      (the main degradation curve)
sensor_count/vividness_r2_vs_m
sensor_count/degradation_table    (downloadable table)
```

---

### Experiment 4: Encoder comparison

**Config:** `configs/sweep/encoder_architecture.yaml`

**Question:** For a fixed M=full_coverage, which encoder architecture best
predicts task condition (Obj A), reconstructs the prompt (Obj B), and
predicts alpha (Obj C)?

**Variables:** encoder_type ∈ {linear, mlp_small, mlp_large, transformer}

**W&B panels (per run + comparison across sweep):**
```
encoder/task_auroc
encoder/recon_cosine_sim
encoder/alpha_r2
encoder/param_count (for efficiency comparison)
```

---

### Experiment 5: LoRA participant generalisation

**Config:** `configs/sweep/lora_participants.yaml`

**Question:** Does an encoder trained on one LoRA variant (participant)
transfer to another?

**Variables:** lora_rank ∈ {0, 2, 4, 8, 16, 32}; train_participants, test_participants

**W&B panels:**
```
lora/within_participant_auroc   (upper bound)
lora/cross_participant_auroc    (generalisation — should be close)
lora/auroc_drop_vs_rank         (does higher rank = harder transfer?)
```

---

## 6. Critical Algorithms

### 6.1 Top-k sparse attention

```python
# Applied to stacked attention tensor [L*B, H, S, S] in one dispatch
flat = attn.reshape(-1, S, S)            # [N, S, S]
k = max(1, int(S * alpha))
topk_vals, topk_idx = torch.topk(flat, k=k, dim=-1)
sparse = torch.zeros_like(flat)
sparse.scatter_(-1, topk_idx, topk_vals)
sparse = sparse / (sparse.sum(-1, keepdim=True) + 1e-8)
```

alpha=1.0 → k=S → all weights kept (dense)
alpha=0.1 → k=⌊0.1S⌋ → only the top 10% of keys per query survive

### 6.2 Floating score

```python
# patch_attn: [B, S, S] — pooled attention over layers and heads in patch
# reference:  [S, S]   — calibrated population mean
flat = patch_attn.reshape(B, S*S)
ref_std = flat.std(0, keepdim=True).clamp(min=1e-8)
z = (flat - reference.reshape(1, S*S)) / ref_std
floating_score = z.pow(2).mean(-1)   # [B]
```

### 6.3 Linear CKA (vectorised — all pairs × all samples in one pass)

The naive implementation loops over B samples, computing O(M²) matrix
products per sample.  The vectorised implementation exploits:

```
HSIC(X, Y) = ||X̃ᵀ Ỹ||²_F  = trace(K_X K_Y)
```

where K = X̃ X̃ᵀ is the centered Gram matrix.  For all M layers simultaneously:

```python
# H: [M, B, S, d]  (M = n_layers+1)
H_c = H - H.mean(2, keepdim=True)                # center over sequence
H_flat = H_c.reshape(M * B, S, d)
K_flat_3d = torch.bmm(H_flat, H_flat.transpose(1,2))  # [M*B, S, S]
K = K_flat_3d.reshape(M, B, S * S)               # [M, B, S²]
K_T = K.permute(1, 0, 2)                         # [B, M, S²]
# All pair HSICs at once:
HSIC_all = torch.bmm(K_T, K_T.transpose(1, 2))   # [B, M, M]
denom = torch.sqrt(
    HSIC_all.diagonal(dim1=1, dim2=2).unsqueeze(2) *   # [B, M, 1]
    HSIC_all.diagonal(dim1=1, dim2=2).unsqueeze(1)     # [B, 1, M]
)
CKA_all = HSIC_all / (denom + 1e-10)             # [B, M, M]
# Upper triangle (row-major) → [B, n_pairs]
```

This replaces a Python loop of B × M²/2 iterations with three batched matmuls.

### 6.4 Variance decomposition (eta²)

Used in the pre-flight diagnostic to confirm features are content-sensitive:

```python
# feats: [A, N, D]  (A alpha levels × N prompts × D features)
grand_mean = feats.mean((0, 1), keepdims=True)    # [1, 1, D]
prompt_means = feats.mean(0, keepdims=True)        # [1, N, D]
alpha_means  = feats.mean(1, keepdims=True)        # [A, 1, D]

ss_total  = ((feats - grand_mean)**2).sum((0, 1))
ss_prompt = A * ((prompt_means - grand_mean)**2).sum((0, 1))
ss_alpha  = N * ((alpha_means  - grand_mean)**2).sum((0, 1))

eta2_prompt = ss_prompt / (ss_total + 1e-30)   # target: > 0.3
eta2_alpha  = ss_alpha  / (ss_total + 1e-30)   # should be < 0.5 for CKA
```

---

## 7. Debugging with 6-scalar pathway metrics

The 6 global pathway statistics are not used for prediction (they are
alpha-dominated), but they are invaluable as debugging instruments.
Log them alongside every experiment to diagnose:

| Metric | What to look for | Warning sign |
|--------|-----------------|--------------|
| `routing_sparsity` | Increases monotonically with alpha | Flat → sparsity injection not working |
| `path_competition_index` | High PCI at alpha* → phase transition | No inflection → weak signal |
| `routing_entropy` | Decreases as alpha decreases | Never decreases → model ignores sparsity |
| `inter_head_divergence` | High = heads specialise | Always low → heads are homogeneous |
| `layer_stability` | High = consistent routing across layers | Very low → unstable routing |
| `path_efficiency` | Tracks alpha closely | Decoupled from alpha → data issue |

Log all 6 as histograms (not just means) at the reference alpha=1.0 and
at alpha=0.2.  The histogram separation validates that routing actually differs.

---

## 8. Phase transition diagram

```
     PCI
      │
  4.0 │                                  ●●●●●●●●
      │                              ●●●●
      │                         ●●●●
  2.0 │                    ●●●●●
      │               ●●●●
      │         ●●●●●
  0.5 │●●●●●●●●
      │
      └──────────────────────────────────────────
      0.05  0.1  0.15  0.2  0.25  0.3   ...  1.0
                        ▲
                      alpha*
                  (phase transition)

      Task AUROC
      │
  0.9 │                              ●●●●●●●●●●
      │                         ●●●●
  0.7 │                    ●●●●
  0.5 │●●●●●●●●●●●●●●●●●●●
      │
      └──────────────────────────────────────────
      0.05  0.1  0.15  0.2  0.25  0.3   ...  1.0
```

The phase transition at alpha* marks the boundary:
- Below alpha*: routing is too sparse to carry task-specific information
- Above alpha*: circuit information is decodable

---

## 9. Connection to real EEG (Phase 6)

Once the synthetic experiments establish minimum sensor count M* and best
encoder architecture, we apply the framework to real EEG:

```
Real EEG recording [91 channels × 1900 timepoints]
         │
         ▼ (standardise to 67 10-10 channels via ZUCO_STANDARDIZER)
         │
         ▼ (patch the channels spatially — group by electrode neighbourhood)
         │
         ▼ (pool within-neighbourhood ERPs → M* floating-score analogues)
         │
         ▼
EEGViewer encoder (trained or fine-tuned from scratch)
         │
         ▼
Task AUROC / vividness R²
```

The connection requires resolving the mapping between:
- LLM network patches (indexed by layer and head)
- EEG electrode neighbourhoods (indexed by scalp position)

This is the Phase 4/6 mapping architecture problem.
