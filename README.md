# EEGSCRFP — EEG Simulation via Sparse Causal Routing in Frozen Transformers

> **Core idea:** A frozen language model with controllable attention sparsity is
> a tractable simulator of inner speech.  Network patches read local routing
> fluctuations — acting as virtual EEG electrodes.  We train encoders on these
> patch signals to answer: *how much circuit information survives a spatially
> coarse readout, and what is the minimum sensor count for reliable decoding?*

---

## Quick orientation

```
TEXT PROMPT
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Frozen LLM  (GPT-2 / GPT-2-medium / ...)            │
│                                                       │
│  alpha = topk_percent  ─────────────────────────►    │
│          1.0 → dense routing  (rich inner speech)     │
│          0.1 → sparse routing (degraded / quiet IS)   │
└────────────────────────┬─────────────────────────────┘
                         │  attention maps [L, B, H, S, S]
                         ▼
┌──────────────────────────────────────────────────────┐
│  M Network Patches  (virtual EEG electrodes)          │
│                                                       │
│  Each patch: contiguous layer window + head subset    │
│  Output: floating score — deviation from mean routing │
│                                                       │
│  M=1  ──────────────────────────────────── M_full=12  │
│  (one electrode)                    (full scalp)      │
└────────────────────────┬─────────────────────────────┘
                         │  [B, M]  floating scores
                         ▼
┌──────────────────────────────────────────────────────┐
│  EEG Encoder  (what we train)                         │
│                                                       │
│  Objective A: task / semantic prediction  → AUROC     │
│  Objective B: prompt reconstruction      → BLEU / sim │
│  Objective C: alpha (sparsity) prediction → R²        │
└──────────────────────────────────────────────────────┘
```

Three axes of experimental variation:
| Axis | What changes | What we learn |
|------|-------------|---------------|
| **Encoder** | MLP vs transformer vs linear probe | Which architecture extracts the most task-relevant signal |
| **Patch count M** | 1 → full coverage | Minimum sensor count for reliable decoding |
| **Participants** | GPT-2 base vs LoRA variants | Whether encoder generalises across "individuals" |

---

## Why this matters

Real EEG records voltage from 91 electrodes placed on the scalp.  Each
electrode integrates neural activity from a spatial cluster of neurons —
it cannot isolate a specific computation.  We model this with network patches:
each patch pools attention weights from a local cluster of (layer, head)
pairs.  The floating score is what such an electrode would observe.

By sweeping M and tracking how quickly decoding accuracy falls, we get:
1. The **minimum sensor count** for inner-speech decoding
2. Validation that our encoder architecture is doing the right thing
3. A synthetic training signal for the EEGViewer encoders (Phases 5–6)

---

## Repository layout

```
EEGSCRFP/
├── src/
│   ├── model/
│   │   └── sparse_attention.py     # GPT-2 + topk sparsity injection
│   ├── metrics/
│   │   ├── pathway_metrics.py      # 6 diagnostic scalars (debug signal)
│   │   ├── cka_metrics.py          # CKA feature extraction (Phase 1–2)
│   │   └── network_patches.py      # Patch sampler + floating-score extractor
│   ├── data/
│   │   ├── dataset.py              # Synthetic prompt generator
│   │   ├── tokenizer.py
│   │   └── narrative_loader.py     # Loads /data/EEG trial-info CSVs
│   ├── projection/
│   │   └── eeg_projector.py        # Linear pathway → EEG projection
│   └── predictor/
│       └── mlp.py                  # MLP prediction heads
├── experiments/
│   ├── phase_transition.py         # alpha sweep + phase transition detection
│   ├── sensor_count.py             # Main Phase 3: M-patch degradation curve
│   └── claim_a_baseline.py         # Phase 2: full-circuit sufficiency
├── scripts/
│   ├── run_phase_transition.py     # CLI for phase_transition experiment
│   ├── run_sensor_count.py         # → use experiments/sensor_count.py directly
│   └── audit_feature_variance.py   # Pre-flight: eta² variance decomposition
├── configs/
│   └── sweep/
│       ├── encoder_architecture.yaml
│       ├── patch_simulation.yaml
│       ├── lora_participants.yaml
│       └── training_objectives.yaml
└── docs/
    ├── WORKPLAN.md                 # 6-phase formal research plan
    ├── EXPERIMENT_DESIGN.md        # Diagrams, hypotheses, algorithms
    └── WANDB_GUIDE.md              # W&B metric reference
```

---

## Running experiments

```bash
conda activate opence

# 0. Pre-flight: verify features are content-sensitive
python scripts/audit_feature_variance.py --data-dir /data/EEG

# 1. Phase transition (how does routing change with alpha?)
python scripts/run_phase_transition.py --data-dir /data/EEG --wandb-project eegscrfp

# 2. Claim A: does full circuit info predict task?
python experiments/claim_a_baseline.py --data-dir /data/EEG --wandb-project eegscrfp

# 3. Claim B: sensor count degradation curve
python experiments/sensor_count.py --data-dir /data/EEG --wandb-project eegscrfp

# 4. Sweep encoder architecture
wandb sweep configs/sweep/encoder_architecture.yaml
wandb agent <sweep-id>

# 5. Sweep patch simulation parameters
wandb sweep configs/sweep/patch_simulation.yaml
wandb agent <sweep-id>
```

---

## Key references

- **CKA**: Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited" https://arxiv.org/abs/1905.00414
- **Top-k sparse attention**: Correia et al. (2019) "Adaptively Sparse Transformers" https://arxiv.org/abs/1909.00015
- **LoRA**: Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models" https://arxiv.org/abs/2106.09685

See `docs/EXPERIMENT_DESIGN.md` for full scientific underpinning and algorithm descriptions.
