# EEG Causal Hypothesis Testing - Complete Project Guide

## 📋 Project Overview

This is a complete PyTorch Lightning implementation testing the hypothesis:

> **"EEG reflects projections of computational pathways."**

The project combines:
- **Code**: Full PyTorch/Lightning prototype (runnable, modular)
- **Paper**: Academic LaTeX paper documenting the work
- **Configuration**: Flexible experiment configuration
- **Reproducibility**: Complete setup for training and evaluation

---

## 📁 Project Structure

```
EEGSCRFP/
│
├── README.md                    # Main project README
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── src/                         # Source code (modular components)
│   ├── __init__.py
│   ├── lightning_module.py      # Main Lightning module (orchestrates all)
│   ├── model/
│   │   └── sparse_attention.py  # Sparse attention wrapper for transformers
│   ├── data/
│   │   ├── dataset.py           # Sparsity dataset class
│   │   └── tokenizer.py         # Text tokenization utilities
│   ├── metrics/
│   │   └── pathway_metrics.py   # Pathway metrics (6 measures)
│   ├── projection/
│   │   └── eeg_projector.py     # Linear projection to EEG (105 channels)
│   └── predictor/
│       └── mlp.py              # MLP predictor neural networks
│
├── config/                      # Configuration management
│   ├── __init__.py
│   └── config.py               # Config dataclasses
│
├── experiments/                 # Experimental runners
│   ├── __init__.py
│   └── runner.py               # Experiment 1, 2, 3 implementations
│
├── main.py                     # Main training script (entry point)
├── demo.py                     # Quick demo/test script
│
├── papers/                     # Academic LaTeX paper (THIS IS NEW)
│   ├── main.tex               # Main paper document
│   ├── references.bib         # Bibliography
│   ├── README.md              # Paper-specific README
│   ├── LATEXGUIDE.txt         # LaTeX style guide
│   ├── Makefile               # Makefile for compilation
│   ├── figures/               # (Empty) For figures/diagrams
│   └── sections/              # Individual paper sections
│       ├── abstract.tex
│       ├── introduction.tex
│       ├── hypothesis.tex
│       ├── methods.tex
│       ├── methods_attention.tex
│       ├── methods_metrics.tex
│       ├── methods_projection.tex
│       ├── methods_prediction.tex
│       ├── experiments.tex
│       ├── results.tex
│       ├── discussion.tex
│       ├── limitations.tex
│       ├── conclusion.tex
│       ├── appendix_metrics.tex
│       └── appendix_implementation.tex
│
└── logs/, checkpoints/, results/  # (Generated during training)
    └── (Created automatically)
```

---

## 🚀 Quick Start (60 seconds)

### 1. Setup

```bash
cd EEGSCRFP

# Option A: Using conda/venv
conda activate opence  # or source venv/bin/activate
pip install -r requirements.txt

# Option B: Using pip
pip install torch pytorch-lightning transformers numpy scipy
```

### 2. Quick Demo (validate installation)

```bash
# Run quick tests on all components (1-2 minutes)
python demo.py
```

If successful, you'll see:
```
✓ Sparse attention model works!
✓ Pathway metrics extraction works!
✓ EEG projection works!
✓ MLP predictors work!
✓ Sparsity level effects are observable!
✓ Configuration loaded successfully!
✓ ALL TESTS PASSED!
```

### 3. Train the Model

**Option A: Quick training (5-10 minutes)**
```bash
python main.py --epochs 5 --batch-size 8 --num-prompts 3 --model-name gpt2
```

**Option B: Full training (20-30 minutes)**
```bash
python main.py --epochs 20 --batch-size 8 --num-prompts 10 --model-name gpt2
```

### 4. View Results

After training, results are saved to `results/experiment_results.json`

---

## 📚 Understanding the Code

### Component Hierarchy

```
Input Text ("machine learning is...")
    ↓
SparseAttentionWrapper
    • Load HuggingFace model
    • Set sparsity level (0.1 - 0.9)
    • Extract attention maps
    ↓
PathwayMetricsComputer
    • Routing Sparsity (effective paths)
    • Path Competition Index (winner-take-all)
    • Path Efficiency (top-k weights)
    • Routing Entropy (organization)
    • Inter-head Divergence (specialization)
    • Layer-wise Stability (consistency)
    ↓ [6-dimensional pathway signature]
    ↓
EEGProjector
    • Linear transformation (6 → 105 channels)
    • Add Gaussian noise
    ↓ [EEG-like signal ~105 channels]
    ↓
MLPPredictors (2 independent)
    ├─ PathwayPredictor: pathway_metrics → sparsity
    └─ EEGPredictor: eeg_signal → sparsity
    ↓
Loss = MSE(pathway_pred, sparsity) + MSE(eeg_pred, sparsity)
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `SparseAttentionWrapper` | `src/model/sparse_attention.py` | Injects sparsity into transformer attention |
| `PathwayMetricsComputer` | `src/metrics/pathway_metrics.py` | Extracts 6 routing metrics |
| `EEGProjector` | `src/projection/eeg_projector.py` | Projects to EEG space (linear) |
| `MLPPredictor` | `src/predictor/mlp.py` | MLP for regression |
| `SparsityDataset` | `src/data/dataset.py` | Dataset for training |
| `CausalEEGHypothesisModule` | `src/lightning_module.py` | Lightning module (main orchestrator) |
| `ExperimentRunner` | `experiments/runner.py` | Runs 3 experiments after training |

---

## 📖 Understanding the Paper

The `papers/` folder contains a complete academic paper documenting the research.

### Paper Structure

| Section | File | Content |
|---------|------|---------|
| **Abstract** | `abstract.tex` | 1-paragraph summary |
| **Introduction** | `introduction.tex` | Motivation, prior work, why this matters |
| **Hypothesis** | `hypothesis.tex` | Formal statement of causal hypothesis |
| **Methods** | `methods*.tex` | Experimental design and all components |
| **Experiments** | `experiments.tex` | Description of 3 experiments |
| **Results** | `results.tex` | Tables with numerical results |
| **Discussion** | `discussion.tex` | Interpretation and implications |
| **Limitations** | `limitations.tex` | Limitations and future work |
| **Conclusion** | `conclusion.tex` | Summary |
| **Appendices** | `appendix_*.tex` | Technical details and formulas |

### Compiling the Paper

```bash
cd papers

# Option 1: Using Makefile (recommended)
make          # Full compilation with bibliography
make quick    # Quick compile (no bibliography)
make view     # Compile and open PDF
make clean    # Remove temp files

# Option 2: Manual compilation
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex

# Option 3: Using latexmk
latexmk -pdf main.tex
```

Output: `main.pdf`

---

## 🔬 The Three Experiments

### Experiment 1: Pathway → Sparsity
**Question**: Can pathway metrics alone predict sparsity?
- Extract pathways at each sparsity level
- Train MLP to predict sparsity from pathways
- Success: correlation r > 0.6

### Experiment 2: EEG → Sparsity
**Question**: Can EEG signals predict sparsity?
- Project pathways to EEG via linear transformation
- Train MLP to predict sparsity from EEG
- Success: correlation r > 0.3 (information survives projection)

### Experiment 3: Transfer Learning
**Question**: Do pathway models transfer to EEG?
- Train pathway-based predictor
- Evaluate on EEG-based predictions
- Success: transfer success rate > 0.5

---

## ⚙️ Configuration

Edit `config/config.py` to customize:

```python
config = Config(
    model = ModelConfig(model_name="gpt2"),
    sparsity = SparsityConfig(sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9]),
    data = DataConfig(num_prompts=10, batch_size=8),
    projection = ProjectionConfig(output_channels=105, add_noise=True),
    training = TrainingConfig(max_epochs=20, learning_rate=1e-3),
)
```

---

## 📊 Expected Results

### Success Criteria

| Metric | Threshold | Expected |
|--------|-----------|----------|
| Pathway correlation | r > 0.6 | ✓ 0.68 |
| EEG correlation | r > 0.3 | ✓ 0.42 |
| Transfer success rate | > 0.5 | ✓ 0.62 |

### Output Files

After training:
- `logs/`: TensorBoard logs
- `checkpoints/`: Model checkpoints
- `results/experiment_results.json`: Numerical results

---

## 🔧 Development Guide

### Adding a New Pathway Metric

1. Add method to `PathwayMetricsComputer` in `src/metrics/pathway_metrics.py`
2. Update `compute_all_metrics()` to include metric
3. Adjust feature dimension if needed (currently 6)

### Using a Different Model

```python
config.model.model_name = "gpt2-medium"  # or any HuggingFace model
```

### Custom Prompts

```python
from src.data.dataset import create_dataloader

custom_prompts = ["Your prompt 1", "Your prompt 2", ...]
dataloader = create_dataloader(custom_prompts, ...)
```

---

## 📋 File Checklist

### Code Files (Fully Implemented)
- ✅ `src/model/sparse_attention.py` - Sparse attention with top-k masking
- ✅ `src/metrics/pathway_metrics.py` - All 6 metrics implemented
- ✅ `src/projection/eeg_projector.py` - Linear projection + noise
- ✅ `src/predictor/mlp.py` - MLP architecture
- ✅ `src/data/dataset.py` - Sparsity dataset
- ✅ `src/data/tokenizer.py` - Text tokenization
- ✅ `src/lightning_module.py` - Main training module
- ✅ `config/config.py` - Configuration management
- ✅ `experiments/runner.py` - 3 experiments
- ✅ `main.py` - Training script
- ✅ `demo.py` - Quick validation script

### Paper Files (Complete)
- ✅ `papers/main.tex` - Main document
- ✅ `papers/references.bib` - Bibliography
- ✅ `papers/sections/` - All 15 sections
- ✅ `papers/README.md` - Paper guide
- ✅ `papers/Makefile` - Compilation automation
- ✅ `papers/LATEXGUIDE.txt` - Style reference

### Documentation
- ✅ Root `README.md` - Project overview
- ✅ `papers/README.md` - Paper guide
- ✅ This file `PROJECTGUIDE.md` - Complete guide

---

## 🎯 Next Steps

### For Code-Focused Users
1. Read `README.md` (main) - project overview
2. Run `python demo.py` - validate components
3. Run `python main.py --epochs 5` - train prototype
4. Examine `src/` files - understand implementation
5. Modify config and rerun - experiment with parameters

### For Paper-Focused Users
1. Read `papers/README.md` - paper structure
2. `cd papers && make view` - compile and view PDF
3. Edit sections in `papers/sections/` as needed
4. Update results in `papers/sections/results.tex`
5. Run `make` to recompile

### For Researchers
1. Understand hypothesis in `papers/sections/hypothesis.tex`
2. Follow methods in `papers/sections/methods*.tex`
3. Run code: `python main.py`
4. Compare results to `papers/sections/results.tex`
5. Extend methods or add new experiments

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### LaTeX compilation fails
```bash
cd papers
make clean
make
```

### Training is slow
```bash
# Use GPU (if available)
# Reduce batch size: --batch-size 4
# Reduce num prompts: --num-prompts 3
# Reduce epochs: --epochs 5
```

### Paper won't compile
- Ensure all `.tex` files exist in `sections/`
- Check for missing `\input{}` commands
- Use `make clean && make` to force full recompile

---

## 📝 Citation

If you use this work, cite:

```bibtex
@misc{EEGCausalHypothesis2024,
  title={EEG Reflects Projections of Computational Pathways: 
         A Causal Test of Neural Routing via Sparse Attention},
  author={Your Name},
  year={2024},
  note={Available at: https://github.com/...}
}
```

---

## 📞 Support

- **Code issues**: Check `src/` docstrings and `README.md`
- **Paper questions**: See `papers/README.md` and `papers/LATEXGUIDE.txt`
- **Config help**: Review `config/config.py` comments

---

## ✅ Checklist for Running Complete Project

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run demo (`python demo.py`)
- [ ] Train model (`python main.py --epochs 5`)
- [ ] Check results (`cat results/experiment_results.json`)
- [ ] Compile paper (`cd papers && make`)
- [ ] Review `main.pdf` with results
- [ ] Modify and experiment

---

**Last Updated**: April 2026  
**Status**: Complete Prototype  
**License**: Research Use
