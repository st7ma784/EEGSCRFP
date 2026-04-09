# EEG Causal Hypothesis Testing - PyTorch Lightning Prototype

A modular PyTorch Lightning prototype for testing the causal hypothesis:

> **"EEG reflects projections of computational pathways."**

## Overview

This prototype tests whether changes in internal neural routing structure (via sparse attention) remain observable after projection to EEG-like signals. It implements:

1. **Sparse Attention Control**: Dynamically modify attention sparsity in a transformer
2. **Pathway Metrics**: Extract 6 routing/pathway metrics from attention patterns
3. **EEG Projection**: Project pathway features to simulated EEG signals (105 channels)
4. **Predictive Tasks**: Train MLPs to predict sparsity from pathway features vs EEG signals
5. **Causal Experiments**: Test 3 key hypotheses about pathway-EEG relationships

## Project Structure

```
EEGSCRFP/
├── src/
│   ├── model/
│   │   └── sparse_attention.py       # Sparse attention wrapper
│   ├── data/
│   │   ├── dataset.py                 # Sparsity dataset
│   │   └── tokenizer.py               # Text tokenization utilities
│   ├── metrics/
│   │   └── pathway_metrics.py         # Pathway metrics computation
│   ├── projection/
│   │   └── eeg_projector.py           # EEG projection layer
│   ├── predictor/
│   │   └── mlp.py                     # MLP predictor
│   └── lightning_module.py            # Main Lightning module
├── config/
│   └── config.py                      # Configuration management
├── experiments/
│   └── runner.py                      # Experiment runners
├── main.py                            # Main training script
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Key Components

### 1. Sparse Attention Model (`src/model/sparse_attention.py`)

Wraps HuggingFace transformers and injects sparsity into attention computation:

- **Top-K Masking**: Keep top k% of attention weights, zero out rest
- **Sparsemax**: Temperature-based control of attention distribution
- Sparsity is the **only intervention** - not leaked into features directly

### 2. Pathway Metrics (`src/metrics/pathway_metrics.py`)

Extracts 6 metrics from attention maps:

1. **Routing Sparsity**: Effective number of active paths (Rényi entropy)
2. **Path Competition Index**: max(attention) / mean(attention)
3. **Path Efficiency**: Energy concentration in top-k weights
4. **Routing Entropy**: Shannon entropy of attention distributions
5. **Inter-head Divergence**: KL divergence between attention heads
6. **Layer-wise Stability**: Cosine similarity between consecutive layers

### 3. EEG Projection (`src/projection/eeg_projector.py`)

Linear projection: `pathway_metrics (6D) -> EEG (105 channels)`

- Optional Gaussian noise injection
- Optional temporal smoothing
- Trained via gradient descent

### 4. Predictors (`src/predictor/mlp.py`)

Two independent MLPs:

- **Pathway Predictor**: Pathway metrics → Sparsity level
- **EEG Predictor**: EEG signals → Sparsity level

Architecture: `input_dim -> hidden_dim_1 -> hidden_dim_2 -> 1`

### 5. Lightning Module (`src/lightning_module.py`)

Orchestrates training:

```
Input Text
    ↓
Sparse Attention Model (frozen)
    ↓
Attention Maps
    ↓
Pathway Metrics
    ↓
EEG Projection (trained)
    ↓
Pathway & EEG Features
    ↓
Two MLPs (trained)
    ↓
Sparsity Predictions
    ↓
MSE Loss: L = L_pathway + L_eeg
```

## Installation

```bash
# Clone/setup
cd EEGSCRFP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training (5 minutes on GPU)

```bash
python main.py \
    --epochs 5 \
    --batch-size 8 \
    --num-prompts 3 \
    --model-name gpt2
```

### Full Training (20-30 minutes on GPU)

```bash
python main.py \
    --epochs 20 \
    --batch-size 4 \
    --num-prompts 10 \
    --model-name gpt2 \
    --learning-rate 0.001
```

### Skip Training, Run Experiments Only

```bash
python main.py --skip-training
```

## Configuration

Edit `config/config.py` to customize:

```python
config = Config(
    model=ModelConfig(model_name="gpt2"),
    sparsity=SparsityConfig(
        sparsity_type="topk",
        sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        topk_percent=0.5,
    ),
    data=DataConfig(
        num_prompts=10,
        num_samples_per_sparsity=5,
        batch_size=8,
    ),
    projection=ProjectionConfig(
        output_channels=105,
        add_noise=True,
        noise_std=0.1,
    ),
    training=TrainingConfig(
        max_epochs=50,
        learning_rate=1e-3,
    ),
)
```

## Experiments

Three key experiments are automatically run after training:

### Experiment 1: Pathway → Sparsity

**Question**: Can pathway metrics alone predict sparsity?

- Extracts pathway features at each sparsity level
- Trains MLP to predict sparsity from pathway metrics
- **Success metric**: High correlation with actual sparsity

### Experiment 2: EEG → Sparsity

**Question**: Can EEG signals predict sparsity?

- Projects pathway metrics to EEG via trained projector
- Trains MLP to predict sparsity from EEG signals
- **Success metric**: EEG retains predictive signal (even if weaker than pathway)

### Experiment 3: Pathway → EEG Transfer

**Question**: Does understanding pathway structure transfer to EEG?

- Collects pathway-EEG pairs across all sparsity levels
- Evaluates both predictors on held-out data
- **Success metric**: Transfer success rate = EEG_corr / Pathway_corr
  - Rate ≈ 1.0 = perfect transfer
  - Rate < 0.2 = poor transfer (signal lost in projection)

## Expected Results

### Success Criteria

✓ **Strong monotonic relationship**: Pathway metrics vary with sparsity level
- Routing sparsity should decrease with attention sparsity
- Path competition should increase with attention sparsity

✓ **EEG retains signal**: EEG-based predictor achieves r > 0.3 correlation
- Linear projection preserves some informational structure
- Noise doesn't completely destroy the signal

✓ **Generalization works**: Transfer success rate > 0.5
- EEG predictions are at least half as good as pathway predictions
- Information survives the dimensional reduction and noise

### Typical Correlations

| Experiment | Expected Correlation | Interpretation |
|-----------|----------------------|-----------------|
| Pathway → Sparsity | r > 0.6 | Strong predictive power |
| EEG → Sparsity | r > 0.3 | Moderate signal retention |
| Transfer rate | > 0.5 | Decent generalization |

## Output

Results are saved to `./results/`:

- `experiment_results.json`: Detailed results including:
  - Sparsity levels tested
  - Predictions from both pathways and EEG
  - Correlation coefficients
  - Transfer success metric

TensorBoard logs: `./logs/causal_eeg_hypothesis/`

Checkpoints: `./checkpoints/`

## Usage Examples

### Example 1: Test on Custom Prompts

Modify the prompts in `src/data/dataset.py`:

```python
custom_prompts = [
    "Your custom prompt here...",
    "Another prompt...",
]

dataloader = create_dataloader(
    prompts=custom_prompts,
    sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    batch_size=8,
)
```

### Example 2: Use Different Sparsity Levels

```python
config.sparsity.sparsity_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
config.data.sparsity_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
```

### Example 3: Analyze Single Prompt

```python
from src.model.sparse_attention import create_sparse_model
from src.metrics.pathway_metrics import compute_pathway_features
from src.data.tokenizer import TextTokenizer

model = create_sparse_model()
tokenizer = TextTokenizer()

prompt = "Your test prompt..."
tokens = tokenizer.tokenize([prompt])

for sparsity_level in [0.1, 0.5, 0.9]:
    model.set_sparsity_level(sparsity_level)
    output = model(tokens['input_ids'], return_attention_maps=True)
    metrics = compute_pathway_features(output['attention_maps'])
    print(f"Sparsity {sparsity_level}: {metrics}")
```

## Hardware Requirements

- **GPU**: NVIDIA (recommended) with CUDA support
- **Memory**: 
  - GPU: ~8GB for training
  - RAM: ~4GB
- **Training time**:
  - Quick demo (5 epochs): ~5 minutes
  - Full run (20 epochs): ~20-30 minutes

## Modular Design

Each component is independently importable:

```python
from src.model.sparse_attention import SparseAttentionWrapper
from src.metrics.pathway_metrics import PathwayMetricsComputer
from src.projection.eeg_projector import EEGProjector
from src.predictor.mlp import MLPPredictor
from config.config import get_default_config
```

## Contributing & Extending

To add new pathway metrics:

1. Add method to `PathwayMetricsComputer` in `src/metrics/pathway_metrics.py`
2. Update `compute_all_metrics()` to include new metric
3. Adjust feature vector dimension in Lightning module

To use different models:

1. Change `config.model.model_name` to any HuggingFace model ID
2. Update hidden size configs if needed
3. Run training

## References

- **PyTorch Lightning**: https://lightning.ai/
- **HuggingFace Transformers**: https://huggingface.co/transformers/
- **Attention is All You Need**: https://arxiv.org/abs/1706.03762
- **Causal Inference in ML**: https://arxiv.org/abs/1901.10912

## License

This project is provided as-is for research purposes.

## Contact & Questions

For questions about the implementation or experimental design, refer to the code documentation and inline comments.
