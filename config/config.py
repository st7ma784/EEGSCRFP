"""Configuration for EEG Causal Hypothesis Testing."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "gpt2"  # Small model for quick iteration
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    max_seq_length: int = 512
    

@dataclass
class SparsityConfig:
    """Sparsity control configuration."""
    sparsity_type: str = "topk"  # "topk" or "sparsemax"
    sparsity_levels: List[float] = None  # e.g., [0.1, 0.3, 0.5, 0.7, 0.9]
    topk_percent: float = 0.5  # Keep top 50% of attention weights
    
    def __post_init__(self):
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]


@dataclass
class DataConfig:
    """Data configuration."""
    num_prompts: int = 10
    num_samples_per_sparsity: int = 5
    sparsity_levels: List[float] = None
    batch_size: int = 8
    num_workers: int = 0
    # When set, load real task prompts and vividness ratings from narrative CSVs
    # instead of the built-in synthetic prompts.  Path should be the root data
    # directory containing per-subject sub-directories (e.g. /data/EEG).
    narrative_data_dir: Optional[str] = None

    def __post_init__(self):
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]


@dataclass
class MetricsConfig:
    """Pathway metrics configuration."""
    compute_routing_sparsity: bool = True
    compute_path_competition: bool = True
    compute_path_efficiency: bool = True
    compute_routing_entropy: bool = True
    compute_inter_head_divergence: bool = True
    compute_layer_stability: bool = True


@dataclass
class ProjectionConfig:
    """EEG projection configuration."""
    output_channels: int = 105
    add_noise: bool = True
    noise_std: float = 0.1
    smoothing_window: Optional[int] = None


@dataclass
class PredictorConfig:
    """MLP predictor configuration."""
    hidden_dims: List[int] = None
    dropout: float = 0.1
    learning_rate: float = 1e-3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    gradient_clip_val: float = 1.0
    log_interval: int = 10
    val_split: float = 0.2
    seed: int = 42


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = None
    sparsity: SparsityConfig = None
    data: DataConfig = None
    metrics: MetricsConfig = None
    projection: ProjectionConfig = None
    predictor: PredictorConfig = None
    training: TrainingConfig = None
    
    # Experiment settings
    experiment_name: str = "causal_eeg_hypothesis"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.sparsity is None:
            self.sparsity = SparsityConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.metrics is None:
            self.metrics = MetricsConfig()
        if self.projection is None:
            self.projection = ProjectionConfig()
        if self.predictor is None:
            self.predictor = PredictorConfig()
        if self.training is None:
            self.training = TrainingConfig()


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
