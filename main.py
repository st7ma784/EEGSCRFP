"""Main training script for EEG causal hypothesis testing."""
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import logging

from config.config import get_default_config, Config
from src.data.dataset import create_dataloader, create_default_prompts
from src.data.tokenizer import get_collate_fn
from src.data.narrative_loader import (
    NarrativeSparsityDataset,
    create_narrative_dataloader,
    load_narrative_records,
    vividness_to_sparsity,
)
from src.lightning_module import create_lightning_module, create_cached_module
from src.data.feature_cache import extract_and_cache, CachedFeaturesDataset
from experiments.runner import run_all_experiments


def setup_logger():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def _build_raw_dataset_and_collate(config: Config):
    """Return the raw (text + sparsity) dataset and its collate_fn.

    Used during Phase 1 (feature extraction) to build the full dataset
    without splitting it into train/val.
    """
    collate_fn = get_collate_fn(
        model_name=config.model.model_name,
        max_length=config.model.max_seq_length,
    )
    if config.data.narrative_data_dir:
        dataset = NarrativeSparsityDataset(config.data.narrative_data_dir)
    else:
        from src.data.dataset import SparsityDataset
        prompts = create_default_prompts(config.data.num_prompts)
        dataset = SparsityDataset(
            prompts=prompts,
            sparsity_levels=config.data.sparsity_levels,
            samples_per_sparsity=config.data.num_samples_per_sparsity,
        )
    return dataset, collate_fn


def create_dataloaders(config: Config, batch_size: int = None):
    """
    Create train and validation dataloaders.

    When ``config.data.narrative_data_dir`` is set, loads real task prompts
    and per-trial vividness-derived sparsity levels from the narrative CSVs.
    Otherwise falls back to the built-in synthetic prompts.

    Args:
        config: Configuration object
        batch_size: Override batch size if provided

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.data.batch_size

    collate_fn = get_collate_fn(
        model_name=config.model.model_name,
        max_length=config.model.max_seq_length,
    )

    if config.data.narrative_data_dir:
        # --- Real narrative data: prompts + vividness-derived sparsity ---
        dataset = NarrativeSparsityDataset(config.data.narrative_data_dir)
    else:
        # --- Synthetic fallback: fixed prompts × sparsity grid ---
        from src.data.dataset import SparsityDataset
        prompts = create_default_prompts(config.data.num_prompts)
        dataset = SparsityDataset(
            prompts=prompts,
            sparsity_levels=config.data.sparsity_levels,
            samples_per_sparsity=config.data.num_samples_per_sparsity,
        )

    # Split into train/val
    train_size = int(len(dataset) * (1 - config.training.val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed),
    )

    use_gpu = torch.cuda.is_available()
    nw = config.data.num_workers
    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=nw,
        pin_memory=use_gpu,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def setup_logging_and_checkpoints(config: Config):
    """
    Setup TensorBoard logger and checkpoints.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (logger, checkpoint_callback)
    """
    # Create directories
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.experiment_name,
        version=0,
    )
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="{epoch:02d}-{val/total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )
    
    return tb_logger, checkpoint_callback


def train(config: Config = None, **kwargs):
    """
    Main training function.
    
    Args:
        config: Configuration object (uses default if None)
        **kwargs: Override config parameters
    """
    logger = setup_logger()
    
    # Load or create config
    if config is None:
        config = get_default_config()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.info(f"Configuration: {config}")
    
    # Set seed
    pl.seed_everything(config.training.seed)

    tb_logger, checkpoint_callback = setup_logging_and_checkpoints(config)
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=config.training.log_interval,
        enable_progress_bar=True,
    )

    if config.data.cache_dir:
        # ------------------------------------------------------------------
        # Two-phase training: extract features once, then train cheaply.
        # ------------------------------------------------------------------
        cache_path = Path(config.data.cache_dir) / "pathway_features.pt"

        if not cache_path.exists():
            # Phase 1 — build the raw dataset + LLM, extract once
            logger.info("Phase 1: extracting pathway features (one-time cost)...")
            raw_dataset, collate_fn = _build_raw_dataset_and_collate(config)
            extraction_model = create_lightning_module(config).sparse_model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extraction_model.to(device).eval()
            extract_and_cache(
                extraction_model, raw_dataset, collate_fn, cache_path,
                device=device, batch_size=config.data.batch_size * 2,
                num_workers=config.data.num_workers,
            )
            del extraction_model  # free LLM memory before training

        # Phase 2 — train on cached [N, 6] features; no LLM needed
        logger.info(f"Phase 2: training on cached features from {cache_path}")
        cached_dataset = CachedFeaturesDataset(cache_path)
        use_gpu = torch.cuda.is_available()
        nw = config.data.num_workers
        loader_kwargs = dict(
            num_workers=nw,
            pin_memory=use_gpu,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )
        train_size = int(len(cached_dataset) * (1 - config.training.val_split))
        val_size = len(cached_dataset) - train_size
        train_ds, val_ds = random_split(
            cached_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.training.seed),
        )
        train_loader = DataLoader(train_ds, batch_size=config.data.batch_size,
                                  shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=config.data.batch_size,
                                shuffle=False, **loader_kwargs)

        lightning_module = create_cached_module(config)

    else:
        # ------------------------------------------------------------------
        # Original single-phase path: LLM on every training step.
        # ------------------------------------------------------------------
        logger.info("Creating dataloaders (LLM on every step — consider --cache-dir)...")
        train_loader, val_loader = create_dataloaders(config)
        lightning_module = create_lightning_module(config)

    logger.info("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    
    logger.info(f"Training completed. Best checkpoint: {checkpoint_callback.best_model_path}")
    
    return lightning_module, trainer


def evaluate_and_run_experiments(
    lightning_module,
    config: Config,
):
    """
    Evaluate trained module and run final experiments.

    When ``config.data.narrative_data_dir`` is set, the experiment runner
    uses the actual task prompts and their per-trial vividness-derived
    sparsity levels.  Otherwise it falls back to the built-in synthetic
    prompts with the configured sparsity grid.

    Args:
        lightning_module: Trained Lightning module
        config: Configuration object
    """
    logger = setup_logger()
    logger.info("Running final experiments...")

    # Extract component models
    sparse_model = lightning_module.sparse_model
    eeg_projector = lightning_module.eeg_projector
    pathway_predictor = lightning_module.pathway_predictor
    eeg_predictor = lightning_module.eeg_predictor

    # Resolve prompts and per-sample sparsity for experiments
    if config.data.narrative_data_dir:
        logger.info(
            f"Loading narrative prompts from '{config.data.narrative_data_dir}' "
            "for experiment evaluation..."
        )
        prompts, raw_vividness = load_narrative_records(config.data.narrative_data_dir)
        experiment_sparsity = vividness_to_sparsity(raw_vividness)
    else:
        prompts = None          # runner will use create_default_prompts
        experiment_sparsity = config.data.sparsity_levels

    # Run experiments
    results = run_all_experiments(
        sparse_model=sparse_model,
        eeg_projector=eeg_projector,
        pathway_predictor=pathway_predictor,
        eeg_predictor=eeg_predictor,
        sparsity_levels=experiment_sparsity,
        num_prompts=config.data.num_prompts,
        results_dir=config.results_dir,
        prompts=prompts,
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("="*80)
    
    exp1 = results['experiment_1_pathway_to_sparsity']
    exp2 = results['experiment_2_eeg_to_sparsity']
    exp3 = results['experiment_3_pathway_to_eeg_transfer']
    
    logger.info(f"\nExperiment 1 (Pathway -> Sparsity):")
    logger.info(f"  Correlation: {exp1['correlation']:.4f}")
    
    logger.info(f"\nExperiment 2 (EEG -> Sparsity):")
    logger.info(f"  Correlation: {exp2['correlation']:.4f}")
    
    logger.info(f"\nExperiment 3 (Pathway -> EEG Transfer):")
    logger.info(f"  Pathway Correlation: {exp3['pathway_correlation']:.4f}")
    logger.info(f"  EEG Correlation: {exp3['eeg_correlation']:.4f}")
    logger.info(f"  Transfer Success Rate: {exp3['transfer_success_rate']:.4f}")
    
    logger.info("\n" + "="*80)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EEG Causal Hypothesis Testing with PyTorch Lightning"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts for testing"
    )
    parser.add_argument(
        "--skip-training",
        action='store_true',
        help="Skip training and only run experiments"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to the narrative EEG data directory containing per-subject "
            "sub-directories with *_trialinfo[_aligned].csv files.  When set, "
            "real task prompts and vividness-derived sparsity levels are used "
            "instead of the built-in synthetic prompts."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for the pre-extracted pathway feature cache.  "
            "Enables two-phase training: the LLM runs once to extract "
            "[N,6] features (Phase 1), then all subsequent epochs train "
            "only the projector and predictors on the cached tensors (Phase 2), "
            "making each training step ~100-1000x cheaper.  "
            "The cache is reused across runs; delete pathway_features.pt to force re-extraction."
        ),
    )

    args = parser.parse_args()

    # Get config
    config = get_default_config()
    config.training.max_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.model.model_name = args.model_name
    config.data.num_prompts = args.num_prompts
    if args.data_dir:
        config.data.narrative_data_dir = args.data_dir
    if args.cache_dir:
        config.data.cache_dir = args.cache_dir
    
    if args.skip_training:
        # Just run experiments (assumes models are already trained)
        logger = setup_logger()
        logger.info("Skipping training, running experiments only...")
    else:
        # Train
        lightning_module, trainer = train(config)
        
        # Evaluate and run experiments
        evaluate_and_run_experiments(lightning_module, config)


if __name__ == "__main__":
    main()
