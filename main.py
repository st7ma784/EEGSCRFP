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
from src.lightning_module import create_lightning_module
from experiments.runner import run_all_experiments


def setup_logger():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_dataloaders(config: Config, batch_size: int = None):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        batch_size: Override batch size if provided
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.data.batch_size
    
    # Create dataset
    prompts = create_default_prompts(config.data.num_prompts)
    dataloader = create_dataloader(
        prompts=prompts,
        sparsity_levels=config.data.sparsity_levels,
        samples_per_sparsity=config.data.num_samples_per_sparsity,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    
    # Get collate function
    collate_fn = get_collate_fn(
        model_name=config.model.model_name,
        max_length=config.model.max_seq_length,
    )
    
    # Convert to use tokenized collate
    from src.data.dataset import SparsityDataset
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
        generator=torch.Generator().manual_seed(config.training.seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
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
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Create Lightning module
    logger.info("Creating Lightning module...")
    lightning_module = create_lightning_module(config)
    
    # Setup logging
    logger.info("Setting up logging...")
    tb_logger, checkpoint_callback = setup_logging_and_checkpoints(config)
    
    # Create trainer
    logger.info("Creating trainer...")
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
    
    # Train
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
    
    # Run experiments
    results = run_all_experiments(
        sparse_model=sparse_model,
        eeg_projector=eeg_projector,
        pathway_predictor=pathway_predictor,
        eeg_predictor=eeg_predictor,
        sparsity_levels=config.data.sparsity_levels,
        num_prompts=config.data.num_prompts,
        results_dir=config.results_dir,
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
    
    args = parser.parse_args()
    
    # Get config
    config = get_default_config()
    config.training.max_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.model.model_name = args.model_name
    config.data.num_prompts = args.num_prompts
    
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
