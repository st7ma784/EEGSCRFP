"""PyTorch Lightning module for EEG causal hypothesis testing."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import pearsonr

from src.model.sparse_attention import SparseAttentionWrapper
from src.metrics.pathway_metrics import compute_pathway_features
from src.projection.eeg_projector import EEGProjector
from src.predictor.mlp import MLPPredictor
from config.config import Config


class CausalEEGHypothesisModule(pl.LightningModule):
    """
    Lightning module for testing causal hypothesis:
    "EEG reflects projections of computational pathways."
    """
    
    def __init__(self, config: Config):
        """
        Initialize the module.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Model components
        self.sparse_model = SparseAttentionWrapper(
            model_name=config.model.model_name,
            sparsity_type=config.sparsity.sparsity_type,
        )
        self.sparse_model.eval()  # No gradient computation for the base model
        
        # EEG projector
        self.eeg_projector = EEGProjector(
            input_dim=6,  # 6 pathway metrics
            output_channels=config.projection.output_channels,
            add_noise=config.projection.add_noise,
            noise_std=config.projection.noise_std,
            smoothing_window=config.projection.smoothing_window,
        )
        
        # Predictors: pathway -> sparsity and EEG -> sparsity
        self.pathway_predictor = MLPPredictor(
            input_dim=6,
            hidden_dims=config.predictor.hidden_dims,
            dropout=config.predictor.dropout,
        )
        
        self.eeg_predictor = MLPPredictor(
            input_dim=config.projection.output_channels,
            hidden_dims=config.predictor.hidden_dims,
            dropout=config.predictor.dropout,
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Metrics tracking
        self.train_metrics = {
            'pathway_loss': [],
            'eeg_loss': [],
            'total_loss': [],
        }
        self.val_metrics = {
            'pathway_loss': [],
            'eeg_loss': [],
            'total_loss': [],
            'pathway_corr': [],
            'eeg_corr': [],
        }
    
    def forward(self, input_ids: torch.Tensor, sparsity_level: float) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire pipeline.
        
        Args:
            input_ids: [batch_size, seq_len]
            sparsity_level: Sparsity level to enforce
            
        Returns:
            Dictionary with pathway features, EEG signal, and predictions
        """
        # Set sparsity
        self.sparse_model.set_sparsity_level(sparsity_level)
        
        # Forward through sparse attention model
        with torch.no_grad():
            model_output = self.sparse_model(
                input_ids=input_ids,
                return_attention_maps=True,
                return_hidden_states=False,
            )
        
        attention_maps = model_output['attention_maps']
        
        # Compute pathway features
        pathway_features_list = []
        for batch_idx in range(input_ids.shape[0]):
            batch_attentions = [attn[batch_idx:batch_idx+1] for attn in attention_maps]
            features = compute_pathway_features(batch_attentions)
            pathway_features_list.append(features)
        
        pathway_features = torch.stack(pathway_features_list)  # [batch, 6]
        
        # Project to EEG
        eeg_signal = self.eeg_projector(pathway_features, training=self.training)
        
        # Predict sparsity from both pathways and EEG
        pathway_pred = self.pathway_predictor(pathway_features)  # [batch, 1]
        eeg_pred = self.eeg_predictor(eeg_signal)  # [batch, 1]
        
        return {
            'pathway_features': pathway_features,
            'eeg_signal': eeg_signal,
            'pathway_pred': pathway_pred.squeeze(-1),
            'eeg_pred': eeg_pred.squeeze(-1),
            'attention_maps': attention_maps,
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            batch: Batch dictionary with 'input_ids' and 'sparsity_level'
            
        Returns:
            Dictionary with loss values
        """
        input_ids = batch['input_ids']
        sparsity_levels = batch['sparsity_level']  # [batch_size]
        
        # Forward pass
        output = self.forward(input_ids, sparsity_levels.mean().item())
        
        # Get predictions
        pathway_pred = output['pathway_pred']  # [batch]
        eeg_pred = output['eeg_pred']  # [batch]
        
        # Compute losses
        pathway_loss = self.mse_loss(pathway_pred, sparsity_levels)
        eeg_loss = self.mse_loss(eeg_pred, sparsity_levels)
        
        total_loss = pathway_loss + eeg_loss
        
        return {
            'pathway_loss': pathway_loss,
            'eeg_loss': eeg_loss,
            'total_loss': total_loss,
            'pathway_pred': pathway_pred.detach(),
            'eeg_pred': eeg_pred.detach(),
            'sparsity_levels': sparsity_levels.detach(),
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss_dict = self._compute_loss(batch)
        
        # Log losses
        self.log('train/pathway_loss', loss_dict['pathway_loss'], prog_bar=True)
        self.log('train/eeg_loss', loss_dict['eeg_loss'], prog_bar=True)
        self.log('train/total_loss', loss_dict['total_loss'], prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        loss_dict = self._compute_loss(batch)
        
        # Log losses
        self.log('val/pathway_loss', loss_dict['pathway_loss'])
        self.log('val/eeg_loss', loss_dict['eeg_loss'])
        self.log('val/total_loss', loss_dict['total_loss'])
        
        # Compute correlations
        pathway_pred = loss_dict['pathway_pred'].cpu().numpy()
        eeg_pred = loss_dict['eeg_pred'].cpu().numpy()
        sparsity_levels = loss_dict['sparsity_levels'].cpu().numpy()
        
        if len(pathway_pred) > 1:
            try:
                pathway_corr, _ = pearsonr(pathway_pred, sparsity_levels)
                eeg_corr, _ = pearsonr(eeg_pred, sparsity_levels)
                
                self.log('val/pathway_corr', pathway_corr)
                self.log('val/eeg_corr', eeg_corr)
            except:
                pass
    
    def configure_optimizers(self):
        """Configure optimizer."""
        # Only train the projector and predictors (not the base model)
        params_to_train = list(self.eeg_projector.parameters()) + \
                         list(self.pathway_predictor.parameters()) + \
                         list(self.eeg_predictor.parameters())
        
        optimizer = torch.optim.AdamW(
            params_to_train,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        return optimizer


def create_lightning_module(config: Config) -> CausalEEGHypothesisModule:
    """
    Create and return Lightning module.
    
    Args:
        config: Configuration object
        
    Returns:
        CausalEEGHypothesisModule instance
    """
    return CausalEEGHypothesisModule(config)
