"""PyTorch Lightning module for EEG causal hypothesis testing."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict
from scipy.stats import pearsonr

from src.model.sparse_attention import SparseAttentionWrapper
from src.metrics.pathway_metrics import compute_pathway_features, PathwayMetricsComputer
from src.projection.eeg_projector import EEGProjector
from src.predictor.mlp import MLPPredictor
from config.config import Config


class CausalEEGHypothesisModule(pl.LightningModule):
    """
    Test causal hypothesis: "EEG reflects projections of computational pathways."

    GPU notes
    ---------
    * ``pathway_metrics_computer`` is an ``nn.Module`` so it moves with ``.to(device)``.
    * ``forward()`` computes pathway features for the entire batch in one
      vectorised pass — no Python loop over batch items.
    * ``EEGProjector.forward()`` uses ``self.training`` internally; no
      ``training=`` kwarg is passed here.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.sparse_model = SparseAttentionWrapper(
            model_name=config.model.model_name,
            sparsity_type=config.sparsity.sparsity_type,
        )
        self.sparse_model.eval()

        # Pathway metrics computer registered as submodule (moves to GPU with self)
        self.pathway_metrics_computer = PathwayMetricsComputer()

        self.eeg_projector = EEGProjector(
            input_dim=6,
            output_channels=config.projection.output_channels,
            add_noise=config.projection.add_noise,
            noise_std=config.projection.noise_std,
            smoothing_window=config.projection.smoothing_window,
        )

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

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        sparsity_level: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Full pipeline forward for an entire batch.

        Args:
            input_ids: [B, S]
            sparsity_level: scalar applied uniformly across the batch
                (uses the batch-mean vividness when training with real data)

        Returns:
            dict with pathway_features [B,6], eeg_signal [B,C],
            pathway_pred [B], eeg_pred [B], attention_maps list[B,H,S,S]
        """
        self.sparse_model.set_sparsity_level(sparsity_level)

        with torch.no_grad():
            model_output = self.sparse_model(
                input_ids=input_ids,
                return_attention_maps=True,
                return_hidden_states=False,
            )

        attention_maps = model_output["attention_maps"]

        # Batched — no Python loop over samples
        pathway_features = self.pathway_metrics_computer(attention_maps)  # [B, 6]
        eeg_signal = self.eeg_projector(pathway_features)                  # [B, C]

        pathway_pred = self.pathway_predictor(pathway_features).squeeze(-1)  # [B]
        eeg_pred = self.eeg_predictor(eeg_signal).squeeze(-1)                # [B]

        return {
            "pathway_features": pathway_features,
            "eeg_signal": eeg_signal,
            "pathway_pred": pathway_pred,
            "eeg_pred": eeg_pred,
            "attention_maps": attention_maps,
        }

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        sparsity_levels = batch["sparsity_level"]            # [B]

        # Single sparsity level per batch — use mean of per-sample vividness
        output = self.forward(input_ids, sparsity_levels.mean().item())

        pathway_loss = self.mse_loss(output["pathway_pred"], sparsity_levels)
        eeg_loss = self.mse_loss(output["eeg_pred"], sparsity_levels)
        total_loss = pathway_loss + eeg_loss

        return {
            "pathway_loss": pathway_loss,
            "eeg_loss": eeg_loss,
            "total_loss": total_loss,
            "pathway_pred": output["pathway_pred"].detach(),
            "eeg_pred": output["eeg_pred"].detach(),
            "sparsity_levels": sparsity_levels.detach(),
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict = self._compute_loss(batch)
        self.log("train/pathway_loss", loss_dict["pathway_loss"], prog_bar=True)
        self.log("train/eeg_loss", loss_dict["eeg_loss"], prog_bar=True)
        self.log("train/total_loss", loss_dict["total_loss"], prog_bar=True)
        return loss_dict["total_loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss_dict = self._compute_loss(batch)
        self.log("val/pathway_loss", loss_dict["pathway_loss"])
        self.log("val/eeg_loss", loss_dict["eeg_loss"])
        self.log("val/total_loss", loss_dict["total_loss"])

        pathway_pred = loss_dict["pathway_pred"].cpu().numpy()
        eeg_pred = loss_dict["eeg_pred"].cpu().numpy()
        sparsity = loss_dict["sparsity_levels"].cpu().numpy()

        if len(pathway_pred) > 1:
            try:
                self.log("val/pathway_corr", pearsonr(pathway_pred, sparsity)[0])
                self.log("val/eeg_corr", pearsonr(eeg_pred, sparsity)[0])
            except Exception:
                pass

    def configure_optimizers(self):
        params = (
            list(self.eeg_projector.parameters())
            + list(self.pathway_predictor.parameters())
            + list(self.eeg_predictor.parameters())
        )
        return torch.optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )


def create_lightning_module(config: Config) -> CausalEEGHypothesisModule:
    return CausalEEGHypothesisModule(config)
