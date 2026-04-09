"""Experiments for testing the causal hypothesis.

GPU notes
---------
* ``run_batch()`` replaces ``run_single_level()``: it tokenises *all* prompts
  (and all noise-repeats) into a single forward pass so every CUDA call is
  maximally occupied.
* In grid-experiment mode every prompt at the same sparsity level is
  processed together — one forward pass per sparsity level instead of one
  per (prompt × repeat).
* In per-sample (vividness) mode each trial has a unique sparsity, so the
  loop over trials is unavoidable, but each iteration is a full-batch
  forward (batch = ``num_repeats`` copies of the single prompt).
* No numpy in the hot path; all tensor ops stay on-device until final
  result serialisation.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
import json
from pathlib import Path

from src.model.sparse_attention import SparseAttentionWrapper
from src.metrics.pathway_metrics import compute_pathway_features
from src.projection.eeg_projector import EEGProjector
from src.predictor.mlp import MLPPredictor
from src.data.dataset import create_default_prompts
from src.data.tokenizer import TextTokenizer


class ExperimentRunner:
    """Runner for causal hypothesis experiments."""

    def __init__(
        self,
        sparse_model: SparseAttentionWrapper,
        eeg_projector: EEGProjector,
        pathway_predictor: MLPPredictor,
        eeg_predictor: MLPPredictor,
        sparsity_levels: List[float],
    ):
        self.sparse_model = sparse_model
        self.eeg_projector = eeg_projector
        self.pathway_predictor = pathway_predictor
        self.eeg_predictor = eeg_predictor
        self.sparsity_levels = sparsity_levels
        self.device = next(sparse_model.parameters()).device

        # Pre-set all sub-modules to eval for inference
        for m in (sparse_model, eeg_projector, pathway_predictor, eeg_predictor):
            m.eval()

    # ------------------------------------------------------------------
    # Core batched forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch(
        self,
        prompts: List[str],
        sparsity_level: float,
        tokenizer: TextTokenizer,
        num_repeats: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward a batch of prompts (optionally repeated) at one sparsity level.

        When ``num_repeats > 1`` each prompt is duplicated ``num_repeats``
        times so all repeats are processed in a single forward pass; the
        returned tensors are averaged over repeats.

        Args:
            prompts: list of N text strings
            sparsity_level: sparsity to inject
            tokenizer: TextTokenizer
            num_repeats: number of stochastic repeats (averaged)

        Returns:
            features: [N, 6] pathway feature matrix (mean over repeats)
            eeg:      [N, C] EEG signal matrix (mean over repeats)
        """
        N = len(prompts)
        self.sparse_model.set_sparsity_level(sparsity_level)

        if num_repeats > 1:
            # Tile: [p0, p1, ..., p0, p1, ...] × num_repeats
            tiled = prompts * num_repeats          # length N * num_repeats
        else:
            tiled = prompts

        tokens = tokenizer.tokenize(tiled)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        output = self.sparse_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention_maps=True,
            return_hidden_states=False,
        )

        # [N*repeats, 6] and [N*repeats, C]
        features = compute_pathway_features(output["attention_maps"])
        eeg = self.eeg_projector(features)

        if num_repeats > 1:
            # Reshape to [repeats, N, ...] then mean over repeats
            features = features.reshape(num_repeats, N, -1).mean(dim=0)  # [N, 6]
            eeg = eeg.reshape(num_repeats, N, -1).mean(dim=0)             # [N, C]

        return features, eeg

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def experiment_1_pathway_to_sparsity(
        self,
        prompts: List[str],
        tokenizer: TextTokenizer,
        per_sample_sparsity: Optional[List[float]] = None,
    ) -> Dict:
        """Pathway metrics → predict sparsity.

        Grid mode (``per_sample_sparsity=None``): all prompts are processed
        together at each sparsity level — one forward pass per level.

        Per-sample mode: each (prompt, sparsity) pair is its own forward.
        """
        sparsity_gt: List[float] = []
        pathway_preds: List[float] = []

        if per_sample_sparsity is not None:
            # Per-trial vividness path — unique sparsity per prompt
            for prompt, sp in zip(prompts, per_sample_sparsity):
                feats, _ = self.run_batch([prompt], sp, tokenizer, num_repeats=1)
                pred = self.pathway_predictor(feats).squeeze(-1)  # [1]
                sparsity_gt.append(sp)
                pathway_preds.append(pred.item())
        else:
            # Grid path — batch all prompts at each sparsity level
            for sp in self.sparsity_levels:
                feats, _ = self.run_batch(prompts, sp, tokenizer, num_repeats=2)  # [N, 6]
                preds = self.pathway_predictor(feats).squeeze(-1)                 # [N]
                sparsity_gt.append(sp)
                pathway_preds.append(preds.mean().item())

        corr, _ = pearsonr(sparsity_gt, pathway_preds)
        return {
            "sparsity_levels": sparsity_gt,
            "pathway_preds": pathway_preds,
            "correlation": float(corr),
        }

    def experiment_2_eeg_to_sparsity(
        self,
        prompts: List[str],
        tokenizer: TextTokenizer,
        per_sample_sparsity: Optional[List[float]] = None,
    ) -> Dict:
        """EEG signal → predict sparsity."""
        sparsity_gt: List[float] = []
        eeg_preds: List[float] = []

        if per_sample_sparsity is not None:
            for prompt, sp in zip(prompts, per_sample_sparsity):
                _, eeg = self.run_batch([prompt], sp, tokenizer, num_repeats=1)
                pred = self.eeg_predictor(eeg).squeeze(-1)
                sparsity_gt.append(sp)
                eeg_preds.append(pred.item())
        else:
            for sp in self.sparsity_levels:
                _, eeg = self.run_batch(prompts, sp, tokenizer, num_repeats=2)  # [N, C]
                preds = self.eeg_predictor(eeg).squeeze(-1)                     # [N]
                sparsity_gt.append(sp)
                eeg_preds.append(preds.mean().item())

        corr, _ = pearsonr(sparsity_gt, eeg_preds)
        return {
            "sparsity_levels": sparsity_gt,
            "eeg_preds": eeg_preds,
            "correlation": float(corr),
        }

    def experiment_3_pathway_to_eeg_generalization(
        self,
        prompts: List[str],
        tokenizer: TextTokenizer,
        per_sample_sparsity: Optional[List[float]] = None,
    ) -> Dict:
        """Evaluate pathway and EEG predictors on the held-out set."""
        all_features: List[torch.Tensor] = []
        all_eeg: List[torch.Tensor] = []
        sparsity_gt: List[float] = []

        if per_sample_sparsity is not None:
            for prompt, sp in zip(prompts, per_sample_sparsity):
                feats, eeg = self.run_batch([prompt], sp, tokenizer, num_repeats=1)
                all_features.append(feats)
                all_eeg.append(eeg)
                sparsity_gt.append(sp)
        else:
            for sp in self.sparsity_levels:
                feats, eeg = self.run_batch(prompts, sp, tokenizer, num_repeats=1)
                all_features.append(feats)   # [N, 6]
                all_eeg.append(eeg)          # [N, C]
                sparsity_gt.extend([sp] * len(prompts))

        features_t = torch.cat(all_features, dim=0)   # [total, 6]
        eeg_t = torch.cat(all_eeg, dim=0)             # [total, C]

        pathway_preds = self.pathway_predictor(features_t).squeeze(-1).cpu().numpy()
        eeg_preds = self.eeg_predictor(eeg_t).squeeze(-1).cpu().numpy()

        pathway_corr, _ = pearsonr(pathway_preds, sparsity_gt)
        eeg_corr, _ = pearsonr(eeg_preds, sparsity_gt)

        return {
            "pathway_correlation": float(pathway_corr),
            "eeg_correlation": float(eeg_corr),
            "transfer_success_rate": float(eeg_corr / (pathway_corr + 1e-8)),
            "sparsity_levels": sparsity_gt,
            "pathway_predictions": pathway_preds.tolist(),
            "eeg_predictions": eeg_preds.tolist(),
        }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_all_experiments(
    sparse_model: SparseAttentionWrapper,
    eeg_projector: EEGProjector,
    pathway_predictor: MLPPredictor,
    eeg_predictor: MLPPredictor,
    sparsity_levels: List[float],
    num_prompts: int = 5,
    results_dir: str = "./results",
    prompts: Optional[List[str]] = None,
) -> Dict:
    """Run all three experiments and serialise results.

    Args:
        sparse_model, eeg_projector, pathway_predictor, eeg_predictor:
            trained model components.
        sparsity_levels: discrete grid (synthetic mode) or per-sample values
            from vividness normalisation (narrative mode).
        num_prompts: number of synthetic prompts when ``prompts`` is None.
        results_dir: output directory.
        prompts: optional list of real narrative prompts; when provided
            ``sparsity_levels`` must contain one entry per prompt.
    """
    tokenizer = TextTokenizer()

    if prompts is not None:
        per_sample_sparsity: Optional[List[float]] = list(sparsity_levels)
        runner_levels = sorted(set(per_sample_sparsity))
        print(
            f"Using {len(prompts)} real narrative prompts with per-trial "
            f"vividness-derived sparsity "
            f"[{min(per_sample_sparsity):.3f}, {max(per_sample_sparsity):.3f}]"
        )
    else:
        prompts = create_default_prompts(num_prompts)
        per_sample_sparsity = None
        runner_levels = sparsity_levels

    runner = ExperimentRunner(
        sparse_model, eeg_projector, pathway_predictor,
        eeg_predictor, runner_levels,
    )

    print("Running Experiment 1: Pathway → Sparsity...")
    exp1 = runner.experiment_1_pathway_to_sparsity(
        prompts, tokenizer, per_sample_sparsity=per_sample_sparsity
    )

    print("Running Experiment 2: EEG → Sparsity...")
    exp2 = runner.experiment_2_eeg_to_sparsity(
        prompts, tokenizer, per_sample_sparsity=per_sample_sparsity
    )

    print("Running Experiment 3: Pathway → EEG Transfer...")
    exp3 = runner.experiment_3_pathway_to_eeg_generalization(
        prompts, tokenizer, per_sample_sparsity=per_sample_sparsity
    )

    all_results = {
        "experiment_1_pathway_to_sparsity": exp1,
        "experiment_2_eeg_to_sparsity": exp2,
        "experiment_3_pathway_to_eeg_transfer": exp3,
    }

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(json.loads(json.dumps(all_results, default=float)), f, indent=2)
    print(f"Results saved to {results_path}")

    return all_results
