"""Experiments for testing the causal hypothesis."""
import torch
import numpy as np
from typing import Dict, List, Tuple
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
        """
        Initialize experiment runner.
        
        Args:
            sparse_model: Sparse attention model
            eeg_projector: EEG projector module
            pathway_predictor: Pathway predictor MLP
            eeg_predictor: EEG predictor MLP
            sparsity_levels: List of sparsity levels to test
        """
        self.sparse_model = sparse_model
        self.eeg_projector = eeg_projector
        self.pathway_predictor = pathway_predictor
        self.eeg_predictor = eeg_predictor
        self.sparsity_levels = sparsity_levels
        
        self.device = next(sparse_model.parameters()).device
    
    def run_single_level(
        self,
        prompt: str,
        sparsity_level: float,
        tokenizer: TextTokenizer,
        num_repeats: int = 3,
    ) -> Dict[str, np.ndarray]:
        """
        Run single sparsity level experiment.
        
        Args:
            prompt: Input text
            sparsity_level: Sparsity level to test
            tokenizer: Text tokenizer
            num_repeats: Number of repetitions
            
        Returns:
            Dictionary with results
        """
        self.sparse_model.set_sparsity_level(sparsity_level)
        self.sparse_model.eval()
        
        pathway_features_all = []
        eeg_signals_all = []
        
        with torch.no_grad():
            for _ in range(num_repeats):
                # Tokenize
                tokens = tokenizer.tokenize([prompt])
                input_ids = tokens['input_ids'].to(self.device)
                
                # Forward through sparse model
                output = self.sparse_model(
                    input_ids=input_ids,
                    return_attention_maps=True,
                    return_hidden_states=False,
                )
                
                # Extract attention and compute pathway features
                attention_maps = output['attention_maps']
                features = compute_pathway_features(attention_maps)
                pathway_features_all.append(features)
                
                # Project to EEG
                eeg_signal = self.eeg_projector(
                    features.unsqueeze(0).to(self.device),
                    training=False
                )
                eeg_signals_all.append(eeg_signal.squeeze(0))
        
        # Average across repeats
        pathway_features_mean = torch.stack(pathway_features_all).mean(dim=0)
        eeg_signals_mean = torch.stack(eeg_signals_all).mean(dim=0)
        
        return {
            'sparsity_level': sparsity_level,
            'pathway_features': pathway_features_mean.cpu().numpy(),
            'eeg_signal': eeg_signals_mean.cpu().numpy(),
        }
    
    def experiment_1_pathway_to_sparsity(
        self,
        prompts: List[str],
        tokenizer: TextTokenizer,
    ) -> Dict:
        """
        Experiment 1: Pathway metrics -> predict sparsity.
        
        Args:
            prompts: List of test prompts
            tokenizer: Text tokenizer
            
        Returns:
            Results dictionary
        """
        results = {
            'sparsity_levels': [],
            'pathway_preds': [],
            'correlations': [],
        }
        
        self.pathway_predictor.eval()
        
        for sparsity_level in self.sparsity_levels:
            preds = []
            
            for prompt in prompts:
                exp_result = self.run_single_level(
                    prompt, sparsity_level, tokenizer, num_repeats=2
                )
                
                features = torch.from_numpy(
                    exp_result['pathway_features']
                ).float().to(self.device)
                
                with torch.no_grad():
                    pred = self.pathway_predictor(features.unsqueeze(0))
                    preds.append(pred.item())
            
            results['sparsity_levels'].append(sparsity_level)
            results['pathway_preds'].append(np.mean(preds))
        
        # Compute correlation
        corr, _ = pearsonr(
            results['sparsity_levels'],
            results['pathway_preds']
        )
        results['correlation'] = corr
        
        return results
    
    def experiment_2_eeg_to_sparsity(
        self,
        prompts: List[str],
        tokenizer: TextTokenizer,
    ) -> Dict:
        """
        Experiment 2: EEG signal -> predict sparsity.
        
        Args:
            prompts: List of test prompts
            tokenizer: Text tokenizer
            
        Returns:
            Results dictionary
        """
        results = {
            'sparsity_levels': [],
            'eeg_preds': [],
            'correlations': [],
        }
        
        self.eeg_predictor.eval()
        
        for sparsity_level in self.sparsity_levels:
            preds = []
            
            for prompt in prompts:
                exp_result = self.run_single_level(
                    prompt, sparsity_level, tokenizer, num_repeats=2
                )
                
                eeg_signal = torch.from_numpy(
                    exp_result['eeg_signal']
                ).float().to(self.device)
                
                with torch.no_grad():
                    pred = self.eeg_predictor(eeg_signal.unsqueeze(0))
                    preds.append(pred.item())
            
            results['sparsity_levels'].append(sparsity_level)
            results['eeg_preds'].append(np.mean(preds))
        
        # Compute correlation
        corr, _ = pearsonr(
            results['sparsity_levels'],
            results['eeg_preds']
        )
        results['correlation'] = corr
        
        return results
    
    def experiment_3_pathway_to_eeg_generalization(
        self,
        prompts: List[str],
        tokenizer: TextTokenizer,
    ) -> Dict:
        """
        Experiment 3: Train on pathway features, test on EEG.
        
        This tests whether understanding pathway structure transfers to EEG.
        
        Args:
            prompts: List of test prompts
            tokenizer: Text tokenizer
            
        Returns:
            Results dictionary
        """
        # Collect pathway-EEG pairs
        pathway_features_all = []
        eeg_signals_all = []
        sparsity_all = []
        
        self.sparse_model.eval()
        
        with torch.no_grad():
            for sparsity_level in self.sparsity_levels:
                for prompt in prompts:
                    exp_result = self.run_single_level(
                        prompt, sparsity_level, tokenizer, num_repeats=1
                    )
                    
                    pathway_features_all.append(exp_result['pathway_features'])
                    eeg_signals_all.append(exp_result['eeg_signal'])
                    sparsity_all.append(sparsity_level)
        
        # Stack into tensors
        pathway_features_tensor = torch.from_numpy(
            np.stack(pathway_features_all)
        ).float()
        eeg_signals_tensor = torch.from_numpy(
            np.stack(eeg_signals_all)
        ).float()
        sparsity_tensor = torch.tensor(sparsity_all, dtype=torch.float32)
        
        # Evaluate on held-out data
        pathway_preds = []
        eeg_preds = []
        
        self.pathway_predictor.eval()
        self.eeg_predictor.eval()
        
        with torch.no_grad():
            pathway_preds = self.pathway_predictor(
                pathway_features_tensor.to(self.device)
            ).squeeze(-1).cpu().numpy()
            
            eeg_preds = self.eeg_predictor(
                eeg_signals_tensor.to(self.device)
            ).squeeze(-1).cpu().numpy()
        
        # Compute correlations
        pathway_corr, _ = pearsonr(pathway_preds, sparsity_all)
        eeg_corr, _ = pearsonr(eeg_preds, sparsity_all)
        transfer_success = eeg_corr / (pathway_corr + 1e-8)
        
        return {
            'pathway_correlation': pathway_corr,
            'eeg_correlation': eeg_corr,
            'transfer_success_rate': transfer_success,
            'sparsity_levels': sparsity_all,
            'pathway_predictions': pathway_preds.tolist(),
            'eeg_predictions': eeg_preds.tolist(),
        }


def run_all_experiments(
    sparse_model: SparseAttentionWrapper,
    eeg_projector: EEGProjector,
    pathway_predictor: MLPPredictor,
    eeg_predictor: MLPPredictor,
    sparsity_levels: List[float],
    num_prompts: int = 5,
    results_dir: str = "./results",
) -> Dict:
    """
    Run all three experiments.
    
    Args:
        sparse_model: Sparse attention model
        eeg_projector: EEG projector
        pathway_predictor: Pathway predictor
        eeg_predictor: EEG predictor
        sparsity_levels: List of sparsity levels
        num_prompts: Number of prompts to test
        results_dir: Directory to save results
        
    Returns:
        Dictionary with all results
    """
    # Setup
    tokenizer = TextTokenizer()
    prompts = create_default_prompts(num_prompts)
    runner = ExperimentRunner(
        sparse_model, eeg_projector, pathway_predictor,
        eeg_predictor, sparsity_levels
    )
    
    # Run experiments
    print("Running Experiment 1: Pathway -> Sparsity...")
    exp1_results = runner.experiment_1_pathway_to_sparsity(prompts, tokenizer)
    
    print("Running Experiment 2: EEG -> Sparsity...")
    exp2_results = runner.experiment_2_eeg_to_sparsity(prompts, tokenizer)
    
    print("Running Experiment 3: Pathway -> EEG Transfer...")
    exp3_results = runner.experiment_3_pathway_to_eeg_generalization(prompts, tokenizer)
    
    # Combine results
    all_results = {
        'experiment_1_pathway_to_sparsity': exp1_results,
        'experiment_2_eeg_to_sparsity': exp2_results,
        'experiment_3_pathway_to_eeg_transfer': exp3_results,
    }
    
    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / "experiment_results.json"
    
    # Convert numpy types for JSON serialization
    results_serializable = json.loads(json.dumps(all_results, default=float))
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return all_results
