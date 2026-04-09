"""Dataset for causal hypothesis testing."""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import numpy as np


class SparsityDataset(Dataset):
    """Dataset for testing sparsity-to-computation mapping."""
    
    def __init__(
        self,
        prompts: List[str],
        sparsity_levels: List[float],
        samples_per_sparsity: int = 5,
    ):
        """
        Initialize dataset.
        
        Args:
            prompts: Fixed list of prompts (content invariant)
            sparsity_levels: List of sparsity levels to test
            samples_per_sparsity: Number of samples per sparsity level
        """
        self.prompts = prompts
        self.sparsity_levels = sparsity_levels
        self.samples_per_sparsity = samples_per_sparsity
        
        # Create samples: (prompt_idx, sparsity_level)
        self.samples = []
        for sparsity_idx, sparsity in enumerate(sparsity_levels):
            for sample_idx in range(samples_per_sparsity):
                prompt_idx = sample_idx % len(prompts)
                self.samples.append({
                    'prompt_idx': prompt_idx,
                    'sparsity_level': sparsity,
                    'sparsity_idx': sparsity_idx,
                })
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'text' and 'sparsity_level'
        """
        sample = self.samples[idx]
        prompt_idx = sample['prompt_idx']
        sparsity_level = sample['sparsity_level']
        
        return {
            'text': self.prompts[prompt_idx],
            'sparsity_level': torch.tensor(sparsity_level, dtype=torch.float32),
            'prompt_idx': prompt_idx,
        }


def create_default_prompts(num_prompts: int = 10) -> List[str]:
    """
    Create fixed set of prompts for testing.
    
    Args:
        num_prompts: Number of prompts to create
        
    Returns:
        List of prompts
    """
    base_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep neural networks learn hierarchical representations of data.",
        "Attention mechanisms have revolutionized modern NLP models.",
        "Transformers process sequences in parallel, improving training efficiency.",
        "Sparse attention patterns can reduce computational complexity.",
        "EEG signals reflect electrical activity of the brain.",
        "Neural correlates of consciousness remain an active research area.",
        "Causal inference in neuroscience requires careful experimental design.",
    ]
    
    if num_prompts <= len(base_prompts):
        return base_prompts[:num_prompts]
    
    # Repeat if needed
    result = base_prompts.copy()
    while len(result) < num_prompts:
        result.extend(base_prompts)
    return result[:num_prompts]


def create_dataloader(
    prompts: List[str],
    sparsity_levels: List[float],
    samples_per_sparsity: int = 5,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create dataloader for sparsity dataset.
    
    Args:
        prompts: List of prompts
        sparsity_levels: List of sparsity levels
        samples_per_sparsity: Number of samples per sparsity level
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = SparsityDataset(prompts, sparsity_levels, samples_per_sparsity)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
