"""Tokenization and data collation utilities."""
import torch
from typing import Dict, List
from transformers import AutoTokenizer


class TextTokenizer:
    """Wrapper for HuggingFace tokenizer."""
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
        }


def collate_text_batch(batch: List[Dict], tokenizer: TextTokenizer) -> Dict[str, torch.Tensor]:
    """
    Collate batch with tokenization.
    
    Args:
        batch: List of samples from dataset
        tokenizer: TextTokenizer instance
        
    Returns:
        Collated batch dictionary
    """
    texts = [sample['text'] for sample in batch]
    sparsity_levels = torch.stack([sample['sparsity_level'] for sample in batch])
    
    # Tokenize
    tokenized = tokenizer.tokenize(texts)
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'sparsity_level': sparsity_levels,
    }


def get_collate_fn(model_name: str = "gpt2", max_length: int = 512):
    """
    Get collate function for DataLoader.
    
    Args:
        model_name: HuggingFace model identifier
        max_length: Maximum sequence length
        
    Returns:
        Collate function
    """
    tokenizer = TextTokenizer(model_name, max_length)
    
    def collate_fn(batch):
        return collate_text_batch(batch, tokenizer)
    
    return collate_fn
